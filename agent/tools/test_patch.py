"""Meta-agent-facing half of the patch-testing skill. Runs inside the
meta-agent's own (sandboxed, no docker) container -- captures the current
session's own diff to swe_task_agent.py, queues a request, then blocks,
polling for the host-side orchestrator's report. See
skills/patch_testing/SKILL.md for the full picture and
domains/deep_swe/patch_test_orchestrator.py for what actually runs the
replicates.
"""
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

# Resolved relative to this file's own location, same pattern as
# agent/tools/compare_generations.py's TASK_LANGUAGES_PATH and
# skills/branching/request_branch.py's REQUESTS_PATH -- inside the
# container this resolves to /hyperagents, the live-bind-mounted root_dir,
# so a write here is immediately visible host-side and vice versa (no copy
# step), which is what makes the synchronous poll-and-block below work at
# all despite this container having no docker access of its own.
REPO_ROOT = Path(__file__).resolve().parents[2]
REQUESTS_DIR = REPO_ROOT / "skills" / "patch_testing" / "requests"
REPORTS_DIR = REPO_ROOT / "skills" / "patch_testing" / "reports"

REPLICATES_HARD_CAP = 5
POLL_INTERVAL_SEC = 15
# Real margin above the ~20 minute expected runtime for a LONE primary
# checkpoint -- but patch_test_orchestrator.py processes checkpoints
# sequentially (control then treatment, one checkpoint after another, to
# avoid stacking too much concurrent Docker load), so wall-clock scales
# linearly with checkpoint count: primary + a couple regression checkpoints
# realistically approaches an hour, not ~20 minutes. Doubled from the
# original 40 minutes so timing out stays a rare safety-valve case (a
# genuinely stuck orchestrator) rather than the routine outcome for any
# request that actually uses regression_checkpoints/n_random_regression.
MAX_WAIT_SEC = 80 * 60


def tool_info():
    return {
        "name": "test_patch",
        "description": """Test your current patch to swe_task_agent.py against a real historical checkpoint before deciding whether to keep it -- reconstructs that checkpoint's own original code and environment, runs it unmodified (control) and with your patch applied on top of that SAME original code (treatment), several times each to full completion, so you can see whether your patch actually causes the change you expect and whether it breaks anything else you didn't intend to touch.

Blocks synchronously for the whole run (order of ~20 minutes) -- this is deliberate, not a bug; see skills/patch_testing/SKILL.md for why. You must state your goal (what you expect to change and why) -- this is what you should compare the actual outcome against once you have the report, and what a memory note about the outcome should cite.

Optionally test additional "regression" checkpoints from elsewhere in the lineage to catch unintended side effects -- pass explicit ones, a count to sample automatically, or both. A checkpoint whose code the patch doesn't cleanly apply to is reported as skipped, not silently misapplied or treated as a failure.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "evals_folder": {
                    "type": "string",
                    "description": "Path to the run's output directory, as you see it (e.g. the /tmp/generate_... path named in your own instructions).",
                },
                "primary_checkpoint": {
                    "type": "object",
                    "description": "The round to test your patch against.",
                    "properties": {
                        "genid": {"type": ["string", "integer"]},
                        "task_id": {"type": "string"},
                        "round": {"type": "integer"},
                    },
                    "required": ["genid", "task_id", "round"],
                },
                "goal": {
                    "type": "string",
                    "description": "What you expect this patch to change at the primary checkpoint, and why -- required, compared against the actual outcome once the report is back.",
                },
                "regression_checkpoints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "genid": {"type": ["string", "integer"]},
                            "task_id": {"type": "string"},
                            "round": {"type": "integer"},
                        },
                        "required": ["genid", "task_id", "round"],
                    },
                    "description": "Optional explicit additional checkpoints to sanity-check against, same shape as primary_checkpoint.",
                },
                "n_random_regression": {
                    "type": "integer",
                    "description": "Optional count of additional checkpoints to sample automatically across the lineage.",
                },
                "replicates_per_arm": {
                    "type": "integer",
                    "default": 3,
                    "description": f"Replicates of each arm (control, treatment) per checkpoint, hard-capped at {REPLICATES_HARD_CAP}.",
                },
            },
            "required": ["evals_folder", "primary_checkpoint", "goal"],
        },
    }


def _current_patch():
    """git diff of swe_task_agent.py against this session's own starting
    commit (see run_meta_agent.py's META_AGENT_BASE_COMMIT) -- the same
    base commit model_patch.diff itself gets computed against at the end of
    this run, just read mid-session here instead."""
    base_commit = os.environ.get("META_AGENT_BASE_COMMIT")
    if not base_commit:
        return None, "META_AGENT_BASE_COMMIT is not set -- can't determine this session's own starting commit."
    try:
        result = subprocess.run(
            ["git", "diff", base_commit, "--", "swe_task_agent.py"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30,
        )
    except Exception as e:
        return None, f"Could not compute the current patch: {e}"
    if result.returncode != 0:
        return None, f"git diff failed: {result.stderr}"
    if not result.stdout.strip():
        return None, "No changes to swe_task_agent.py yet this session -- write the patch first, then call test_patch."
    return result.stdout, None


def _format_report(report):
    lines = [f"Patch test {report.get('request_id')} -- status: {report.get('status')}"]
    if report.get("error"):
        lines.append(f"Error: {report['error']}")
    lines.append(f"Goal: {report.get('goal')}")
    lines.append("")
    for cp in report.get("checkpoints", []):
        lines.append(f"[{cp.get('role')}] gen_{cp.get('genid')} / {cp.get('task_id')} round {cp.get('round')}")
        if not cp.get("patch_applies", True):
            lines.append(f"  patch does not apply cleanly to this checkpoint's own code -- skipped ({cp.get('patch_apply_detail', '')})")
            lines.append("")
            continue
        control = cp.get("control") or []
        treatment = cp.get("treatment") or []
        lines.append(f"  control  (original code, {len(control)} replicate(s)):")
        for r in control:
            lines.append(f"    reward={r.get('reward')}  {r.get('detail', '')}")
        lines.append(f"  treatment (patch applied, {len(treatment)} replicate(s)):")
        for r in treatment:
            lines.append(f"    reward={r.get('reward')}  {r.get('detail', '')}")
        lines.append("")
    return "\n".join(lines)


def tool_function(evals_folder, primary_checkpoint, goal, regression_checkpoints=None, n_random_regression=0, replicates_per_arm=3):
    if not goal or not str(goal).strip():
        return "Error: goal is required -- state what you expect this patch to change at the primary checkpoint, and why."

    patch_text, error = _current_patch()
    if error:
        return f"Error: {error}"

    request_id = uuid.uuid4().hex[:12]
    request = {
        "request_id": request_id,
        "queued_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "evals_folder": evals_folder,
        "primary_checkpoint": primary_checkpoint,
        "goal": goal,
        "regression_checkpoints": regression_checkpoints or [],
        "n_random_regression": max(0, int(n_random_regression or 0)),
        "replicates_per_arm": max(1, min(int(replicates_per_arm or 3), REPLICATES_HARD_CAP)),
        "patch_text": patch_text,
        "status": "pending",
    }

    try:
        REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
        (REQUESTS_DIR / f"{request_id}.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    except Exception as e:
        return f"Error: could not queue request: {e}"

    report_path = REPORTS_DIR / request_id / "report.json"
    waited = 0
    while waited < MAX_WAIT_SEC:
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                return _format_report(report)
            except Exception:
                pass  # partially-written file (report writer isn't atomic against a mid-write read) -- keep waiting
        time.sleep(POLL_INTERVAL_SEC)
        waited += POLL_INTERVAL_SEC

    return (
        f"test_patch request {request_id} is still running after {MAX_WAIT_SEC // 60} minutes -- no result yet. "
        f"Check back later this session or a future one: the report will appear at "
        f"skills/patch_testing/reports/{request_id}/report.json once it's done."
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("evals_folder")
    parser.add_argument("genid")
    parser.add_argument("task_id")
    parser.add_argument("round", type=int)
    parser.add_argument("goal")
    args = parser.parse_args()
    print(tool_function(
        args.evals_folder,
        {"genid": args.genid, "task_id": args.task_id, "round": args.round},
        args.goal,
    ))
