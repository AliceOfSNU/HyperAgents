"""Host-side half of the branching skill (see skills/branching/SKILL.md for
the meta-agent-facing picture). Processes pending requests in
branch_requests.jsonl: probe k branches cheaply, judge which ones actually
diverge, extend the survivors to a real budget, write a report.

Runs with the same trust boundary as domains/deep_swe/harness.py's own
evaluation runs -- full docker/Pier access, invoked from generate_loop.py
after a generation's own meta-agent session ends (see that module's own
process_pending_branch_requests() call site). The meta-agent's own
container never runs any of this directly -- see request_branch.py and
this module's own docstring in generate_loop.py's wiring for why.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from agent.llm import DEEPSEEK_FLASH_MODEL, get_response_from_llm
from agent.trajectory_replay import load_trajectory_rounds
from utils.common import extract_jsons

from .harness import _reconstruct_code  # noqa: reused deliberately, see _build_code_root_dir's docstring

K_HARD_CAP = 5
PROBE_MAX_TOOL_CALLS = 2
FULL_MAX_TOOL_CALLS = 40  # per surviving branch
TOTAL_FULL_ROLLOUT_BUDGET = 3  # max surviving branches that get a full rollout, regardless of judge output
PROBE_AGENT_TIMEOUT_MULTIPLIER = 0.15
FULL_AGENT_TIMEOUT_MULTIPLIER = 0.3
FULL_VERIFIER_TIMEOUT_MULTIPLIER = 1.0
BRANCH_TEMPERATURE = 0.7

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUESTS_PATH = REPO_ROOT / "branch_requests.jsonl"
REPORTS_DIR = REPO_ROOT / "skills" / "branching" / "reports"


def _log(message):
    print(f"[branching] {message}", file=sys.stderr, flush=True)


def _load_pending_requests():
    if not REQUESTS_PATH.exists():
        return []
    requests = []
    for line in REQUESTS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            requests.append(json.loads(line))
        except json.JSONDecodeError:
            _log(f"skipping malformed line in {REQUESTS_PATH}: {line[:200]}")
    return requests


def _already_processed(request_id):
    return (REPORTS_DIR / request_id / "report.json").exists()


def _translate_container_path(container_path, run_output_dir):
    """Meta-agent containers see eval_path copied in at
    /tmp/<basename(run_output_dir)> (see generate_loop.py's own
    copy_to_container call for eval_path) -- translate that container-local
    view back to the real host path this process can actually read."""
    basename = os.path.basename(os.path.normpath(run_output_dir))
    container_prefix = f"/tmp/{basename}"
    if container_path.startswith(container_prefix):
        return str(run_output_dir).rstrip("/") + container_path[len(container_prefix):]
    return container_path  # already host-relative/absolute (e.g. a manually queued/test request)


_GENID_RE = re.compile(r"/gen_(initial|\d+)/deep_swe_eval/")


def _parse_run_and_genid(chat_history_host_path):
    """Everything about which run/generation a trial belongs to is encoded
    in its own path (outputs/generate_<run_id>/gen_<N>/deep_swe_eval/eval/
    <trial>/agent/chat_history.json) -- more reliable than the trial's own
    config.json, whose agent.kwargs.root_dir points at a tempfile.
    TemporaryDirectory that's long gone by the time this runs (see
    _build_code_root_dir)."""
    match = _GENID_RE.search(chat_history_host_path)
    if not match:
        raise ValueError(f"Could not find /gen_<N>/deep_swe_eval/ in path: {chat_history_host_path}")
    genid_str = match.group(1)
    genid = "initial" if genid_str == "initial" else int(genid_str)
    run_output_dir = chat_history_host_path[:match.start()]
    return run_output_dir, genid


def _load_task_context(chat_history_host_path):
    trial_dir = Path(chat_history_host_path).resolve().parents[1]  # .../<trial>/agent/chat_history.json -> .../<trial>
    config = json.loads((trial_dir / "config.json").read_text(encoding="utf-8"))
    task_path = config["task"]["path"]
    instruction = (Path(task_path) / "instruction.md").read_text(encoding="utf-8")
    return {
        "task_path": task_path,
        "task_id": Path(task_path).name,
        "model": config["agent"]["model_name"],
        "instruction": instruction,
    }


def _build_code_root_dir(run_output_dir, genid, dest_dir):
    """Reconstruct the exact code snapshot that generation actually ran with
    -- config.json's own agent.kwargs.root_dir is a tempfile.
    TemporaryDirectory harness.py's compute_node_utility already cleaned up
    by the time this runs later, so it can't be reused directly. Reuses
    harness.py's own _reconstruct_code (same BASELINE_FILES + patch-chain
    replay real evaluation already relies on) rather than duplicating it."""
    root_dir = os.path.join(run_output_dir, "gen_initial", "hyperagents")
    _reconstruct_code(root_dir, run_output_dir, genid, dest_dir)


def _pier_run(cmd_extra, jobs_dir, job_name, timeout_sec=1800):
    cmd = [
        "pier", "run",
        "--jobs-dir", str(jobs_dir),
        "--job-name", job_name,
        "--quiet",
    ] + cmd_extra
    hyperagents_root = str(REPO_ROOT)
    run_env = {**os.environ, "PYTHONPATH": hyperagents_root + os.pathsep + os.environ.get("PYTHONPATH", "")}
    _log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=hyperagents_root, env=run_env, capture_output=True, text=True, timeout=timeout_sec)
    if result.returncode != 0:
        _log(f"pier run ({job_name}) exited {result.returncode}. stderr tail:\n{result.stderr[-3000:]}")
    return result


def _load_branch_trajectories(jobs_dir, job_name):
    """Every attempt under jobs_dir/job_name/*/agent/chat_history.json --
    -k/--n-attempts produces one trial dir per attempt, each independently
    (and randomly) named. Order isn't meaningful; callers treat these as an
    unordered set of k branches."""
    job_dir = Path(jobs_dir) / job_name
    branches = []
    if not job_dir.exists():
        return branches
    for trial_dir in sorted(job_dir.iterdir()):
        chat_history_path = trial_dir / "agent" / "chat_history.json"
        if not chat_history_path.exists():
            continue
        try:
            rounds = load_trajectory_rounds(chat_history_path)
        except Exception as e:
            _log(f"failed to load {chat_history_path}: {e}")
            continue
        branches.append({"trial_dir": trial_dir, "rounds": rounds})
    return branches


def _judge_probes(instruction, note, branches, model=DEEPSEEK_FLASH_MODEL):
    """Ask an LLM which probe continuations represent genuinely different
    strategies (worth a full rollout) vs redundant restatements of the same
    action. Returns the list of branch indices to keep, capped at
    TOTAL_FULL_ROLLOUT_BUDGET regardless of how many the judge picks --
    that cap is enforced by the caller, not trusted from the judge's own
    output."""
    if len(branches) <= 1:
        return list(range(len(branches)))

    summaries = []
    for i, b in enumerate(branches):
        actions = []
        for r in b["rounds"]:
            for call in r.get("act") or []:
                actions.append(f"{call['name']}({call['arguments']})")
        summaries.append(f"Branch {i}:\n" + "\n".join(actions[:10]) or f"Branch {i}: (no actions taken)")

    prompt = f"""A coding agent is partway through this task:
{instruction[:2000]}

The investigator's own note on why this branch point was chosen: {note}

{len(branches)} independent continuations were sampled from the exact same starting point (same prefix, same environment), each given a small budget to take a first move. Here is what each one did:

{chr(10).join(summaries)}

Identify which continuations represent genuinely DIFFERENT strategies worth exploring further with a full budget, versus which are redundant restatements of essentially the same action as another branch already in the list (keep only one representative of each distinct strategy). Be conservative -- if in doubt whether two branches differ meaningfully, treat them as the same strategy and drop the redundant one, since a full rollout costs real budget.

Respond in JSON format:
<json>
{{"keep_indices": [list of branch indices (integers) to keep], "reasoning": "one sentence per kept index explaining what's distinct about it"}}
</json>"""

    try:
        response, _, _ = get_response_from_llm(prompt, model=model, temperature=0.0)
        parsed_list = extract_jsons(response)
        parsed = parsed_list[0] if parsed_list else None
        if parsed is None:
            # extract_jsons only matches <json>/```json fenced blocks; this
            # model doesn't reliably use the wrapper for a plain-text judge
            # call (no native tool-calling involved here) -- confirmed live,
            # it sometimes just returns bare JSON. Fall back to parsing the
            # whole response before giving up.
            try:
                parsed = json.loads(response.strip())
            except json.JSONDecodeError:
                parsed = None
        if parsed is None:
            _log(f"judge response had no parseable JSON, keeping all branches as fallback: {response[:500]}")
            return list(range(len(branches)))
        keep = parsed.get("keep_indices", [])
        keep = [i for i in keep if isinstance(i, int) and 0 <= i < len(branches)]
        return keep or list(range(len(branches)))
    except Exception as e:
        _log(f"judge call failed ({e}), keeping all branches as fallback")
        return list(range(len(branches)))


def _extend_rounds(prefix_rounds, branch_round, probe_rounds):
    """Concatenate the original prefix (1..branch_round) with a surviving
    probe's own new rounds, renumbered to continue from branch_round+1 --
    so the full-rollout phase genuinely continues the probed direction
    (extends it) rather than resampling a fresh, unrelated draw from the
    same branch point. See SKILL.md's honesty section: this is the one
    place fidelity to "the specific thing the probe did" actually matters,
    since the judge's reasoning was about that specific probe."""
    extended = [r for r in prefix_rounds if r["round"] <= branch_round]
    next_round_num = branch_round + 1
    for r in probe_rounds:
        extended.append({**r, "round": next_round_num})
        next_round_num += 1
    return extended, next_round_num - 1


def _probe_outcome_summary(rounds):
    actions = sum(len(r.get("act") or []) for r in rounds)
    final_plan = next((r.get("plan") for r in reversed(rounds) if r.get("plan")), "")
    return {"actions_taken": actions, "final_plan_excerpt": (final_plan or "")[:300]}


def process_one_request(request):
    request_id = request["request_id"]
    report_dir = REPORTS_DIR / request_id
    report = {
        "request": request,
        "branches": [],
        "dropped_for_budget": [],
        "status": "failed",
        "error": None,
    }

    dest_dir = None
    try:
        run_output_dir, genid = _parse_run_and_genid(request["source_chat_history"])
        host_chat_history = _translate_container_path(request["source_chat_history"], run_output_dir)
        task_ctx = _load_task_context(host_chat_history)
        prefix_rounds = load_trajectory_rounds(host_chat_history)

        k = min(request.get("k", 3), K_HARD_CAP)

        dest_dir = tempfile.mkdtemp(prefix=f"branch_{request_id}_")
        _build_code_root_dir(run_output_dir, genid, dest_dir)

        with tempfile.TemporaryDirectory(prefix=f"branch_jobs_{request_id}_") as jobs_dir:
            # --- Probe phase: cheap, k attempts, no verification ---
            probe_result = _pier_run(
                [
                    "-p", task_ctx["task_path"],
                    "--agent-import-path", "domains.deep_swe.branch_pier_agent:BranchRolloutPierAgent",
                    "-m", task_ctx["model"],
                    "--ak", f"root_dir={dest_dir}",
                    "--ak", f"source_chat_history_path={host_chat_history}",
                    "--ak", f"branch_round={request['branch_round']}",
                    "--ak", f"continuation_max_tool_calls={PROBE_MAX_TOOL_CALLS}",
                    "--ak", f"temperature={BRANCH_TEMPERATURE}",
                    "-e", "docker",
                    "-k", str(k),
                    "-n", str(min(k, 4)),
                    "--agent-timeout-multiplier", str(PROBE_AGENT_TIMEOUT_MULTIPLIER),
                    "--disable-verification",
                ],
                jobs_dir=jobs_dir, job_name="probe",
            )
            probe_branches = _load_branch_trajectories(jobs_dir, "probe")
            if not probe_branches:
                report["error"] = f"probe phase produced no branch trajectories (pier exit {probe_result.returncode})"
                report["status"] = "failed"
                _write_report(report_dir, report)
                return report

            keep_indices = _judge_probes(task_ctx["instruction"], request.get("note", ""), probe_branches)
            if len(keep_indices) > TOTAL_FULL_ROLLOUT_BUDGET:
                report["dropped_for_budget"] = [
                    {"branch": i, "reason": f"judge kept {len(keep_indices)} branches, cap is {TOTAL_FULL_ROLLOUT_BUDGET}"}
                    for i in keep_indices[TOTAL_FULL_ROLLOUT_BUDGET:]
                ]
                keep_indices = keep_indices[:TOTAL_FULL_ROLLOUT_BUDGET]

            dropped_indices = [i for i in range(len(probe_branches)) if i not in keep_indices]
            for i in dropped_indices:
                report["branches"].append({
                    "index": i, "kept_for_full_rollout": False,
                    "probe_summary": _probe_outcome_summary(probe_branches[i]["rounds"]),
                })

            # --- Full rollout: only for judge-selected, budget-capped survivors ---
            for i in keep_indices:
                extended_rounds, extended_branch_round = _extend_rounds(
                    prefix_rounds, request["branch_round"], probe_branches[i]["rounds"],
                )
                extended_source_path = Path(jobs_dir) / f"extended_source_{i}.json"
                extended_source_path.write_text(json.dumps(extended_rounds), encoding="utf-8")

                with tempfile.TemporaryDirectory(prefix=f"branch_full_jobs_{request_id}_{i}_") as full_jobs_dir:
                    full_result = _pier_run(
                        [
                            "-p", task_ctx["task_path"],
                            "--agent-import-path", "domains.deep_swe.branch_pier_agent:BranchRolloutPierAgent",
                            "-m", task_ctx["model"],
                            "--ak", f"root_dir={dest_dir}",
                            "--ak", f"source_chat_history_path={extended_source_path}",
                            "--ak", f"branch_round={extended_branch_round}",
                            "--ak", f"continuation_max_tool_calls={FULL_MAX_TOOL_CALLS}",
                            "--ak", f"temperature={BRANCH_TEMPERATURE}",
                            "-e", "docker",
                            "--agent-timeout-multiplier", str(FULL_AGENT_TIMEOUT_MULTIPLIER),
                            "--verifier-timeout-multiplier", str(FULL_VERIFIER_TIMEOUT_MULTIPLIER),
                        ],
                        jobs_dir=full_jobs_dir, job_name="full",
                    )
                    full_branches = _load_branch_trajectories(full_jobs_dir, "full")
                    branch_report = {"index": i, "kept_for_full_rollout": True, "probe_summary": _probe_outcome_summary(probe_branches[i]["rounds"])}
                    if full_branches:
                        branch_dir = report_dir / f"branch_{i}"
                        branch_dir.mkdir(parents=True, exist_ok=True)
                        src_trial_dir = full_branches[0]["trial_dir"]
                        for fname in ("chat_history.json", "chat_history.md"):
                            src = src_trial_dir / "agent" / fname
                            if src.exists():
                                (branch_dir / fname).write_bytes(src.read_bytes())
                        branch_report["trajectory_dir"] = str(branch_dir)
                        branch_report["full_rollout_summary"] = _probe_outcome_summary(full_branches[0]["rounds"])
                        result_path = src_trial_dir / "result.json"
                        if result_path.exists():
                            try:
                                branch_report["reward"] = json.loads(result_path.read_text()).get("verifier_result", {}).get("rewards")
                            except Exception:
                                pass
                    else:
                        branch_report["error"] = f"full rollout produced no trajectory (pier exit {full_result.returncode})"
                    report["branches"].append(branch_report)

        report["status"] = "done"
    except Exception as e:
        report["error"] = str(e)
        report["status"] = "failed"
        _log(f"request {request_id} failed: {e}")
    finally:
        if dest_dir and os.path.isdir(dest_dir):
            shutil.rmtree(dest_dir, ignore_errors=True)

    _write_report(report_dir, report)
    return report


def _write_report(report_dir, report):
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def process_pending_branch_requests():
    """Entry point called from generate_loop.py after a generation's own
    meta-agent session ends. Silently a no-op (not even a log line) when
    there's nothing to do -- this runs every generation regardless of
    whether the meta-agent ever used the skill."""
    pending = [r for r in _load_pending_requests() if not _already_processed(r["request_id"])]
    if not pending:
        return
    _log(f"processing {len(pending)} pending branch request(s)")
    for request in pending:
        process_one_request(request)


if __name__ == "__main__":
    process_pending_branch_requests()
