"""Harness for the deep-swe domain: drives the `pier` CLI (github.com/
datacurve-ai/pier) against a fixed scoring subset of deep-swe tasks, using
domains/deep_swe/pier_agent.py to run HyperAgents' own swe_task_agent.py
inside each task's container. Scoring is real test execution (F2P/P2P via
Pier's own verifier), not an LLM judge -- see domains/deep_swe/config.py's
docstring for the full picture, including why network access must stay
scoped to only the LLM provider's API domain.

Node utility per generation: reconstruct swe_task_agent.py (and the shared
agent/, utils/ code) exactly as of `genid` by replaying its patch chain
host-side (no container needed for this -- it's plain text patching), then
run the whole scoring subset through `pier run` in one job. node_utility is
the mean `reward` (SWE-bench-style binary resolved/not) across the subset.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from utils.gl_utils import get_patch_files, update_node_metadata

from .config import (
    AGENT_MODEL,
    AGENT_TIMEOUT_MULTIPLIER,
    BASELINE_FILES,
    DEEPSWE_TASKS_DIR,
    SCORING_SUBSET_PATH,
    VERIFIER_TIMEOUT_MULTIPLIER,
)

# Host-side only: nothing else in this module imports litellm (that only
# happens inside the per-task containers, via the uploaded swe_task_agent.py),
# so DEEPSEEK_API_KEY/ANTHROPIC_API_KEY from .env wouldn't otherwise reach
# os.environ before `pier run` is launched as a subprocess -- same reasoning
# as domains/research/harness.py's own load_dotenv() call.
load_dotenv()

N_CONCURRENT = 4


def _log(message):
    print(f"[deep_swe] {message}", file=sys.stderr, flush=True)


def _reconstruct_code(root_dir, output_dir, genid, dest_dir):
    """Copy BASELINE_FILES from root_dir into dest_dir, then replay genid's
    own patch chain on top -- the host-side equivalent of domains/research/
    harness.py's _copy_baseline_code + _apply_patch_chain, just without a
    container (patch application is plain text manipulation either way)."""
    for rel in BASELINE_FILES:
        src = Path(root_dir) / rel.rstrip("/")
        if not src.exists():
            continue
        dst = Path(dest_dir) / rel.rstrip("/")
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    for patch_file in get_patch_files(output_dir, genid):
        with open(patch_file, "rb") as f:
            result = subprocess.run(
                ["patch", "-p1", "-f"], cwd=dest_dir, stdin=f,
                capture_output=True, text=False,
            )
        if result.returncode != 0:
            _log(
                f"patch {patch_file} applied with errors (tolerated, matching "
                f"the -f/force behavior other domains already rely on): "
                f"{result.stdout.decode(errors='ignore')[:500]}"
            )

    _fetch_wheels(dest_dir)


def _fetch_wheels(dest_dir):
    """Pre-download requirements-deep_swe.txt as manylinux wheels, host-side
    (full internet access here, no restrictions) -- confirmed live that
    `pip install` *inside* a task container fails silently: the agent's
    network_allowlist() only opens the LLM provider's own API domain, never
    pypi.org, and that's deliberate (see config.py's docstring) -- widening
    it to cover pip during the agent's own run phase would hand the
    interactive coding agent a disguised general-purpose internet channel
    (e.g. `pip install --index-url <arbitrary-url>`). pier_agent.py's
    setup() installs from these wheels with `--no-index`, so the container
    never needs network access for its own dependencies at all."""
    req_path = Path(dest_dir) / "requirements-deep_swe.txt"
    if not req_path.exists():
        return
    wheels_dir = Path(dest_dir) / "_wheels"
    wheels_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3", "-m", "pip", "download",
        "-r", str(req_path), "-d", str(wheels_dir),
        "--only-binary=:all:", "--python-version", "3.12",
        "--platform", "manylinux2014_x86_64", "--implementation", "cp",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        _log(f"pip download failed (agent container will be missing deps): {result.stderr[-2000:]}")


def _load_trial_results(jobs_dir, job_name):
    """Read every trial's own result.json under jobs_dir/job_name/ -- more
    reliable than the aggregate result.json's stats.evals[...].metrics[]
    array, which isn't keyed by task_id at all."""
    job_dir = Path(jobs_dir) / job_name
    records = []
    if not job_dir.exists():
        return records
    for trial_dir in sorted(job_dir.iterdir()):
        result_path = trial_dir / "result.json"
        if not trial_dir.is_dir() or not result_path.exists():
            continue
        try:
            trial = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as e:
            _log(f"Failed to parse {result_path}: {e}")
            continue

        # task_name comes back as "datacurve/<task-id>" (Harbor's [task] name
        # field) -- strip the prefix so it matches scoring_subset.json's
        # plain task ids exactly.
        task_id = (trial.get("task_name") or trial_dir.name).rsplit("/", 1)[-1]
        rewards = ((trial.get("verifier_result") or {}).get("rewards")) or {}
        exception = trial.get("exception_info")
        records.append({
            "task_id": task_id,
            "task_agent_ok": exception is None,
            "reward": float(rewards.get("reward", 0.0)),
            "partial": float(rewards.get("partial", 0.0)),
            "f2p": rewards.get("f2p"),
            "p2p": rewards.get("p2p"),
            "f2p_total": rewards.get("f2p_total"),
            "p2p_total": rewards.get("p2p_total"),
            "exception": (exception or {}).get("exception_message"),
        })
    return records


def compute_node_utility(root_dir, output_dir, genid, model=AGENT_MODEL, max_workers=N_CONCURRENT):
    """Run the candidate swe_task_agent.py (as of `genid`) on the fixed
    scoring subset via a single `pier run` job, score with real test
    execution, and return (node_utility, per_task_records)."""
    scoring_subset = json.loads(SCORING_SUBSET_PATH.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix=f"deep_swe_gen{genid}_") as dest_dir:
        _reconstruct_code(root_dir, output_dir, genid, dest_dir)

        jobs_dir = Path(output_dir) / f"gen_{genid}" / "deep_swe_eval"
        job_name = "eval"
        jobs_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "pier", "run",
            "-p", str(DEEPSWE_TASKS_DIR),
            "--agent-import-path", "domains.deep_swe.pier_agent:HyperAgentsSweAgent",
            "-m", model,
            "--ak", f"root_dir={dest_dir}",
            "-e", "docker",
            "--jobs-dir", str(jobs_dir),
            "--job-name", job_name,
            "-n", str(max_workers),
            "--agent-timeout-multiplier", str(AGENT_TIMEOUT_MULTIPLIER),
            "--verifier-timeout-multiplier", str(VERIFIER_TIMEOUT_MULTIPLIER),
            "--quiet",
        ]
        for task_id in scoring_subset:
            cmd += ["-i", task_id]

        _log(f"Running: {' '.join(cmd)}")
        hyperagents_root = str(Path(__file__).resolve().parents[2])
        # pier is a console-script entry point, not `python -m`/a script run
        # from this cwd -- its own process never adds our repo root to
        # sys.path on its own, so --agent-import-path's plain
        # importlib.import_module("domains.deep_swe.pier_agent") would fail
        # without PYTHONPATH set explicitly here (confirmed by reading
        # pier's own agent factory: it does nothing beyond import_module).
        run_env = {**os.environ, "PYTHONPATH": hyperagents_root + os.pathsep + os.environ.get("PYTHONPATH", "")}
        result = subprocess.run(cmd, cwd=hyperagents_root, env=run_env, capture_output=True, text=True)
        if result.returncode != 0:
            _log(f"pier run exited {result.returncode}. stderr tail:\n{result.stderr[-4000:]}")

        per_task_records = _load_trial_results(jobs_dir, job_name)

    node_utility = (
        sum(r["reward"] for r in per_task_records) / len(per_task_records)
        if per_task_records else 0.0
    )

    report_path = Path(output_dir) / f"gen_{genid}" / "deep_swe_eval" / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "node_utility": node_utility,
        "per_task": per_task_records,
    }, indent=2), encoding="utf-8")

    return node_utility, per_task_records


def run_harness_deep_swe(root_dir, output_dir, genid):
    """Called only when the parent self-modification diff was actually
    approved and non-empty (see generate_loop.py's own run_eval gate) --
    otherwise the code is byte-identical to its already-scored parent and
    re-scoring would be pure waste, same reasoning as domains/research/
    harness.py's own run_harness_research."""
    node_utility, _ = compute_node_utility(root_dir, output_dir, genid)
    update_node_metadata(output_dir, genid, {"deep_swe_node_utility": node_utility})
    return node_utility
