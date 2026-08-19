"""One-time setup script: generates the fixed 30-task evaluator-promotion
anchor. Not run automatically by generate_loop.py -- run this once, by hand,
before starting a research-domain experiment.

For each of the 30 anchor tasks (domains/research/subsets/anchor_subset.json),
a real report is produced via ResearchClawBench's own `researchharness`
baseline agent (through its `rcb-eval` / `evaluation.cli_eval` CLI), split
15/15 across two DeepSeek capability tiers so the anchor has real score
variance. Each report is then scored with our own real_score (Claude via the
real anthropic SDK, text-only checklist items) and the whole set is cached to
ANCHOR_CACHE_PATH for reuse at every evaluator-promotion checkpoint --
generated once, never regenerated.

Prerequisites (not installed/verified by this script):
  - `pip install researchharness` in whatever environment runs this script
  - SERPER_KEY, JINA_KEY, MINERU_TOKEN, ANTHROPIC_API_KEY in the environment
  - ANCHOR_AGENT_API_BASE / ANCHOR_AGENT_MODEL_STRONG / ANCHOR_AGENT_MODEL_WEAK
    (domains/research/config.py) matching DeepSeek's actual API -- these are
    best-effort defaults, not verified against DeepSeek's docs/dashboard.

This is a real, billed run against 30 tasks on two live agent backends plus
30 x (several checklist items) live Claude calls. Consider running
`python -m evaluation.cli_eval <config> --dry-run --skip-secret-check --no-score`
(from RCB_ROOT) on a trimmed config first to sanity-check the setup before
running this for real.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from dotenv import load_dotenv

from .claude_scorer import score_item_with_claude, score_report_text_items
from .config import (
    ANCHOR_AGENT_API_BASE,
    ANCHOR_AGENT_MODEL_STRONG,
    ANCHOR_AGENT_MODEL_WEAK,
    ANCHOR_CACHE_PATH,
    ANCHOR_SUBSET_PATH,
    CLAUDE_JUDGE_MODEL,
    RCB_ROOT,
)
from .workspace_utils import read_checklist

# Explicit path (not bare load_dotenv()) so this resolves regardless of the
# caller's cwd -- same fix domains/research/harness.py needed: nothing else
# in this host-side import chain triggers python-dotenv's implicit .env
# loading (that only happens inside the containers, as a side effect of
# importing litellm). RCB's own evaluation/cli_eval.py separately calls
# load_dotenv() pointed at its own evaluation/.env, but python-dotenv
# doesn't override already-set vars, so populating os.environ here first is
# sufficient for the subprocess env passed to it below.
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


def _run_cli_eval_batch(task_ids, agent_model_name):
    """Run ResearchHarness on `task_ids` via rcb-eval, return the batch
    workspace directory (workspaces/cli_runs/cli_<...>/) it wrote to."""
    config = {
        "name": f"anchor_gen_{agent_model_name}",
        "agent_model": {
            "name_env": "AGENT_MODEL_NAME",
            "api_base_env": "AGENT_API_BASE",
            "api_key_env": "AGENT_API_KEY",
        },
        "tasks": [{"id": t, "repeats": 1} for t in task_ids],
        "repeats_per_task": 1,
        "max_concurrent_runs": 3,
        "researchharness": {
            "max_rounds": 500,
            "max_runtime_seconds": 10800,
            "llm_request_timeout_seconds": 1200,
            "webfetch_tool_timeout_seconds": 300,
            "readpdf_tool_timeout_seconds": 300,
            "max_output_tokens": 16384,
            "max_input_tokens": 131072,
            "compact_trigger_tokens": "96k",
        },
        # We score with our own real_score afterward, not RCB's judge.
        "judge_model": {
            "enabled": False,
            "name_env": "JUDGE_MODEL_NAME",
            "api_base_env": "JUDGE_API_BASE",
            "api_key_env": "JUDGE_API_KEY",
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(config, tmp)
        config_path = tmp.name

    env = os.environ.copy()
    env["AGENT_MODEL_NAME"] = agent_model_name
    env["AGENT_API_BASE"] = ANCHOR_AGENT_API_BASE
    env["AGENT_API_KEY"] = os.environ["DEEPSEEK_API_KEY"]

    workspaces_before = set((RCB_ROOT / "workspaces" / "cli_runs").glob("cli_*")) if (RCB_ROOT / "workspaces" / "cli_runs").exists() else set()

    result = subprocess.run(
        [sys.executable, "-m", "evaluation.cli_eval", config_path],
        cwd=str(RCB_ROOT), env=env, capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"rcb-eval failed for {agent_model_name} (exit {result.returncode})")

    workspaces_after = set((RCB_ROOT / "workspaces" / "cli_runs").glob("cli_*"))
    new_batches = workspaces_after - workspaces_before
    if not new_batches:
        raise RuntimeError("rcb-eval produced no new batch directory under workspaces/cli_runs/")
    return max(new_batches, key=lambda p: p.stat().st_mtime)


def _collect_reports(batch_dir):
    """Map task_id -> (report_text, instructions) for every completed run
    in a cli_eval batch directory."""
    reports = {}
    for run_dir in Path(batch_dir).iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "_meta.json"
        report_path = run_dir / "report" / "report.md"
        instructions_path = run_dir / "INSTRUCTIONS.md"
        if not (meta_path.exists() and report_path.exists()):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        reports[meta["task_id"]] = (
            report_path.read_text(encoding="utf-8"),
            instructions_path.read_text(encoding="utf-8") if instructions_path.exists() else "",
        )
    return reports


def generate_anchor_cache():
    anchor_tasks = json.loads(ANCHOR_SUBSET_PATH.read_text(encoding="utf-8"))
    strong_tasks, weak_tasks = anchor_tasks[:15], anchor_tasks[15:]

    reports = {}
    for task_ids, model in ((strong_tasks, ANCHOR_AGENT_MODEL_STRONG), (weak_tasks, ANCHOR_AGENT_MODEL_WEAK)):
        batch_dir = _run_cli_eval_batch(task_ids, model)
        reports.update(_collect_reports(batch_dir))

    missing = [t for t in anchor_tasks if t not in reports]
    if missing:
        print(f"Warning: {len(missing)} anchor task(s) produced no report and will be skipped: {missing}", file=sys.stderr)

    cache_entries = []
    for task_id in anchor_tasks:
        if task_id not in reports:
            continue
        report_text, instructions = reports[task_id]
        checklist = read_checklist(task_id)

        def _claude_score_fn(rt, item, instr):
            return score_item_with_claude(rt, item, instr, model=CLAUDE_JUDGE_MODEL)

        items, _score = score_report_text_items(report_text, checklist, instructions, _claude_score_fn)
        cache_entries.append({
            "task_id": task_id,
            "report_text": report_text,
            "instructions": instructions,
            "real_items": [{"index": it["index"], "score": it["score"]} for it in items],
        })

    ANCHOR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANCHOR_CACHE_PATH.write_text(json.dumps({"reports": cache_entries}, indent=2), encoding="utf-8")
    print(f"Wrote {len(cache_entries)} anchor reports to {ANCHOR_CACHE_PATH}")


if __name__ == "__main__":
    generate_anchor_cache()
