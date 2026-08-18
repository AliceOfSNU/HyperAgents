"""Docker-sandboxed harness for the research (ResearchClawBench) domain.

Every execution of meta-agent-authored code -- both task_agent.py (which
uses bash/editor tools, i.e. runs arbitrary shell commands) and
evaluator_agent.py (a single LLM judgment today, but just as editable) --
happens inside a fresh, disposable container built from the same
"hyperagents" image the meta-agent itself runs in. Nothing from this file
ever runs meta-agent code on the host.

Node utility per generation: run the candidate task_agent.py against a fixed
5-10 task "scoring subset", score each resulting report with (a) real_score
(Claude via the real anthropic SDK, domains/research/claude_scorer.py --
trusted, non-evolvable, safe to run on the host) and (b) the epoch's frozen
incumbent evaluator_agent.py (sandboxed, reconstructed from its own patch
chain, which may be from an earlier generation than the current one).
Combined score per report = real_score + evaluator_score; node utility is
the mean over the scoring subset.

Every EPOCH_LENGTH_GENERATIONS generations, a checkpoint compares the
incumbent evaluator against CHALLENGER_SAMPLE_SIZE random evaluator_agent.py
snapshots from generations since the last checkpoint, scored against the
fixed 30-task anchor set (never regenerated), and promotes whichever has the
best pooled per-item agreement (negative MAE) with the anchor's real scores.
"""

import json
import os
import random
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import docker
from dotenv import load_dotenv

from utils.common import load_json_file
from utils.constants import REPO_NAME
from utils.docker_utils import (
    copy_from_container,
    copy_to_container,
    safe_log,
    setup_logger,
)
from utils.gl_utils import get_patch_files, get_node_metadata_key, update_node_metadata

from .claude_scorer import score_item_with_claude, score_report_text_items
from .config import (
    ANCHOR_CACHE_PATH,
    CHALLENGER_SAMPLE_SIZE,
    CLAUDE_JUDGE_MODEL,
    EPOCH_LENGTH_GENERATIONS,
    RCB_ROOT,
    RCB_TASKS_DIR,
    SCORING_SUBSET_PATH,
)
from .workspace_utils import read_instructions, read_report

sys.path.insert(0, str(RCB_ROOT))
from evaluation.instructions_tmpl import INSTRUCTIONS_TEMPLATE  # noqa: E402
from evaluation.utils import load_task_info  # noqa: E402

# Host-side: nothing else here imports litellm (that only happens inside the
# containers), so DEEPSEEK_API_KEY/ANTHROPIC_API_KEY from .env wouldn't
# otherwise reach os.environ before _agent_env_vars() reads them.
load_dotenv()

IMAGE_NAME = "hyperagents"
AGENT_MODEL = "deepseek/deepseek-v4-flash"  # task_agent / evaluator_agent backend during search
CONTAINER_WORKSPACE = f"/{REPO_NAME}/workspace"
# Unlike domains/polyglot/harness.py (which destabilized the host running 5+
# parallel containers, each doing a heavy pip install), this domain's
# containers are LLM-bound, not compute-bound, so the host tolerates far
# more concurrency. Set to the scoring subset size (see subsets/scoring_subset.json)
# so every task in a generation runs concurrently -- keep these two in sync.
#
# NB: an early run at MAX_WORKERS=10 saw every task fail with
# "AuthenticationError: ... Authentication Fails (governor)", which first
# looked like a DeepSeek-side concurrency limit. It wasn't -- the real bug
# was _agent_env_vars() reading os.getenv() on the host, where nothing
# triggers python-dotenv's implicit .env loading (that only happens as an
# import side effect of litellm, which runs inside the containers, not on
# the host), so every container was silently getting a garbage
# DEEPSEEK_API_KEY regardless of concurrency. Fixed by load_dotenv() above;
# see _agent_env_vars()'s docstring.
MAX_WORKERS = 5
# Small stagger so workers don't all open their first LLM connection in the
# same instant -- cheap insurance, not load-bearing now that the real bug
# above is fixed.
WORKER_STAGGER_S = 1

# Files copied into every container, reconstructed from a patch chain on top.
BASELINE_FILES = [
    "task_agent.py",
    "run_task_agent.py",
    "evaluator_agent.py",
    "run_evaluator_agent.py",
    "requirements-research.txt",
    "agent/",
    "utils/",
    "domains/research/config.py",
    "domains/research/claude_scorer.py",
]


# ---------------------------------------------------------------------------
# Container plumbing
# ---------------------------------------------------------------------------

def _log_container_output(exec_result):
    """Log a container exec's output without raising on a nonzero exit code
    -- unlike utils.docker_utils.log_container_output, which always raises.
    We want to tolerate individual failures (a patch hunk that doesn't apply,
    a pip install hiccup, an agent run that times out) and decide what to do
    about them ourselves, not have the whole task/evaluator run abort here."""
    output = exec_result.output
    if isinstance(output, bytes):
        safe_log(f"Container output: {output.decode(errors='ignore')}")
    else:
        for chunk in output:
            if chunk:
                safe_log(f"Container output: {chunk.decode(errors='ignore').strip()}")


def _fresh_container(docker_client, name):
    try:
        existing = docker_client.containers.get(name)
        existing.remove(force=True)
    except docker.errors.NotFound:
        pass
    container = docker_client.containers.run(
        IMAGE_NAME, command="sleep infinity", detach=True,
        network_mode="host", name=name,
    )
    return container


def _cleanup(container):
    try:
        container.stop(timeout=5)
    except Exception:
        pass
    try:
        container.remove(force=True)
    except Exception as e:
        safe_log(f"Error removing container {container.name}: {e}")


def _copy_baseline_code(container, root_dir):
    for rel in BASELINE_FILES:
        src = os.path.join(root_dir, rel)
        if os.path.exists(src):
            copy_to_container(container, src, f"/{REPO_NAME}/{rel.rstrip('/')}")


def _apply_patch_chain(container, patch_files):
    for patch_file in patch_files:
        copy_to_container(container, patch_file, f"/{REPO_NAME}/parent_patch.txt")
        exec_result = container.exec_run(
            f"/bin/sh -c 'patch -p1 -f < /{REPO_NAME}/parent_patch.txt'", workdir=f"/{REPO_NAME}"
        )
        _log_container_output(exec_result)
        container.exec_run(f"rm -f /{REPO_NAME}/parent_patch.txt", workdir="/")


def _pip_install(container):
    exec_result = container.exec_run(
        f"python -m pip install -r /{REPO_NAME}/requirements-research.txt", workdir="/"
    )
    _log_container_output(exec_result)


def _agent_env_vars():
    """API keys for the containers' litellm/anthropic calls. Unlike the
    container side (where importing litellm has the side effect of loading
    .env), nothing on the host otherwise triggers python-dotenv, so
    load_dotenv() is called explicitly at module import time below. Missing
    keys fail loudly here instead of silently shipping the string "None" as
    a container env var (which is what a bare os.getenv() fallback did
    before -- every request then reached DeepSeek/Anthropic with a garbage
    key and came back as an opaque remote auth error)."""
    keys = {
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    }
    missing = [k for k, v in keys.items() if not v]
    if missing:
        raise RuntimeError(f"Missing required API key(s) in host environment/.env: {missing}")
    return keys


# ---------------------------------------------------------------------------
# Host-side task setup (the only place that touches the RCB checkout)
# ---------------------------------------------------------------------------

def _build_instructions(task_id, container_workspace_path):
    task_info = load_task_info(task_id)
    task_desc = task_info.get("task", "")
    data_parts = []
    for d in task_info.get("data", []):
        ws_path = d.get("path", "").lstrip("./")
        data_type = d.get("type", "")
        type_str = f" [{data_type}]" if data_type else ""
        data_parts.append(f"- **{d['name']}**{type_str} (`{ws_path}`): {d.get('description', '')}")
    data_text = "\n".join(data_parts) if data_parts else "No specific data files."
    return INSTRUCTIONS_TEMPLATE.format(
        workspace=container_workspace_path, task_desc=task_desc, data_text=data_text
    )


def _prepare_local_workspace(task_id, local_dir):
    """Stage data/related_work/INSTRUCTIONS.md on the host; this whole
    directory is later copied wholesale into the container."""
    local_dir = Path(local_dir)
    task_dir = RCB_TASKS_DIR / task_id
    local_dir.mkdir(parents=True, exist_ok=True)
    if (task_dir / "data").exists():
        shutil.copytree(task_dir / "data", local_dir / "data", dirs_exist_ok=True)
    if (task_dir / "related_work").exists():
        shutil.copytree(task_dir / "related_work", local_dir / "related_work", dirs_exist_ok=True)
    for dirname in ("code", "outputs", "report", "report/images"):
        (local_dir / dirname).mkdir(parents=True, exist_ok=True)
    instructions = _build_instructions(task_id, CONTAINER_WORKSPACE)
    (local_dir / "INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-task execution
# ---------------------------------------------------------------------------

def run_task_agent_sandboxed(docker_client, root_dir, patch_files, task_id, model, local_out_dir, timeout_s=3600):
    """Run task_agent.py (reconstructed from `patch_files`) on `task_id`
    inside a fresh container. Copies the completed workspace (report/,
    INSTRUCTIONS.md, _meta.json, ...) to `local_out_dir` on the host.
    Returns True on a clean (possibly task-unsuccessful) run, False if the
    container-level execution itself failed (timeout, crash)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    container = _fresh_container(docker_client, f"research-task-{task_id}-{stamp}")
    ok = True
    try:
        _copy_baseline_code(container, root_dir)
        _apply_patch_chain(container, patch_files)
        _pip_install(container)

        with tempfile.TemporaryDirectory() as tmp:
            local_ws = Path(tmp) / "workspace"
            _prepare_local_workspace(task_id, local_ws)
            copy_to_container(container, str(local_ws), CONTAINER_WORKSPACE)

        cmd = [
            "timeout", str(timeout_s),
            "python", f"/{REPO_NAME}/run_task_agent.py",
            "--workspace", CONTAINER_WORKSPACE,
            "--task_id", task_id,
            "--chat_history_file", f"/{REPO_NAME}/{task_id}_chat.md",
            "--model", model,
        ]
        exec_result = container.exec_run(cmd, environment=_agent_env_vars(), workdir=f"/{REPO_NAME}")
        _log_container_output(exec_result)
        if exec_result.exit_code != 0:
            ok = False

        copy_from_container(container, CONTAINER_WORKSPACE, str(local_out_dir))
    except Exception as e:
        safe_log(f"Error running task agent on {task_id}: {e}")
        ok = False
    finally:
        _cleanup(container)
    return ok


def run_evaluator_sandboxed(docker_client, root_dir, evaluator_patch_files, task_id, workspace_local_dir, model, timeout_s=600):
    """Score `workspace_local_dir`'s report against `task_id`'s checklist
    using evaluator_agent.py reconstructed from `evaluator_patch_files`
    (the epoch's frozen incumbent, or a promotion-checkpoint challenger --
    NOT necessarily the same generation as the task agent that produced the
    report). Returns {"task_id", "items", "score"} or None on failure."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    container = _fresh_container(docker_client, f"research-eval-{task_id}-{stamp}")
    try:
        _copy_baseline_code(container, root_dir)
        _apply_patch_chain(container, evaluator_patch_files)
        _pip_install(container)

        report_dir = Path(workspace_local_dir) / "report"
        instructions_path = Path(workspace_local_dir) / "INSTRUCTIONS.md"
        container.exec_run(f"mkdir -p {CONTAINER_WORKSPACE}/report", workdir="/")
        if report_dir.exists():
            copy_to_container(container, str(report_dir), f"{CONTAINER_WORKSPACE}/report")
        if instructions_path.exists():
            copy_to_container(container, str(instructions_path), f"{CONTAINER_WORKSPACE}/INSTRUCTIONS.md")

        checklist = load_json_file(str(RCB_TASKS_DIR / task_id / "target_study" / "checklist.json"))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(checklist, tmp)
            checklist_local_path = tmp.name
        copy_to_container(container, checklist_local_path, f"/{REPO_NAME}/checklist.json")
        os.unlink(checklist_local_path)

        cmd = [
            "timeout", str(timeout_s),
            "python", f"/{REPO_NAME}/run_evaluator_agent.py",
            "--workspace", CONTAINER_WORKSPACE,
            "--checklist_path", f"/{REPO_NAME}/checklist.json",
            "--task_id", task_id,
            "--model", model,
            "--out", f"/{REPO_NAME}/eval_result.json",
        ]
        exec_result = container.exec_run(cmd, environment=_agent_env_vars(), workdir=f"/{REPO_NAME}")
        _log_container_output(exec_result)
        if exec_result.exit_code != 0:
            return None

        with tempfile.TemporaryDirectory() as tmp:
            local_result_path = Path(tmp) / "eval_result.json"
            copy_from_container(container, f"/{REPO_NAME}/eval_result.json", str(local_result_path))
            return json.loads(local_result_path.read_text(encoding="utf-8"))
    except Exception as e:
        safe_log(f"Error running evaluator on {task_id}: {e}")
        return None
    finally:
        _cleanup(container)


# ---------------------------------------------------------------------------
# Epoch state
# ---------------------------------------------------------------------------

def _epoch_state_path(output_dir):
    # Scoped per-run (under output_dir), not a fixed module-level path --
    # each generate_loop.py run gets its own generation lineage/patch
    # history, so a shared path would leak an old run's incumbent_genid
    # into a new run where that genid doesn't even correspond to the same
    # patches.
    return Path(output_dir) / "research_epoch_state.json"


def _load_epoch_state(output_dir):
    path = _epoch_state_path(output_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"incumbent_genid": "initial", "last_checkpoint_genid": 0, "evaluator_snapshots": {}}


def _save_epoch_state(output_dir, state):
    path = _epoch_state_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _evaluator_patch_files(output_dir, genid):
    """The patch chain needed to reconstruct evaluator_agent.py exactly as
    of `genid` -- same ancestor-chain-replay mechanism used for task_agent.py,
    since evaluator_agent.py travels in the same per-generation patches."""
    return get_patch_files(output_dir, genid)


# ---------------------------------------------------------------------------
# Node utility (called every generation)
# ---------------------------------------------------------------------------

def _process_scoring_task(root_dir, patch_files, incumbent_patch_files, task_id, model, gen_eval_dir, log_path, stagger_s=0):
    # Spread out the first DeepSeek call each task makes -- see MAX_WORKERS's
    # comment: firing every task's first LLM call in the same instant tripped
    # DeepSeek's own concurrency governor, independent of MAX_WORKERS itself.
    if stagger_s:
        time.sleep(stagger_s)

    # Each worker thread needs its own thread-local logger (matching
    # generate_loop.py's eval_agent_worker: "Re-setup logger because of
    # threading") -- setup_logger() keys off threading.get_ident().
    setup_logger(str(log_path))

    # A fresh docker.DockerClient() per worker thread, NOT the client shared
    # by the caller: docker-py's client wraps one requests.Session/APIClient
    # that isn't safe for concurrent use across threads. Sharing it across
    # a ThreadPoolExecutor caused exec/copy calls from different threads to
    # cross-talk -- e.g. Astronomy_000's local_out_dir ending up with
    # Math_000's workspace (mismatched _meta.json task_id) -- when this was
    # first parallelized. One client per thread avoids that entirely.
    docker_client = docker.DockerClient()
    try:
        local_out_dir = gen_eval_dir / task_id
        ok = run_task_agent_sandboxed(docker_client, root_dir, patch_files, task_id, model, local_out_dir)
        if not ok:
            return {
                "task_id": task_id, "real_score": 0.0, "evaluator_score": 0.0, "combined": 0.0,
                "task_agent_ok": False, "real_items": [], "evaluator_items": [], "report_empty": True,
            }

        report_text = read_report(local_out_dir) or ""
        if not report_text.strip():
            # Nothing worth scoring -- an empty report would just make Claude
            # and the evaluator grade every checklist item ~0 anyway, so skip
            # both LLM-scoring calls rather than spending real API cost on a
            # foregone conclusion. report_empty is also what
            # compute_node_utility's gate checks to decide whether to bail
            # out of the rest of the scoring subset early.
            return {
                "task_id": task_id, "real_score": 0.0, "evaluator_score": 0.0, "combined": 0.0,
                "task_agent_ok": True, "real_items": [], "evaluator_items": [], "report_empty": True,
            }

        instructions = read_instructions(local_out_dir)
        checklist = load_json_file(str(RCB_TASKS_DIR / task_id / "target_study" / "checklist.json"))

        def _claude_score_fn(rt, item, instr):
            return score_item_with_claude(rt, item, instr, model=CLAUDE_JUDGE_MODEL)

        real_items, real_score = score_report_text_items(report_text, checklist, instructions, _claude_score_fn)

        eval_result = run_evaluator_sandboxed(docker_client, root_dir, incumbent_patch_files, task_id, local_out_dir, model)
        evaluator_score = eval_result["score"] if eval_result else 0.0
        evaluator_items = eval_result["items"] if eval_result else []

        combined = real_score + evaluator_score
        return {
            "task_id": task_id, "real_score": real_score, "evaluator_score": evaluator_score,
            "combined": combined, "task_agent_ok": True,
            # Per-item breakdown -- real_items carries Claude's real_score
            # reasoning per checklist item, evaluator_items carries the
            # evolving evaluator_agent.py's own reasoning for the same
            # items, so a dashboard/report can show both side by side.
            "real_items": real_items, "evaluator_items": evaluator_items, "report_empty": False,
        }
    finally:
        docker_client.close()


def compute_node_utility(docker_client, root_dir, output_dir, genid, model=AGENT_MODEL, max_workers=MAX_WORKERS):
    """Run the candidate task_agent.py (as of `genid`) on the fixed scoring
    subset (up to `max_workers` tasks in parallel, each in its own sandboxed
    container on its own docker.DockerClient()), score each report with
    real_score + the epoch's frozen incumbent evaluator, and return
    (node_utility, per_task_records).

    Gated: the first min(3, len(scoring_subset)) tasks run first; if every
    one of them comes back with an empty report, that's a strong signal
    task_agent.py is broken in a way a few more attempts won't fix, so the
    remaining tasks are skipped entirely (forced to a flat zero) rather than
    burning a full task-agent + evaluator run on each of them for a
    foregone conclusion. The remaining tasks only run if at least one gate
    task actually produced a report.

    `docker_client` (the caller's client, used by non-parallel code paths
    elsewhere in this module) is intentionally NOT reused by the worker
    threads below -- see _process_scoring_task's docstring comment on why
    each thread needs its own client."""
    scoring_subset = load_json_file(str(SCORING_SUBSET_PATH))
    patch_files = get_patch_files(output_dir, genid)

    epoch_state = _load_epoch_state(output_dir)
    incumbent_genid = epoch_state["incumbent_genid"]
    incumbent_patch_files = _evaluator_patch_files(output_dir, incumbent_genid)

    gen_eval_dir = Path(output_dir) / f"gen_{genid}" / "research_eval" / "eval_run"
    gen_eval_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / f"gen_{genid}" / "research_eval" / "generate.log"

    def _run_tasks(task_ids, stagger_start_idx):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _process_scoring_task, root_dir, patch_files,
                    incumbent_patch_files, task_id, model, gen_eval_dir, log_path,
                    (stagger_start_idx + i) * WORKER_STAGGER_S,
                )
                for i, task_id in enumerate(task_ids)
            ]
            return [future.result() for future in as_completed(futures)]

    gate_size = min(3, len(scoring_subset))
    gate_subset, remaining_subset = scoring_subset[:gate_size], scoring_subset[gate_size:]

    gate_records = _run_tasks(gate_subset, 0)
    all_gate_empty = bool(gate_records) and all(r["report_empty"] for r in gate_records)

    if all_gate_empty:
        safe_log(
            f"gen_{genid}: all {len(gate_records)} gate tasks "
            f"({[r['task_id'] for r in gate_records]}) produced empty reports -- "
            f"forcing node_utility to 0.0 and skipping the remaining "
            f"{len(remaining_subset)} scoring tasks without running them."
        )
        skipped_records = [
            {
                "task_id": task_id, "real_score": 0.0, "evaluator_score": 0.0, "combined": 0.0,
                "task_agent_ok": False, "real_items": [], "evaluator_items": [],
                "report_empty": True, "skipped_by_gate": True,
            }
            for task_id in remaining_subset
        ]
        per_task_records = gate_records + skipped_records
    else:
        remaining_records = _run_tasks(remaining_subset, gate_size) if remaining_subset else []
        per_task_records = gate_records + remaining_records

    node_utility = sum(r["combined"] for r in per_task_records) / len(per_task_records) if per_task_records else 0.0

    report_path = Path(output_dir) / f"gen_{genid}" / "research_eval" / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "node_utility": node_utility,
        "incumbent_evaluator_genid": incumbent_genid,
        "per_task": per_task_records,
    }, indent=2), encoding="utf-8")

    return node_utility, per_task_records


def _rescore_task_with_evaluator(root_dir, evaluator_patch_files, task_id, model, local_out_dir, prior_record):
    """Re-score one already-completed scoring task against a (newly
    promoted) evaluator, WITHOUT re-running the task agent -- reuses
    local_out_dir's already-persisted report/INSTRUCTIONS.md (written by the
    original run_task_agent_sandboxed call and never cleaned up) and
    prior_record's real_score/real_items, which come from Claude and don't
    depend on evaluator_agent.py at all. Only evaluator_score/evaluator_items
    actually change here."""
    if not prior_record.get("task_agent_ok") or prior_record.get("report_empty"):
        return prior_record  # nothing to re-score; no report was ever produced (incl. skipped-by-gate tasks)
    docker_client = docker.DockerClient()
    try:
        eval_result = run_evaluator_sandboxed(docker_client, root_dir, evaluator_patch_files, task_id, local_out_dir, model)
        evaluator_score = eval_result["score"] if eval_result else 0.0
        evaluator_items = eval_result["items"] if eval_result else []
        return {
            **prior_record,
            "evaluator_score": evaluator_score,
            "evaluator_items": evaluator_items,
            "combined": prior_record["real_score"] + evaluator_score,
        }
    finally:
        docker_client.close()


def _rescore_node_utility(root_dir, output_dir, genid, new_incumbent_genid, model=AGENT_MODEL, max_workers=MAX_WORKERS):
    """Re-score gen_{genid}'s existing report.json against new_incumbent_genid,
    in place, so a later select_parent never compares this node's node_utility
    (computed under the OLD incumbent) against nodes scored under the new one.
    No-op if this generation was never scored, or is already up to date."""
    report_path = Path(output_dir) / f"gen_{genid}" / "research_eval" / "report.json"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("incumbent_evaluator_genid") == new_incumbent_genid:
        return

    evaluator_patch_files = _evaluator_patch_files(output_dir, new_incumbent_genid)
    gen_eval_dir = Path(output_dir) / f"gen_{genid}" / "research_eval" / "eval_run"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _rescore_task_with_evaluator, root_dir, evaluator_patch_files,
                record["task_id"], model, gen_eval_dir / record["task_id"], record,
            )
            for record in report["per_task"]
        ]
        new_per_task = [future.result() for future in as_completed(futures)]

    new_node_utility = sum(r["combined"] for r in new_per_task) / len(new_per_task) if new_per_task else 0.0

    # Append-only so the dashboard/a curious human can see what this node
    # used to score under a prior incumbent, rather than silently losing it.
    report.setdefault("rescore_history", []).append({
        "as_of_incumbent_genid": report.get("incumbent_evaluator_genid"),
        "node_utility": report.get("node_utility"),
    })
    report["node_utility"] = new_node_utility
    report["incumbent_evaluator_genid"] = new_incumbent_genid
    report["per_task"] = new_per_task
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _backfill_node_utility_on_promotion(root_dir, output_dir, new_incumbent_genid, current_genid, model=AGENT_MODEL, max_workers=MAX_WORKERS):
    """After an evaluator promotion, re-score every completed generation's
    node_utility (1..current_genid) against the new incumbent, reusing each
    generation's already-persisted task-agent reports -- so select_parent is
    always comparing node_utility scores produced by the same evaluator,
    instead of a mix of pre- and post-promotion scores. One generation's
    rescore failing doesn't block the rest; each is independent and this is
    itself best-effort (see maybe_run_evaluator_promotion's own docstring)."""
    for genid in range(1, current_genid + 1):
        try:
            _rescore_node_utility(root_dir, output_dir, genid, new_incumbent_genid, model=model, max_workers=max_workers)
        except Exception as e:
            safe_log(f"Rescoring gen_{genid} against new incumbent gen_{new_incumbent_genid} failed, leaving its node_utility as-is: {e}")


# ---------------------------------------------------------------------------
# Evaluator promotion checkpoint (every EPOCH_LENGTH_GENERATIONS generations)
# ---------------------------------------------------------------------------

def _score_evaluator_against_anchor(docker_client, root_dir, evaluator_patch_files, model=AGENT_MODEL):
    """Score a candidate evaluator (frozen incumbent or challenger) against
    the fixed anchor cache. Returns (pooled_score, breakdown):
    pooled_score is the negative MAE across every (task, item) pair (higher,
    closer to 0, is better agreement); breakdown is a per-task list of
    per-item {index, content, weight, predicted, real, abs_error,
    evaluator_reasoning} dicts, for dashboards/reports to render the full
    N-tasks x #rubric-items comparison, not just the pooled number."""
    anchor_cache = json.loads(ANCHOR_CACHE_PATH.read_text(encoding="utf-8"))
    abs_errors = []
    breakdown = []
    for entry in anchor_cache["reports"]:
        task_id = entry["task_id"]
        with tempfile.TemporaryDirectory() as tmp:
            local_ws = Path(tmp) / "workspace"
            (local_ws / "report").mkdir(parents=True, exist_ok=True)
            (local_ws / "report" / "report.md").write_text(entry["report_text"], encoding="utf-8")
            (local_ws / "INSTRUCTIONS.md").write_text(entry["instructions"], encoding="utf-8")

            eval_result = run_evaluator_sandboxed(docker_client, root_dir, evaluator_patch_files, task_id, local_ws, model)
        if eval_result is None:
            continue
        predicted_by_index = {item["index"]: item for item in eval_result["items"]}
        task_items = []
        for real_item in entry["real_items"]:
            predicted_item = predicted_by_index.get(real_item["index"])
            if predicted_item is None:
                continue
            predicted = predicted_item["score"]
            abs_error = abs(predicted - real_item["score"])
            abs_errors.append(abs_error)
            task_items.append({
                "index": real_item["index"],
                "content": predicted_item.get("content", ""),
                "weight": predicted_item.get("weight"),
                "predicted": predicted,
                "real": real_item["score"],
                "abs_error": abs_error,
                "evaluator_reasoning": predicted_item.get("reasoning", ""),
            })
        breakdown.append({"task_id": task_id, "items": task_items})

    pooled_score = -(sum(abs_errors) / len(abs_errors)) if abs_errors else float("-inf")
    return pooled_score, breakdown


def maybe_run_evaluator_promotion(docker_client, root_dir, output_dir, current_genid):
    """At every checkpoint (every EPOCH_LENGTH_GENERATIONS generations),
    compare the incumbent evaluator against CHALLENGER_SAMPLE_SIZE random
    challengers from generations since the last checkpoint, and promote
    whichever has the best anchor agreement. No-op between checkpoints.

    This is a nice-to-have checkpoint on top of hours of already-completed
    generation work -- it must never be able to take down the orchestrating
    generate_loop.py process (this crashed a real run: the anchor cache is
    built by a separate one-time script, domains/research/anchor.py, and a
    checkpoint can land before that script finishes). Any failure here is
    logged and skipped, leaving last_checkpoint_genid unchanged so the next
    checkpoint retries the same (and any newer) candidates."""
    if not isinstance(current_genid, int) or current_genid % EPOCH_LENGTH_GENERATIONS != 0:
        return

    if not ANCHOR_CACHE_PATH.exists():
        safe_log(
            f"Evaluator promotion skipped at gen_{current_genid}: anchor cache not "
            f"found at {ANCHOR_CACHE_PATH} (still being generated?). Incumbent "
            "evaluator unchanged; will retry at the next checkpoint."
        )
        return

    try:
        epoch_state = _load_epoch_state(output_dir)
        since_last = list(range(epoch_state["last_checkpoint_genid"] + 1, current_genid + 1))
        since_last = [g for g in since_last if g != epoch_state["incumbent_genid"]]
        challengers = random.sample(since_last, min(CHALLENGER_SAMPLE_SIZE, len(since_last)))

        candidates = {epoch_state["incumbent_genid"]: None}
        for genid in challengers:
            candidates[genid] = None

        candidate_results = {}
        best_genid, best_score = epoch_state["incumbent_genid"], None
        for genid, _ in candidates.items():
            patch_files = _evaluator_patch_files(output_dir, genid)
            score, breakdown = _score_evaluator_against_anchor(docker_client, root_dir, patch_files)
            safe_log(f"Evaluator promotion candidate gen_{genid}: anchor score {score}")
            candidate_results[str(genid)] = {"anchor_score": score, "breakdown": breakdown}
            if best_score is None or score > best_score:
                best_genid, best_score = genid, score

        incumbent_before = epoch_state["incumbent_genid"]
        promoted = best_genid != incumbent_before
        if promoted:
            safe_log(f"Evaluator promotion: gen_{best_genid} replaces gen_{incumbent_before} as incumbent")

        # Full per-candidate breakdown for this checkpoint (dashboard/report
        # detail view) plus a compact, cumulative promotion_history entry in
        # epoch_state (dashboard/report can flag "evaluator swapped" without
        # having to scan every generation's own file).
        promotion_path = Path(output_dir) / f"gen_{current_genid}" / "research_eval" / "evaluator_promotion.json"
        promotion_path.parent.mkdir(parents=True, exist_ok=True)
        promotion_path.write_text(json.dumps({
            "checkpoint_genid": current_genid,
            "incumbent_before": incumbent_before,
            "incumbent_after": best_genid,
            "promoted": promoted,
            "candidates": candidate_results,
        }, indent=2), encoding="utf-8")

        epoch_state["incumbent_genid"] = best_genid
        epoch_state["last_checkpoint_genid"] = current_genid
        epoch_state.setdefault("promotion_history", []).append({
            "checkpoint_genid": current_genid,
            "incumbent_before": incumbent_before,
            "incumbent_after": best_genid,
            "promoted": promoted,
        })
        _save_epoch_state(output_dir, epoch_state)

        # Every generation scored before this checkpoint used incumbent_before
        # as its evaluator; without this, select_parent would keep comparing
        # those frozen scores against new generations scored by best_genid --
        # apples to oranges. Reuses each generation's already-persisted
        # task-agent reports, so this only re-runs the (cheap) evaluator, not
        # the expensive task-agent loop.
        if promoted:
            _backfill_node_utility_on_promotion(root_dir, output_dir, best_genid, current_genid)
    except Exception as e:
        safe_log(f"Evaluator promotion at gen_{current_genid} failed, skipping this checkpoint: {e}")


# ---------------------------------------------------------------------------
# Top-level entrypoint, analogous to run_harness_polyglot
# ---------------------------------------------------------------------------

def run_harness_research(root_dir, output_dir, genid, run_eval=True):
    """run_eval=False (an empty/unapproved self-modification diff -- nothing
    about the agent code actually changed from its parent) skips the
    expensive part: compute_node_utility would just re-run the same 5-task
    scoring subset against code identical to the already-scored parent,
    burning a real task-agent + evaluator pass per task for zero new
    information (confirmed live: this was happening on every empty-diff
    generation all session). maybe_run_evaluator_promotion still runs
    unconditionally -- it self-gates on current_genid's own checkpoint
    schedule (every EPOCH_LENGTH_GENERATIONS), and skipping it here too would
    silently delay that schedule whenever a generation happened to produce
    an empty diff, rather than just skipping the one redundant score."""
    log_path = Path(output_dir) / f"gen_{genid}" / "research_eval" / "generate.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logger(str(log_path))

    docker_client = docker.DockerClient()
    node_utility = None
    if run_eval:
        node_utility, _ = compute_node_utility(docker_client, root_dir, output_dir, genid)
        update_node_metadata(output_dir, genid, {"research_node_utility": node_utility})
    else:
        safe_log(f"gen_{genid}: skipping node utility scoring -- no approved self-modification diff to evaluate.")
    maybe_run_evaluator_promotion(docker_client, root_dir, output_dir, genid)
    return node_utility
