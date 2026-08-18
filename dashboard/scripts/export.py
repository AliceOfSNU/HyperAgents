"""Reads every outputs/generate_*/ run directory and writes compact,
dashboard-ready JSON into dashboard/export/. Pure local transformation --
knows nothing about Drive. upload.py consumes this directory's contents,
injects Drive links for already-uploaded log files, and pushes everything
to Drive.

Run standalone for a one-off export, or via run_loop.py for the scheduled
pipeline.
"""
import json
import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
EXPORT_DIR = Path(__file__).resolve().parent.parent / "export"

TRACEBACK_MARKER = "Traceback (most recent call last):"
# Known-benign noise: litellm/asyncio leave this trailing traceback on
# interpreter shutdown after every container run; it's cosmetic, not a real
# failure (diagnosed directly against a live run -- see harness.py history).
BENIGN_TRACEBACK_SIGNATURES = ("Event loop is closed", "BaseSubprocessTransport.__del__")
STALE_RUNNING_THRESHOLD_S = 10 * 60  # no file activity in this long -> not "running"


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None


def _parse_run_args(run_dir):
    """Pull max_generation/domains out of generate_loop.log's own debug
    print of its argv (there's no other persisted record of these)."""
    log_text = _read_text(run_dir / "generate_loop.log") or ""
    m = re.search(r"max_generation=(\d+)", log_text)
    max_generation = int(m.group(1)) if m else None
    m = re.search(r"domains=(\[[^\]]*\])", log_text)
    domains = m.group(1) if m else None
    return max_generation, domains


def _load_archive(run_dir):
    path = run_dir / "archive.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return []
    return json.loads(lines[-1])["archive"]


def _gen_dir(run_dir, genid):
    return run_dir / f"gen_{genid}"


def _load_gen_metadata(run_dir, genid):
    return _read_json(_gen_dir(run_dir, genid) / "metadata.json")


def _load_gen_report(run_dir, genid):
    return _read_json(_gen_dir(run_dir, genid) / "research_eval" / "report.json")


def _load_gen_promotion(run_dir, genid):
    return _read_json(_gen_dir(run_dir, genid) / "research_eval" / "evaluator_promotion.json")


def _model_patch_size(run_dir, genid):
    p = _gen_dir(run_dir, genid) / "agent_output" / "model_patch.diff"
    return p.stat().st_size if p.exists() else 0


def _extract_genuine_tracebacks(text):
    positions = [m.start() for m in re.finditer(re.escape(TRACEBACK_MARKER), text)]
    snippets = []
    for pos in positions:
        snippet = text[pos: pos + 800].strip()
        if any(sig in snippet for sig in BENIGN_TRACEBACK_SIGNATURES):
            continue
        snippets.append(snippet)
    return snippets


def _find_errors(run_dir, genids, limit=10):
    """Scan each generation's log files, plus the run-level crash.log
    (written by generate_loop.py's own top-level exception handler -- see
    its comment), for genuine traceback snippets, skipping the known-benign
    asyncio shutdown noise that follows essentially every container run."""
    errors = []
    crash_text = _read_text(run_dir / "crash.log")
    if crash_text:
        for snippet in _extract_genuine_tracebacks(crash_text):
            errors.append({"genid": None, "source": "crash.log", "snippet": snippet})
    for genid in genids:
        for rel in ("generate.log", "research_eval/generate.log"):
            text = _read_text(_gen_dir(run_dir, genid) / rel)
            if not text:
                continue
            snippets = _extract_genuine_tracebacks(text)
            if snippets:
                errors.append({"genid": genid, "source": rel, "snippet": snippets[-1]})
    return errors[-limit:]


def _is_process_alive_for_run(run_dir, run_id):
    """Prefer run_dir/run.pid (written by generate_loop.py itself at
    startup) -- checking /proc/<pid> directly is exact, unlike mtime-based
    staleness, which goes quiet for long, healthy stretches whenever a
    generation's container is mid-run (see generate_loop.py's comment where
    run.pid is written). Falls back to a pgrep substring match for older
    runs from before run.pid existed -- best-effort only, since run_id
    doesn't appear in the command line unless --run_id was passed
    explicitly at launch."""
    pid_file = run_dir / "run.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
        except ValueError:
            return False
        return Path(f"/proc/{pid}").exists()
    try:
        out = subprocess.run(["pgrep", "-af", "generate_loop.py"], capture_output=True, text=True, timeout=5)
    except Exception:
        return False
    return run_id in out.stdout


def _most_recent_mtime(run_dir):
    latest = 0.0
    for p in run_dir.rglob("*"):
        try:
            latest = max(latest, p.stat().st_mtime)
        except OSError:
            continue
    return latest


def _determine_status(run_dir, run_id, max_generation, current_genid, latest_report_exists):
    if max_generation is not None and current_genid is not None and current_genid >= max_generation and latest_report_exists:
        return "completed"
    if (run_dir / "run.pid").exists():
        # Exact signal (see _is_process_alive_for_run) -- no mtime fallback
        # needed or wanted, since mtime staleness is expected/healthy here.
        return "running" if _is_process_alive_for_run(run_dir, run_id) else "stalled_or_crashed"
    # Legacy run from before run.pid existed: best-effort only.
    if _is_process_alive_for_run(run_dir, run_id):
        return "running"
    if time.time() - _most_recent_mtime(run_dir) < STALE_RUNNING_THRESHOLD_S:
        return "running"
    return "stalled_or_crashed"


def _find_pending_pr(run_dir, genids):
    """A generation's PR is "pending" if it has a pr_number but metadata
    doesn't (yet) show it approved -- i.e. run_pr_review_gate is still
    polling. metadata.json is only written once generate() returns, and it
    blocks on the PR decision, so in practice a truly pending PR means this
    generation's metadata.json doesn't exist yet at all while a PR was
    already opened -- check the generation's own generate.log for the
    "Opened PR" line as the source of truth regardless of metadata state."""
    for genid in reversed(genids):
        if not isinstance(genid, int):
            continue
        log_text = _read_text(_gen_dir(run_dir, genid) / "generate.log") or ""
        m = re.search(r"Opened PR #(\d+) for gen \d+: (\S+)", log_text)
        if not m:
            continue
        meta = _load_gen_metadata(run_dir, genid)
        if meta is not None and meta.get("pr_approved") is not None:
            continue  # already resolved
        return {"genid": genid, "number": int(m.group(1)), "url": m.group(2)}
    return None


def _agent_scores(report):
    if not report:
        return None, None, None
    per_task = report.get("per_task", [])
    if not per_task:
        return report.get("node_utility"), None, None
    real_avg = sum(t.get("real_score", 0) for t in per_task) / len(per_task)
    eval_avg = sum(t.get("evaluator_score", 0) for t in per_task) / len(per_task)
    return report.get("node_utility"), real_avg, eval_avg


def _find_anchor_breakdown_for_genid(run_dir, genid, all_genids):
    """A genid's evaluator only gets scored against the anchor if it was
    ever a checkpoint candidate (incumbent or sampled challenger) -- search
    every checkpoint's evaluator_promotion.json for an entry."""
    for check_genid in all_genids:
        if not isinstance(check_genid, int) or check_genid % 3 != 0:
            continue
        promotion = _load_gen_promotion(run_dir, check_genid)
        if not promotion:
            continue
        candidate = promotion.get("candidates", {}).get(str(genid))
        if candidate:
            return {"checkpoint_genid": check_genid, **candidate}
    return None


def build_agent_detail(run_dir, run_id, genid, parent_genid, all_genids):
    meta = _load_gen_metadata(run_dir, genid) or {}
    report = _load_gen_report(run_dir, genid)
    node_utility, real_avg, eval_avg = _agent_scores(report)
    has_diff = _model_patch_size(run_dir, genid) > 0

    log_local = _gen_dir(run_dir, genid) / "research_eval" / "generate.log"
    return {
        "run_id": run_id,
        "genid": genid,
        "parent_genid": parent_genid,
        "has_diff": has_diff,
        "pr": {
            "number": meta.get("pr_number"),
            "url": meta.get("pr_url"),
            "approved": meta.get("pr_approved"),
        } if meta.get("pr_number") else None,
        "node_utility": node_utility,
        "real_score_avg": real_avg,
        "evaluator_score_avg": eval_avg,
        "incumbent_evaluator_genid": report.get("incumbent_evaluator_genid") if report else None,
        "per_task": report.get("per_task", []) if report else [],
        "evaluator_anchor_breakdown": _find_anchor_breakdown_for_genid(run_dir, genid, all_genids),
        "log_local_path": str(log_local.relative_to(REPO_ROOT)) if log_local.exists() else None,
        "log_drive_link": None,  # filled in by upload.py from its ID cache
    }


def build_run_export(run_dir):
    run_id = run_dir.name
    max_generation, domains = _parse_run_args(run_dir)
    genids = _load_archive(run_dir)
    numeric_genids = [g for g in genids if isinstance(g, int)]
    current_genid = max(numeric_genids) if numeric_genids else None

    last_report_exists = current_genid is not None and _load_gen_report(run_dir, current_genid) is not None
    status = _determine_status(run_dir, run_id, max_generation, current_genid, last_report_exists)

    epoch_state = _read_json(run_dir / "research_epoch_state.json") or {
        "incumbent_genid": "initial", "last_checkpoint_genid": 0, "promotion_history": [],
    }

    tree = []
    scores = []
    agents = []
    parent_by_genid = {"initial": None}
    for genid in genids:
        meta = _load_gen_metadata(run_dir, genid) or {}
        parent_genid = meta.get("parent_genid", parent_by_genid.get(genid))
        parent_by_genid[genid] = parent_genid
        report = _load_gen_report(run_dir, genid)
        node_utility = report.get("node_utility") if report else None
        if node_utility is not None:
            scores.append(node_utility)
        tree.append({"genid": genid, "parent_genid": parent_genid, "score": node_utility})
        if genid != "initial":
            agent_detail = build_agent_detail(run_dir, run_id, genid, parent_genid, genids)
            agents.append({
                "genid": genid,
                "parent_genid": parent_genid,
                "has_diff": agent_detail["has_diff"],
                "node_utility": agent_detail["node_utility"],
                "pr": agent_detail["pr"],
            })
            out_path = EXPORT_DIR / f"agent_{run_id}_{genid}.json"
            out_path.write_text(json.dumps(agent_detail, indent=2), encoding="utf-8")

    run_export = {
        "run_id": run_id,
        "domain": "research",
        "status": status,
        "current_genid": current_genid,
        "max_generation": max_generation,
        "max_score": max(scores) if scores else None,
        "avg_score": (sum(scores) / len(scores)) if scores else None,
        "errors": _find_errors(run_dir, [g for g in genids if isinstance(g, int)]),
        "pending_pr": _find_pending_pr(run_dir, genids),
        "epoch": {
            "incumbent_genid": epoch_state.get("incumbent_genid"),
            "last_checkpoint_genid": epoch_state.get("last_checkpoint_genid"),
            "promotion_history": epoch_state.get("promotion_history", []),
        },
        "tree": tree,
        "agents": agents,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (EXPORT_DIR / f"run_{run_id}.json").write_text(json.dumps(run_export, indent=2), encoding="utf-8")
    return run_export


def export_all():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    run_summaries = []
    if OUTPUTS_DIR.exists():
        for run_dir in sorted(OUTPUTS_DIR.glob("generate_*")):
            if not run_dir.is_dir():
                continue
            try:
                run_export = build_run_export(run_dir)
            except Exception as e:
                print(f"Skipping {run_dir.name}: {e}")
                continue
            run_summaries.append({
                "run_id": run_export["run_id"],
                "status": run_export["status"],
                "current_genid": run_export["current_genid"],
                "max_generation": run_export["max_generation"],
                "max_score": run_export["max_score"],
                "avg_score": run_export["avg_score"],
                "pending_pr": run_export["pending_pr"],
            })
    index = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": run_summaries}
    (EXPORT_DIR / "runs_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


if __name__ == "__main__":
    result = export_all()
    print(f"Exported {len(result['runs'])} run(s) to {EXPORT_DIR}")
