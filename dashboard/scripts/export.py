"""Reads every outputs/generate_*/ run directory and writes compact,
dashboard-ready JSON into dashboard/export/. Pure local transformation --
knows nothing about Drive. upload.py consumes this directory's contents,
injects Drive links for already-uploaded log files, and pushes everything
to Drive.

Run standalone for a one-off export, or via run_loop.py for the scheduled
pipeline.
"""
import ast
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
    """Pull max_generation/domain out of generate_loop.log's own debug
    print of its argv (there's no other persisted record of these). Every
    run so far passes exactly one domain via --domains, so the eval
    directory name (f"{domain}_eval", matching utils/gl_utils.py's own
    eval_dirname convention and each domain's harness.py) and the exported
    "domain" field both key off this single value -- defaulting to
    "research" only for pre-existing runs from before this parsing existed."""
    log_text = _read_text(run_dir / "generate_loop.log") or ""
    m = re.search(r"max_generation=(\d+)", log_text)
    max_generation = int(m.group(1)) if m else None
    m = re.search(r"domains=(\[[^\]]*\])", log_text)
    domain = "research"
    if m:
        try:
            parsed = ast.literal_eval(m.group(1))
            if parsed:
                domain = parsed[0]
        except (ValueError, SyntaxError):
            pass
    return max_generation, domain


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


def _load_gen_report(run_dir, genid, domain):
    return _read_json(_gen_dir(run_dir, genid) / f"{domain}_eval" / "report.json")


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


def _find_errors(run_dir, genids, domain, limit=10):
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
        for rel in ("generate.log", f"{domain}_eval/generate.log"):
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


def _scan_pr_from_log(run_dir, genid):
    """Scan a generation's own generate.log for the "Opened PR" line -- the
    source of truth for PR info before metadata.json exists (metadata.json
    is only written once generate() returns, and generate() blocks on the
    PR review gate while a PR is pending)."""
    log_text = _read_text(_gen_dir(run_dir, genid) / "generate.log") or ""
    m = re.search(r"Opened PR #(\d+) for gen \d+: (\S+)", log_text)
    if not m:
        return None
    return {"number": int(m.group(1)), "url": m.group(2)}


def _find_pending_pr(run_dir, genids):
    """A generation's PR is "pending" if it has a pr_number but metadata
    doesn't (yet) show it approved -- i.e. run_pr_review_gate is still
    polling. metadata.json is only written once generate() returns, and it
    blocks on the PR decision, so in practice a truly pending PR means this
    generation's metadata.json doesn't exist yet at all while a PR was
    already opened -- check the generation's own generate.log via
    _scan_pr_from_log as the source of truth regardless of metadata state."""
    for genid in reversed(genids):
        if not isinstance(genid, int):
            continue
        pr = _scan_pr_from_log(run_dir, genid)
        if not pr:
            continue
        meta = _load_gen_metadata(run_dir, genid)
        if meta is not None and meta.get("pr_approved") is not None:
            continue  # already resolved
        return {"genid": genid, **pr}
    return None


def _agent_scores(report):
    if not report:
        return None, None
    per_task = report.get("per_task", [])
    if not per_task:
        return report.get("node_utility"), None
    real_avg = sum(t.get("real_score", 0) for t in per_task) / len(per_task)
    return report.get("node_utility"), real_avg


def build_agent_detail(run_dir, run_id, genid, parent_genid, all_genids, domain):
    meta = _load_gen_metadata(run_dir, genid) or {}
    report = _load_gen_report(run_dir, genid, domain)
    node_utility, real_avg = _agent_scores(report)
    has_diff = _model_patch_size(run_dir, genid) > 0

    if meta.get("pr_number"):
        pr = {
            "number": meta.get("pr_number"),
            "url": meta.get("pr_url"),
            "approved": meta.get("pr_approved"),
        }
    else:
        # metadata.json doesn't exist yet (generation still in-flight,
        # blocked on the PR review gate) -- fall back to the log so a
        # genuinely open PR still surfaces here, not just in pending_pr.
        log_pr = _scan_pr_from_log(run_dir, genid)
        pr = {"number": log_pr["number"], "url": log_pr["url"], "approved": None} if log_pr else None

    log_local = _gen_dir(run_dir, genid) / f"{domain}_eval" / "generate.log"
    return {
        "run_id": run_id,
        "genid": genid,
        "parent_genid": parent_genid,
        "has_diff": has_diff,
        "pr": pr,
        "node_utility": node_utility,
        "real_score_avg": real_avg,
        # No more co-evolving evaluator agent -- node_utility is real_avg
        # directly now. Kept as explicit nulls (rather than dropping the
        # keys) so dashboard/web's existing AgentDetail type/rendering,
        # which already treats all three as optional, doesn't need to
        # change to stay correct.
        "evaluator_score_avg": None,
        "incumbent_evaluator_genid": None,
        "per_task": report.get("per_task", []) if report else [],
        "evaluator_anchor_breakdown": None,
        "log_local_path": str(log_local.relative_to(REPO_ROOT)) if log_local.exists() else None,
        "log_drive_link": None,  # filled in by upload.py from its ID cache
    }


def _find_in_flight_genid(run_dir, run_id, known_genids):
    """The generation currently being processed doesn't appear in
    archive.jsonl until generate() returns -- and generate() blocks on the
    PR review gate while a PR is pending -- so while a PR is open, the
    in-flight generation is invisible to _load_archive() entirely (its
    genid isn't in the returned list at all, unlike a resolved generation
    whose metadata.json just doesn't exist yet). Detect it directly from
    disk instead: the highest-numbered gen_N directory not already in
    known_genids, but only if the run's own process is still alive --
    otherwise this is just an abandoned generation from a crashed/killed
    run, not something genuinely in progress."""
    if not _is_process_alive_for_run(run_dir, run_id):
        return None
    known = {g for g in known_genids if isinstance(g, int)}
    candidates = []
    for p in run_dir.glob("gen_*"):
        if not p.is_dir():
            continue
        suffix = p.name[len("gen_"):]
        if suffix.isdigit() and int(suffix) not in known:
            candidates.append(int(suffix))
    return max(candidates) if candidates else None


def _read_parent_genid_breadcrumb(run_dir, genid):
    """Read the parent_genid.txt breadcrumb generate_loop.py writes the
    instant a generation's directory is created (see its own comment) --
    the only source for an in-flight generation's parent before its
    metadata.json exists. Older runs from before this breadcrumb existed
    simply won't have one; that's fine, the caller treats a missing parent
    as unknown rather than failing."""
    text = _read_text(_gen_dir(run_dir, genid) / "parent_genid.txt")
    if text is None:
        return None
    text = text.strip()
    return int(text) if text.lstrip("-").isdigit() else text


def build_run_export(run_dir):
    run_id = run_dir.name
    max_generation, domain = _parse_run_args(run_dir)
    genids = _load_archive(run_dir)

    in_flight_genid = _find_in_flight_genid(run_dir, run_id, genids)
    in_flight_parent = None
    if in_flight_genid is not None:
        in_flight_parent = _read_parent_genid_breadcrumb(run_dir, in_flight_genid)
        genids = genids + [in_flight_genid]

    numeric_genids = [g for g in genids if isinstance(g, int)]
    current_genid = max(numeric_genids) if numeric_genids else None

    last_report_exists = current_genid is not None and _load_gen_report(run_dir, current_genid, domain) is not None
    status = _determine_status(run_dir, run_id, max_generation, current_genid, last_report_exists)

    epoch_state = _read_json(run_dir / "research_epoch_state.json") or {
        "incumbent_genid": "initial", "last_checkpoint_genid": 0, "promotion_history": [],
    }

    tree = []
    scores = []
    agents = []
    parent_by_genid = {"initial": None}
    if in_flight_genid is not None:
        # Pre-populated so the loop below picks it up via its existing
        # meta.get("parent_genid", parent_by_genid.get(genid)) fallback,
        # exactly as if a metadata.json with just this field existed.
        parent_by_genid[in_flight_genid] = in_flight_parent
    for genid in genids:
        meta = _load_gen_metadata(run_dir, genid) or {}
        parent_genid = meta.get("parent_genid", parent_by_genid.get(genid))
        parent_by_genid[genid] = parent_genid
        report = _load_gen_report(run_dir, genid, domain)
        node_utility = report.get("node_utility") if report else None
        if node_utility is not None:
            scores.append(node_utility)
        tree.append({"genid": genid, "parent_genid": parent_genid, "score": node_utility})
        if genid != "initial":
            agent_detail = build_agent_detail(run_dir, run_id, genid, parent_genid, genids, domain)
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
        "domain": domain,
        "status": status,
        "current_genid": current_genid,
        "max_generation": max_generation,
        "max_score": max(scores) if scores else None,
        "avg_score": (sum(scores) / len(scores)) if scores else None,
        "errors": _find_errors(run_dir, [g for g in genids if isinstance(g, int)], domain),
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
