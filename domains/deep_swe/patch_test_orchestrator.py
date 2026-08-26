"""Host-side half of skills/patch_testing (see skills/patch_testing/SKILL.md
for the meta-agent-facing picture, and agent/tools/test_patch.py for the
container-side request/poll half). Processes pending requests under
<root_dir>/skills/patch_testing/requests/: for each checkpoint (primary,
then any regression checkpoints), reconstructs that checkpoint's own
original generation code, runs it unmodified (control) and with the
request's patch applied on top of that SAME snapshot (treatment) --
replicates_per_arm times each, to full completion with a real reward -- and
writes a report.

Runs CONCURRENTLY with the meta-agent's own session (not deferred to after,
like skills/branching's own branch_orchestrator.py) -- see
generate_loop.py's own watch_loop() call site, started on a background
thread right before the blocking container.exec_run() that runs the
meta-agent, so a synchronous test_patch tool call can actually see a result
before that session ends. Full docker/Pier access, same trust boundary as
domains/deep_swe/harness.py's own evaluation runs -- the meta-agent's own
container never runs any of this directly.

Deliberately takes root_dir AND output_dir as explicit parameters from
generate_loop.py's own scope, rather than inferring a host path from a
container-side path string the way branch_orchestrator.py's
_translate_container_path/_parse_run_and_genid do -- that inference was
confirmed broken (it only ever round-trips a path that was already
host-side, not an actual /tmp/<run> container path, and has never been
exercised live since skills/branching has zero real invocations). Knowing
which run's own output_dir this orchestrator is servicing outright, because
generate_loop.py already has that value in scope when it starts this
watcher, sidesteps the whole translation problem instead of fixing it.
"""
import json
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from agent.trajectory_stats import find_trial_dir

from .harness import _reconstruct_code

K_HARD_CAP = 5
REPLICATES_HARD_CAP = 5
ROLLOUT_AGENT_TIMEOUT_MULTIPLIER = 1.0  # full runs, not truncated probes -- see SKILL.md
ROLLOUT_VERIFIER_TIMEOUT_MULTIPLIER = 1.0
POLL_LOOP_INTERVAL_SEC = 5


def _log(message):
    print(f"[patch_testing] {message}", file=sys.stderr, flush=True)


def _requests_dir(root_dir):
    return Path(root_dir) / "skills" / "patch_testing" / "requests"


def _reports_dir(root_dir):
    return Path(root_dir) / "skills" / "patch_testing" / "reports"


def _load_pending_requests(root_dir):
    reqs_dir = _requests_dir(root_dir)
    if not reqs_dir.is_dir():
        return []
    pending = []
    for path in sorted(reqs_dir.glob("*.json")):
        try:
            request = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            _log(f"skipping malformed request file {path}")
            continue
        if not _already_processed(root_dir, request.get("request_id")):
            pending.append(request)
    return pending


def _already_processed(root_dir, request_id):
    return (_reports_dir(root_dir) / str(request_id) / "report.json").exists()


def _find_checkpoint_trial(output_dir, genid, task_id):
    genid_str = str(genid)
    gen_dir_name = genid_str if genid_str.startswith("gen_") else f"gen_{genid_str}"
    eval_dir = Path(output_dir) / gen_dir_name / "deep_swe_eval" / "eval"
    return find_trial_dir(eval_dir, task_id) if eval_dir.is_dir() else None


def _load_task_context(trial_dir):
    config = json.loads((trial_dir / "config.json").read_text(encoding="utf-8"))
    task_path = config["task"]["path"]
    instruction = (Path(task_path) / "instruction.md").read_text(encoding="utf-8")
    return {
        "task_path": task_path,
        "task_id": Path(task_path).name,
        "model": config["agent"]["model_name"],
        "instruction": instruction,
    }


def _sample_random_checkpoints(output_dir, n, exclude=()):
    """Uniformly sample n (genid, task_id, round) checkpoints from whatever
    evaluated generations exist under output_dir -- only rounds carrying a
    recorded "state" field are eligible (see swe_task_agent.py's own
    state-machine docstring: only trajectories produced after that refactor
    are checkpoint-testable), so older generations are silently skipped as
    candidates, not errored on."""
    candidates = []
    exclude_keys = {(str(c["genid"]), c["task_id"], c["round"]) for c in exclude}
    for gen_dir in Path(output_dir).glob("gen_*"):
        eval_dir = gen_dir / "deep_swe_eval" / "eval"
        if not eval_dir.is_dir():
            continue
        genid = gen_dir.name[len("gen_"):]
        for trial_dir in eval_dir.iterdir():
            chat_history_path = trial_dir / "agent" / "chat_history.json"
            if not chat_history_path.exists():
                continue
            try:
                rounds = json.loads(chat_history_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            task_id = trial_dir.name.split("__", 1)[0]
            for record in rounds:
                if record.get("state") is None:
                    continue
                key = (genid, task_id, record.get("round"))
                if key in exclude_keys:
                    continue
                candidates.append({"genid": genid, "task_id": task_id, "round": record.get("round")})
    random.shuffle(candidates)
    return candidates[:n]


def _build_original_code_dir(root_dir, output_dir, genid, dest_dir):
    """Reconstruct the exact code snapshot that generation actually ran with
    -- reuses harness.py's own _reconstruct_code (BASELINE_FILES copy +
    that generation's own patch chain replay + wheel prefetch), same as
    branch_orchestrator.py's _build_code_root_dir does, since it's the same
    operation for the same reason."""
    _reconstruct_code(root_dir, output_dir, genid, dest_dir)


def _check_and_apply_patch(dest_dir, patch_text):
    """Reconstructed code dirs are plain copied file trees, not git repos
    (see harness.py's _reconstruct_code, which applies patches via the
    plain `patch` command, not `git apply`) -- so patch application here
    uses the same `patch -p1` convention, not git. Returns
    (applies: bool, detail: str). Checks with --dry-run first so a patch
    that doesn't cleanly apply to this checkpoint's own code (e.g. it
    depends on something added after this generation) is reported as such,
    never partially or incorrectly applied."""
    check = subprocess.run(
        ["patch", "-p1", "--dry-run", "-f"], cwd=dest_dir,
        input=patch_text, capture_output=True, text=True, timeout=60,
    )
    if check.returncode != 0:
        return False, (check.stdout or check.stderr or "patch --dry-run failed")[:500]
    apply_result = subprocess.run(
        ["patch", "-p1", "-f"], cwd=dest_dir,
        input=patch_text, capture_output=True, text=True, timeout=60,
    )
    if apply_result.returncode != 0:
        return False, (apply_result.stdout or apply_result.stderr or "patch apply failed after a clean --dry-run (unexpected)")[:500]
    return True, "applied cleanly"


def _pier_run(cmd_extra, jobs_dir, job_name, timeout_sec):
    cmd = ["pier", "run", "--jobs-dir", str(jobs_dir), "--job-name", job_name, "--quiet"] + cmd_extra
    _log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    if result.returncode != 0:
        _log(f"pier run ({job_name}) exited {result.returncode}. stderr tail:\n{result.stderr[-3000:]}")
    return result


def _load_replicate_rewards(jobs_dir, job_name):
    """One entry per trial dir under jobs_dir/job_name/ -- real reward from
    that replicate's own result.json (harness.py's own sole source of
    truth for a trial's outcome; verifier/reward.json is a verified
    byte-identical duplicate that's sometimes absent, e.g. on an infra
    failure, so this never reads it instead), or an explicit error detail
    if the replicate never produced one (infra failure, timeout, crash)
    rather than silently omitting it."""
    job_dir = Path(jobs_dir) / job_name
    out = []
    if not job_dir.is_dir():
        return out
    for trial_dir in sorted(job_dir.iterdir()):
        result_path = trial_dir / "result.json"
        if not result_path.exists():
            out.append({"reward": None, "detail": f"no result.json in {trial_dir.name} -- replicate likely errored before completing"})
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as e:
            out.append({"reward": None, "detail": f"result.json unreadable: {e}"})
            continue
        verifier_result = result.get("verifier_result")
        if verifier_result is None:
            exc = (result.get("exception_info") or {}).get("exception_message", "")
            out.append({"reward": None, "detail": f"infra failure, not an agent-quality signal: {exc[:300]}"})
            continue
        rewards = verifier_result.get("rewards") or {}
        out.append({"reward": rewards.get("reward"), "detail": f"partial={rewards.get('partial')}"})
    return out


def _run_arm(task_ctx, code_dir, source_chat_history_path, checkpoint_round, replicates, jobs_dir, job_name, model, agent_timeout_multiplier, timeout_sec):
    _pier_run(
        [
            "-p", task_ctx["task_path"],
            "--agent-import-path", "domains.deep_swe.patch_test_pier_agent:PatchTestPierAgent",
            "-m", model,
            "--ak", f"root_dir={code_dir}",
            "--ak", f"source_chat_history_path={source_chat_history_path}",
            "--ak", f"checkpoint_round={checkpoint_round}",
            "--ak", "temperature=0.7",
            "-e", "docker",
            "-k", str(replicates),
            "-n", str(min(replicates, 4)),
            "--agent-timeout-multiplier", str(agent_timeout_multiplier),
            "--verifier-timeout-multiplier", str(ROLLOUT_VERIFIER_TIMEOUT_MULTIPLIER),
        ],
        jobs_dir=jobs_dir, job_name=job_name, timeout_sec=timeout_sec,
    )
    return _load_replicate_rewards(jobs_dir, job_name)


def _process_checkpoint(root_dir, output_dir, role, checkpoint, patch_text, replicates_per_arm, rollout_timeout_sec):
    genid, task_id, round_num = checkpoint["genid"], checkpoint["task_id"], checkpoint["round"]
    result = {"role": role, "genid": genid, "task_id": task_id, "round": round_num}

    trial_dir = _find_checkpoint_trial(output_dir, genid, task_id)
    if trial_dir is None:
        result["patch_applies"] = False
        result["patch_apply_detail"] = f"no trial found for gen_{genid}/{task_id} under {output_dir}"
        return result

    try:
        task_ctx = _load_task_context(trial_dir)
    except Exception as e:
        result["patch_applies"] = False
        result["patch_apply_detail"] = f"could not load task context: {e}"
        return result

    source_chat_history_path = trial_dir / "agent" / "chat_history.json"

    control_dir = tempfile.mkdtemp(prefix=f"patchtest_control_{genid}_")
    treatment_dir = None
    try:
        _build_original_code_dir(root_dir, output_dir, genid, control_dir)

        with tempfile.TemporaryDirectory(prefix=f"patchtest_jobs_{genid}_") as jobs_dir:
            result["control"] = _run_arm(
                task_ctx, control_dir, source_chat_history_path, round_num, replicates_per_arm,
                jobs_dir, "control", task_ctx["model"], ROLLOUT_AGENT_TIMEOUT_MULTIPLIER, rollout_timeout_sec,
            )

        treatment_dir = tempfile.mkdtemp(prefix=f"patchtest_treatment_{genid}_")
        shutil.copytree(control_dir, treatment_dir, dirs_exist_ok=True)
        applies, detail = _check_and_apply_patch(treatment_dir, patch_text)
        result["patch_applies"] = applies
        result["patch_apply_detail"] = detail
        if not applies:
            return result

        with tempfile.TemporaryDirectory(prefix=f"patchtest_jobs_treat_{genid}_") as jobs_dir:
            result["treatment"] = _run_arm(
                task_ctx, treatment_dir, source_chat_history_path, round_num, replicates_per_arm,
                jobs_dir, "treatment", task_ctx["model"], ROLLOUT_AGENT_TIMEOUT_MULTIPLIER, rollout_timeout_sec,
            )
    except Exception as e:
        result["error"] = str(e)
        _log(f"checkpoint gen_{genid}/{task_id}#{round_num} ({role}) failed: {e}")
    finally:
        shutil.rmtree(control_dir, ignore_errors=True)
        if treatment_dir:
            shutil.rmtree(treatment_dir, ignore_errors=True)

    return result


def process_one_request(root_dir, output_dir, request, rollout_timeout_sec=2400):
    request_id = request["request_id"]
    report_dir = _reports_dir(root_dir) / request_id
    report = {"request_id": request_id, "goal": request.get("goal"), "status": "failed", "error": None, "checkpoints": []}

    try:
        replicates_per_arm = max(1, min(int(request.get("replicates_per_arm", 3)), REPLICATES_HARD_CAP))
        patch_text = request.get("patch_text")
        if not patch_text:
            report["error"] = "request carried no patch_text"
            _write_report(report_dir, report)
            return report

        primary = request["primary_checkpoint"]
        checkpoints = [("primary", primary)]

        for cp in request.get("regression_checkpoints") or []:
            checkpoints.append(("regression", cp))

        n_random = int(request.get("n_random_regression") or 0)
        if n_random > 0:
            already = [primary] + (request.get("regression_checkpoints") or [])
            for cp in _sample_random_checkpoints(output_dir, n_random, exclude=already):
                checkpoints.append(("regression", cp))

        for role, cp in checkpoints:
            report["checkpoints"].append(
                _process_checkpoint(root_dir, output_dir, role, cp, patch_text, replicates_per_arm, rollout_timeout_sec)
            )

        report["status"] = "done"
    except Exception as e:
        report["error"] = str(e)
        report["status"] = "failed"
        _log(f"request {request_id} failed: {e}")

    _write_report(report_dir, report)
    return report


def _write_report(report_dir, report):
    report_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = report_dir / "report.json.tmp"
    final_path = report_dir / "report.json"
    tmp_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    tmp_path.replace(final_path)  # atomic swap -- test_patch.py polls for final_path and must never see a partial write


def watch_loop(root_dir, output_dir, stop_event, rollout_timeout_sec=2400):
    """Entry point run on a background thread from generate_loop.py, for
    the duration of one generation's own meta-agent session (started right
    before the blocking container.exec_run() that runs it, stopped right
    after). Polls for new requests every POLL_LOOP_INTERVAL_SEC -- silent
    when there's nothing to do, matching branch_orchestrator.py's own
    "runs every generation regardless of whether the skill was ever used"
    posture."""
    while not stop_event.is_set():
        for request in _load_pending_requests(root_dir):
            if stop_event.is_set():
                break
            _log(f"processing patch-test request {request.get('request_id')}")
            try:
                process_one_request(root_dir, output_dir, request, rollout_timeout_sec=rollout_timeout_sec)
            except Exception as e:
                _log(f"unhandled error processing request {request.get('request_id')}: {e}")
        stop_event.wait(POLL_LOOP_INTERVAL_SEC)
