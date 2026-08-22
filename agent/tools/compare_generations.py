"""Tabular cross-generation summary for the deep_swe domain: per-generation
scores plus cheap behavioral metrics (rounds, planning frequency, read/write/
test-run mix, token usage, per-language breakdown), computed from files
already readable inside the meta-agent's own container (report.json,
metadata.json, and the visible subset's chat_history.json trajectories under
--evals_folder). Exists so the meta-agent doesn't have to open every prior
generation's report.json and chat_history.json by hand to see how they
compare.

Held-out ("val") per-task detail is deliberately never read here -- only the
already-exposed metadata.json aggregate (deep_swe_val_node_utility). The raw
deep_swe_eval_val/ directory is stripped before this container ever sees it
(generate_loop.py's own exclusion rule); this tool doesn't try to work around
that, it just surfaces what's already visible in one place.
"""
import json
import re
from pathlib import Path

# Resolved relative to this file's own location rather than imported from
# domains.deep_swe.config -- that package-relative import only resolves when
# something puts the repo root on sys.path (e.g. `python -m agent.tools...`
# from the repo root). run_meta_agent.py itself is launched as a plain
# script (`python run_meta_agent.py ...` from within /hyperagents/), so the
# import silently failed in the one context that actually matters, and the
# try/except below turned that into a silent "all languages show as ?"
# instead of an error. Same fixed-parents[2] pattern already used by
# domains/deep_swe/branch_pier_agent.py for the same reason.
TASK_LANGUAGES_PATH = Path(__file__).resolve().parents[2] / "domains" / "deep_swe" / "subsets" / "task_languages.json"


def tool_info():
    return {
        "name": "compare_generations",
        "description": """Print a table comparing every prior generation in this lineage: visible and held-out scores, plus cheap behavioral metrics -- avg rounds/task, planning-note frequency, bash-command read/write/test-run mix, token usage (when available), and a per-language score breakdown.

Reads only files already under `evals_folder` (report.json, metadata.json, and the visible subset's own chat_history.json trajectories) -- nothing this tool does grants access beyond what you could already read by hand with bash, it just aggregates it.

Behavioral metrics are heuristic, not exact:
- read/write/test-run counts come from regex-classifying each bash tool call's command string (the task agent only exposes one generic `bash` tool, no separate read/write/test tool names).
- "reasoning token %" is only populated for generations run after token-usage capture was added to agent/llm.py -- older generations show "n/a".

`genids` optionally limits which generations to include (e.g. [1, 3, 4]); omit it to compare everything found under evals_folder.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "evals_folder": {
                    "type": "string",
                    "description": "Path to the run's output directory, as you see it (e.g. the /tmp/generate_... path named in your own instructions).",
                },
                "genids": {
                    "type": "array",
                    "items": {"type": ["string", "integer"]},
                    "description": "Optional subset of generation ids to compare (e.g. [1, 3, 4]). Omit to compare every generation found.",
                },
            },
            "required": ["evals_folder"],
        },
    }


# Checked in this order (first match wins) so e.g. `go test ./... > out.txt`
# counts as a test run, not a write.
_TEST_PATTERNS = [
    re.compile(r"\bgo\s+(test|vet|build)\b"),
    re.compile(r"\bpytest\b"),
    re.compile(r"\bpython3?\s+-m\s+(pytest|unittest)\b"),
    re.compile(r"\b(npm|yarn|pnpm)\s+(run\s+)?(test|build)\b"),
    re.compile(r"\bjest\b"),
    re.compile(r"\bcargo\s+(test|build|check)\b"),
    re.compile(r"\btsc\b"),
]
_WRITE_PATTERNS = [
    re.compile(r"\bsed\s+-i\b"),
    re.compile(r"\bgit\s+apply\b"),
    re.compile(r"\bpatch\s+-p"),
    re.compile(r"\bgofmt\s+-w\b"),
    re.compile(r"\bmv\s"),
    re.compile(r"\brm\s"),
    re.compile(r"\bmkdir\s"),
    re.compile(r"<<\s*['\"]?[A-Za-z_]+['\"]?"),  # heredoc -- usually writing a file
    re.compile(r"[^2\d]>\s*[^&(]"),  # redirection into a file (roughly excludes 2>&1-style fd dups)
]
_READ_PATTERNS = [
    re.compile(r"\bcat\s"),
    re.compile(r"\bgrep\b"),
    re.compile(r"\brg\b"),
    re.compile(r"\bfind\s"),
    re.compile(r"\bls\b"),
    re.compile(r"\bsed\s+-n\b"),
    re.compile(r"\bhead\s"),
    re.compile(r"\btail\s"),
    re.compile(r"\bwc\s"),
    re.compile(r"\bgit\s+(status|diff|log|show|branch)\b"),
    re.compile(r"\bpwd\b"),
]


def _classify_bash(cmd):
    for pat in _TEST_PATTERNS:
        if pat.search(cmd):
            return "test"
    for pat in _WRITE_PATTERNS:
        if pat.search(cmd):
            return "write"
    for pat in _READ_PATTERNS:
        if pat.search(cmd):
            return "read"
    return "other"


def _task_stats(chat_history_path):
    try:
        rounds = json.loads(chat_history_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    n_rounds = len(rounds)
    n_plan = sum(1 for r in rounds if (r.get("plan") or "").strip())
    counts = {"test": 0, "write": 0, "read": 0, "other": 0}
    prompt_tok = completion_tok = reasoning_tok = 0
    has_usage = False
    for r in rounds:
        usage = r.get("usage") or {}
        if any(usage.get(k) is not None for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens")):
            has_usage = True
            prompt_tok += usage.get("prompt_tokens") or 0
            completion_tok += usage.get("completion_tokens") or 0
            reasoning_tok += usage.get("reasoning_tokens") or 0
        for call in (r.get("act") or []):
            if call.get("name") == "bash":
                cmd = (call.get("arguments") or {}).get("command", "")
                counts[_classify_bash(cmd)] += 1
    return {
        "n_rounds": n_rounds,
        "plan_frac": (n_plan / n_rounds) if n_rounds else 0.0,
        "counts": counts,
        "prompt_tok": prompt_tok if has_usage else None,
        "completion_tok": completion_tok if has_usage else None,
        "reasoning_tok": reasoning_tok if has_usage else None,
    }


def _gen_sort_key(name):
    # "gen_initial" first, then gen_1, gen_2, ... numerically.
    suffix = name[len("gen_"):]
    return (0, -1) if suffix == "initial" else (1, int(suffix))


def _find_trajectory(eval_dir, task_id):
    """Trial directories are named "<task_id>__<random suffix>", not a bare
    task_id match (confirmed live against real output) -- glob for it rather
    than assuming an exact path, matching harness.py's own more thorough
    task_name-based lookup in spirit if not in mechanism."""
    exact = eval_dir / task_id / "agent" / "chat_history.json"
    if exact.exists():
        return exact
    matches = sorted(eval_dir.glob(f"{task_id}__*/agent/chat_history.json"))
    return matches[0] if matches else None


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt_score(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "-"


def _fmt_pct(x):
    return f"{x*100:.0f}%" if isinstance(x, (int, float)) else "-"


def tool_function(evals_folder, genids=None):
    try:
        base = Path(evals_folder)
        if not base.is_dir():
            return f"Error: '{evals_folder}' is not a directory."

        languages = _load_json(TASK_LANGUAGES_PATH) or {}

        wanted = {str(g) for g in genids} if genids else None
        gen_dirs = sorted(
            (d for d in base.glob("gen_*") if d.is_dir()),
            key=lambda d: _gen_sort_key(d.name),
        )

        rows = []
        for gdir in gen_dirs:
            suffix = gdir.name[len("gen_"):]
            if wanted and suffix not in wanted:
                continue

            report = _load_json(gdir / "deep_swe_eval" / "report.json")
            if report is None:
                continue  # not evaluated (yet), e.g. an empty-diff or in-flight generation

            metadata = _load_json(gdir / "metadata.json") or {}
            per_task = report.get("per_task") or []

            n_rounds_total = n_tasks_with_traj = 0
            plan_frac_sum = 0.0
            counts_total = {"test": 0, "write": 0, "read": 0, "other": 0}
            tok_prompt = tok_completion = tok_reasoning = 0
            any_usage = False
            lang_scores = {}  # language -> [rewards]

            for t in per_task:
                task_id = t.get("task_id")
                reward = t.get("reward")
                lang = languages.get(task_id, "?")
                lang_scores.setdefault(lang, []).append(reward if isinstance(reward, (int, float)) else 0.0)

                traj_path = _find_trajectory(gdir / "deep_swe_eval" / "eval", task_id)
                stats = _task_stats(traj_path) if traj_path else None
                if stats is None:
                    continue
                n_tasks_with_traj += 1
                n_rounds_total += stats["n_rounds"]
                plan_frac_sum += stats["plan_frac"]
                for k in counts_total:
                    counts_total[k] += stats["counts"][k]
                if stats["prompt_tok"] is not None:
                    any_usage = True
                    tok_prompt += stats["prompt_tok"]
                    tok_completion += stats["completion_tok"]
                    tok_reasoning += stats["reasoning_tok"]

            avg_rounds = (n_rounds_total / n_tasks_with_traj) if n_tasks_with_traj else None
            avg_plan_frac = (plan_frac_sum / n_tasks_with_traj) if n_tasks_with_traj else None
            reads, writes, tests = counts_total["read"], counts_total["write"], counts_total["test"]
            edit_read = (writes / reads) if reads else (float("inf") if writes else None)
            avg_tests = (tests / n_tasks_with_traj) if n_tasks_with_traj else None
            reasoning_pct = (tok_reasoning / tok_completion * 100) if (any_usage and tok_completion) else None

            lang_summary = " ".join(
                f"{lang}:{sum(rs)/len(rs):.2f}({len(rs)})"
                for lang, rs in sorted(lang_scores.items())
            )

            rows.append({
                "gen": gdir.name,
                "parent": metadata.get("parent_genid", "-"),
                "visible": report.get("node_utility"),
                "heldout": metadata.get("deep_swe_val_node_utility"),
                "avg_rounds": avg_rounds,
                "plan_pct": avg_plan_frac,
                "edit_read": edit_read,
                "avg_tests": avg_tests,
                "reasoning_pct": reasoning_pct,
                "langs": lang_summary,
            })

        if not rows:
            return f"No evaluated generations found under '{evals_folder}' (checked for gen_*/deep_swe_eval/report.json)."

        header = f"{'Gen':<12} {'Parent':<7} {'Visible':<8} {'Held-out':<9} {'AvgRnd':<7} {'Plan%':<6} {'Edit/Read':<10} {'Tests/Task':<11} {'ReasonTok%':<11} Languages(score(n))"
        lines = [header, "-" * len(header)]
        for r in rows:
            edit_read_str = "-" if r["edit_read"] is None else ("inf" if r["edit_read"] == float("inf") else f"{r['edit_read']:.2f}")
            reasoning_str = "n/a" if r["reasoning_pct"] is None else f"{r['reasoning_pct']:.0f}%"
            avg_rounds_str = "-" if r["avg_rounds"] is None else f"{r['avg_rounds']:.1f}"
            avg_tests_str = "-" if r["avg_tests"] is None else f"{r['avg_tests']:.1f}"
            lines.append(
                f"{r['gen']:<12} {str(r['parent']):<7} {_fmt_score(r['visible']):<8} {_fmt_score(r['heldout']):<9} "
                f"{avg_rounds_str:<7} {_fmt_pct(r['plan_pct']):<6} {edit_read_str:<10} {avg_tests_str:<11} {reasoning_str:<11} {r['langs']}"
            )
        lines.append("")
        lines.append(
            "Visible/Held-out are node_utility (mean reward) on the train/val task subsets. "
            "Edit/Read is write-classified / read-classified bash commands (heuristic -- see this tool's own description). "
            "Tests/Task is avg test-or-build-runner invocations per task. ReasonTok% is reasoning_tokens/completion_tokens "
            "summed across the visible subset's trajectories, \"n/a\" for generations predating token-usage capture."
        )
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python compare_generations.py <evals_folder> [genid ...]")
    else:
        print(tool_function(sys.argv[1], sys.argv[2:] or None))
