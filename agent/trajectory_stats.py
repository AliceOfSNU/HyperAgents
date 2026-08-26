"""Shared trial-lookup/classification plumbing -- used by
agent/tools/compare_generations.py and domains/deep_swe/patch_test_orchestrator.py.
Pure functions, no LLM calls, no I/O beyond reading the paths callers hand
it. Same "shared helper outside agent/tools/" pattern as
agent/trajectory_replay.py and agent/memory_store.py, required because
agent/tools/__init__.py's load_tools() rejects any .py file there lacking
both tool_info()/tool_function().

Deliberately narrow: this module used to also carry a mechanical
"credit-assignment" layer (round_budget_events/candidate_pivot_rounds/
verdict_from_reward, behind a trajectory_facts tool) -- removed. Two
reasons, both real: (1) it was tied to deep_swe/SWE-bench's own reward
shape (f2p/p2p, ctrf.json's message format) and would need rewriting for
every different benchmark, despite being pitched as a general capability;
(2) surfacing "candidates" to the meta-agent -- even purely mechanically
flagged ones, with no LLM judgment in them -- still steers which rounds it
looks at first, when the actual trajectory and eval scores are already
fully readable by the agent's own bash/editor access. Better for it to read
the raw material and decide what's salient itself than to be handed a
pre-narrowed list. See memory: dgm_h_credit_assignment for the fuller
history if this gets reconsidered later.
"""
import re
from pathlib import Path

# Checked in this order (first match wins) so e.g. `go test ./... > out.txt`
# counts as a test run, not a write. Moved verbatim from
# agent/tools/compare_generations.py's own former _TEST_PATTERNS etc., now
# shared so both that tool and patch_test_orchestrator.py classify bash
# commands identically instead of maintaining two copies.
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


def classify_bash(cmd):
    """"test" / "write" / "read" / "other", by regex on the command string --
    the task agent only exposes one generic bash tool, no separate
    read/write/test tool names to key off of instead."""
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


def find_trial_dir(eval_dir, task_id):
    """Trial directories are named "<task_id>__<random suffix>" -- except
    Pier truncates the task_id part to a fixed length for long task names
    (confirmed live: exactly 33 characters, e.g.
    "boa-hierarchical-evaluation-cancellation" -> directory prefix
    "boa-hierarchical-evaluation-canc"), so a plain f"{task_id}__*" glob
    silently finds nothing for any task_id longer than that. Matches by
    prefix relationship instead of a fixed length (future-proof if Pier's
    own truncation length ever changes): a directory's own prefix (its name
    before the last "__") must be a prefix of the real task_id, in either
    direction. Looks for the directory itself (via result.json, which every
    trial has), not for chat_history.json specifically -- see
    find_trajectory below for that narrower, chat_history.json-only variant
    (kept deliberately un-fixed, see its own docstring)."""
    eval_dir = Path(eval_dir)
    exact = eval_dir / task_id
    if (exact / "result.json").exists():
        return exact
    candidates = []
    for result_path in eval_dir.glob("*/result.json"):
        dir_name = result_path.parent.name
        prefix = dir_name.rsplit("__", 1)[0] if "__" in dir_name else dir_name
        if task_id.startswith(prefix) or prefix.startswith(task_id):
            candidates.append(result_path.parent)
    if not candidates:
        return None
    # Prefer the longest (most specific) prefix match if more than one
    # directory's prefix happens to be a prefix of task_id.
    candidates.sort(key=lambda p: len(p.name), reverse=True)
    return candidates[0]


def find_trajectory(eval_dir, task_id):
    """Moved verbatim from agent/tools/compare_generations.py's own former
    _find_trajectory -- kept independent from find_trial_dir above (rather
    than built on top of it) so this tool's byte-for-byte output can't shift
    from an edge case where the two globs would disagree (e.g. multiple
    trial dirs for one task_id, one missing chat_history.json but sorting
    first) -- shouldn't happen in practice, but this function's whole job is
    matching compare_generations.py's prior behavior exactly, not being
    clever about it."""
    eval_dir = Path(eval_dir)
    exact = eval_dir / task_id / "agent" / "chat_history.json"
    if exact.exists():
        return exact
    matches = sorted(eval_dir.glob(f"{task_id}__*/agent/chat_history.json"))
    return matches[0] if matches else None
