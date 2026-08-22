"""Paths and constants for the deep-swe domain.

deep-swe (github.com/datacurve-ai/deep-swe) is a Harbor/Pier-format SWE
benchmark: 113 feature/bugfix/enhancement tasks against real open-source
repos (TypeScript/Go/Python/Rust/JavaScript), each graded by actually running
the repo's test suite -- F2P (tests that must newly pass) and P2P (tests that
must not regress) -- rather than an LLM judge. See domains/deep_swe/harness.py
for how this gets driven via the `pier` CLI (github.com/datacurve-ai/pier,
installed as the `datacurve-pier` package).

Unlike domains/research (where the ground-truth rubric only had to be kept
out of the meta-agent's local filesystem view), deep-swe's task repo,
including the oracle solution.patch for every task, is publicly fetchable
from GitHub (confirmed live: `raw.githubusercontent.com/datacurve-ai/deep-swe/
main/tasks/<id>/solution/solution.patch` returns 200 with no auth). Hiding
local files does nothing against that. The real mitigation is the one the
benchmark's own task.toml already declares: `network_mode = "no-network"` for
the agent. domains/deep_swe/pier_agent.py's network_allowlist() must never
open more than the LLM provider's own API domain, and the SWE task agent
itself is never given the fetch_url tool (see swe_task_agent.py).
"""

import os
from pathlib import Path

# deep-swe is a sibling checkout, not a subpackage of this repo.
# Override with DEEPSWE_ROOT if it lives somewhere else.
DEEPSWE_ROOT = Path(os.environ.get("DEEPSWE_ROOT", Path(__file__).resolve().parents[3] / "deep-swe"))
DEEPSWE_TASKS_DIR = DEEPSWE_ROOT / "tasks"

# Fixed task-pool split, chosen once and never resampled -- see
# domains/deep_swe/subsets/scoring_subset.json. Hand-picked as the smallest
# F2P+P2P test counts across the 113-task corpus (computed directly from
# each task's tests/config.json) while keeping some language diversity, since
# the benchmark ships no curated "lite"/"verified" subset of its own and
# wall-clock time is dominated by running each repo's real test suite.
SUBSETS_DIR = Path(__file__).resolve().parent / "subsets"
SCORING_SUBSET_PATH = SUBSETS_DIR / "scoring_subset.json"

# A second, disjoint 8-task subset (verified: zero overlap with
# scoring_subset.json), same "smallest test count" selection logic with a
# deliberate lean toward language spread (3 go / 3 typescript / 2 python --
# skipped rust/javascript entirely since their cheapest remaining options
# were 3-10x pricier than everything else in range). This is deep-swe's
# "val" split (see utils/domain_utils.py's get_domain_splits) -- generation
# metadata.json exposes its aggregate score, but the raw eval directory
# (deep_swe_eval_val/) is stripped from what any meta-agent's own container
# ever sees, by the SAME existing generate_loop.py exclusion rule
# (copy_prev_eval_to_container's *_eval_val*/*_eval_test* prune) that
# research/polyglot's own val/test splits already rely on -- confirmed live
# by reading that function before wiring this up, not assumed. The point:
# the meta-agent gets rich diagnostic detail on the visible 8 (what it reads
# to decide what to fix) but whether a fix survives into the lineage is
# decided by held-out generalization (select_parent already prefers "val"
# over "train" scoring when a domain has a val split -- this hooks into
# that, not a new selection mechanism), not by how well the visible 8 do.
HELDOUT_SUBSET_PATH = SUBSETS_DIR / "heldout_subset.json"

# Static task_id -> language mapping for every task in scoring_subset.json and
# heldout_subset.json (16 total), read once from each task's own task.toml
# ([metadata].language) and baked in here rather than looked up live: the
# meta-agent's own container never has DEEPSWE_TASKS_DIR mounted (only
# BASELINE_FILES is), so agent/tools/compare_generations.py -- which DOES run
# inside that container -- needs this available as a plain repo file. No
# information-hiding concern in exposing it (see this module's own docstring:
# the whole deep-swe task repo, including solutions, is already publicly
# fetchable).
TASK_LANGUAGES_PATH = SUBSETS_DIR / "task_languages.json"

AGENT_MODEL = "deepseek/deepseek-v4-flash"

# The benchmark's own per-task defaults (task.toml: 5400s/90min agent budget,
# 1800s/30min verifier budget) are sized for headline leaderboard runs, not a
# fast self-modification loop. Passed to `pier run` as
# --agent-timeout-multiplier / --verifier-timeout-multiplier.
AGENT_TIMEOUT_MULTIPLIER = 0.3
VERIFIER_TIMEOUT_MULTIPLIER = 1.0

# LLM provider API domains to allow through the agent's otherwise
# network_mode="no-network" container -- see this module's own docstring for
# why this must stay as narrow as possible (never add github.com or any
# general-purpose domain here).
PROVIDER_API_DOMAINS = {
    "deepseek": ["api.deepseek.com"],
    "anthropic": ["api.anthropic.com"],
    "openai": ["api.openai.com"],
    "gemini": [".googleapis.com"],
}

# Files copied into every task container -- an explicit allowlist, not a
# wholesale repo copy, matching domains/research/harness.py's own
# BASELINE_FILES rationale (keep container contents minimal and auditable).
# Lives here (not in pier_agent.py) so harness.py -- which runs in this
# project's own venv, not the separate Python that `pier` itself is
# installed under -- can read it without importing anything from the `pier`
# package at all.
BASELINE_FILES = [
    "swe_task_agent.py",
    "run_swe_task_agent.py",
    "requirements-deep_swe.txt",
    "agent/",
    "utils/",
]
