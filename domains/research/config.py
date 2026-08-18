"""Paths and constants for the research (ResearchClawBench) domain."""

import os
from pathlib import Path

# ResearchClawBench is a sibling checkout, not a subpackage of this repo.
# Override with RCB_ROOT if it lives somewhere else.
RCB_ROOT = Path(os.environ.get("RCB_ROOT", Path(__file__).resolve().parents[3] / "ResearchClawBench"))
RCB_TASKS_DIR = RCB_ROOT / "tasks"

# Judge model for real_score, called via the real Anthropic SDK directly
# (RCB's own scorer is hard-wired to the OpenAI SDK shape and can't reach
# Claude -- see domains/research/claude_scorer.py).
CLAUDE_JUDGE_MODEL = "claude-sonnet-5"

# Anchor-generation agent backends (deliberately two different capability
# tiers so the 30-task anchor set has real score variance for evaluator
# candidates to be distinguished on). These are RAW DeepSeek API model IDs
# (rcb-eval talks OpenAI-compatible HTTP directly, not through litellm), which
# may differ from the "deepseek/deepseek-v4-pro" litellm-routing strings used
# elsewhere in this repo -- double check against DeepSeek's own docs/dashboard
# before running domains/research/anchor.py for the first time.
ANCHOR_AGENT_MODEL_STRONG = os.environ.get("ANCHOR_AGENT_MODEL_STRONG", "deepseek-v4-pro")
ANCHOR_AGENT_MODEL_WEAK = os.environ.get("ANCHOR_AGENT_MODEL_WEAK", "deepseek-v4-flash")
ANCHOR_AGENT_API_BASE = os.environ.get("ANCHOR_AGENT_API_BASE", "https://api.deepseek.com/v1")

# Fixed task-pool split, chosen once and never resampled (see
# domains/research/subsets/scoring_subset.json / anchor_subset.json).
SUBSETS_DIR = Path(__file__).resolve().parent / "subsets"
SCORING_SUBSET_PATH = SUBSETS_DIR / "scoring_subset.json"
ANCHOR_SUBSET_PATH = SUBSETS_DIR / "anchor_subset.json"

# Cached anchor artifacts (fixed reports + real per-item Claude scores),
# generated once by domains/research/anchor.py.
ANCHOR_CACHE_PATH = SUBSETS_DIR / "anchor_cache.json"

# Epoch cadence for evaluator promotion.
EPOCH_LENGTH_GENERATIONS = 3
CHALLENGER_SAMPLE_SIZE = 3

# Root-level, meta-agent-editable files (same level as task_agent.py so they
# survive the ancestor-chain patch replay).
EVALUATOR_AGENT_FILENAME = "evaluator_agent.py"
RUN_EVALUATOR_AGENT_FILENAME = "run_evaluator_agent.py"
