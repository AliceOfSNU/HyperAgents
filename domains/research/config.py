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

# Fixed task pool, chosen once and never resampled (see
# domains/research/subsets/scoring_subset.json).
SUBSETS_DIR = Path(__file__).resolve().parent / "subsets"
SCORING_SUBSET_PATH = SUBSETS_DIR / "scoring_subset.json"
