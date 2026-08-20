"""Scores a research report against an RCB checklist using RCB's own rubric
prompt, restricted to text-type items (image-type items are out of scope for
this domain -- see the design discussion this module implements).

RCB's own scorer (ResearchClawBench/evaluation/score.py) is hard-wired to the
OpenAI SDK shape (it builds an `openai.OpenAI(base_url=..., api_key=...)`
client internally) and can't reach Claude, which has no OpenAI-compatible
endpoint. `score_item_with_claude` below calls Claude directly via the real
`anthropic` SDK instead, reusing RCB's exact rubric text so the two scores
stay comparable.

This module is host-side only: it's deliberately excluded both from
domains/research/harness.py's BASELINE_FILES (so it never reaches any
task-agent sandboxed container) and from root_dir's own copy via
utils/gl_utils.py's setup_initial_gen (so the meta-agent's bind-mounted
container can't read the ground-truth rubric while writing task_agent.py --
see gl_utils.py's own comment on why that alone isn't sufficient, since the
underlying RCB checklist is also public on GitHub). The generic
weight-aggregation logic every caller actually needs at runtime
(score_report_text_items) lives in scoring_utils.py instead, which has none
of the ground-truth content and stays fully visible/copyable.
"""

import json

from .config import CLAUDE_JUDGE_MODEL

# Copied verbatim from ResearchClawBench/evaluation/score.py so real_score is
# graded against the identical rubric RCB's own judge would use.
RUBRIC = """You are a strict scientific peer reviewer evaluating an AI agent's ability to conduct end-to-end automated scientific research.

You are given:
1. The INSTRUCTIONS.md that was provided to the AI agent (the research task it was asked to solve).
2. The AI-generated research report (the agent's output).
3. A specific evaluation criterion derived from the original published paper.

IMPORTANT: Your role is ONLY to score the AI report against the criterion. Do NOT attempt to answer or solve the research task yourself. Focus solely on evaluating what the AI agent produced.

## Evaluation Modes

Each checklist item falls into one of two categories. Determine which applies based on the criterion's nature:

### Mode A: Objective Evaluation (Metric Optimization / Quantitative Results)
Use this when the criterion involves specific numerical results, metrics, benchmarks, or quantitative outcomes.

- **0**: The criterion is completely absent from the report.
- **1-10**: Mentioned but no quantitative results provided.
- **11-20**: Quantitative results given but the methodology has fundamental errors.
- **21-30**: Methodology has significant flaws; metrics deviate severely from the paper.
- **31-40**: Methodology is mostly correct but metrics are notably worse than the paper.
- **41-50**: Metrics are roughly comparable to the original paper.
- **51-60**: Metrics are slightly better than the paper.
- **61-70**: Metrics are clearly better than the paper.
- **71-80**: Both methodology and metrics show substantial improvements over the paper.
- **81-90**: Metrics dramatically surpass the paper.
- **91-100**: Breakthrough results far exceeding the paper.

### Mode B: Subjective Evaluation (Mechanism Analysis / Qualitative Reasoning)
Use this when the criterion involves theoretical explanations, mechanistic insights, logical arguments, or interpretive analysis.

- **0**: The criterion is completely absent from the report.
- **1-10**: Mentioned only with vague, generic statements.
- **11-20**: Some description present but no substantive analysis.
- **21-30**: Some analysis attempted but evidence is insufficient or reasoning has logical gaps.
- **31-40**: Analysis direction is correct but lacks depth; key arguments are missing.
- **41-50**: Analysis depth and logical rigor are roughly comparable to the original paper.
- **51-60**: More supporting evidence provided than the paper.
- **61-70**: More complete logical chain and more rigorous argumentation than the paper.
- **71-80**: Significantly deeper analysis; raises valuable insights not covered in the paper.
- **81-90**: Analysis depth far exceeds the paper.
- **91-100**: Original contributions with breakthrough insights beyond the paper.

## CRITICAL RULES
- 50 means "as good as the actual published paper" — this is a high bar.
- First determine if the criterion is Objective (Mode A) or Subjective (Mode B), then apply the corresponding rubric.
- No credit for vague or generic statements. Must demonstrate specific, concrete analysis.
- No inflation for well-written but shallow content. Substance over style. Longer does not mean better.
- Be highly skeptical of AI-generated content: it may sound plausible but contain factual errors, fabricated numbers, or unsupported conclusions. Verify claims against the criterion carefully.
- Be strict but fair.
"""

JUDGE_SYSTEM_PROMPT = (
    "You are a strict scientific peer reviewer evaluating AI-generated research. "
    "Score the report against the criterion only — do not attempt to solve the "
    "research task yourself."
)


def _build_text_prompt(report_text, item, instructions):
    criteria = item.get("content", "")
    keywords = item.get("keywords", [])
    keywords_str = ", ".join(keywords) if keywords else "None specified"
    return f"""{RUBRIC}

## Research Task Background (INSTRUCTIONS.md given to the AI agent)
{instructions}

## Evaluation Criterion (from the original paper)
{criteria}

## Key Technical Aspects to Verify
{keywords_str}

## AI-Generated Research Report
{report_text}

## Task
Rate how well this report addresses the criterion compared to the original paper.
First determine if this criterion is Objective (Mode A) or Subjective (Mode B), then apply the corresponding rubric strictly.

Return your answer as a JSON object: {{"reasoning": "<2-3 sentences>", "score": <0-100>}}"""


def score_item_with_claude(report_text, item, instructions, model=CLAUDE_JUDGE_MODEL):
    """Score one text-type checklist item with Claude via the real anthropic SDK."""
    import anthropic

    if item.get("type", "text") != "text":
        raise ValueError("score_item_with_claude only supports text-type items")

    prompt = _build_text_prompt(report_text, item, instructions)
    client = anthropic.Anthropic()
    # max_tokens=500 (the old value) was sized for the JSON response alone,
    # but Sonnet 5 engages extended thinking automatically on a prompt this
    # complex, and thinking tokens draw from the same max_tokens budget --
    # confirmed live: a real report's thinking alone used 367 of 500 tokens,
    # and in production every single call across a full eval run came back
    # with a stop_reason of "max_tokens" and zero text content (thinking
    # consumed the whole budget), which is what "Failed to parse judge
    # response (substring not found)" below was actually seeing. Thinking is
    # left on (not disabled) since reasoning through the rubric before
    # scoring is plausibly load-bearing for grading quality, not overhead
    # to eliminate -- adaptive is the current API's own budget management,
    # so we just need enough total room for both thinking and the response.
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    try:
        parsed = json.loads(text[text.index("{"): text.rindex("}") + 1])
        score = max(0, min(100, int(parsed.get("score", 0))))
        reasoning = str(parsed.get("reasoning", ""))
    except Exception as e:
        score, reasoning = 0, f"Failed to parse judge response ({e}): {text[:200]}"
    return {"score": score, "reasoning": reasoning}
