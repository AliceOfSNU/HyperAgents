import json

from agent.base_agent import AgentSystem
from agent.llm import get_response_from_llm
from utils.common import extract_jsons

# Seeded from ResearchClawBench's own rubric (evaluation/score.py) so the
# starting evaluator has a reasonable prior; the meta-agent is free to edit
# this from here.
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

# Same system prompt used by the trusted Claude scorer
# (domains/research/claude_scorer.py). The evaluator's value is agreement with
# that scorer's fixed anchor, so keeping every prompt component as close as
# possible to the anchor judge minimizes avoidable mismatch from formatting
# and framing differences.
JUDGE_SYSTEM_PROMPT = (
    "You are a strict scientific peer reviewer evaluating AI-generated research. "
    "Score the report against the criterion only — do not attempt to solve the "
    "research task yourself."
)


def _build_evaluator_prompt(report_text, instructions, criterion, keywords):
    keywords_str = ", ".join(keywords) if keywords else "None specified"
    return f"""{RUBRIC}

## Research Task Background (INSTRUCTIONS.md given to the AI agent)
{instructions}

## Evaluation Criterion (from the original paper)
{criterion}

## Key Technical Aspects to Verify
{keywords_str}

## AI-Generated Research Report
{report_text}

## Task
Rate how well this report addresses the criterion compared to the original paper.
First determine if this criterion is Objective (Mode A) or Subjective (Mode B), then apply the corresponding rubric strictly.

Return your answer as a JSON object: {{"reasoning": "<2-3 sentences>", "score": <0-100>}}"""


def _parse_score_response(response):
    """Extract a score/reasoning dict from any plausible judge response."""
    try:
        extracted = extract_jsons(response)
        if extracted:
            return extracted[-1]
    except Exception:
        pass

    # Some backends ignore the <json> wrapper or return text around the JSON.
    # Match claude_scorer's own fallback: parse the first {...} span in the
    # raw response.
    try:
        start = response.index("{")
        end = response.rindex("}") + 1
        return json.loads(response[start:end])
    except Exception:
        return None


class EvaluatorAgent(AgentSystem):
    def forward(self, inputs):
        """
        Scores one ResearchClawBench checklist item (text-type only) against
        an AI-generated research report -- a co-evolving stand-in for
        ResearchClawBench's own peer-reviewer judge.

        This role shares a workspace and a meta-agent with task_agent.py
        (the research agent it grades). It is periodically checkpointed
        against a fixed, independent ground-truth anchor and only promoted
        if it beats the current incumbent -- see domains/research/harness.py.
        Its own score on any given report never affects its own promotion,
        only agreement with that anchor does, so there's no direct incentive
        to grade leniently.
        """
        instruction = _build_evaluator_prompt(
            report_text=inputs.get("report_text", ""),
            instructions=inputs.get("instructions", ""),
            criterion=inputs.get("criterion", ""),
            keywords=inputs.get("keywords", []),
        )

        response, new_msg_history, _info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[],
            system=JUDGE_SYSTEM_PROMPT,
        )

        prediction = {"score": 0, "reasoning": "Failed to parse evaluator response."}
        item = _parse_score_response(response)
        if item:
            prediction = {
                "score": max(0, min(100, int(item.get("score", 0)))),
                "reasoning": str(item.get("reasoning", "")),
            }
        return prediction, new_msg_history
