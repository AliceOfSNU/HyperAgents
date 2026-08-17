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

        Args:
            inputs (dict): {
                "report_text": str,
                "instructions": str,     # INSTRUCTIONS.md background given to the task agent
                "criterion": str,        # checklist item's "content" field
                "keywords": list[str],   # checklist item's "keywords" field
            }

        Returns:
            tuple:
                - prediction (dict): {"score": int 0-100, "reasoning": str}
                - new_msg_history (list): message history of the interaction.
        """
        keywords_str = ", ".join(inputs.get("keywords", [])) or "None specified"
        instruction = f"""{RUBRIC}

## Research Task Background (INSTRUCTIONS.md given to the AI agent)
{inputs.get('instructions', '')}

## Evaluation Criterion (from the original paper)
{inputs['criterion']}

## Key Technical Aspects to Verify
{keywords_str}

## AI-Generated Research Report
{inputs['report_text']}

## Task
Rate how well this report addresses the criterion compared to the original paper.

## Scoring Procedure (follow strictly)
1. First determine if this criterion is Objective (Mode A) or Subjective (Mode B); state which mode you are applying.
2. Search the report for concrete, verifiable evidence relevant to the criterion: exact numbers, metric values, named methods, specific claims. Quote the exact sentence/figure/table from the report that contains each piece of evidence.
3. If the criterion asks for quantitative results but the report gives no real numbers (or only vague/borrowed numbers), that is a low score regardless of how polished the text is. If the report fabricates or hand-waves results, score it near 0.
4. Compare against the published-paper bar (50 = as good as the published paper). Be strict and skeptical of AI-generated-sounding filler.

Respond in JSON format with the following schema:
<json>
{{
    "mode": "A" or "B",
    "evidence": "<2-4 verbatim quotes/specific numbers from the report you based your score on>",
    "reasoning": "<2-4 sentences explaining the score>",
    "score": <integer 0-100>
}}
</json>"""

        response, new_msg_history, _info = get_response_from_llm(
            msg=instruction, model=self.model, msg_history=[],
        )

        prediction = {"score": 0, "reasoning": "Failed to parse evaluator response."}
        try:
            extracted = extract_jsons(response)
            item = extracted[-1] if extracted else None
            if item is None:
                # Some backends (observed with DeepSeek) ignore the <json>
                # wrapper and return a bare JSON object -- extract_jsons only
                # matches <json>...</json> or ```json fences, so fall back to
                # parsing the whole response directly.
                item = json.loads(response.strip())
            if item:
                prediction = {
                    "score": max(0, min(100, int(item.get("score", 0)))),
                    "reasoning": str(item.get("reasoning", "")),
                }
        except Exception as e:
            self.log(f"Error extracting evaluator score: {e}")

        return prediction, new_msg_history
