import json

from agent.base_agent import AgentSystem
from agent.llm import get_response_from_llm
from utils.common import extract_jsons

# Reuse the exact rubric and system prompt as the trusted anchor scorer
# (domains/research/claude_scorer.py) so evaluator predictions stay as close
# as possible to the fixed ground-truth anchor. A local fallback is kept only
# so this module can still be imported in isolation during development.
try:
    from domains.research.claude_scorer import (
        JUDGE_SYSTEM_PROMPT,
        RUBRIC,
        _build_text_prompt as _anchor_build_text_prompt,
    )
except Exception:  # pragma: no cover - fallback only
    JUDGE_SYSTEM_PROMPT = (
        "You are a strict scientific peer reviewer evaluating AI-generated research. "
        "Score the report against the criterion only — do not attempt to solve the "
        "research task yourself."
    )
    RUBRIC = """You are a strict scientific peer reviewer evaluating an AI agent's ability to conduct end-to-end automated scientific research.

## CRITICAL RULES
- 50 means "as good as the actual published paper" — this is a high bar.
- First determine if the criterion is Objective (Mode A) or Subjective (Mode B), then apply the corresponding rubric.
- No credit for vague or generic statements. Must demonstrate specific, concrete analysis.
- No inflation for well-written but shallow content. Substance over style.
- Be highly skeptical of AI-generated content.
- Be strict but fair.
"""


class EvaluatorAgent(AgentSystem):
    def forward(self, inputs):
        """
        Scores one ResearchClawBench checklist item (text-type only) against
        an AI-generated research report -- a co-evolving stand-in for
        ResearchClawBench's own peer-reviewer judge.

        The prompt is intentionally identical to the trusted anchor scorer's
        prompt (same rubric, same system prompt, same bare-JSON output shape),
        differing only in the backend model. This keeps evaluator predictions
        aligned with the fixed anchor distribution and removes a whole class
        of avoidable parsing/wrapper failures.

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
        try:
            # Prefer the anchor scorer's own prompt builder: this guarantees the
            # evaluator sees byte-for-byte the same prompt as the fixed
            # ground-truth judge, so any remaining score gap is model ability,
            # not a wrapper/prompt divergence.
            instruction = _anchor_build_text_prompt(
                inputs["report_text"],
                {"content": inputs["criterion"], "keywords": inputs.get("keywords", []), "type": "text"},
                inputs.get("instructions", ""),
            )
        except Exception:
            # Fallback only for isolated development imports; the sandboxed
            # evaluator container always ships domains/research/claude_scorer.py.
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
First determine if this criterion is Objective (Mode A) or Subjective (Mode B), then apply the corresponding rubric strictly.

Return your answer as a JSON object: {{"reasoning": "<2-3 sentences>", "score": <0-100>}}"""

        response, new_msg_history, _info = get_response_from_llm(
            msg=instruction,
            model=self.model,
            msg_history=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT}],
        )

        prediction = {"score": 0, "reasoning": "Failed to parse evaluator response."}
        try:
            extracted = extract_jsons(response)
            item = extracted[-1] if extracted else None
            if item is None:
                # The anchor prompt asks for a bare JSON object; some backends
                # still wrap it or add prose. Parse the largest brace-delimited
                # object rather than failing on whitespace/prose.
                text = response.strip()
                start, end = text.find("{"), text.rfind("}")
                if start >= 0 and end > start:
                    item = json.loads(text[start:end + 1])
            if item:
                raw_score = item.get("score", 0)
                try:
                    score = max(0, min(100, int(round(float(raw_score)))))
                except Exception:
                    score = 0
                prediction = {
                    "score": score,
                    "reasoning": str(item.get("reasoning", "")),
                }
        except Exception as e:
            self.log(f"Error extracting evaluator score: {e}")

        return prediction, new_msg_history
