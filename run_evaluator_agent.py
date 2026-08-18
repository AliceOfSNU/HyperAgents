import argparse
import json
import tempfile
from pathlib import Path

from evaluator_agent import EvaluatorAgent
from domains.research.claude_scorer import score_report_text_items

# Runs inside a sandboxed container. Unlike run_task_agent.py this doesn't
# execute arbitrary tool calls today (evaluator_agent.py is a single LLM
# judgment, no tools_available), but it's just as meta-agent-editable as
# task_agent.py, so it gets the same sandboxing rather than a safety
# guarantee that erodes the moment someone adds tool use to it.
#
# The checklist and report are pre-copied into the container by the host
# harness (domains/research/harness.py) -- no ResearchClawBench checkout
# dependency in here.


def _make_score_fn(model):
    def score_fn(report_text, item, instructions):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
            chat_history_file = tmp.name
        agent = EvaluatorAgent(model=model, chat_history_file=chat_history_file)
        prediction, _ = agent.forward({
            "report_text": report_text,
            "instructions": instructions,
            "criterion": item.get("content", ""),
            "keywords": item.get("keywords", []),
        })
        return prediction
    return score_fn


def main():
    parser = argparse.ArgumentParser(description="Score a report against a checklist with evaluator_agent.py.")
    parser.add_argument("--workspace", required=True, help="Contains report/report.md and INSTRUCTIONS.md")
    parser.add_argument("--checklist_path", required=True, help="Path to the task's checklist.json, pre-copied by the host")
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--model", required=False, default="deepseek/deepseek-v4-flash")
    parser.add_argument("--out", required=True, help="Path to write the resulting {task_id, items, score} JSON")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    report_path = workspace / "report" / "report.md"
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    instructions_path = workspace / "INSTRUCTIONS.md"
    instructions = instructions_path.read_text(encoding="utf-8") if instructions_path.exists() else ""
    with open(args.checklist_path, "r", encoding="utf-8") as f:
        checklist = json.load(f)

    items, score = score_report_text_items(report_text, checklist, instructions, _make_score_fn(args.model))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"task_id": args.task_id, "items": items, "score": score}, f, indent=2)


if __name__ == "__main__":
    main()
