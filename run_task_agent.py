import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from task_agent import TaskAgent

# Runs inside a sandboxed container. The workspace (data/, related_work/,
# INSTRUCTIONS.md, and the code/outputs/report scaffold) is expected to
# already be populated by the host-side harness before this script runs --
# see domains/research/harness.py, which builds INSTRUCTIONS.md from
# ResearchClawBench's own template (host has the RCB checkout; the
# container deliberately does not).


def write_meta(workspace, task_id, run_id, status, agent_name="hyperagents_task_agent"):
    meta = {
        "task_id": task_id,
        "run_id": run_id,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "status": status,
        "workspace": str(workspace),
        "agent_name": agent_name,
    }
    with open(Path(workspace) / "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run the research task agent inside an already-populated workspace.")
    parser.add_argument("--workspace", required=True, help="Pre-populated workspace (data/, related_work/, INSTRUCTIONS.md, code/, outputs/, report/)")
    parser.add_argument("--task_id", required=True, help="ResearchClawBench task ID, recorded in _meta.json")
    parser.add_argument("--chat_history_file", required=True)
    parser.add_argument("--model", required=False, default="deepseek/deepseek-v4-flash")
    parser.add_argument("--run_id", required=False, default=None)
    args = parser.parse_args()

    chat_history_file = str(Path(args.chat_history_file).resolve())
    workspace = str(Path(args.workspace).resolve())
    run_id = args.run_id or f"{args.task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    instructions_path = Path(workspace, "INSTRUCTIONS.md")
    instructions = instructions_path.read_text(encoding="utf-8")
    for dirname in ("code", "outputs", "report", "report/images"):
        Path(workspace, dirname).mkdir(parents=True, exist_ok=True)

    write_meta(workspace, args.task_id, run_id, "running")

    prev_cwd = os.getcwd()
    os.chdir(workspace)
    try:
        agentic_system = TaskAgent(model=args.model, chat_history_file=chat_history_file)
        prediction, _ = agentic_system.forward({"instructions": instructions, "task_id": args.task_id, "workspace": workspace})
    finally:
        os.chdir(prev_cwd)

    write_meta(workspace, args.task_id, run_id, "completed" if prediction == "done" else "failed")


if __name__ == "__main__":
    main()
