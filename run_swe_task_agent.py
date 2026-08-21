import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from swe_task_agent import SweTaskAgent

# Runs inside the deep-swe task's own container (pulled/built by Pier from
# task.toml's docker_image -- see domains/deep_swe/pier_agent.py, which is
# what invokes this script via environment.exec()). The repo is already
# checked out at repo_dir, pinned at the task's base_commit, by the image
# itself -- this script never touches ResearchClawBench-style workspace
# staging, there's no data/related_work/report scaffold here, just a repo.


def write_meta(repo_dir, task_id, status):
    meta = {
        "task_id": task_id,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "status": status,
        "agent_name": "hyperagents_swe_task_agent",
    }
    with open(Path(repo_dir) / "_hyperagents_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run the SWE task agent inside an already-checked-out repo.")
    parser.add_argument("--repo_dir", required=True, help="Path to the repo checkout (task.toml's /app)")
    parser.add_argument("--instruction_file", required=True, help="Path to the task's instruction text")
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--chat_history_file", required=True)
    parser.add_argument("--model", required=False, default="deepseek/deepseek-v4-flash")
    args = parser.parse_args()

    chat_history_file = str(Path(args.chat_history_file).resolve())
    repo_dir = str(Path(args.repo_dir).resolve())
    instruction = Path(args.instruction_file).read_text(encoding="utf-8")

    write_meta(repo_dir, args.task_id, "running")

    prev_cwd = os.getcwd()
    os.chdir(repo_dir)
    try:
        agentic_system = SweTaskAgent(model=args.model, chat_history_file=chat_history_file)
        prediction, _ = agentic_system.forward({"instructions": instruction})
    finally:
        os.chdir(prev_cwd)

    write_meta(repo_dir, args.task_id, "completed" if prediction == "done" else "failed")


if __name__ == "__main__":
    main()
