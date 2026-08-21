import argparse
import os
from pathlib import Path

from agent.trajectory_replay import load_trajectory_rounds
from domains.deep_swe.branch_task_agent import BranchRolloutAgent

# Runs inside a fresh deep-swe task container, same as run_swe_task_agent.py
# -- see domains/deep_swe/branch_pier_agent.py, which uploads the source
# trajectory prefix alongside the usual baseline code before invoking this.


def main():
    parser = argparse.ArgumentParser(description="Replay a trajectory prefix inside an already-checked-out repo, then continue with fresh sampling.")
    parser.add_argument("--repo_dir", required=True, help="Path to the repo checkout (task.toml's /app)")
    parser.add_argument("--instruction_file", required=True, help="Path to the task's instruction text")
    parser.add_argument("--chat_history_file", required=True, help="Where this branch's OWN trajectory gets written")
    parser.add_argument("--model", required=False, default="deepseek/deepseek-v4-flash")
    parser.add_argument("--source_chat_history", required=True, help="Path to the uploaded source trajectory to replay a prefix of")
    parser.add_argument("--branch_round", type=int, required=True, help="Last round (1-based, inclusive) of source_chat_history to replay before diverging")
    parser.add_argument("--original_max_tool_calls", type=int, default=100, help="max_tool_calls the SOURCE trajectory was run with -- needed to reconstruct round 1's exact first turn")
    parser.add_argument("--continuation_max_tool_calls", type=int, required=True, help="Budget for THIS branch's own new rounds, from the branch point onward")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    chat_history_file = str(Path(args.chat_history_file).resolve())
    repo_dir = str(Path(args.repo_dir).resolve())
    instruction = Path(args.instruction_file).read_text(encoding="utf-8")
    source_rounds = load_trajectory_rounds(args.source_chat_history)

    prev_cwd = os.getcwd()
    os.chdir(repo_dir)
    try:
        agentic_system = BranchRolloutAgent(model=args.model, chat_history_file=chat_history_file)
        agentic_system.forward({
            "instructions": instruction,
            "source_rounds": source_rounds,
            "branch_round": args.branch_round,
            "original_max_tool_calls": args.original_max_tool_calls,
            "continuation_max_tool_calls": args.continuation_max_tool_calls,
            "temperature": args.temperature,
        })
    finally:
        os.chdir(prev_cwd)


if __name__ == "__main__":
    main()
