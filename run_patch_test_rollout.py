import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from agent.trajectory_replay import load_trajectory_rounds
from domains.deep_swe.patch_test_task_agent import PatchTestRolloutAgent

# Runs inside a fresh deep-swe task container, same pattern as
# run_swe_task_agent.py and skills/branching's own run_branch_rollout.py --
# see domains/deep_swe/patch_test_orchestrator.py, which uploads the source
# trajectory prefix alongside the usual baseline code (control: that
# checkpoint's own original swe_task_agent.py; treatment: the same file
# with the meta-agent's patch applied) before invoking this.


def write_meta(repo_dir, task_id, status):
    meta = {
        "task_id": task_id,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "status": status,
        "agent_name": "hyperagents_patch_test_rollout",
    }
    with open(Path(repo_dir) / "_hyperagents_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Replay a trajectory prefix inside an already-checked-out repo, then continue with THIS container's own swe_task_agent.py to full completion.",
    )
    parser.add_argument("--repo_dir", required=True, help="Path to the repo checkout (task.toml's /app)")
    parser.add_argument("--instruction_file", required=True, help="Path to the task's instruction text")
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--chat_history_file", required=True, help="Where this replicate's OWN trajectory gets written")
    parser.add_argument("--model", required=False, default="deepseek/deepseek-v4-flash")
    parser.add_argument("--source_chat_history", required=True, help="Path to the uploaded source trajectory to replay a prefix of")
    parser.add_argument("--checkpoint_round", type=int, required=True, help="Round (1-based) whose own recorded state this replicate continues from -- rounds before it are replayed for real, this round onward is decided fresh by this container's own code")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    chat_history_file = str(Path(args.chat_history_file).resolve())
    repo_dir = str(Path(args.repo_dir).resolve())
    instruction = Path(args.instruction_file).read_text(encoding="utf-8")
    source_rounds = load_trajectory_rounds(args.source_chat_history)

    write_meta(repo_dir, args.task_id, "running")

    prev_cwd = os.getcwd()
    os.chdir(repo_dir)
    try:
        agentic_system = PatchTestRolloutAgent(model=args.model, chat_history_file=chat_history_file)
        prediction, _ = agentic_system.forward({
            "instructions": instruction,
            "source_rounds": source_rounds,
            "checkpoint_round": args.checkpoint_round,
            "temperature": args.temperature,
        })
    finally:
        os.chdir(prev_cwd)

    write_meta(repo_dir, args.task_id, "completed" if prediction == "done" else "failed")


if __name__ == "__main__":
    main()
