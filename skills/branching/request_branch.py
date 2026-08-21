"""Meta-agent-facing half of the branching skill. Runs inside the
meta-agent's own (sandboxed, no docker) container -- validates a branch
request and queues it; never runs anything itself. See SKILL.md for the
full picture, including why this is deferred and what a report looks like.
"""
import argparse
import json
import os
import time
import uuid
from pathlib import Path

K_HARD_CAP = 5

# Repo-root file, alongside memory.jsonl -- appended to, never rewritten, so
# a request queued mid-session survives exactly like any other file edit
# (through the meta-agent's own patch chain) even though nothing processes
# it until the host does, after this session ends.
REQUESTS_PATH = Path(__file__).resolve().parents[2] / "branch_requests.jsonl"


def _load_rounds(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Queue a branching-skill request. See skills/branching/SKILL.md.")
    parser.add_argument("--source-chat-history", required=True, dest="source_chat_history")
    parser.add_argument("--branch-round", required=True, type=int, dest="branch_round")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--note", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.source_chat_history):
        print(f"Error: {args.source_chat_history} does not exist (from this container's own view).")
        return 1

    try:
        rounds = _load_rounds(args.source_chat_history)
    except Exception as e:
        print(f"Error: could not parse {args.source_chat_history} as a chat_history.json trajectory: {e}")
        return 1

    matching = [r for r in rounds if r.get("round") == args.branch_round]
    if not matching:
        available = sorted(r.get("round") for r in rounds if "round" in r)
        print(f"Error: round {args.branch_round} not found in this trajectory. Available rounds: {available}")
        return 1
    if not matching[0].get("act"):
        print(
            f"Error: round {args.branch_round} has no tool call (act is empty) -- nothing to branch from. "
            "This is likely the trajectory's own trailing summary round; pick an earlier round with a real action."
        )
        return 1

    if args.k < 1:
        print("Error: --k must be at least 1.")
        return 1
    if args.k > K_HARD_CAP:
        print(f"Note: --k {args.k} exceeds the hard cap of {K_HARD_CAP}; the host will clamp it down to {K_HARD_CAP} when processing.")

    if not args.note.strip():
        print("Error: --note must not be empty -- see SKILL.md for why.")
        return 1

    request = {
        "request_id": uuid.uuid4().hex[:12],
        "queued_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_chat_history": args.source_chat_history,
        "branch_round": args.branch_round,
        "k": min(args.k, K_HARD_CAP),
        "note": args.note,
        "status": "pending",
    }
    with open(REQUESTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(request, ensure_ascii=False) + "\n")

    print(
        f"Queued branch request {request['request_id']} (round {args.branch_round}, k={request['k']}). "
        "Results won't be available this session -- see SKILL.md's 'This is deferred' section. "
        f"Check skills/branching/reports/{request['request_id']}/ in a later session."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
