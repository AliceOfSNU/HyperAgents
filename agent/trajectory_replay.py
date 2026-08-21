"""Reconstruct a chat_with_agent(plan_act_observe=True) trajectory's prefix --
both the message history a later call should see, and (container-side only)
the actual tool-call side effects that produced it.

Two separate things a trajectory prefix means, and this module deliberately
keeps them separate:

- The message history the model saw. agent/llm.py's get_response_from_llm
  only ever stores an assistant turn's final `content` + raw `tool_calls`,
  never any hidden/internal reasoning -- confirmed by reading it directly.
  So splicing a round's recorded plan/act/observe back in as literal text
  reproduces exactly what a later round in the *same* run would have seen;
  there's no richer state a continuous run has that a spliced replay lacks.
  reconstruct_msg_history does only this: pure, no I/O, no side effects.

- The environment the tool calls actually ran against. Splicing text back
  into the message history does nothing to a fresh container's filesystem --
  the recorded tool *results* are just strings. To get a matching filesystem
  state, the recorded tool calls have to be genuinely re-executed against
  the new environment. replay_tool_calls does this, and only this: it never
  touches message history, and it deliberately ignores whatever fresh output
  the re-execution produces (logged for debugging, nothing else) -- the
  spliced message history above is what the model conditions on, not this.

Byte-identical reproduction isn't the goal (see the branching skill's own
discussion of why it's unreachable -- model sampling, verifier/test
flakiness); "functionally equivalent, built from the same actions" is.
"""
import json

from agent.llm_withtools import PLAN_ACT_OBSERVE_INSTRUCTION


def reconstruct_msg_history(rounds, upto_round, initial_instruction, max_tool_calls=100, plan_act_observe=True):
    """rounds: a chat_history.json-style list of {"round","plan","act","observe"}
    records. upto_round: last round INDEX (1-based, inclusive) to include.

    chat_history.json never stores the original instruction text -- only the
    round records chat_with_agent's own loop produces after that first call.
    Without prepending it back, the reconstructed history would start with a
    bare assistant message and no task instruction at all (most providers
    reject a conversation that doesn't open with a user/system turn, and even
    where they don't, the model would be missing the actual task). Rebuilds
    it exactly as chat_with_agent's own first call would have constructed it
    -- see agent/llm_withtools.py's own max_tool_calls/plan_act_observe
    banner-appending logic, replicated here rather than imported since it's
    two lines of string formatting, not worth a shared entry point for.
    max_tool_calls/plan_act_observe must match what the ORIGINAL run actually
    used (not this replay's own continuation budget) for the reconstructed
    first turn to match what round 1 actually responded to.

    Returns a msg_history list in the same "text"-keyed storage format
    agent/llm.py's get_response_from_llm produces/consumes (NOT the
    "content"-keyed format litellm itself expects -- that conversion happens
    inside get_response_from_llm, same as for any other stored history)."""
    first_turn = initial_instruction
    if max_tool_calls > 0:
        first_turn = f"{first_turn}\n\n[You have up to {max_tool_calls} tool calls available for this task -- plan accordingly.]"
    if plan_act_observe:
        first_turn = f"{first_turn}\n\n{PLAN_ACT_OBSERVE_INSTRUCTION}"
    msg_history = [{"role": "user", "text": first_turn}]
    for record in rounds:
        if record["round"] > upto_round:
            break
        act = record.get("act") or []
        if not act:
            # A terminal round with plan text but no tool call (e.g. the
            # trajectory's own final "Done." summary) -- nothing to splice
            # as an assistant-with-tool-calls turn; only relevant if this is
            # itself the exact cutoff, which callers shouldn't request since
            # there's nothing left to branch from.
            continue
        raw_tool_calls = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                },
            }
            for call in act
        ]
        msg_history.append({
            "role": "assistant",
            "text": record.get("plan") or "",
            "tool_calls": raw_tool_calls,
        })
        for obs in record.get("observe") or []:
            msg_history.append({
                "role": "tool",
                "tool_call_id": obs["tool_call_id"],
                "text": str(obs["result"]),
            })
    return msg_history


def replay_tool_calls(rounds, upto_round, tools_dict, logging=print):
    """Container-side only: re-execute every tool call recorded in rounds
    1..upto_round, in order, against whatever environment this process is
    actually running in -- reconstructing matching filesystem/git state for
    a fresh container. tools_dict is the same {name: {"function": ...}} shape
    agent/llm_withtools.py's process_tool_call already expects (from
    agent.tools.load_tools) -- reused rather than reimplemented so replay
    executes through the exact same tool code a live round would.

    Deliberately returns nothing and never touches message history -- see
    this module's docstring for why the recorded text (not fresh
    re-execution output) is what a later round should see."""
    for record in rounds:
        if record["round"] > upto_round:
            break
        for call in record.get("act") or []:
            tool_name = call["name"]
            if tool_name not in tools_dict:
                logging(f"replay: skipping unknown tool '{tool_name}' from round {record['round']}")
                continue
            try:
                tools_dict[tool_name]["function"](**call["arguments"])
            except Exception as e:
                logging(f"replay: round {record['round']} tool '{tool_name}' raised during replay: {e}")


def load_trajectory_rounds(chat_history_path):
    with open(chat_history_path, "r", encoding="utf-8") as f:
        return json.load(f)
