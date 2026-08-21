from agent.llm import get_response_from_llm
from agent.tools import load_tools


PLAN_ACT_OBSERVE_INSTRUCTION = (
    "Work in explicit Plan-Act-Observe rounds. Each turn, first state your "
    "plan as your text response -- the next logical step(s) and why, given "
    "the problem and everything so far -- THEN make the tool call(s) that "
    "carry it out. Always include this plan text even when you're about to "
    "call a tool; a turn with a tool call and no plan text is incomplete. "
    "From your second round onward, begin your plan by explicitly observing "
    "the result of your previous action -- did it actually work, did it "
    "help, what changed, was the outcome what you expected -- before "
    "deciding what to do next. Don't just plan forward as if the last "
    "action's outcome doesn't matter."
)


def _to_litellm_tool(tool_info):
    """Convert this codebase's tool_info() shape (name/description/input_schema,
    already a proper JSON schema -- see e.g. agent/tools/edit.py) into the
    OpenAI/litellm function-calling tool format."""
    return {
        "type": "function",
        "function": {
            "name": tool_info["name"],
            "description": tool_info["description"],
            "parameters": tool_info["input_schema"],
        },
    }


def process_tool_call(tools_dict, tool_name, tool_input):
    try:
        if tool_name in tools_dict:
            return tools_dict[tool_name]['function'](**tool_input)
        else:
            return f"Error: Tool '{tool_name}' not found"
    except Exception as e:
        return f"Error executing tool '{tool_name}': {str(e)}"


def _format_round(round_record):
    """Render one {"round", "plan", "act", "observe"} record as a structured
    Markdown block -- the log-file counterpart of the trajectory list,
    replacing the old flat Input:/Output:/Tool output: lines."""
    plan = (round_record["plan"] or "").strip() or "(no plan text)"
    lines = [f"## Round {round_record['round']}", "", "**Plan:**", plan]
    if round_record["act"]:
        lines += ["", "**Act:**"]
        lines += [f"- {call['name']}({call['arguments']})" for call in round_record["act"]]
    if round_record["observe"]:
        lines += ["", "**Observe:**"]
        lines += [f"- {obs['name']} -> {repr(obs['result'])}" for obs in round_record["observe"]]
    return "\n".join(lines)


def chat_with_agent(
    msg,
    model="claude-4-sonnet-genai",
    msg_history=None,
    logging=print,
    tools_available=[],  # Empty list means no tools, 'all' means all tools
    multiple_tool_calls=False,  # Whether to allow multiple tool calls in a single response
    max_tool_calls=40,  # Maximum number of tool calls allowed in a single response, -1 for unlimited
    plan_act_observe=False,  # See docstring
    temperature=0.0,  # Every existing caller relies on the old implicit 0.0
    # default (deterministic-as-possible tool use); real sampling diversity
    # (temperature>0) is opt-in, currently only meaningful for
    # skills/branching's sibling rollouts, which need rounds to actually
    # diverge from an identical starting point.
):
    """Uses the model provider's native function-calling (via litellm's
    `tools=` parameter) rather than a prompt-engineered text format. Real
    runs surfaced three distinct failure shapes from a free-text
    `<json>{"tool_name": ...}</json>` convention -- missing wrapper tags,
    corrupted closing tags, and an entirely different tool-call syntax the
    model substituted on its own -- all from the model simply not reliably
    reproducing arbitrary formatting instructions. Native tool-calling moves
    that enforcement into the provider's own constrained decoding, so there's
    no text convention left for the model to get wrong.

    Every turn is recorded as one Plan-Act-Observe round: the model's own
    text response for that turn is the "plan" (no extra call needed to
    extract it -- native tool-calling already returns text and tool_calls as
    separate fields on the same message), the tool_calls it emits are the
    "act", and their results are the "observe". The full trajectory (one
    record per round) is always assembled internally and always used to
    write structured Plan/Act/Observe log blocks via `logging`, in place of
    the old flat Input:/Output:/Tool output: lines -- a pure logging
    improvement with no effect on the model's behavior, so it applies
    regardless of `plan_act_observe`.

    `plan_act_observe=True` additionally (a) prompts the model to always
    state its plan before acting (PLAN_ACT_OBSERVE_INSTRUCTION) and (b)
    returns `(new_msg_history, trajectory)` instead of just
    `new_msg_history`. Defaults to False so existing callers that expect a
    single list back (baselines/*, this module's own __main__) are
    unaffected -- opted into explicitly by meta_agent.py and task_agent.py,
    the two callers this is meant for."""
    get_response_fn = get_response_from_llm
    if msg_history is None:
        msg_history = []
    new_msg_history = msg_history
    trajectory = []

    try:
        all_tools = load_tools(logging=logging, names=tools_available)
        tools_dict = {tool['info']['name']: tool for tool in all_tools}
        litellm_tools = [_to_litellm_tool(tool['info']) for tool in all_tools] or None
        num_tool_calls = 0

        # Tell the model its own tool-call ceiling up front -- previously
        # this was purely a silent harness-side cap: the model had no way to
        # know one existed until it was already cut off. Observed live,
        # repeatedly: full budgets spent entirely on read-only exploration
        # before ever acting, then cut off mid-investigation with no warning.
        if max_tool_calls > 0 and all_tools:
            msg = f"{msg}\n\n[You have up to {max_tool_calls} tool calls available for this task -- plan accordingly.]"
        if plan_act_observe and all_tools:
            msg = f"{msg}\n\n{PLAN_ACT_OBSERVE_INSTRUCTION}"

        # Call API
        logging(f"Input: {repr(msg)}")
        response, new_msg_history, info = get_response_fn(
            msg=msg,
            model=model,
            msg_history=new_msg_history,
            tools=litellm_tools,
            temperature=temperature,
        )
        tool_calls = info.get("tool_calls") or []

        # A response cut off mid-generation (e.g. hit max_tokens) may carry no
        # tool_calls at all, or an incomplete one -- ask the model to retry
        # rather than silently treating a truncated response as final. This
        # retry is plumbing/error-recovery, not a real round, so it isn't
        # recorded in the trajectory.
        if info.get("finish_reason") == "length" and not tool_calls:
            logging("Error: Output context exceeded. Please try again.")
            response, new_msg_history, info = get_response_fn(
                msg="Error: Output context exceeded. Please try again.",
                model=model,
                msg_history=new_msg_history,
                tools=litellm_tools,
                temperature=temperature,
            )
            logging(f"Retried after truncation. Output: {repr(response)}")
            tool_calls = info.get("tool_calls") or []

        round_num = 0
        while True:
            round_num += 1
            round_record = {"round": round_num, "plan": response, "act": [], "observe": []}

            if not tool_calls:
                # Terminal round: the model responded with plan text only,
                # no tool call -- nothing to act on or observe.
                trajectory.append(round_record)
                logging(_format_round(round_record))
                break

            if max_tool_calls > 0 and num_tool_calls >= max_tool_calls:
                # The assistant message already in new_msg_history still has
                # these tool_calls attached and unanswered -- the API rejects
                # any later message on this history until every tool_call_id
                # gets a matching "tool" response. Answer them here instead of
                # just breaking, or the returned history is left invalid for
                # any caller that resumes it later (confirmed live: exactly
                # what broke task_agent.py's own continuation pass, which
                # reuses this history when the first pass ends without a
                # report -- "insufficient tool messages following tool_calls
                # message").
                logging("Error: Maximum number of tool calls reached.")
                skip_text = "Skipped: maximum tool call budget reached."
                round_record["act"] = [
                    {"id": call["id"], "name": call["name"], "arguments": call["arguments"]}
                    for call in tool_calls
                ]
                round_record["observe"] = [
                    {"tool_call_id": call["id"], "name": call["name"], "result": skip_text}
                    for call in tool_calls
                ]
                new_msg_history = new_msg_history + [
                    {"role": "tool", "tool_call_id": call["id"], "text": skip_text}
                    for call in tool_calls
                ]
                trajectory.append(round_record)
                logging(_format_round(round_record))
                break

            calls_to_run = tool_calls if multiple_tool_calls else tool_calls[:1]
            tool_result_msgs = []

            for call in calls_to_run:
                tool_name = call["name"]
                tool_input = call["arguments"]
                tool_output = str(process_tool_call(tools_dict, tool_name, tool_input))
                num_tool_calls += 1
                tool_result_msgs.append({"role": "tool", "tool_call_id": call["id"], "text": tool_output})
                round_record["act"].append({"id": call["id"], "name": tool_name, "arguments": tool_input})
                round_record["observe"].append({"tool_call_id": call["id"], "name": tool_name, "result": tool_output})

            # The API requires a "tool" response for every tool_call_id the
            # model emitted this turn, even ones not executed here (when
            # multiple_tool_calls=False truncates to just the first) --
            # otherwise the next request fails validation over the orphaned call.
            executed_ids = {call["id"] for call in calls_to_run}
            for call in tool_calls:
                if call["id"] not in executed_ids:
                    skip_text = "Skipped: only one tool call is processed per turn."
                    tool_result_msgs.append({"role": "tool", "tool_call_id": call["id"], "text": skip_text})
                    round_record["act"].append({"id": call["id"], "name": call["name"], "arguments": call["arguments"]})
                    round_record["observe"].append({"tool_call_id": call["id"], "name": call["name"], "result": skip_text})

            # Keep the running remaining-budget count visible after every
            # round (not just at the start) -- appended to the last tool
            # result of this round since the model reads every "tool"
            # message from a round together before its next turn.
            if max_tool_calls > 0:
                remaining = max_tool_calls - num_tool_calls
                tool_result_msgs[-1]["text"] += f"\n\n[{remaining} of {max_tool_calls} tool calls remaining -- plan accordingly.]"
                round_record["observe"][-1]["result"] = tool_result_msgs[-1]["text"]

            trajectory.append(round_record)
            logging(_format_round(round_record))

            # Get tool response -- msg=None: the tool-result messages above
            # are the continuation, not a new user turn.
            response, new_msg_history, info = get_response_fn(
                msg=None,
                model=model,
                msg_history=new_msg_history + tool_result_msgs,
                tools=litellm_tools,
                temperature=temperature,
            )
            tool_calls = info.get("tool_calls") or []

    except Exception as e:
        logging(f"Error: {str(e)}")
        raise e

    if plan_act_observe:
        return new_msg_history, trajectory
    return new_msg_history

if __name__ == "__main__":
    msg = """hello"""
    new_msg_history = chat_with_agent(msg)
