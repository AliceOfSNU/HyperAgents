from agent.llm import get_response_from_llm
from agent.tools import load_tools


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


def chat_with_agent(
    msg,
    model="claude-4-sonnet-genai",
    msg_history=None,
    logging=print,
    tools_available=[],  # Empty list means no tools, 'all' means all tools
    multiple_tool_calls=False,  # Whether to allow multiple tool calls in a single response
    max_tool_calls=40,  # Maximum number of tool calls allowed in a single response, -1 for unlimited
):
    """Uses the model provider's native function-calling (via litellm's
    `tools=` parameter) rather than a prompt-engineered text format. Real
    runs surfaced three distinct failure shapes from a free-text
    `<json>{"tool_name": ...}</json>` convention -- missing wrapper tags,
    corrupted closing tags, and an entirely different tool-call syntax the
    model substituted on its own -- all from the model simply not reliably
    reproducing arbitrary formatting instructions. Native tool-calling moves
    that enforcement into the provider's own constrained decoding, so there's
    no text convention left for the model to get wrong."""
    get_response_fn = get_response_from_llm
    if msg_history is None:
        msg_history = []
    new_msg_history = msg_history

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

        # Call API
        logging(f"Input: {repr(msg)}")
        response, new_msg_history, info = get_response_fn(
            msg=msg,
            model=model,
            msg_history=new_msg_history,
            tools=litellm_tools,
        )
        logging(f"Output: {repr(response)}")
        tool_calls = info.get("tool_calls") or []

        # A response cut off mid-generation (e.g. hit max_tokens) may carry no
        # tool_calls at all, or an incomplete one -- ask the model to retry
        # rather than silently treating a truncated response as final.
        if info.get("finish_reason") == "length" and not tool_calls:
            logging("Error: Output context exceeded. Please try again.")
            response, new_msg_history, info = get_response_fn(
                msg="Error: Output context exceeded. Please try again.",
                model=model,
                msg_history=new_msg_history,
                tools=litellm_tools,
            )
            logging(f"Output: {repr(response)}")
            tool_calls = info.get("tool_calls") or []

        while tool_calls:
            # Check for max tool calls
            if max_tool_calls > 0 and num_tool_calls >= max_tool_calls:
                logging("Error: Maximum number of tool calls reached.")
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
                new_msg_history = new_msg_history + [
                    {"role": "tool", "tool_call_id": call["id"], "text": "Skipped: maximum tool call budget reached."}
                    for call in tool_calls
                ]
                break

            calls_to_run = tool_calls if multiple_tool_calls else tool_calls[:1]
            tool_result_msgs = []

            for call in calls_to_run:
                tool_name = call["name"]
                tool_input = call["arguments"]
                tool_output = process_tool_call(tools_dict, tool_name, tool_input)
                num_tool_calls += 1
                logging(f"Tool output: {tool_name}({tool_input}) -> {repr(tool_output)}")
                tool_result_msgs.append({
                    "role": "tool", "tool_call_id": call["id"], "text": str(tool_output),
                })

            # The API requires a "tool" response for every tool_call_id the
            # model emitted this turn, even ones not executed here (when
            # multiple_tool_calls=False truncates to just the first) --
            # otherwise the next request fails validation over the orphaned call.
            executed_ids = {call["id"] for call in calls_to_run}
            for call in tool_calls:
                if call["id"] not in executed_ids:
                    tool_result_msgs.append({
                        "role": "tool", "tool_call_id": call["id"],
                        "text": "Skipped: only one tool call is processed per turn.",
                    })

            # Keep the running remaining-budget count visible after every
            # round (not just at the start) -- appended to the last tool
            # result of this round since the model reads every "tool"
            # message from a round together before its next turn.
            if max_tool_calls > 0:
                remaining = max_tool_calls - num_tool_calls
                tool_result_msgs[-1]["text"] += f"\n\n[{remaining} of {max_tool_calls} tool calls remaining -- plan accordingly.]"

            # Get tool response -- msg=None: the tool-result messages above
            # are the continuation, not a new user turn.
            response, new_msg_history, info = get_response_fn(
                msg=None,
                model=model,
                msg_history=new_msg_history + tool_result_msgs,
                tools=litellm_tools,
            )
            logging(f"Output: {repr(response)}")
            tool_calls = info.get("tool_calls") or []

    except Exception as e:
        logging(f"Error: {str(e)}")
        raise e

    return new_msg_history

if __name__ == "__main__":
    msg = """hello"""
    new_msg_history = chat_with_agent(msg)
