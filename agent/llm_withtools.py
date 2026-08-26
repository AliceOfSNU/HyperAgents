from agent.llm import get_response_from_llm
from agent.tools import load_tools


PLAN_ACT_OBSERVE_INSTRUCTION = (
    "Work in explicit Plan-Act-Observe rounds. Each turn, first state your "
    "plan as your text response -- the next logical step(s) and why, given "
    "the problem and everything so far -- THEN make the tool call(s) that "
    "carry it out. Always include this plan text even when you're about to "
    "call a tool; a turn with a tool call and no plan text is incomplete. "
    "Write your plan as ordinary prose, the way you'd explain your thinking "
    "out loud to a colleague -- not as structured or machine-readable data "
    "of any kind. And making a tool call means actually invoking the tool "
    "through the tool-calling mechanism itself -- describing, naming, or "
    "outlining an action in your prose is not the same as calling it, and a "
    "turn that only does that will be treated as if you made no call at "
    "all. From your second round onward, begin your plan by explicitly "
    "observing the result of your previous action -- did it actually work, "
    "did it help, what changed, was the outcome what you expected -- before "
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


def build_kickoff_message(msg, max_tool_calls, tools_available, plan_act_observe, logging=print):
    """Decorates a new "turn" message the same way chat_with_agent's own
    first call always has -- appends the visible tool-call budget (if any
    tools are available and a budget is set) and, if plan_act_observe, the
    Plan-Act-Observe instruction. A caller starts a new turn (rather than
    continuing an existing conversation) by appending
    {"role": "user", "text": build_kickoff_message(...)} to context before
    calling step() -- this is how chat_with_agent's own wrapper starts a
    conversation, and how swe_task_agent.py builds its own initial context
    or a phase-transition kickoff, without duplicating this logic."""
    all_tools = load_tools(logging=logging, names=tools_available)
    if max_tool_calls > 0 and all_tools:
        msg = f"{msg}\n\n[You have up to {max_tool_calls} tool calls available for this task -- plan accordingly.]"
    if plan_act_observe and all_tools:
        msg = f"{msg}\n\n{PLAN_ACT_OBSERVE_INSTRUCTION}"
    return msg


def step(state, context, logging=print):
    """One Plan-Act-Observe round: makes exactly one new LLM call using the
    CURRENT context (never a pre-fetched response -- context already
    contains everything, including any new "turn" a caller wants to start,
    e.g. an initial instruction or a phase-transition kickoff message,
    which the caller appends as a {"role": "user", "text": ...} entry
    before calling step(), the same way get_response_from_llm's own
    msg=None convention already works) and, if it returned tool_calls,
    executes them for real. Always real execution -- no mode where a round
    can be "processed" without actually running its tool call(s).

    `state` carries everything needed to decide what happens this round
    (model, tools_available, temperature, max_tool_calls,
    multiple_tool_calls, num_tool_calls so far, round_num) plus whatever
    extra bookkeeping a caller adds (e.g. swe_task_agent.py's own phase
    tracking) -- step() only reads/writes its own keys (see _STATE_KEYS)
    and passes any others through untouched, so (state, context) together
    are a complete, replayable checkpoint: no out-of-band info is needed to
    resume from one, and a caller's own extra fields survive round after
    round without step() having to know about them.

    Returns (state', context'): context' has this round's messages
    appended. state' has num_tool_calls/round_num updated, "done" (True
    once this round ended the conversation -- no tool call, or the hard
    tool-call-budget cutoff), "hard_cutoff" (True only for the budget-cutoff
    case), and "last_round" (this round's own {"round","plan","act",
    "observe","usage","state"} record -- "state" inside IT is a snapshot of
    the INCOMING state, i.e. what this round actually had available when it
    made its decision, with "last_round" itself excluded from that snapshot
    so recorded state never nests a round's entire history inside itself).
    `logging` is a plain callable (defaults to print), never part of state
    -- state must stay JSON-serializable for replay, and where log output
    goes doesn't affect what decision gets made.
    """
    model = state["model"]
    tools_available = state.get("tools_available") or []
    temperature = state.get("temperature", 0.0)
    max_tool_calls = state.get("max_tool_calls", 40)
    multiple_tool_calls = state.get("multiple_tool_calls", False)
    num_tool_calls = state.get("num_tool_calls", 0)
    round_num = state.get("round_num", 0) + 1

    state_snapshot = {k: v for k, v in state.items() if k != "last_round"}

    all_tools = load_tools(logging=logging, names=tools_available)
    tools_dict = {tool['info']['name']: tool for tool in all_tools}
    litellm_tools = [_to_litellm_tool(tool['info']) for tool in all_tools] or None

    response, context, info = get_response_from_llm(
        msg=None, model=model, msg_history=context, tools=litellm_tools, temperature=temperature,
    )
    tool_calls = info.get("tool_calls") or []
    usage = info.get("usage")

    # A response cut off mid-generation (e.g. hit max_tokens) may carry no
    # tool_calls at all, or an incomplete one -- ask the model to retry
    # rather than silently treating a truncated response as final. Applied
    # to every round now (a deliberate, minor behavior change from the
    # pre-extraction code, which only ever retried the very first call of a
    # conversation -- an asymmetry with nothing explaining it, that looked
    # like an oversight rather than a deliberate choice; extending the same
    # protection to every round is strictly safer, not a new failure mode).
    if info.get("finish_reason") == "length" and not tool_calls:
        logging("Error: Output context exceeded. Please try again.")
        response, context, info = get_response_from_llm(
            msg="Error: Output context exceeded. Please try again.",
            model=model, msg_history=context, tools=litellm_tools, temperature=temperature,
        )
        logging(f"Retried after truncation. Output: {repr(response)}")
        tool_calls = info.get("tool_calls") or []
        usage = info.get("usage")

    # A response with tools available but no tool_calls used to be treated
    # as unconditionally terminal -- but confirmed live, repeatedly (two
    # different local Qwen model sizes acting as the foodtruck meta-agent):
    # a model can narrate a clear, specific intended action (or even just
    # echo a bare JSON-wrapped plan fragment) and simply never emit the
    # paired tool_call, which silently ended the whole session on round 1 or
    # 3 with the vast majority of its tool-call budget unused -- not the
    # model genuinely finishing, just failing to pair stated intent with the
    # actual call. One bounded nudge-and-retry: only when tools were
    # actually available this round (an agent with none configured, e.g.
    # TaskAgent's default INTERACTIVE_TOOLS=[], is SUPPOSED to end this way
    # every time -- nothing to nudge), and only once (if the retry also
    # comes back with no tool_calls, that's accepted as genuinely
    # finished -- avoids an infinite ping-pong with a model that's truly
    # done or persistently confused).
    if all_tools and not tool_calls:
        logging("Note: no tool call was made despite tools being available -- nudging for one retry.")
        response, context, info = get_response_from_llm(
            msg=(
                "You didn't make a tool call this round. If your plan above "
                "described an action, make that tool call now. If you're "
                "genuinely finished, you don't need to do anything else."
            ),
            model=model, msg_history=context, tools=litellm_tools, temperature=temperature,
        )
        logging(f"Retried after no tool call. Output: {repr(response)}")
        tool_calls = info.get("tool_calls") or []
        usage = info.get("usage")

    round_record = {"round": round_num, "plan": response, "act": [], "observe": [], "usage": usage, "state": state_snapshot}
    new_state = {**state, "round_num": round_num}

    if not tool_calls:
        # Terminal round: the model responded with plan text only, no tool
        # call -- nothing to act on or observe.
        logging(_format_round(round_record))
        new_state["last_round"] = round_record
        new_state["done"] = True
        new_state["hard_cutoff"] = False
        return new_state, context

    if max_tool_calls > 0 and num_tool_calls >= max_tool_calls:
        # The assistant message already in context still has these
        # tool_calls attached and unanswered -- the API rejects any later
        # message on this history until every tool_call_id gets a matching
        # "tool" response. Answer them here instead of just stopping, or
        # the returned context is left invalid for any caller that resumes
        # it later (confirmed live pre-refactor: exactly what broke
        # task_agent.py's own continuation pass, which reuses this history
        # when the first pass ends without a report -- "insufficient tool
        # messages following tool_calls message").
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
        context = context + [
            {"role": "tool", "tool_call_id": call["id"], "text": skip_text}
            for call in tool_calls
        ]
        logging(_format_round(round_record))
        new_state["last_round"] = round_record
        new_state["done"] = True
        new_state["hard_cutoff"] = True
        return new_state, context

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

    # The API requires a "tool" response for every tool_call_id the model
    # emitted this turn, even ones not executed here (when
    # multiple_tool_calls=False truncates to just the first) -- otherwise
    # the next request fails validation over the orphaned call.
    executed_ids = {call["id"] for call in calls_to_run}
    for call in tool_calls:
        if call["id"] not in executed_ids:
            skip_text = "Skipped: only one tool call is processed per turn."
            tool_result_msgs.append({"role": "tool", "tool_call_id": call["id"], "text": skip_text})
            round_record["act"].append({"id": call["id"], "name": call["name"], "arguments": call["arguments"]})
            round_record["observe"].append({"tool_call_id": call["id"], "name": call["name"], "result": skip_text})

    # Keep the running remaining-budget count visible after every round (not
    # just at the start) -- appended to the last tool result of this round
    # since the model reads every "tool" message from a round together
    # before its next turn.
    if max_tool_calls > 0:
        remaining = max_tool_calls - num_tool_calls
        tool_result_msgs[-1]["text"] += f"\n\n[{remaining} of {max_tool_calls} tool calls remaining -- plan accordingly.]"
        round_record["observe"][-1]["result"] = tool_result_msgs[-1]["text"]

    logging(_format_round(round_record))

    new_state["num_tool_calls"] = num_tool_calls
    new_state["last_round"] = round_record
    new_state["done"] = False
    new_state["hard_cutoff"] = False
    return new_state, context + tool_result_msgs


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
    # skills/branching's sibling rollouts and skills/patch_testing's
    # replicated checkpoint tests, which need rounds to actually diverge
    # from an identical starting point.
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
    the two callers this is meant for.

    A thin wrapper around step() (see its own docstring): builds the
    initial state/context, then just calls step() in a loop until it
    reports "done" -- this is round-granular replayability's whole point,
    letting a caller like swe_task_agent.py drive the same step() function
    itself with its own richer per-round state, one round at a time, instead
    of only ever getting a whole conversation back at once."""
    if msg_history is None:
        msg_history = []

    trajectory = []
    context = msg_history

    try:
        # Tell the model its own tool-call ceiling up front -- previously
        # this was purely a silent harness-side cap: the model had no way to
        # know one existed until it was already cut off. Observed live,
        # repeatedly: full budgets spent entirely on read-only exploration
        # before ever acting, then cut off mid-investigation with no warning.
        decorated_msg = build_kickoff_message(msg, max_tool_calls, tools_available, plan_act_observe, logging=logging)

        logging(f"Input: {repr(decorated_msg)}")
        context = context + [{"role": "user", "text": decorated_msg}]

        state = {
            "model": model,
            "tools_available": tools_available,
            "temperature": temperature,
            "max_tool_calls": max_tool_calls,
            "multiple_tool_calls": multiple_tool_calls,
            "num_tool_calls": 0,
            "round_num": 0,
        }
        while True:
            state, context = step(state, context, logging=logging)
            trajectory.append(state["last_round"])
            if state.get("done"):
                break

    except Exception as e:
        logging(f"Error: {str(e)}")
        raise e

    if plan_act_observe:
        return context, trajectory
    return context

if __name__ == "__main__":
    msg = """hello"""
    new_msg_history = chat_with_agent(msg)
