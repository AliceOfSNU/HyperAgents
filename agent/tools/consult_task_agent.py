import importlib
import json
import os
import sys


def tool_info():
    return {
        "name": "consult_task_agent",
        "description": """Run the CURRENT task_agent.py (the one you are evolving) on a sample input, and see exactly what it does -- its prediction and its full reasoning/tool-use trace -- instead of only reading its code or an aggregate report.json score.

Use this before deciding what to change: watching a live success or failure is usually more informative than static code review after the fact. This is the same task_agent.py that gets evaluated for real, so if it's broken you'll see the actual crash/behavior here.

* `inputs_json` must be a JSON object matching the shape a real harness passes for the domain you're testing, e.g. '{"domain": "paper_review", "paper_text": "..."}'. Check the relevant domains/<domain>/ code (format_input_dict, or an evaluator's run_episode) for the exact keys expected.
* This makes a real LLM call under the hood and costs real tokens/time -- use it deliberately, not in a loop over many inputs.
* Always re-imports task_agent.py fresh, so it reflects whatever you most recently edited.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "inputs_json": {
                    "type": "string",
                    "description": "JSON-encoded dict to pass as TaskAgent.forward(inputs)'s argument.",
                },
                "model": {
                    "type": "string",
                    "description": "Optional. litellm-format model string (e.g. 'deepseek/deepseek-v4-flash') to run the task agent with. Omit to use task_agent.py's own default.",
                },
            },
            "required": ["inputs_json"],
        },
    }


def tool_function(inputs_json, model=None):
    try:
        inputs = json.loads(inputs_json)
    except json.JSONDecodeError as e:
        return f"Error: inputs_json is not valid JSON: {e}"

    repo_root = os.getcwd()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        import task_agent as task_agent_module
        importlib.reload(task_agent_module)  # pick up whatever was just edited
        TaskAgent = task_agent_module.TaskAgent
    except Exception as e:
        return f"Error: could not import the current task_agent.py: {e}"

    try:
        kwargs = {"chat_history_file": "/tmp/consult_task_agent_chat_history.md"}
        if model:
            kwargs["model"] = model
        agent = TaskAgent(**kwargs)
        prediction, msg_history = agent.forward(inputs)
    except Exception as e:
        return f"task_agent.py raised an exception when run on this input: {e}"

    trace = "\n\n".join(
        f"[{m.get('role', '?')}]: {str(m.get('text', m))[:2000]}" for m in (msg_history or [])
    )
    return f"Prediction:\n{prediction}\n\nFull reasoning/tool-use trace:\n{trace}"


if __name__ == "__main__":
    # Example usage
    import sys as _sys
    if len(_sys.argv) < 2:
        print("Usage: python consult_task_agent.py '<inputs_json>' [model]")
    else:
        print(tool_function(_sys.argv[1], _sys.argv[2] if len(_sys.argv) > 2 else None))
