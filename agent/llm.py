import backoff
import os
from typing import Tuple
import requests
import litellm
from dotenv import load_dotenv
import json

load_dotenv()

MAX_TOKENS = 16384

CLAUDE_MODEL = "anthropic/claude-sonnet-4-5-20250929"
CLAUDE_HAIKU_MODEL = "anthropic/claude-3-haiku-20240307"
CLAUDE_35NEW_MODEL = "anthropic/claude-3-5-sonnet-20241022"
OPENAI_MODEL = "openai/gpt-4o"
OPENAI_MINI_MODEL = "openai/gpt-4o-mini"
OPENAI_O3_MODEL = "openai/o3"
OPENAI_O3MINI_MODEL = "openai/o3-mini"
OPENAI_O4MINI_MODEL = "openai/o4-mini"
OPENAI_GPT52_MODEL = "openai/gpt-5.2"
OPENAI_GPT5_MODEL = "openai/gpt-5"
OPENAI_GPT5MINI_MODEL = "openai/gpt-5-mini"
GEMINI_3_MODEL = "gemini/gemini-3-pro-preview"
GEMINI_MODEL = "gemini/gemini-2.5-pro"
GEMINI_FLASH_MODEL = "gemini/gemini-2.5-flash"
DEEPSEEK_MODEL = "deepseek/deepseek-v4-pro"
DEEPSEEK_FLASH_MODEL = "deepseek/deepseek-v4-flash"

litellm.drop_params=True

REQUEST_TIMEOUT_SEC = 180  # See get_response_from_llm's own completion_kwargs
# for why this exists at all -- comfortably under max_time below so a single
# hang still leaves room for a retry within the overall backoff budget.

@backoff.on_exception(
    backoff.expo,
    (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, litellm.exceptions.Timeout),
    max_time=600,
    max_value=60,
)
def get_response_from_llm(
    msg,
    model: str = OPENAI_MODEL,
    temperature: float = 0.0,
    max_tokens: int = MAX_TOKENS,
    msg_history=None,
    tools=None,
    tool_choice="auto",
) -> Tuple[str, list, dict]:
    """`msg=None` skips appending a new user turn and just sends msg_history
    as-is -- used by chat_with_agent to continue a conversation after
    appending tool-result messages, without inventing an extra user turn.

    `tools`, if given, is a list of OpenAI/litellm-format function-calling
    tool definitions (`{"type": "function", "function": {"name", "description",
    "parameters"}}`). When provided, the returned info dict carries
    `tool_calls` (a normalized `[{"id", "name", "arguments"}, ...]`, "arguments"
    already parsed from JSON) and `finish_reason`, and the assistant message
    appended to history carries the provider's own raw `tool_calls` alongside
    `text` -- needed verbatim on the next call so the API can match up the
    "tool" role responses that follow it."""
    if msg_history is None:
        msg_history = []

    # Convert text to content, compatible with LITELLM API
    msg_history = [
        {**m, "content": m.pop("text")} if "text" in m else m
        for m in msg_history
    ]

    new_msg_history = msg_history if msg is None else msg_history + [{"role": "user", "content": msg}]

    # Build kwargs - handle model-specific requirements
    completion_kwargs = {
        "model": model,
        "messages": new_msg_history,
        # Without this, a request that never gets a response (confirmed
        # live: sockets stuck in CLOSE_WAIT, the remote end had already
        # hung up) just blocks forever -- litellm has no default timeout of
        # its own. This turns that into litellm.exceptions.Timeout, which
        # the backoff decorator above now retries instead of hanging.
        "timeout": REQUEST_TIMEOUT_SEC,
    }
    if tools:
        completion_kwargs["tools"] = tools
        completion_kwargs["tool_choice"] = tool_choice

    # GPT-5 and GPT-5-mini only support default temperature (1), skip it
    # GPT-5.2 supports temperature
    if model in ["openai/gpt-5", "openai/gpt-5-mini"]:
        pass  # Don't set temperature
    else:
        completion_kwargs["temperature"] = temperature

    # GPT-5 models require max_completion_tokens instead of max_tokens
    if "gpt-5" in model:
        completion_kwargs["max_completion_tokens"] = max_tokens
    else:
        # Claude Haiku has a 4096 token limit
        if "claude-3-haiku" in model:
            completion_kwargs["max_tokens"] = min(max_tokens, 4096)
        else:
            completion_kwargs["max_tokens"] = max_tokens

    response = litellm.completion(**completion_kwargs)
    message = response['choices'][0]['message']  # pyright: ignore
    response_text = message.get("content") or ""
    finish_reason = response['choices'][0].get("finish_reason")  # pyright: ignore

    assistant_msg = {"role": "assistant", "content": message.get("content")}
    raw_tool_calls = message.get("tool_calls") or []
    parsed_tool_calls = []
    if raw_tool_calls:
        # Preserved verbatim (not re-serialized) -- the API needs to see
        # exactly what the model emitted on the next call, to match it up
        # against the "tool" role responses that will follow it.
        assistant_msg["tool_calls"] = raw_tool_calls
        for tc in raw_tool_calls:
            try:
                arguments = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            parsed_tool_calls.append({
                "id": tc["id"], "name": tc["function"]["name"], "arguments": arguments,
            })

    new_msg_history.append(assistant_msg)

    # Convert content to text, compatible with MetaGen API
    new_msg_history = [
        {**m, "text": m.pop("content")} if "content" in m else m
        for m in new_msg_history
    ]

    return response_text, new_msg_history, {"tool_calls": parsed_tool_calls, "finish_reason": finish_reason}


if __name__ == "__main__":
    msg = 'Hello there!'
    models = [
        ("CLAUDE_MODEL", CLAUDE_MODEL),
        ("CLAUDE_HAIKU_MODEL", CLAUDE_HAIKU_MODEL),
        ("CLAUDE_35NEW_MODEL", CLAUDE_35NEW_MODEL),
        ("OPENAI_MODEL", OPENAI_MODEL),
        ("OPENAI_MINI_MODEL", OPENAI_MINI_MODEL),
        ("OPENAI_O3_MODEL", OPENAI_O3_MODEL),
        ("OPENAI_O3MINI_MODEL", OPENAI_O3MINI_MODEL),
        ("OPENAI_O4MINI_MODEL", OPENAI_O4MINI_MODEL),
        ("OPENAI_GPT52_MODEL", OPENAI_GPT52_MODEL),
        ("OPENAI_GPT5_MODEL", OPENAI_GPT5_MODEL),
        ("OPENAI_GPT5MINI_MODEL", OPENAI_GPT5MINI_MODEL),
        ("GEMINI_3_MODEL", GEMINI_3_MODEL),
        ("GEMINI_MODEL", GEMINI_MODEL),
        ("GEMINI_FLASH_MODEL", GEMINI_FLASH_MODEL),
        ("DEEPSEEK_MODEL", DEEPSEEK_MODEL),
        ("DEEPSEEK_FLASH_MODEL", DEEPSEEK_FLASH_MODEL),
    ]
    for name, model in models:
        print(f"\n{'='*50}")
        print(f"Testing {name}: {model}")
        print('='*50)
        try:
            output_msg, msg_history, info = get_response_from_llm(msg, model=model)
            print(f"OK: {output_msg[:100]}...")
        except Exception as e:
            print(f"FAIL: {str(e)[:200]}")
