import backoff
import os
import uuid
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
# Served locally via Ollama (network_mode="host" containers reach the host's
# own localhost:11434 directly, no extra Docker networking needed -- see
# utils/docker_utils.py's build_container). "ollama_chat/" (not "ollama/")
# specifically -- litellm's chat-completions-shaped route, needed for this
# codebase's multi-turn message history + native tool-calling to work.
QWEN_LOCAL_MODEL = "ollama_chat/qwen3:8b"
# Served remotely (Vast.ai rental A100 80GB) via an SSH -L tunnel from this
# host's own localhost:21434 -- picked instead of the default 11434 because
# this machine's own local Ollama daemon (serving QWEN_LOCAL_MODEL above) is
# already bound to 11434, and a different port avoids fighting over it (no
# passwordless sudo to stop that systemd-managed daemon). network_mode="host"
# containers share this host's loopback, so localhost:21434 reaches the
# tunnel transparently the same way localhost:11434 does for QWEN_LOCAL_MODEL
# -- but litellm does NOT pick this nonstandard port up automatically (an
# OLLAMA_API_BASE env var is silently ignored; confirmed live), so
# get_response_from_llm below passes api_base explicitly, keyed off this
# model string, rather than relying on any env var.
QWEN_LOCAL_LARGE_MODEL = "ollama_chat/qwen3.8:27b"
# Also on the same rented A100 80GB. Unlike QWEN_LOCAL_LARGE_MODEL, this
# model's "thinking" is NOT toggleable -- confirmed live, twice, that
# think=False has zero effect; every response includes a full reasoning
# trace regardless (this is an R1-distill, trained to always reason, and
# Ollama's own model card for it lists no <think> stop token the way
# qwen3.8:27b's does). Kept anyway per direct instruction to measure real
# cost against deepseek-v4-flash rather than assume it's a bad fit.
DEEPSEEK_R1_70B_LOCAL_MODEL = "ollama_chat/deepseek-r1:70b"
_OLLAMA_API_BASE_BY_MODEL = {
    QWEN_LOCAL_LARGE_MODEL: "http://localhost:21434",
    DEEPSEEK_R1_70B_LOCAL_MODEL: "http://localhost:21434",
}
# deepseek-r1:70b's own tool-calling turned out unreliable via Ollama
# regardless of model size (confirmed live: deepseek-r1:32b failed 5/5
# trials too) -- traced to Ollama's bundled chat template never rendering
# the available tools into the prompt at all, a gap in the model's own
# distributed Modelfile, not a capability limit. Switched this specific
# model to vLLM instead (served on the same rented A100, a fresh instance
# each time -- Vast.ai destroys the box outright if outbid, nothing
# persists), which lets us supply a corrected chat template directly. Even
# then, vLLM's own built-in tool-call parser cannot be trusted for this
# checkpoint: confirmed live across multiple parser/template combinations
# that the model reliably determines the right function+arguments but
# wraps them inconsistently (<response>, ```xml, bare JSON -- never
# reliably vLLM's own expected <tool_call> tag), and DeepSeek's native
# token-based format (<｜tool▁calls▁begin｜>...) isn't parseable at all
# here since this checkpoint's tokenizer is Qwen2's, which never
# registered those as real special tokens. get_response_from_llm below
# therefore renders tools via a corrected Hermes-style template server-side
# (informs the model what's available) but parses the response itself,
# leniently, rather than trusting vLLM's own parser.
DEEPSEEK_R1_32B_VLLM_MODEL = "hosted_vllm/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
_HOSTED_VLLM_API_BASE_BY_MODEL = {
    DEEPSEEK_R1_32B_VLLM_MODEL: "http://localhost:28000/v1",
}


def _extract_lenient_tool_call(text):
    """Scan raw text for a {"name": ..., "arguments": {...}} JSON object,
    regardless of what it's wrapped in (XML tags, code fences, bare, prose
    around it) -- confirmed live this recovers DEEPSEEK_R1_32B_VLLM_MODEL's
    tool calls reliably: its own JSON payload (function name + arguments)
    is consistently correct, only the surrounding wrapper varies, none of
    which match vLLM's own strict <tool_call>-tag parser. Tries every '{'
    in the text as a possible start and uses JSONDecoder.raw_decode to
    parse just that one balanced object, ignoring anything before or after
    it -- a naive regex would break on the nested braces inside
    "arguments" itself. Returns the first valid match's parsed dict
    ({"name": str, "arguments": dict}), or None if nothing matches."""
    if not text:
        return None
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and isinstance(obj.get("arguments"), dict):
            return obj
    return None

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
        # DEEPSEEK_R1_32B_VLLM_MODEL's server is deliberately run WITHOUT
        # --enable-auto-tool-choice (see its own comment -- we parse tool
        # calls ourselves instead of trusting vLLM's built-in parser), and
        # vLLM rejects tool_choice="auto" outright when that flag isn't
        # set. Simply omitting tool_choice doesn't avoid this -- confirmed
        # live litellm's hosted_vllm provider injects "auto" by default
        # whenever tools is non-empty, regardless of what this code sends.
        # "none" is the one value vLLM accepts without the flag -- and the
        # custom chat template still renders `tools` into the prompt either
        # way, since that's driven by `tools` being present, not by
        # tool_choice's value.
        completion_kwargs["tool_choice"] = "none" if model == DEEPSEEK_R1_32B_VLLM_MODEL else tool_choice

    # GPT-5 and GPT-5-mini only support default temperature (1), skip it
    # GPT-5.2 supports temperature
    if model in ["openai/gpt-5", "openai/gpt-5-mini"]:
        pass  # Don't set temperature
    else:
        completion_kwargs["temperature"] = temperature

    # Qwen3 (like DeepSeek's models) is a hybrid reasoning model that defaults
    # to an extended <think> pass before answering -- confirmed live this can
    # burn the entire max_tokens budget on reasoning alone with zero actual
    # content left (the exact failure mode that motivated moving off
    # deepseek-v4-flash for this domain). Ollama's own "think" request field,
    # passed through by litellm, turns that off -- confirmed live this cuts a
    # trivial call from ~4s/172 tokens to ~0.2s/2 tokens. Gated on the model
    # string so this is never sent to any other provider.
    if model.startswith("ollama"):
        completion_kwargs["think"] = False
        if model in _OLLAMA_API_BASE_BY_MODEL:
            completion_kwargs["api_base"] = _OLLAMA_API_BASE_BY_MODEL[model]
        if model == QWEN_LOCAL_MODEL:
            # Ollama's own runtime default context window is 4096 regardless
            # of what the model itself supports (qwen3:8b supports up to
            # 40960) -- confirmed live this is a real, separate allocation
            # from the model weights, sized against this machine's own 8GB
            # GPU: 16384 keeps ~77% of the KV cache on GPU (confirmed live
            # via `ollama ps`), vs. 32768 which drops to ~53% GPU and
            # measurably slows every call. Long meta-agent sessions can
            # still exceed this and lose early context (Ollama truncates
            # oldest-first, not a crash) -- a real capability/speed tradeoff
            # inherent to this hardware, not tuned away.
            completion_kwargs["num_ctx"] = 16384
        # QWEN_LOCAL_LARGE_MODEL (remote A100 80GB) gets no num_ctx override:
        # confirmed live Ollama's own VRAM-aware default already picks this
        # model's full native 262144 -- overriding down would only throw
        # away context this deployment has plenty of room for.
        elif model == DEEPSEEK_R1_70B_LOCAL_MODEL:
            # Unlike QWEN_LOCAL_LARGE_MODEL, this one's own weights (42GB
            # Q4_K_M) plus Ollama's VRAM-aware default context (131072)
            # together slightly EXCEED this box's 78.8GB VRAM -- confirmed
            # live via `ollama ps` this spills ~5% onto CPU, which collapses
            # generation from ~20 tok/s to ~3 tok/s (a single CPU-resident
            # layer bottlenecks the whole forward pass for a model this
            # size, unlike the earlier num_ctx tradeoffs above which only
            # traded KV-cache speed, not this catastrophically). 32768
            # confirmed live keeps the whole model+cache on GPU.
            completion_kwargs["num_ctx"] = 32768
    elif model.startswith("hosted_vllm"):
        # No "think" request field here -- that's Ollama-specific; vLLM has
        # no equivalent, and this model's reasoning can't be suppressed
        # regardless (server-side --reasoning-parser separates it into its
        # own field instead, see DEEPSEEK_R1_32B_VLLM_MODEL's own comment).
        # No num_ctx either -- vLLM's context length is fixed at server
        # startup (--max-model-len), not a per-request option.
        if model in _HOSTED_VLLM_API_BASE_BY_MODEL:
            completion_kwargs["api_base"] = _HOSTED_VLLM_API_BASE_BY_MODEL[model]
        # litellm's hosted_vllm provider requires a non-empty api_key even
        # though vLLM's own default server doesn't check one -- any
        # non-empty string satisfies the client-side check.
        completion_kwargs["api_key"] = "dummy"

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

    # DEEPSEEK_R1_32B_VLLM_MODEL's server-side chat template appends its EOS
    # token (<|im_end|>) onto the generated content -- confirmed live in the
    # foodtruck baseline, where the task agent's commands came back as "buy
    # beef 5<|im_end|>" / "check recipes<|im_end|>" and every one of those
    # parsed as an invalid action (the env's command parser is exact). This
    # was independently found and fixed by a real meta-agent generation
    # (deepseek-v4-flash, foodtruck_r1_32b_vllm gen_1) -- ported the fix here
    # directly rather than waiting to rediscover it. Strip trailing
    # chat-template special tokens here, at the single choke point every
    # provider's text passes through, so no downstream caller
    # (extract_command, interactive parsers, etc.) ever has to defend against
    # them again. Only trailing occurrences are removed -- the tokens are
    # terminators, never legitimately mid-string.
    for _ in range(8):  # a handful of repeats is plenty; loop guards pathological cases
        stripped = response_text.rstrip()
        if stripped.endswith("<|im_end|>"):
            stripped = stripped[: -len("<|im_end|>")].rstrip()
        elif stripped.endswith("<|endoftext|>"):
            stripped = stripped[: -len("<|endoftext|>")].rstrip()
        else:
            break
        response_text = stripped

    finish_reason = response['choices'][0].get("finish_reason")  # pyright: ignore

    # Best-effort: not every provider/model reports usage, and reasoning_tokens
    # specifically is only present for reasoning-capable models. `response`
    # supports dict-style access (confirmed above), but the nested usage/
    # details objects may be litellm's own Usage class instead of a plain
    # dict, so probe both access styles rather than assuming one.
    def _field(obj, key):
        if obj is None:
            return None
        try:
            return obj[key]
        except (TypeError, KeyError, IndexError):
            return getattr(obj, key, None)

    usage = _field(response, "usage")
    details = _field(usage, "completion_tokens_details")
    usage_out = {
        "prompt_tokens": _field(usage, "prompt_tokens"),
        "completion_tokens": _field(usage, "completion_tokens"),
        "total_tokens": _field(usage, "total_tokens"),
        "reasoning_tokens": _field(details, "reasoning_tokens"),
    }

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
    elif tools and model == DEEPSEEK_R1_32B_VLLM_MODEL:
        # See DEEPSEEK_R1_32B_VLLM_MODEL's own comment and
        # _extract_lenient_tool_call's docstring -- this model's native
        # tool_calls never comes back populated for this checkpoint, but
        # the raw text reliably contains a correct {"name", "arguments"}
        # payload anyway.
        lenient = _extract_lenient_tool_call(response_text)
        if lenient:
            synthetic_id = f"lenient_{uuid.uuid4().hex[:8]}"
            parsed_tool_calls.append({
                "id": synthetic_id, "name": lenient["name"], "arguments": lenient["arguments"],
            })
            # Mirror the native tool_calls shape (content=None, a
            # tool_calls list) into the history we hand back too, so a
            # later turn sees exactly what a real native tool-calling
            # response would have looked like, not this checkpoint's own
            # inconsistent raw wrapper text.
            assistant_msg["content"] = None
            assistant_msg["tool_calls"] = [{
                "id": synthetic_id, "type": "function",
                "function": {"name": lenient["name"], "arguments": json.dumps(lenient["arguments"])},
            }]

    new_msg_history.append(assistant_msg)

    # Convert content to text, compatible with MetaGen API
    new_msg_history = [
        {**m, "text": m.pop("content")} if "content" in m else m
        for m in new_msg_history
    ]

    return response_text, new_msg_history, {"tool_calls": parsed_tool_calls, "finish_reason": finish_reason, "usage": usage_out}


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
