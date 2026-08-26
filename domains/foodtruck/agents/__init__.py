from task_agent import TaskAgent
from agent.llm import DEEPSEEK_R1_32B_VLLM_MODEL


class AgentFactory:
    """Factory class for creating agents based on configuration.

    Mirrors domains/arc_agi3/agents/AgentFactory: the task agent being evolved
    by the DGM-H is always `task_agent.TaskAgent` at the repo root, this just
    wires it up with the model and per-episode chat log used for this domain.

    Model history for this domain, each swap driven by confirmed-live
    evidence (see agent/llm.py for the constants): deepseek-v4-flash
    (matching the reference agent's own default in 11-766-hw2's
    simple_llm_agent.py, and deep_swe's AGENT_MODEL) -- dropped after a
    single 3-seed baseline eval took ~2 hours, traced to Flash's own
    reasoning-token overhead. QWEN_LOCAL_MODEL (local Ollama qwen3:8b) --
    dramatically faster (~2 min baseline eval) but the meta-agent role
    specifically failed outright: 10/10 generations produced empty diffs,
    root-caused to the model narrating an intended tool call without ever
    issuing it, which this codebase's chat_with_agent reads as "session
    done". QWEN_LOCAL_LARGE_MODEL (qwen3.8:27b on a rented A100 80GB) --
    strong, fast task-agent baseline (ROI 0.396), but the SAME meta-agent
    failure mode persisted despite several harness/prompt fixes, root cause
    never conclusively identified. DEEPSEEK_R1_70B_LOCAL_MODEL (Ollama)
    could call tools correctly but only 0/10 times in live trials -- traced
    to Ollama's own distributed chat template never rendering the available
    tools into the prompt at all, not a model capability gap. Now
    DEEPSEEK_R1_32B_VLLM_MODEL (same checkpoint family, served via vLLM
    instead of Ollama, on the same rented box) -- vLLM lets us supply a
    corrected template, and this model then reliably determines the right
    function+arguments but wraps the JSON inconsistently rather than in
    vLLM's own expected tag format, so get_response_from_llm parses the
    response itself rather than trusting vLLM's built-in tool-call parser
    (see agent/llm.py's own comments on both). The meta-agent for this run
    is deepseek-v4-flash instead (see generate_loop.py's own
    --meta_agent_model flag) -- swap the model string here if a different
    model is preferred."""

    def __init__(self, config):
        self.config = config

    def create_agent(self, chat_history_file):
        return TaskAgent(model=DEEPSEEK_R1_32B_VLLM_MODEL, chat_history_file=chat_history_file)
