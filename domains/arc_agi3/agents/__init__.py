from task_agent import TaskAgent
from agent.llm import OPENAI_MINI_MODEL


class AgentFactory:
    """Factory class for creating agents based on configuration.

    Mirrors domains/balrog/agents/AgentFactory: the task agent being evolved by the
    DGM-H is always `task_agent.TaskAgent` at the repo root, this just wires it up
    with the model and per-episode chat log used for this domain.
    """

    def __init__(self, config):
        self.config = config

    def create_agent(self, chat_history_file):
        return TaskAgent(model=OPENAI_MINI_MODEL, chat_history_file=chat_history_file)
