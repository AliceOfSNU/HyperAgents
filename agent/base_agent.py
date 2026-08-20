import json
from abc import ABC, abstractmethod
from pathlib import Path

from agent.llm import OPENAI_MODEL
from utils.thread_logger import ThreadLoggerManager


class AgentSystem(ABC):
    def __init__(
        self,
        model=OPENAI_MODEL,
        chat_history_file='./outputs/chat_history.md',
    ):
        self.model = model
        self.chat_history_file = chat_history_file

        # Initialize logger and store it in thread-local storage
        self.logger_manager = ThreadLoggerManager(log_file=chat_history_file)
        self.log = self.logger_manager.log

        # Clear the log file
        with open(chat_history_file, 'w') as f:
            f.write('')

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def save_trajectory(self, trajectory):
        """Persist the Plan-Act-Observe trajectory (chat_with_agent's
        plan_act_observe=True return value) as JSON alongside
        chat_history_file -- the structured, machine-readable counterpart to
        the human-readable Markdown log the same rounds are already written
        to via self.log."""
        path = Path(self.chat_history_file).with_suffix(".json")
        path.write_text(json.dumps(trajectory, indent=2), encoding="utf-8")
        return path
