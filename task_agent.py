import os

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent

class TaskAgent(AgentSystem):
    def forward(self, inputs):
        """
        A research agent that independently conducts a ResearchClawBench-style
        scientific research task: explores the provided data and related work,
        writes and runs analysis code, and produces a publication-quality
        report/report.md with figures.

        Args:
            inputs (dict): {"instructions": str} -- the fully-rendered task
                instructions (ResearchClawBench's INSTRUCTIONS.md template,
                already filled in with the workspace path, task description,
                and data manifest). The current working directory is expected
                to already be that workspace (run_task_agent.py sets this up).

        Returns:
            tuple:
                - prediction (str): "done" if report/report.md exists, else a short failure note.
                - new_msg_history (list): full message history of the interaction.
        """
        instruction = inputs["instructions"]
        new_msg_history = chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available='all',
            multiple_tool_calls=True,
            max_tool_calls=100,
        )

        prediction = "done" if os.path.exists("report/report.md") else "incomplete: report/report.md not found"
        return prediction, new_msg_history
