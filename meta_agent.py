# Copyright (c) Meta Platforms, Inc. and affiliates.

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent

class MetaAgent(AgentSystem):
    def forward(self, repo_path, eval_path, iterations_left=None):
        """
        A meta agent that recursively self-improves.

        Args:
            repo_path (str): The path to the repository.
            eval_path (str): The path to previously generated agents and their evaluation results.
            iterations_left (int, optional): The number of remaining iterations in which the meta agent will be invoked in future. Defaults to None.
        """
        instruction = f"""Modify any part of the codebase at `{repo_path}`.

Context:
- Path to previously generated agents and their evaluation results: `{eval_path}`.
- Number of remaining iterations in which you will be invoked in future: {iterations_left if iterations_left is not None else 'unknown'}.

Improve the system's ability to generate better agents for downstream tasks. Analyze the current code and the evaluation results, then make targeted edits to improve the agent generation and self-improvement mechanisms. Be concise and focused."""

        new_msg_history = chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available='all',
            multiple_tool_calls=True,
            max_tool_calls=60,
        )
