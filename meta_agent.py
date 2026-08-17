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
        instruction = f"""Modify the codebase at `{repo_path}` to improve its ability to generate better agents for downstream tasks.

Context:
- Path to previously generated agents and their evaluation results: `{eval_path}`.
- Number of remaining iterations in which you will be invoked in future: {iterations_left if iterations_left is not None else 'unknown'}.

Your job is to analyze the current code and the evaluation results, then make targeted, high-leverage edits to the agent generation / self-improvement mechanisms. Prioritise changes that:
1. Make the generated downstream agents (e.g. `task_agent.py`, its prompt, tool usage, and environment handling) more capable and reliable.
2. Improve the meta-agent loop (`run_meta_agent.py`, `generate_loop.py`, selection/archive logic) so that future iterations get better feedback and explore more promising lineages.
3. Fix any bugs or inefficiencies you notice in the agent infrastructure (tools, Docker orchestration, evaluation plumbing).

When changing tool implementations, especially `agent/tools/bash.py`, preserve functional correctness. After editing, verify your changes with quick smoke tests / compilation checks (e.g. import the modified module and run a simple command). Avoid over-aggressive simplifications that break existing functionality.

Be concise and focused: make a small number of high-quality edits rather than many speculative ones."""

        new_msg_history = chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available='all',
            multiple_tool_calls=True,
            max_tool_calls=60,
        )
