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
        # Build a rich instruction that includes the repository path, the path to
        # previously generated agents and their evaluation results, and the number
        # of remaining iterations. This enables the meta-agent to inspect past
        # attempts and evaluation feedback to make targeted improvements.
        instruction = f"""You are the self-improving meta-agent for {repo_path}.

Your job is to improve the codebase so that it can generate better task-solving agents.

Key paths:
- Repository root: `{repo_path}`
- Previously generated agents and evaluation results: `{eval_path}`
- Iterations left in this run: {iterations_left if iterations_left is not None else 'unknown'}

You should:
1. Explore the repository and the evaluation results under `{eval_path}`.
2. Identify weaknesses in the current codebase (e.g., insufficient prompts, missing tooling, fragile parsing, or poor agent scaffolding).
3. Modify any part of the codebase to iteratively improve the agent generation pipeline.
4. Keep your changes focused, coherent, and well-documented.

Modify any part of the codebase at `{repo_path}`."""

        new_msg_history = chat_with_agent(instruction, model=self.model, msg_history=[], logging=self.log, tools_available='all')
