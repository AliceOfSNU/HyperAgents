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
        instruction = (
            f"Modify any part of the codebase at `{repo_path}`. "
            "Consider using the consult_task_agent tool to actually run the current "
            "task_agent.py on a sample input before deciding what to change -- "
            "seeing a live trace is often more informative than reading code or "
            "report.json scores alone.\n\n"
            "If this codebase also has an evaluator_agent.py, it co-evolves "
            "alongside task_agent.py under your same edits, and its own quality is "
            "checked separately against a fixed ground-truth anchor -- see how it's "
            "used in eval_path before assuming it's already good. It's just as "
            "editable and just as worth improving as the task agent it grades.\n\n"
            "Check /tmp/steering/ (it may not exist or may be empty -- that's fine). "
            "A human overseeing this run may leave files there for you: notes, PDFs, "
            "links, anything. Use the read_pdf and fetch_url tools if you want to "
            "read something in there. Whether any of it is relevant right now, and "
            "whether to act on it, is entirely your judgment call."
        )

        new_msg_history = chat_with_agent(instruction, model=self.model, msg_history=[], logging=self.log, tools_available='all')
