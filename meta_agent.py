# Copyright (c) Meta Platforms, Inc. and affiliates.

import os

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent

class MetaAgent(AgentSystem):
    def forward(self, repo_path, eval_path, iterations_left=None, parent_genid=None):
        """
        A meta agent that recursively self-improves.

        Args:
            repo_path (str): The path to the repository.
            eval_path (str): The path to previously generated agents and their evaluation results.
            iterations_left (int, optional): The number of remaining iterations in which the meta agent will be invoked in future. Defaults to None.
            parent_genid (int, optional): The generation id this run's codebase was built from, used to point out where that generation's own self-modification attempt lives under eval_path.
        """
        instruction = (
            f"Modify any part of the codebase at `{repo_path}`. "
            "If this codebase also has an evaluator_agent.py, it co-evolves "
            "alongside task_agent.py under your same edits, and its own quality is "
            "checked separately against a fixed ground-truth anchor -- see how it's "
            "used in eval_path before assuming it's already good. It's just as "
            "editable and just as worth improving as the task agent it grades.\n\n"
            "Check /tmp/steering/ (it may not exist or may be empty -- that's fine). "
            "A human overseeing this run may leave files there for you: notes, PDFs, "
            "links, anything. Use the read_pdf and fetch_url tools if you want to "
            "read something in there. Whether any of it is relevant right now, and "
            "whether to act on it, is entirely your judgment call.\n\n"
            f"`{eval_path}` holds the full history of every earlier generation in this "
            "lineage: each gen_N/ has that generation's own self-modification attempt "
            "(agent_output/model_patch.diff, and agent_output/meta_agent_chat_history.md "
            "-- the raw conversation an earlier version of you had) plus its evaluation "
            "results. Also note that the agent harness code itself (e.g. "
            "agent/llm_withtools.py's chat_with_agent, which every agent in this repo, "
            "including you right now, runs through) is just as much yours to inspect "
            "and improve as task_agent.py or evaluator_agent.py are."
        )

        parent_note = self._parent_empty_diff_note(eval_path, parent_genid)
        if parent_note:
            instruction += f"\n\n{parent_note}"

        # chat_with_agent's own default (40) was repeatedly exhausted purely
        # on read-only exploration across several real generations -- gen_1's
        # own prior attempt read essentially the whole repo (harness code,
        # prior generations' diffs and eval breakdowns, anchor cache) and
        # never reached the editor tool. Doubled explicitly here (matching
        # task_agent.py's own explicit max_tool_calls) rather than raising
        # chat_with_agent's shared default, which would also silently affect
        # the unrelated baseline callers that rely on it.
        new_msg_history = chat_with_agent(instruction, model=self.model, msg_history=[], logging=self.log, tools_available='all', max_tool_calls=80)

    @staticmethod
    def _parent_empty_diff_note(eval_path, parent_genid):
        """If the generation this run builds on produced an empty
        self-modification diff, say so plainly and point at where its raw
        attempt lives -- deliberately doesn't say why, so it's on the agent
        to actually investigate rather than being handed the diagnosis."""
        if not parent_genid or parent_genid < 1:
            return None
        parent_output_dir = os.path.join(eval_path, f"gen_{parent_genid}", "agent_output")
        patch_path = os.path.join(parent_output_dir, "model_patch.diff")
        if not os.path.exists(patch_path) or os.path.getsize(patch_path) > 0:
            return None
        return (
            f"Note: gen_{parent_genid}, the generation you're building on, produced an "
            f"empty self-modification diff -- its raw attempt is at `{parent_output_dir}`. "
            "Might be worth understanding why before continuing."
        )
