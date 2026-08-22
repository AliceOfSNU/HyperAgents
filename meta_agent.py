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
            f"Modify any part of the codebase at `{repo_path}` -- including "
            "creating new files or directories where that fits better than "
            "editing what's already there.\n\n"
            "Check /tmp/steering/ (it may not exist or may be empty -- that's fine). "
            "A human overseeing this run may leave files there for you: notes, PDFs, "
            "links, anything. Use the read_pdf and fetch_url tools if you want to "
            "read something in there. Whether any of it is relevant right now, and "
            "whether to act on it, is entirely your judgment call.\n\n"
            f"`{eval_path}` holds the full history of every earlier generation in this "
            "lineage: each gen_N/ has that generation's own self-modification attempt "
            "(agent_output/model_patch.diff, agent_output/meta_agent_chat_history.md -- "
            "an earlier version of you, as a sequence of Plan/Act/Observe rounds -- and "
            "agent_output/meta_agent_chat_history.json, the same rounds as structured "
            "data) plus its evaluation results. Also note that the agent harness code "
            "itself (e.g. "
            "agent/llm_withtools.py's chat_with_agent, which every agent in this repo, "
            "including you right now, runs through) is just as much yours to inspect "
            "and improve as the task-facing agent code (task_agent.py, "
            "swe_task_agent.py) is.\n\n"
            "The compare_generations tool prints a table of every prior "
            "generation's scores plus cheap behavioral metrics (rounds/task, "
            "planning frequency, read/write/test-run mix, token usage where "
            "available, per-language breakdown) in one call, instead of you "
            "opening each generation's report.json and chat_history.json by "
            f"hand -- point it at `{eval_path}`.\n\n"
            f"`{repo_path}/memory.jsonl` is yours too -- a plain file, readable "
            "directly with bash or the editor tool same as any other, and free "
            "to read in full any time you want (whether to, when, is your call). "
            "Notes are structured (title, description, content, type, files, "
            "which generation(s) it's about -- see memory_append's own "
            "description for the exact schema), one note per issue (if you "
            "make several unrelated changes this generation, write a separate "
            "note for each rather than summarizing all of them in one), and "
            "immutable, with one "
            "exception: right after memory_append, its result includes related "
            "earlier notes it already found using your note's own content -- if "
            "any are genuinely related, memory_link can attach relation-typed "
            "links (resolves/develops/contradicts/other) from the note you just "
            "wrote to them, but only that note; it's for finishing what you just "
            "created, not revising older notes. memory_search is for everything "
            "else: ad-hoc lookup any time you want notes about something "
            "specific instead of reading everything, showing both a note's own "
            "links and its backlinks (other notes that point at it).\n\n"
            f"`{repo_path}/skills/` holds packaged capabilities beyond your "
            "built-in tools -- each is a directory with its own SKILL.md "
            "(what it's for, when to reach for it, how to invoke it) plus "
            "whatever scripts it needs. Check there for something relevant "
            "before assuming you have to do a thing entirely by hand. "
            "skills/branching/ is the first one: sampling several real "
            "continuations from a specific point in a task-agent trajectory "
            "to find out which choice actually mattered, instead of guessing "
            "from reading one trajectory alone -- see its own SKILL.md before "
            "using it, including why it's deferred (you queue it, a later "
            "session reads the result) and what it costs."
        )

        parent_note = self._parent_empty_diff_note(eval_path, parent_genid)
        if parent_note:
            instruction += f"\n\n{parent_note}"

        # chat_with_agent's own default (40) was repeatedly exhausted purely
        # on read-only exploration across several real generations -- gen_1's
        # own prior attempt read essentially the whole repo (harness code,
        # prior generations' diffs and eval breakdowns) and never reached
        # the editor tool. Doubled explicitly here (matching
        # task_agent.py's own explicit max_tool_calls) rather than raising
        # chat_with_agent's shared default, which would also silently affect
        # the unrelated baseline callers that rely on it.
        new_msg_history, trajectory = chat_with_agent(
            instruction, model=self.model, msg_history=[], logging=self.log,
            tools_available='all', max_tool_calls=80, plan_act_observe=True,
        )
        self.save_trajectory(trajectory)

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
