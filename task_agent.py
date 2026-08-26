import os

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent, step


class TaskAgent(AgentSystem):
    """Shared task agent -- used directly by the research domain (one-shot,
    inputs={"instructions": str}) and, via domains/harness.py's generic
    dispatch, by every domain outside deep_swe's own dedicated
    swe_task_agent.py: the CSV domains (search_arena/paper_review/imo_*,
    each with their own one-shot inputs shape) and the interactive
    gym-style domains (arc_agi3/balrog/genesis/foodtruck -- forward()
    called once per environment step, on the SAME agent instance, for a
    whole episode).

    Dispatches on whether `inputs` carries a "step" key -- the one signal
    shared by every interactive/gym-style caller (arc_agi3's and
    foodtruck's own inputs dicts both include it, see their evaluator.py's
    own run_episode) and absent from every one-shot caller, research
    included. NOTE: the one-shot path below only actually matches
    research's own inputs shape (inputs["instructions"]) -- confirmed live
    that search_arena/paper_review/imo_* pass entirely different keys
    (messages_a/messages_b, paper_text, problem/solution/...) and already
    crash on that line, independent of anything here. Not fixed by this
    change -- pre-existing, out of scope, flagged for whoever picks up the
    CSV-domain path next."""

    # Per-turn budget for the interactive path: how many step() calls this
    # agent may make while deciding ONE action for the CURRENT environment
    # step (e.g. one round to recall something relevant from its own
    # history, another to commit to the actual action), before being
    # forced to finalize with whatever it last said. Deliberately small
    # and separate from any whole-episode limit -- most turns will take
    # exactly one round in practice, since INTERACTIVE_TOOLS is empty by
    # default and step()'s own termination rule (no tools loaded means the
    # model can never emit a tool call) means every turn naturally
    # finishes after one round unless a future generation gives this agent
    # something to call.
    MAX_STEPS_PER_TURN = 5
    INTERACTIVE_TOOLS = []
    INTERACTIVE_TEMPERATURE = 0.7

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._interactive_state = None
        self._interactive_context = None
        self._interactive_trajectory = None

    def forward(self, inputs):
        if "step" in inputs:
            return self._forward_interactive(inputs)
        return self._forward_one_shot(inputs)

    def _forward_one_shot(self, inputs):
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
        new_msg_history, trajectory = chat_with_agent(
            instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available='all',
            multiple_tool_calls=True,
            max_tool_calls=100,
            plan_act_observe=True,
        )
        self.save_trajectory(trajectory)

        prediction = "done" if os.path.exists("report/report.md") else "incomplete: report/report.md not found"
        return prediction, new_msg_history

    def _forward_interactive(self, inputs):
        """
        For a per-environment-step domain: called once per step, 
        on the SAME agent instance
        for a whole episode -- state/context persist across calls as
        instance attributes (initialized lazily on the first call, not
        reset), so this agent has a real memory of everything that
        happened earlier in the episode, not just the current inputs.
        round_num (inside state) is part of that persisted state too, so
        it counts up across the WHOLE episode rather than restarting at 0
        on every call -- a global counter, not a per-call one.

        Calls agent.llm_withtools.step() directly (not chat_with_agent) so
        this loop -- and only this loop -- owns the "how many rounds does
        it take to decide THIS turn's action" decision. That's genuinely
        task-specific business logic (how much a domain's agent should
        reason before acting), not something chat_with_agent's own shared
        loop should have to know about.

        Args:
            inputs (dict): whatever the calling evaluator's own run_episode
                builds -- always includes "step" (the dispatch signal
                above) and "domain"; the rest is
                domain-specific and just gets rendered into this turn's
                message as-is (see _build_turn_message).

        Returns:
            tuple:
                - response_text (str): the model's own text response for
                  this turn -- an interactive caller parses
                  this into a concrete action.
                - context (list): the full running message history.
        """
        if self._interactive_state is None:
            self._interactive_state = {
                "model": self.model,
                "tools_available": self.INTERACTIVE_TOOLS,
                "temperature": self.INTERACTIVE_TEMPERATURE,
                "max_tool_calls": -1,  # unlimited -- MAX_STEPS_PER_TURN below is what actually bounds one turn
                "multiple_tool_calls": False,
                "num_tool_calls": 0,
                "round_num": 0,
            }
            self._interactive_context = []
            self._interactive_trajectory = []

        turn_message = self._build_turn_message(inputs, is_first_turn=not self._interactive_trajectory)
        self._interactive_context = self._interactive_context + [{"role": "user", "text": turn_message}]

        response_text = None
        for _ in range(self.MAX_STEPS_PER_TURN):
            self._interactive_state, self._interactive_context = step(
                self._interactive_state, self._interactive_context, logging=self.log,
            )
            round_record = self._interactive_state["last_round"]
            self._interactive_trajectory.append(round_record)
            response_text = round_record["plan"]
            if self._interactive_state.get("done"):
                break
        else:
            self.log(
                f"Hit MAX_STEPS_PER_TURN ({self.MAX_STEPS_PER_TURN}) without the model "
                "finalizing on its own -- using its last response as this turn's action."
            )

        self.save_trajectory(self._interactive_trajectory)
        return response_text, self._interactive_context

    @staticmethod
    def _build_turn_message(inputs, is_first_turn):
        """Compact, domain-agnostic rendering of one turn's inputs -- rules
        (if the caller provides them) are only included on the first turn,
        not repeated every step, since they'd otherwise accumulate as
        redundant text in a context that already only ever grows."""
        parts = []
        if is_first_turn and inputs.get("rules"):
            parts.append(str(inputs["rules"]))
        for key, value in inputs.items():
            if key in ("domain", "step", "rules") or value is None:
                continue
            parts.append(f"{key}: {value}")
        return "\n\n".join(parts) if parts else str(inputs)
