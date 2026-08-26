import subprocess
import time

from agent.base_agent import AgentSystem
from agent.llm_withtools import build_kickoff_message, step


def task_forward(state, context, logging=print):
    """One round of SweTaskAgent's own task loop: agent.llm_withtools.step()
    for the actual LLM-turn mechanics, plus phase-transition bookkeeping on
    top. Currently a single "first_pass" phase -- across many real
    generations the meta-agent has independently re-discovered and added
    review/adversarial follow-up passes on top of this baseline; this
    genesis version deliberately doesn't re-import that evolved behavior,
    since the point of self-improvement is for the meta-agent to keep
    finding it on its own round-granular, checkpoint-testable footing, not
    for a human to hand the same behavior back in as a fixed baseline.

    (state, context) together are a complete, replayable checkpoint (see
    step()'s own docstring) -- this function reads/writes "phase" and
    "elapsed_seconds" on top of what step() itself manages, and passes
    everything else through untouched. "elapsed_seconds" is an ACCUMULATED
    value (this call's own measured wall-clock cost added to whatever was
    already there), never a diff against an unreplayable start_time -- a
    checkpoint replayed later has to pick up the clock where THIS run's own
    history left it, not reset to zero, or elapsed-time gating (a future
    phase might add one, the way ADVERSARIAL_MAX_ELAPSED_SECONDS existed in
    evolved generations) would silently misjudge a replay as having taken
    no time at all so far.
    """
    started = time.monotonic()
    new_state, new_context = step(state, context, logging=logging)
    new_state["elapsed_seconds"] = state.get("elapsed_seconds", 0.0) + (time.monotonic() - started)

    if new_state.get("done"):
        new_state["phase"] = "done"

    return new_state, new_context


class SweTaskAgent(AgentSystem):
    """A coding agent for the deep-swe domain (see domains/deep_swe/): given a
    feature/bugfix instruction and a real open-source repo checked out at
    /app, uses bash + the editor tool to make the change and commit it. The
    task is scored by actually running the repo's test suite against
    whatever ends up committed on HEAD (see domains/deep_swe/config.py's
    docstring), not by an LLM judge -- there's no report to write, just a
    working commit.

    Built around task_forward(state, context), a round-granular step
    function (see its own docstring and agent.llm_withtools.step()) rather
    than one opaque multi-round chat_with_agent() call -- this is what
    makes a checkpoint from any round of a real trajectory replayable: feed
    the recorded (state, context) from that round into any version of
    task_forward (including a patched one) and see what it actually does,
    instead of only ever being able to read a finished trajectory and guess.

    Deliberately NOT given the fetch_url tool: the benchmark's own task.toml
    already declares network_mode="no-network" for the agent (its own
    solution.patch files are publicly fetchable from GitHub, so this is the
    real defense, not an afterthought -- see domains/deep_swe/config.py)."""

    MAX_TOOL_CALLS = 100
    TOOLS_AVAILABLE = ["bash", "editor"]

    def forward(self, inputs):
        instruction = inputs["instructions"]
        initial_head = self._current_head()

        state, context = self._initial_state_and_context(instruction, initial_head)
        trajectory = []
        while state.get("phase") != "done":
            state, context = task_forward(state, context, logging=self.log)
            trajectory.append(state["last_round"])
        self.save_trajectory(trajectory)

        self._ensure_committed()
        # "Done" means HEAD moved from where it started -- not just "some
        # commit exists", since the base image's own git history already has
        # commits (the real repo cloned and pinned at base_commit) before the
        # agent does anything at all.
        prediction = "done" if self._current_head() != initial_head else "incomplete: no commit produced"
        return prediction, context

    def _initial_state_and_context(self, instruction, initial_head):
        decorated = build_kickoff_message(
            instruction, self.MAX_TOOL_CALLS, self.TOOLS_AVAILABLE, plan_act_observe=True, logging=self.log,
        )
        context = [{"role": "user", "text": decorated}]
        state = {
            "phase": "first_pass",
            "model": self.model,
            "tools_available": self.TOOLS_AVAILABLE,
            "multiple_tool_calls": True,
            "temperature": 0.0,
            "max_tool_calls": self.MAX_TOOL_CALLS,
            "num_tool_calls": 0,
            "round_num": 0,
            "initial_head": initial_head,
            "elapsed_seconds": 0.0,
        }
        return state, context

    def _current_head(self):
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=".", capture_output=True, text=True, timeout=30,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def _ensure_committed(self):
        """Safety net: commit any uncommitted changes left in the workspace.

        The interactive phase above may end (early stop, exception, tool-call
        budget) with real edits on disk that were never committed -- and
        Pier's [[verifier.collect]] hook only ever sees `git diff base_commit
        HEAD`, so an uncommitted edit scores identically to no edit at all.
        Mirrors the same "don't let a forgotten final step erase real work"
        reasoning as the fallback report-writer in the research domain's task
        agent (PR #17, independently discovered by the meta-agent -- worth
        giving this agent the same safety net from the start rather than
        waiting for it to be rediscovered here too)."""
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain"], cwd=".", capture_output=True, text=True, timeout=30,
            )
            if status.returncode != 0:
                self.log(f"git status failed, cannot verify commit state: {status.stderr}")
                return
            if status.stdout.strip():
                subprocess.run(["git", "add", "-A"], cwd=".", timeout=60)
                commit = subprocess.run(
                    ["git", "-c", "user.name=swe_task_agent", "-c", "user.email=swe_task_agent@local",
                     "commit", "-q", "-m", "Auto-commit: uncommitted changes at end of run"],
                    cwd=".", capture_output=True, text=True, timeout=60,
                )
                if commit.returncode != 0:
                    self.log(f"Auto-commit failed: {commit.stderr}")
        except Exception as e:
            self.log(f"_ensure_committed failed: {e}")
