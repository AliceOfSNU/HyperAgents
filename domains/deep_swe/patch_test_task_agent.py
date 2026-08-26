"""Container-side half of skills/patch_testing. Runs inside a fresh deep-swe
task container carrying a SPECIFIC generation's own swe_task_agent.py --
control: that generation's code unmodified; treatment: the meta-agent's
patch applied on top of that SAME original snapshot. Which version is
actually present in this container is decided host-side by
domains/deep_swe/patch_test_orchestrator.py, never by this module -- this
just imports whatever swe_task_agent.py it finds and runs its task_forward,
so control and treatment run through literally identical code here.

`import swe_task_agent` at module level, deliberately -- this module is
imported (by run_patch_test_rollout.py) while the process's cwd is still
CODE_DIR (where swe_task_agent.py was uploaded, see pier_agent.py's own
BASELINE_FILES upload), BEFORE main() changes cwd to the task repo's own
/app. Importing it later, inside forward(), would fail to find it (cwd
would already be /app by then) -- same reason run_swe_task_agent.py's own
`from swe_task_agent import SweTaskAgent` happens at its module top, not
inside main()."""
import subprocess

from agent.base_agent import AgentSystem
from agent.tools import load_tools
from agent.trajectory_replay import reconstruct_msg_history, replay_tool_calls

import swe_task_agent  # noqa: E402 -- see module docstring for why this must stay at module level


class PatchTestRolloutAgent(AgentSystem):
    """Replays a recorded trajectory's prefix for real up to a checkpoint
    round (same replay machinery skills/branching's own branch_task_agent.py
    uses), then continues using THIS container's own swe_task_agent.py
    (task_forward) from the checkpoint's own recorded state -- not a
    hardcoded tool/budget config -- so the continuation genuinely picks up
    what that round actually had available. Runs to full completion (a real
    reward, not a probe -- see skills/patch_testing/SKILL.md for why this
    skill doesn't do the cheap-probe-then-extend staging skills/branching
    does)."""

    def forward(self, inputs):
        instruction = inputs["instructions"]
        source_rounds = inputs["source_rounds"]
        checkpoint_round = inputs["checkpoint_round"]
        temperature = inputs.get("temperature", 0.7)

        checkpoint_state = self._checkpoint_state(source_rounds, checkpoint_round)
        tools_available = checkpoint_state.get("tools_available") or ["bash", "editor"]
        tools_dict = {t["info"]["name"]: t for t in load_tools(logging=self.log, names=tools_available)}

        self._ensure_git_identity()
        initial_head = checkpoint_state.get("initial_head") or self._current_head()

        # Replay everything BEFORE the checkpoint round for real (rounds
        # 1..checkpoint_round-1) -- round checkpoint_round itself is what
        # THIS run (control or treatment) decides fresh, not what the
        # source trajectory recorded for it.
        prefix_round = checkpoint_round - 1
        replay_tool_calls(source_rounds, prefix_round, tools_dict, logging=self.log)
        context = reconstruct_msg_history(
            source_rounds, prefix_round, initial_instruction=instruction,
            max_tool_calls=checkpoint_state.get("max_tool_calls", 100), plan_act_observe=True,
        )

        state = dict(checkpoint_state)
        state.pop("last_round", None)  # belongs to the ORIGINAL run's own round, not this continuation
        state["temperature"] = temperature  # real sampling diversity across replicates -- the whole point
        state.setdefault("phase", "first_pass")

        trajectory = []
        while state.get("phase") != "done":
            state, context = swe_task_agent.task_forward(state, context, logging=self.log)
            trajectory.append(state["last_round"])
        self.save_trajectory(trajectory)

        self._ensure_committed()
        prediction = "done" if self._current_head() != initial_head else "incomplete: no commit produced"
        return prediction, context

    @staticmethod
    def _checkpoint_state(source_rounds, checkpoint_round):
        for record in source_rounds:
            if record.get("round") == checkpoint_round:
                state = record.get("state")
                if state is None:
                    raise ValueError(
                        f"round {checkpoint_round} has no recorded 'state' -- only trajectories produced "
                        "after the state-machine refactor are checkpoint-testable."
                    )
                return dict(state)
        raise ValueError(f"round {checkpoint_round} not found in source trajectory.")

    def _current_head(self):
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=".", capture_output=True, text=True, timeout=30,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def _ensure_git_identity(self):
        for key, value in (
            ("user.name", "patch_test_rollout_agent"),
            ("user.email", "patch_test_rollout_agent@local"),
            ("commit.gpgsign", "false"),
        ):
            try:
                subprocess.run(["git", "config", key, value], cwd=".", capture_output=True, text=True, timeout=30)
            except Exception as e:
                self.log(f"git config {key} failed: {e}")

    def _ensure_committed(self):
        """Same safety net as SweTaskAgent's own -- see its docstring."""
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
                    ["git", "-c", "user.name=patch_test_rollout_agent", "-c", "user.email=patch_test_rollout_agent@local",
                     "commit", "-q", "-m", "Auto-commit: uncommitted changes at end of patch-test rollout"],
                    cwd=".", capture_output=True, text=True, timeout=60,
                )
                if commit.returncode != 0:
                    self.log(f"Auto-commit failed: {commit.stderr}")
        except Exception as e:
            self.log(f"_ensure_committed failed: {e}")
