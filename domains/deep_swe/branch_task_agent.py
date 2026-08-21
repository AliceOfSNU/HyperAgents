import subprocess

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent
from agent.tools import load_tools
from agent.trajectory_replay import reconstruct_msg_history, replay_tool_calls


class BranchRolloutAgent(AgentSystem):
    """Runs inside a fresh deep-swe task container (same image as the
    original trial -- see domains/deep_swe/branch_pier_agent.py), replays a
    recorded trajectory's prefix against it, then continues with fresh
    sampling from that point. This is the skills/branching skill's
    container-side half; see skills/branching/SKILL.md for the full picture
    (why branch, how the probe/judge/full-rollout phases fit together, the
    cost cap) -- this class only knows how to run ONE branch, once.

    Deliberately doesn't reuse SweTaskAgent's own class or prompt text:
    swe_task_agent.py is the artifact under study, and its content differs
    per trial (it's exactly what generation N's own self-modification
    produced) -- depending on its specific internals here would mean this
    replay machinery silently drifts out of sync with whatever the current
    generation's agent actually does. Reusing the same tool set and
    Plan-Act-Observe conventions (bash+editor, multiple_tool_calls,
    plan_act_observe) captures what actually matters for comparing
    divergent next actions, without coupling to swe_task_agent.py's own
    evolving prompt engineering.
    """

    def forward(self, inputs):
        instruction = inputs["instructions"]
        source_rounds = inputs["source_rounds"]
        branch_round = inputs["branch_round"]
        original_max_tool_calls = inputs.get("original_max_tool_calls", 100)
        continuation_max_tool_calls = inputs.get("continuation_max_tool_calls", 20)
        temperature = inputs.get("temperature", 0.7)

        tools_dict = {t["info"]["name"]: t for t in load_tools(logging=self.log, names=["bash", "editor"])}
        self._ensure_git_identity()
        initial_head = self._current_head()

        replay_tool_calls(source_rounds, branch_round, tools_dict, logging=self.log)
        msg_history = reconstruct_msg_history(
            source_rounds, branch_round, initial_instruction=instruction,
            max_tool_calls=original_max_tool_calls, plan_act_observe=True,
        )

        new_msg_history, trajectory = chat_with_agent(
            None,  # continuing the reconstructed conversation, not a new user turn
            model=self.model,
            msg_history=msg_history,
            logging=self.log,
            tools_available=['bash', 'editor'],
            multiple_tool_calls=True,
            max_tool_calls=continuation_max_tool_calls,
            plan_act_observe=True,
            temperature=temperature,
        )
        self.save_trajectory(trajectory)

        self._ensure_committed()
        prediction = "done" if self._current_head() != initial_head else "incomplete: no commit produced"
        return prediction, new_msg_history

    def _current_head(self):
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=".", capture_output=True, text=True, timeout=30,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None

    def _ensure_git_identity(self):
        for key, value in (("user.name", "branch_rollout_agent"), ("user.email", "branch_rollout_agent@local"), ("commit.gpgsign", "false")):
            try:
                subprocess.run(["git", "config", key, value], cwd=".", capture_output=True, text=True, timeout=30)
            except Exception as e:
                self.log(f"git config {key} failed: {e}")

    def _ensure_committed(self):
        """Same safety net as SweTaskAgent's own -- see its docstring. Kept
        independent (not imported) for the same reason the rest of this
        class doesn't depend on swe_task_agent.py -- see this module's own
        docstring."""
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
                    ["git", "-c", "user.name=branch_rollout_agent", "-c", "user.email=branch_rollout_agent@local",
                     "commit", "-q", "-m", "Auto-commit: uncommitted changes at end of branch rollout"],
                    cwd=".", capture_output=True, text=True, timeout=60,
                )
                if commit.returncode != 0:
                    self.log(f"Auto-commit failed: {commit.stderr}")
        except Exception as e:
            self.log(f"_ensure_committed failed: {e}")
