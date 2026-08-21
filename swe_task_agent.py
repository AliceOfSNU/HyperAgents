import subprocess

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent


class SweTaskAgent(AgentSystem):
    """A coding agent for the deep-swe domain (see domains/deep_swe/): given a
    feature/bugfix instruction and a real open-source repo checked out at
    /app, uses bash + the editor tool to make the change and commit it. The
    task is scored by actually running the repo's test suite against
    whatever ends up committed on HEAD (see domains/deep_swe/config.py's
    docstring), not by an LLM judge -- there's no report to write, just a
    working commit.

    Deliberately NOT given the fetch_url tool: the benchmark's own task.toml
    already declares network_mode="no-network" for the agent (its own
    solution.patch files are publicly fetchable from GitHub, so this is the
    real defense, not an afterthought -- see domains/deep_swe/config.py)."""

    def forward(self, inputs):
        instruction = inputs["instructions"]
        # The benchmark instruction names no checkout path, and gen_1 runs
        # repeatedly spent their first tool call(s) discovering it (`cd /repo`
        # -> no such directory).  It is always /app in these containers and
        # the bash session already starts there.
        instruction += (
            "\n\nThe repository to modify is already checked out at /app; "
            "your bash shell starts in /app."
        )
        initial_head = self._current_head()
        self._ensure_git_identity()

        new_msg_history, trajectory = [], []
        try:
            new_msg_history, trajectory = chat_with_agent(
                instruction,
                model=self.model,
                msg_history=[],
                logging=self.log,
                tools_available=['bash', 'editor'],
                multiple_tool_calls=True,
                # Gen_1's scoring runs repeatedly hit the 100-call cap while
                # still mid-implementation (helm, boa) or while chasing the
                # last failing F2P test.  Agent durations at 100 calls were
                # 6-13min against a 27min container budget, so 120 leaves
                # ample wall-clock margin for the extra LLM/tool rounds.
                max_tool_calls=120,
                plan_act_observe=True,
            )
        except Exception as e:
            # A transient LLM/tool error mid-run must not discard every
            # uncommitted edit on disk.  Log it, then let the finally block
            # below preserve the partial work in a commit -- exactly the same
            # safety net as a tool-call-budget stop or an early finish.
            self.log(f"Agent loop failed: {e}")
        finally:
            try:
                self.save_trajectory(trajectory)
            except Exception as e:
                # Preserving the commit is more important than preserving the
                # trajectory JSON; a logging artifact must not block the
                # safety-net commit below.
                self.log(f"save_trajectory failed: {e}")
            self._ensure_committed()

        # "Done" means HEAD moved from where it started -- not just "some
        # commit exists", since the base image's own git history already has
        # commits (the real repo cloned and pinned at base_commit) before the
        # agent does anything at all.
        prediction = "done" if self._current_head() != initial_head else "incomplete: no commit produced"
        return prediction, new_msg_history

    def _ensure_git_identity(self):
        """Set a repo-local committer identity so the model's own `git commit`
        calls work without it first having to discover/configure one.

        `_ensure_committed` already passes -c user.name/-c user.email for its
        own fallback commit, but that does not help the agent's *interactive*
        commits.  Live runs repeatedly burned several tool calls on the
        otherwise-surprising "Author identity unknown" failure before the
        agent fixed it manually (or before the budget ran out)."""
        try:
            subprocess.run(["git", "config", "user.name", "swe_task_agent"], cwd=".", timeout=30)
            subprocess.run(["git", "config", "user.email", "swe_task_agent@local"], cwd=".", timeout=30)
        except Exception as e:
            self.log(f"_ensure_git_identity failed: {e}")

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
