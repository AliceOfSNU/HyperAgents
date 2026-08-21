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
        instruction = self._build_instruction(inputs["instructions"])
        initial_head = self._current_head()
        try:
            new_msg_history, trajectory = chat_with_agent(
                instruction,
                model=self.model,
                msg_history=[],
                logging=self.log,
                tools_available=['bash', 'editor'],
                multiple_tool_calls=True,
                max_tool_calls=100,
                plan_act_observe=True,
            )
            self.save_trajectory(trajectory)
        except Exception as e:
            # A mid-loop API/tool failure must not throw away edits that were
            # already made on disk before the failure.
            self.log(f"chat_with_agent failed: {e}")
            new_msg_history = []
        finally:
            self._ensure_committed()
        # "Done" means HEAD moved from where it started -- not just "some
        # commit exists", since the base image's own git history already has
        # commits (the real repo cloned and pinned at base_commit) before the
        # agent does anything at all.
        prediction = "done" if self._current_head() != initial_head else "incomplete: no commit produced"
        return prediction, new_msg_history

    @staticmethod
    def _build_instruction(task):
        return (
            "You are an expert software engineer completing a coding task inside the "
            "repository at the current working directory.\n"
            "Work step by step:\n"
            "1. Explore the repository and identify the relevant source files and tests.\n"
            "2. Reproduce the current behavior or failure if that helps.\n"
            "3. Make a minimal, correct change using the editor tool.\n"
            "4. Run the relevant tests with bash and iterate until they pass.\n"
            "5. Review your change with `git diff` and `git status`.\n"
            "6. Commit your final change with a clear commit message. The verifier only "
            "sees committed changes, so never finish with uncommitted work.\n\n"
            f"--- TASK ---\n{task}"
        )

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
