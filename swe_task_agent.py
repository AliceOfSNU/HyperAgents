import subprocess

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent


SWE_INSTRUCTION_TEMPLATE = """\
You are an autonomous software engineering agent working in a Docker container.
The target repository is checked out at /app, and /app is your current working
directory. Solve the task below by editing the repository with the editor tool,
running commands with the bash tool, and committing your final changes.

Task:
{task}

Recommended workflow:
1. Explore the repo just enough to understand the relevant code paths.
2. Reproduce the bug or locate the missing behavior, including any existing
   tests that currently fail or should pass after your change.
3. Implement the smallest correct change with the editor tool.
4. Run the most relevant test commands (and, when cheap, a broader test slice)
   and iterate until they pass. Inspect failures and `git diff` after edits.
5. When satisfied, commit all of your changes with
   `git add -A && git -c user.name=swe_task_agent -c user.email=swe_task_agent@local commit -m "..."` (any
   uncommitted changes at the end of the run will also be committed for you).

Constraints:
- The only tools available are `bash` and `editor`; there is no internet access.
- Do not alter the repository's git history beyond committing your final change.
- Avoid huge, untargeted changes; keep the diff as small and focused as possible.
- If a command hangs, use short, bounded commands and background long-running
  processes with `&`.
"""

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
        task_instruction = inputs["instructions"]
        initial_head = self._current_head()
        instruction = SWE_INSTRUCTION_TEMPLATE.format(task=task_instruction)
        new_msg_history = []
        trajectory = []
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
        except Exception as e:
            # A provider/transport failure mid-loop should not erase edits
            # that are already on disk: commit them and let the verifier
            # score whatever made it in, rather than crashing out before
            # _ensure_committed() runs.
            self.log(f"chat_with_agent raised: {e}")
        finally:
            self._ensure_committed()
            self.save_trajectory(trajectory)

        # If the first pass stopped quickly without producing a commit, give
        # the agent one focused continuation on the same history. This is
        # deliberately cheap (only when very few tool calls were used) and
        # targets the common failure where the model ends its turn too early,
        # before actually editing anything.
        tool_calls_used = sum(len(round_record["act"]) for round_record in trajectory)
        if self._current_head() == initial_head and new_msg_history and tool_calls_used <= 15:
            self.log("No commit after first pass; giving the agent one focused continuation.")
            try:
                new_msg_history, extra_trajectory = chat_with_agent(
                    "You have not committed any change yet. If the task is not "
                    "yet solved, continue using bash/editor and commit when the "
                    "change is ready. If you already made changes, commit them now.",
                    model=self.model,
                    msg_history=new_msg_history,
                    logging=self.log,
                    tools_available=['bash', 'editor'],
                    multiple_tool_calls=True,
                    max_tool_calls=40,
                    plan_act_observe=True,
                )
                trajectory.extend(extra_trajectory)
            except Exception as e:
                self.log(f"continuation pass raised: {e}")
            finally:
                self._ensure_committed()
                self.save_trajectory(trajectory)

        # "Done" means HEAD moved from where it started -- not just "some
        # commit exists", since the base image's own git history already has
        # commits (the real repo cloned and pinned at base_commit) before the
        # agent does anything at all.
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
