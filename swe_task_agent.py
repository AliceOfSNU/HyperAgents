import os
import subprocess

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent


SWE_TASK_SUFFIX = """
[You are the coding agent that must actually fix this repository at /app.
Work in a tight, purposeful loop:
1. Read the instruction, then inspect the repo with targeted commands
   (`git status`, `ls`, `find`, `grep -rn`, `sed -n`) rather than printing
   whole large files.
2. Find the relevant tests and run them once to see the current, failing
   baseline before editing anything.
3. Make the smallest correct change with the editor tool. Prefer
   `str_replace` over rewriting whole files.
4. Run the relevant tests after each change. If a command may be slow, run
   it in the background and tail its log.
5. Periodically commit working intermediate states; git user.name and
   user.email are already configured for this repo, so plain `git commit`
   works.
6. Before declaring done, run the narrowest test command that covers the
   changed behavior and make sure it passes. Then check `git status --short`
   and commit any remaining work.

Do not commit `_hyperagents_meta.json` or other task-runner artifacts.
Do not waste calls on repeated full-file `cat`s; use line ranges and grep.
Your goal is a committed, test-passing change on HEAD.]
"""

SWE_REVIEW_PROMPT = """
[Phase 2: verification and completion.
Re-read the original issue now. If your fix is not complete yet, finish it.
Then run the most relevant tests. Think about edge cases or acceptance
criteria the issue implies that you may not have handled yet; add focused
tests and code for them. Fix any failures, rerun the tests, and commit.
If everything already passes and is committed, do not change code; just
reply DONE.]
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
        instruction = inputs["instructions"] + SWE_TASK_SUFFIX
        self._ensure_git_identity()
        initial_head = self._current_head()
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
                max_tool_calls=140,
                plan_act_observe=True,
            )

            # Checkpoint whatever the first pass produced before starting the
            # review pass. If the review pass hits the wall-clock timeout,
            # Pier only grades committed HEAD -- never leave a finished first
            # solution uncommitted and therefore unscored.
            self._ensure_committed()

            # A deliberately separate review pass after the first solution.
            # gen_1 showed two costly patterns: agents that ran out of budget
            # mid-fix, and agents that stopped early with budget left but
            # missed one or two hidden acceptance tests. A fresh user turn
            # with explicit permission to continue and then verify catches
            # both -- it re-focuses the model on the original issue after
            # the first implementation pass.
            try:
                review_history, review_trajectory = chat_with_agent(
                    SWE_REVIEW_PROMPT,
                    model=self.model,
                    msg_history=new_msg_history,
                    logging=self.log,
                    tools_available=['bash', 'editor'],
                    multiple_tool_calls=True,
                    max_tool_calls=60,
                    plan_act_observe=True,
                )
                new_msg_history = review_history
                trajectory.extend(review_trajectory)
            except Exception as e:
                self.log(f"review pass raised: {e}")
        except Exception as e:
            # A mid-run LLM/tool exception shouldn't throw away edits already
            # on disk: still auto-commit below and let the verifier grade
            # whatever HEAD contains, exactly like the normal early-stop path.
            self.log(f"chat_with_agent raised: {e}")
        finally:
            self._ensure_committed()
            self.save_trajectory(trajectory)

        # "Done" means HEAD moved from where it started -- not just "some
        # commit exists", since the base image's own git history already has
        # commits (the real repo cloned and pinned at base_commit) before the
        # agent does anything at all.
        prediction = "done" if self._current_head() != initial_head else "incomplete: no commit produced"
        return prediction, new_msg_history

    def _ensure_git_identity(self):
        """Set local git identity before the interactive phase.

        gen_1's task logs showed the same avoidable failure over and over:
        the model would finish its edits, run `git commit`, and only then hit
        "Author identity unknown", forcing it to spend more tool calls setting
        user.name/user.email and committing again. Setting the repo-local
        identity up front makes plain `git commit` work the first time. The
        change lives only in `.git/config`, never in `git diff base HEAD`, so
        it cannot affect scoring."""
        try:
            subprocess.run(
                ["git", "config", "--global", "--add", "safe.directory", os.getcwd()],
                capture_output=True, text=True, timeout=30,
            )
        except Exception as e:
            self.log(f"git config safe.directory failed: {e}")
        for key, value in (("user.name", "swe_task_agent"), ("user.email", "swe_task_agent@local"), ("commit.gpgsign", "false")):
            try:
                subprocess.run(
                    ["git", "config", key, value], cwd=".",
                    capture_output=True, text=True, timeout=30,
                )
            except Exception as e:
                self.log(f"git config {key} failed: {e}")

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
                # _hyperagents_meta.json may still exist in the repo workspace
                # (leftover from an older runner or a manual run). It is not
                # part of the fix; keep it out of the commit Pier grades.
                if os.path.exists("_hyperagents_meta.json"):
                    subprocess.run(
                        ["git", "reset", "-q", "--", "_hyperagents_meta.json"],
                        cwd=".", timeout=30,
                    )
                commit = subprocess.run(
                    ["git", "-c", "user.name=swe_task_agent", "-c", "user.email=swe_task_agent@local",
                     "commit", "-q", "-m", "Auto-commit: uncommitted changes at end of run"],
                    cwd=".", capture_output=True, text=True, timeout=60,
                )
                if commit.returncode != 0:
                    self.log(f"Auto-commit failed: {commit.stderr}")
        except Exception as e:
            self.log(f"_ensure_committed failed: {e}")
