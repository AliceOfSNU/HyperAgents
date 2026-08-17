import os
import re

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent

# Structured research-methodology guidance prepended to every task. This keeps
# the agent honest about the end-to-end scientific workflow (read -> plan ->
# code -> real results -> report) and, crucially, about *actually writing*
# report/report.md before it declares success.
WORKFLOW_PROMPT = """You are an autonomous AI research scientist working inside a fresh, sandboxed workspace with bash and file-editing tools. Your job is to complete a real scientific research task end-to-end and produce a publication-quality Markdown report at report/report.md (inside the workspace).

## Recommended workflow (adapt as needed)
1. Orient yourself: run `pwd` and `ls -la` to see the workspace layout, then read INSTRUCTIONS.md (given below) and any data/related-work files it points to.
2. Understand what the original paper did: read the related work, identify the research question, method, and metrics.
3. Make a concrete plan: what code will you write, what analyses/metrics will you report, what figures will you produce?
4. Write and run your analysis code under the code/ directory. Save figures to report/images/.
5. Iterate until you have real quantitative results. Every number in the report must come from code you actually executed - never fabricate or hand-wave results. Use tools (e.g. bash) to run scripts and capture their output.
6. Write the final report to report/report.md (run `pwd` first if unsure about the absolute path). Follow a standard scientific structure: Abstract/Summary, Introduction/Background, Methods, Results (with real numbers, tables, and figures embedded via Markdown/image links), and Discussion/Conclusion.
7. Self-verify: before finishing, confirm report/report.md exists and contains substantive, specific content (not placeholders). If it does not, keep working until it does.

## Rules
- Actually execute code and analyses; do not just describe what you would do.
- Read the actual data files rather than guessing their format.
- Include the concrete numbers/figures you produced - the report is graded on substance, not length.
- Always create the report at the exact path report/report.md (i.e. <workspace>/report/report.md).
"""


def _detect_workspace(instruction, cwd):
    """Best-effort detection of the task workspace root.

    ResearchClawBench's INSTRUCTIONS.md embeds an absolute container path such
    as `/hyperagents/workspace` in the text. The harness always chdirs into
    that workspace before invoking us, so the relative path `report/report.md`
    is correct there -- but a path explicitly written into the instructions is
    more salient to the model, so when one is present we resolve it and use it
    as the canonical report location for both the prompt and the existence
    check.
    """
    m = re.search(r"(`|'|\")(/\S*?/workspace)\1", instruction) or re.search(r"(/\S*?/workspace)", instruction)
    if m:
        # First pattern: group 2 is the quoted absolute path. Second pattern:
        # group 1 is the bare absolute path. Use the one that actually matched.
        candidate = m.group(2) if m.lastindex == 2 else m.group(1)
        if candidate.startswith("/"):
            return candidate
    # Fall back to the process cwd (the harness's container workspace).
    return cwd


class TaskAgent(AgentSystem):
    def forward(self, inputs):
        """
        A research agent that independently conducts a ResearchClawBench-style
        scientific research task: explores the provided workspace data and
        related work, writes and runs analysis code, and produces a
        publication-quality report/report.md with real results and figures.

        Args:
            inputs (dict): {"instructions": str} -- the fully-rendered task
                instructions (ResearchClawBench's INSTRUCTIONS.md template,
                already filled in with the workspace path, task description,
                and data manifest). The current working directory is expected
                to already be that workspace (run_task_agent.py sets this up).

        Returns:
            tuple:
                - prediction (str): "done" if report/report.md exists, else a
                  short failure note.
                - new_msg_history (list): full message history of the
                  interaction (including any completion-retry turns).
        """
        instruction = inputs["instructions"]
        workspace = _detect_workspace(instruction, os.getcwd())
        report_path = os.path.join(workspace, "report", "report.md")

        initial_prompt = f"""{WORKFLOW_PROMPT}

The workspace root for this task is:
{workspace}

Use the relative path report/report.md from inside that workspace (i.e.
{report_path}) whenever creating the final report.

## Research Task Instructions
{instruction}
"""

        new_msg_history = chat_with_agent(
            initial_prompt,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available='all',
            multiple_tool_calls=True,
            max_tool_calls=100,
        )

        # The model sometimes stops before actually writing the report. Give it
        # a bounded number of targeted "keep going" turns (with full context
        # retained) so a near-complete run still lands. Each retry re-uses the
        # accumulated msg_history - the model can see exactly what it did and
        # what still needs to happen.
        max_retries = 2
        for attempt in range(max_retries):
            if os.path.exists(report_path):
                break
            follow_up = (
                f"Your work is not complete yet: {os.path.abspath(report_path)} does not exist. "
                "Please finish the task now: run any remaining analyses and write the final "
                f"Markdown report to {os.path.abspath(report_path)} (create the directory if "
                "needed). Do not stop until that file exists and contains real, substantive results."
            )
            self.log(f"Report not found; completion retry {attempt + 1}/{max_retries}.")
            new_msg_history = chat_with_agent(
                follow_up,
                model=self.model,
                msg_history=new_msg_history,
                logging=self.log,
                tools_available='all',
                multiple_tool_calls=True,
                max_tool_calls=100,
            )

        prediction = "done" if os.path.exists(report_path) else "incomplete: report/report.md not found"
        return prediction, new_msg_history
