import os
import re
from pathlib import Path

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent


WORKFLOW_PROMPT = """You are an autonomous scientific research agent working on a ResearchClawBench-style task.

Follow this disciplined workflow end-to-end. Do NOT skip steps or stop early.

1. **Orient** -- Read INSTRUCTIONS.md carefully. Identify the research question, the data files (with formats/units), what related work is provided, and every quantitative deliverable the task expects.

2. **Explore** -- Inspect the actual data before choosing a method: check shapes, columns, value ranges, missing values, sample rows. Read relevant files in related_work/ for baselines and expected metrics. Prefer the bash editor/view tools for quick inspection.

3. **Plan** -- Write a brief concrete plan (e.g. code/PLAN.md, or your first message) that maps every expected deliverable to a specific analysis step. Make sure each quantitative number you intend to report has a step that actually computes it.

4. **Implement** -- Create clean, runnable scripts under code/ using the editor tool (for long scripts) and run them with the bash tool. Save all intermediate artifacts under outputs/ and figures under report/images/. Install missing packages with pip if needed.

5. **Execute & verify** -- Run your scripts and confirm they complete with no silent failures (check exit codes and printed outputs). Verify every number you plan to report is actually produced by a script -- never invent or approximate results.

6. **Report** -- Write a publication-quality Markdown report at report/report.md in the workspace, containing:
   - A title and a clear restatement of the research question.
   - A Methods section describing your approach concretely (model/algorithm, hyperparameters, data splits, evaluation protocol).
   - A Results section with **concrete quantitative values** (numbers, tables, error bars/uncertainties, comparisons to any baselines named in the task). Vague statements like \"good performance\" without numbers are unacceptable.
   - At least one figure saved under report/images/ (e.g. PNG) and referenced from the report with a relative path.
   - A short Conclusion relating your results back to the original paper/baselines.

7. **Self-review before finishing** -- Re-read report/report.md as a strict peer reviewer. Confirm every requirement in INSTRUCTIONS.md is addressed, all numbers are specific and traceable to your scripts, and figures exist and are referenced. If anything is missing or weak, fix it with more tool calls.

Only finish when report/report.md exists in the workspace and passes your self-review.
"""


def _candidate_workspaces(instruction):
    """Return absolute paths where the agent's workspace/report may live.

    In the real harness run_task_agent.py chdir()s into the workspace before
    calling forward(), so the CWD check usually suffices. We additionally scan
    the rendered INSTRUCTIONS.md for the workspace path (the template embeds
    the absolute workspace path) so the completion check is robust even when
    the process CWD is not the workspace (e.g. consult_task_agent runs).
    """
    candidates = []
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, "report", "report.md")):
        candidates.append(cwd)

    # INSTRUCTIONS_TEMPLATE fills in an absolute workspace path such as
    # "/hyperagents/workspace". Match the directory itself or parent-of-report
    # patterns to avoid false positives from unrelated prose.
    for m in re.finditer(r"(?:`|\s)(/[A-Za-z0-9_./-]*?/workspace)(?:`|\s|\n|$)", instruction):
        p = m.group(1).strip()
        if p not in candidates and os.path.isabs(p):
            candidates.append(p)
    # Also catch the report path spelled out directly.
    for m in re.finditer(r"(?:`|\s)(/[A-Za-z0-9_./-]*?/report/report\.md)(?:`|\s|\n|$)", instruction):
        p = str(Path(m.group(1).strip()).parent.parent)
        if p not in candidates and os.path.isabs(p):
            candidates.append(p)

    candidates.append(cwd)
    return candidates


def _completion_status(instruction):
    for ws in _candidate_workspaces(instruction):
        if Path(ws, "report", "report.md").exists():
            return "done"
    return "incomplete: report/report.md not found"


class TaskAgent(AgentSystem):
    def forward(self, inputs):
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
        new_msg_history = chat_with_agent(
            WORKFLOW_PROMPT + "\n\n" + instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available='all',
            multiple_tool_calls=True,
            max_tool_calls=100,
        )

        prediction = _completion_status(instruction)
        return prediction, new_msg_history
