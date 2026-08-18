import os
import re
from pathlib import Path

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent


# Research-protocol scaffolding prepended to the raw INSTRUCTIONS.md.
# The RCB template already forbids stopping early; this adds an explicit
# execution protocol tuned to the way reports are actually scored (weighted
# text checklist items -- concrete, paper-matching numbers and real figures
# matter far more than prose volume).
PROTOCOL = """\
## Agent execution protocol (hard requirements)

Follow this protocol strictly. Until `report/report.md` is complete and you
have verified it yourself, every response MUST end with at least one real tool
call -- a plan-only message is treated as task completion and will be judged a
failure.

1. INVENTORY: List the workspace: `bash: ls -R`. Note every file in `data/`
   (ground-truth inputs) and every PDF in `related_work/` (the papers whose
   results you must reproduce/match).
2. STUDY THE PAPER: Use `read_pdf` on the relevant papers. Extract the paper's
   specific quantitative claims (headline numbers, tables, figure values).
   Keep a numbered list of the exact metrics you must reproduce.
3. ANALYZE THE DATA: Open every data file with the `view`/`bash` tools. Know
   exactly what columns exist and their units. Cross-check data values against
   paper claims. Data is ground truth: if a number conflicts with the paper,
   trust the data and say so explicitly in the report.
4. CODE: Write your analysis as reusable scripts in `code/` and run them.
   Save every intermediate computed number to `outputs/` (e.g. JSON/CSV).
5. FIGURES: Generate at least 2-4 real figures from your ACTUALLY COMPUTED
   numbers (matplotlib/seaborn) and save them to `report/images/` as PNGs.
   Never reference an image that does not exist on disk.
6. REPORT: Write `report/report.md` with these sections in order: Abstract;
   1. Introduction; 2. Data Overview; 3. Methodology; 4. Results (every claim
   backed by a number you actually computed; reference figures as
   `images/....png`); 5. Discussion & Validation; 6. Conclusion;
   7. Reproducibility (code + outputs used).
7. VERIFY YOURSELF: Before finishing, run `bash: ls -la report report/images`
   and `wc -l report/report.md`. Confirm every referenced image exists. If any
   section is missing, any number is unbacked, or any image is missing, fix it.
8. FINISH: Only when report/report.md is complete, substantive, and has real
   figures, give a short final summary (a text-only message is then allowed).

CRITICAL -- NEVER FABRICATE: every quantitative claim in the report must be
traceable to (a) a value read from the dataset, or (b) a number you computed
with code and saved in outputs/. If you cannot reproduce a paper number,
state the discrepancy explicitly. Honesty about a mismatch is scored far above
invented precision.
"""


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
                - prediction (str): "done" if report/report.md exists and is
                  substantive, else a short failure note.
                - new_msg_history (list): full message history of the interaction.
        """
        instruction = inputs["instructions"]
        full_instruction = PROTOCOL + "\n\n" + instruction

        new_msg_history = chat_with_agent(
            full_instruction,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available='all',
            multiple_tool_calls=True,
            max_tool_calls=120,
        )

        # Belief is not enough: verify the deliverable actually exists and is
        # non-trivial. If the agent stopped without a real report, spend one
        # bounded repair round telling it exactly what is missing.
        prediction = self._verify_report()
        if prediction != "done":
            self.log(f"Report verification failed ({prediction}); attempting one bounded repair round.")
            repair_prompt = (
                "VERIFICATION FAILED -- YOUR DELIVERABLE IS INCOMPLETE\n\n"
                f"{prediction}\n\n"
                "Do NOT summarize, do NOT say you are done. Resume work immediately with "
                "real tool calls and actually create/fix the missing pieces so that "
                "report/report.md is a complete research report with real figures. "
                "If the report exists but is thin, improve it substantially before stopping."
            )
            new_msg_history = chat_with_agent(
                repair_prompt,
                model=self.model,
                msg_history=new_msg_history,
                logging=self.log,
                tools_available='all',
                multiple_tool_calls=True,
                max_tool_calls=60,
            )
            prediction = self._verify_report()

        return prediction, new_msg_history

    def _verify_report(self):
        report = Path("report/report.md")
        if not report.exists():
            return "incomplete: report/report.md not found"
        text = report.read_text(encoding="utf-8", errors="replace")
        if len(text.strip()) < 800:
            return "incomplete: report/report.md is too short to be a substantive research report"

        referenced = set(re.findall(r"images/[A-Za-z0-9_\-./]+\.png", text))
        if referenced:
            missing = [img for img in referenced if not (Path("report") / img).exists()]
            if missing:
                return f"incomplete: report/report.md references missing image(s): {missing}"

        images_dir = Path("report/images")
        if images_dir.exists():
            pngs = list(images_dir.glob("*.png"))
            if not pngs:
                return "incomplete: report/images/ exists but contains no PNG figures"
        else:
            return "incomplete: report/images/ directory missing (figures are mandatory)"

        return "done"
