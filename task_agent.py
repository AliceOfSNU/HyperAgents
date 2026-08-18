import os

from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent


# Concrete research workflow appended to every task instruction. The plain
# "go do the task" prompt produced too many runs where the model explored
# forever and never wrote report/report.md (or wrote it empty), which the
# real Claude judge and the co-evolving evaluator both correctly zeroed.
# This turns the instruction into an explicit, verifiable protocol that
# forces a non-empty publication-quality report to exist before the agent
# returns.
RESEARCH_WORKFLOW = """\
Follow this concrete research workflow and do not return until `report/report.md` exists and is non-empty.

## Phase 0 -- Orient (2-5 tool calls)
1. Run `pwd` and `ls -la`, then read `INSTRUCTIONS.md` fully.
2. Inspect every file in `data/` (e.g. `find data -type f`, then `head`/`cat` each CSV/JSON/npy/text file, and python one-liners that print shapes, ranges, and columns).
3. Inspect `related_work/` (abstracts / papers). Identify the key equations, methods, and quantitative claims you need to reproduce or extend.

## Phase 1 -- Plan (1 tool call)
Think step by step, then state a concrete plan: which analyses, which scripts, which outputs, and which task objectives each step addresses. Do not write code before stating the plan.

## Phase 2 -- Implement & Analyze (repeat until done)
- Write analysis code under `code/` (e.g. `code/analyze.py`) using `cat > code/... <<'EOF'`.
- Run it with `python code/...` and capture the printed results (persist key numbers to `outputs/results.json` or print them).
- If a figure is required, generate it with matplotlib and save it under `report/images/`.
- Verify outputs exist with `ls -la outputs report/images`.

## Phase 3 -- Write the report (REQUIRED, never skip)
Create `report/report.md` with `cat > report/report.md <<'EOF' ... EOF`. The report MUST contain ALL of these sections:
1. **Title and Abstract** -- one paragraph summarising the task, the method, and the headline result.
2. **Introduction** -- background from the related work you read and why the task matters.
3. **Methods** -- exact equations (LaTeX), algorithms, data preprocessing, hyperparameters, and how every result was computed. Reproduce the paper's notation faithfully.
4. **Results** -- every quantitative result you obtained, with numbers in the same units/scales as the paper, plus tables and figures (reference `images/...`). Explicitly compare to the paper (e.g. "we obtain X vs. the paper's Y").
5. **Discussion** -- interpret the results, explain the mechanism, acknowledge limitations.
6. **Figures** -- if you generated any, embed them with `![caption](images/foo.png)` and confirm the file exists.
7. **Key technical details** -- call out every specific equation/claim/trick that a strict rubric would check (dataset sizes, layer counts, specific monomer names, exact confidence levels, etc.). The scoring rubric rewards matching the paper's specific numbers and terminology.

## Phase 4 -- Verify (REQUIRED, do not skip)
- Run `ls -la report/report.md` and `wc -l report/report.md`.
- Run `sed -n '1,120p' report/report.md` (or `cat report/report.md`) to confirm it is non-empty and complete.
- If the report is missing or empty, immediately write it again with full content. Do not finish until `report/report.md` exists and is non-empty.
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
                - prediction (str): "done" if report/report.md exists, else a short failure note.
                - new_msg_history (list): full message history of the interaction.
        """
        instruction = inputs["instructions"]
        prompt = instruction + "\n\n" + RESEARCH_WORKFLOW
        new_msg_history = chat_with_agent(
            prompt,
            model=self.model,
            msg_history=[],
            logging=self.log,
            tools_available='all',
            multiple_tool_calls=True,
            max_tool_calls=150,
        )

        prediction = "done" if os.path.exists("report/report.md") else "incomplete: report/report.md not found"
        return prediction, new_msg_history
