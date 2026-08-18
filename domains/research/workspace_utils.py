"""Read RCB task/workspace artifacts needed for scoring: the report text, the
task's checklist, and the INSTRUCTIONS.md background given to the task agent.
"""

import json
from pathlib import Path

from .config import RCB_TASKS_DIR


def read_report(workspace):
    """Read report/report.md from a completed RCB run workspace."""
    workspace = Path(workspace)
    report_path = workspace / "report" / "report.md"
    if report_path.exists():
        return report_path.read_text(encoding="utf-8", errors="replace")
    report_dir = workspace / "report"
    if report_dir.exists():
        for md in report_dir.glob("*.md"):
            return md.read_text(encoding="utf-8", errors="replace")
    return None


def read_instructions(workspace):
    instructions_path = Path(workspace) / "INSTRUCTIONS.md"
    if instructions_path.exists():
        return instructions_path.read_text(encoding="utf-8", errors="replace")
    return ""


def read_checklist(task_id):
    checklist_path = RCB_TASKS_DIR / task_id / "target_study" / "checklist.json"
    with open(checklist_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_meta(workspace):
    meta_path = Path(workspace) / "_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)
