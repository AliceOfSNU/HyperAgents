"""Human-in-the-loop review gate for DGM-H self-modifications.

Every self-modification a hyperagent makes to itself is a real git diff
(model_patch.diff, see run_meta_agent.py). This module turns each one into a
GitHub pull request against the user's own fork before it's allowed to become
part of the archive lineage: generate() pushes the diff as a branch, opens a
PR describing the idea behind the change, and blocks until a human merges
(accept) or closes (reject) it.

Branch/commit operations reuse `root_dir`'s `.git` -- the per-run copy of the
hyperagent codebase that generate_loop.py already builds via
setup_initial_gen() -- since that copy's `.git` carries the same remotes as
the main clone it was copied from, so we can push without a second clone.
Each run gets its own branch namespace `dgmh/<run_id>/...` so multiple
runs/forks don't collide.

Important: `root_dir`'s own *working tree* is bind-mounted live into the
current generation's container (see generate_loop.py's build_container) and
is actively dirty with the meta agent's uncommitted edits at the point
generate() calls into this module -- generate()'s own cleanup
(`git reset --hard`) hasn't run yet. Checking out a different branch directly
in root_dir at this moment would race with that live state. So every
operation that touches the working tree (checkout/apply/commit -- anything
beyond read-only plumbing like `branch`/`push`/`remote`) runs inside an
isolated `git worktree` instead: a second, independent working directory
backed by the same object database, safe to check out/dirty/discard without
touching root_dir's own checkout at all.
"""
import json
import os
import re
import subprocess
import time

from agent.llm import DEEPSEEK_FLASH_MODEL, get_response_from_llm
from utils.common import extract_jsons
from utils.gl_utils import filter_patch_by_files


def _run(cmd, cwd=None, check=True, input=None):
    result = subprocess.run(cmd, cwd=cwd, input=input, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}):\nstdout={result.stdout}\nstderr={result.stderr}")
    return result


def _get_pr_worktree(root_dir, run_id):
    """Create (if needed) and return the path to an isolated git worktree used
    for all PR-branch checkout/apply/commit operations, kept separate from
    root_dir's own live, container-mounted working tree. Reused across
    generations within a run; detached HEAD so it's never tied to one branch."""
    worktree_dir = f"{root_dir.rstrip('/')}_pr_worktree_{run_id}"
    if not os.path.isdir(worktree_dir):
        _run(["git", "worktree", "add", "--detach", worktree_dir], cwd=root_dir)
    return worktree_dir


def get_remote_slug(repo_dir, remote="fork"):
    """Return '<owner>/<repo>' for a git remote, by parsing its URL."""
    url = _run(["git", "remote", "get-url", remote], cwd=repo_dir).stdout.strip()
    match = re.search(r"[:/]([^/:]+)/([^/]+?)(\.git)?$", url)
    if not match:
        raise ValueError(f"Could not parse owner/repo from remote URL: {url}")
    return f"{match.group(1)}/{match.group(2)}"


def push_root_branch(root_dir, root_commit, run_id, remote="fork"):
    """Create and push the branch marking the start of this run's self-modification
    lineage (i.e. the initial hyperagent, before any self-modification). Returns
    the branch name; idempotent, safe to call again for a resumed run."""
    branch = f"dgmh/{run_id}/root"
    _run(["git", "branch", "-f", branch, root_commit], cwd=root_dir)
    _run(["git", "push", "-f", remote, branch], cwd=root_dir)
    return branch


def commit_and_push_generation(root_dir, run_id, genid, parent_branch, patch_content, remote="fork"):
    """Branch off parent_branch, apply this generation's own diff (already filtered
    to exclude domains/ changes) as a single commit, and push it.

    Returns the new branch name. Raises RuntimeError if the patch fails to apply
    -- callers should treat that the same as any other invalid/non-compiling
    generation (i.e. don't add it to the archive as a valid parent).
    """
    branch = f"dgmh/{run_id}/gen_{genid}"
    worktree_dir = _get_pr_worktree(root_dir, run_id)

    _run(["git", "fetch", remote, parent_branch], cwd=worktree_dir)
    _run(["git", "checkout", "-f", "-B", branch, f"{remote}/{parent_branch}"], cwd=worktree_dir)
    _run(["git", "clean", "-fdx"], cwd=worktree_dir)

    apply_result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", "-"],
        cwd=worktree_dir, input=patch_content, capture_output=True, text=True,
    )
    if apply_result.returncode != 0:
        raise RuntimeError(f"Patch failed to apply for gen {genid}: {apply_result.stderr}")

    _run(["git", "add", "--all"], cwd=worktree_dir)
    _run(
        [
            "git", "-c", "user.name=dgm-h-agent", "-c", "user.email=dgm-h-agent@users.noreply.github.com",
            "commit", "-m", f"gen_{genid}: self-modification",
        ],
        cwd=worktree_dir,
    )
    _run(["git", "push", "-f", remote, branch], cwd=worktree_dir)
    return branch


def generate_pr_description(patch_content, model=DEEPSEEK_FLASH_MODEL):
    """Ask the model to explain, in PR form, the idea behind a self-modification diff."""
    instruction = f"""A self-improving coding agent just modified its own codebase. Here is the diff:
```
{patch_content}
```

Write a GitHub pull request title and description explaining what changed and, more
importantly, the *idea* behind the change: what problem it's trying to solve, and what
strategy it's betting on. Be concise and specific to this diff, not generic commentary.

Respond in JSON format:
<json>
{{
    "title": "...",
    "body": "..."
}}
</json>"""
    # This is a nice-to-have summary, not a critical decision -- never let a failure
    # here (model/provider issues, rate limits, etc.) cost an otherwise-valid
    # generation its PR. Fall back to a generic title/body with the raw diff.
    try:
        _response, new_msg_history, _info = get_response_from_llm(msg=instruction, model=model, msg_history=[])
        extracted = extract_jsons(new_msg_history[-1]["text"])
        title, body = extracted[-1]["title"], extracted[-1]["body"]
    except Exception as e:
        title = "Self-modification"
        body = f"(Could not auto-generate a description for this diff: {e})\n\n```diff\n{patch_content[:4000]}\n```"
    return title, body


def open_pr(repo_slug, base_branch, head_branch, title, body):
    result = _run([
        "gh", "pr", "create",
        "--repo", repo_slug,
        "--base", base_branch,
        "--head", head_branch,
        "--title", title,
        "--body", body,
    ])
    url = result.stdout.strip().splitlines()[-1]
    pr_number = int(url.rstrip("/").split("/")[-1])
    return pr_number, url


def wait_for_pr_decision(repo_slug, pr_number, poll_interval=30, logging=print):
    """Block until the PR is merged or closed. Returns True if merged (approved)."""
    logging(f"Waiting for human review of PR #{pr_number} ({repo_slug})...")
    while True:
        state = json.loads(
            _run(["gh", "pr", "view", str(pr_number), "--repo", repo_slug, "--json", "state"]).stdout
        )["state"]
        if state == "MERGED":
            logging(f"PR #{pr_number} merged -- applying self-modification.")
            return True
        if state == "CLOSED":
            logging(f"PR #{pr_number} closed without merging -- rejecting self-modification.")
            return False
        time.sleep(poll_interval)


def run_pr_review_gate(root_dir, run_id, genid, parent_branch, raw_patch_content, remote="fork", poll_interval=30, logging=print):
    """High-level orchestration for one generation's review gate.

    Filters raw_patch_content down to the lineage-relevant diff (excluding
    domains/, which never gets applied to the archive either -- see
    apply_diffs_container), pushes it as a branch off parent_branch, opens a
    PR describing it, and blocks until a human merges or closes it.

    Returns (approved, branch_name, pr_number, pr_url). If the diff has no
    lineage-relevant content, short-circuits to (True, parent_branch, None, None)
    since there is nothing to review.
    """
    patch_content = filter_patch_by_files(raw_patch_content, ["domains/"])
    if not patch_content.strip():
        logging("No lineage-relevant changes in this generation's diff; skipping PR.")
        return True, parent_branch, None, None

    branch = commit_and_push_generation(root_dir, run_id, genid, parent_branch, patch_content, remote=remote)
    title, body = generate_pr_description(patch_content)
    repo_slug = get_remote_slug(root_dir, remote=remote)
    pr_number, pr_url = open_pr(repo_slug, base_branch=parent_branch, head_branch=branch, title=title, body=body)
    logging(f"Opened PR #{pr_number} for gen {genid}: {pr_url}")
    approved = wait_for_pr_decision(repo_slug, pr_number, poll_interval=poll_interval, logging=logging)
    return approved, branch, pr_number, pr_url
