---
name: patch_testing
description: Test a code patch to swe_task_agent.py against a real historical checkpoint before deciding whether to keep it -- reconstructs that checkpoint's own original code and environment, runs it unmodified (control) and with your patch applied on top of that same original code (treatment), several times each to full completion.
---

# Patch testing

## What this is for

Reading a trajectory tells you what happened once, with whatever code was
active at the time. It doesn't tell you what a DIFFERENT version of that
code -- the patch you just wrote -- would have done at the same point,
starting from the same real state. This skill answers that directly: given
a checkpoint (a specific round of a specific task's trajectory), it
reconstructs that generation's own original code and the real environment
state at that round, then runs it forward twice -- once with the original
code unmodified (control), once with your patch applied on top of that
SAME original snapshot (treatment) -- several times each, to full
completion with a real reward signal.

The control baseline matters more than it looks: it's the checkpoint's own
original code, not whatever the current parent generation's code happens to
look like. If the checkpoint is from an older ancestor, "current unpatched"
code could differ from what actually produced that checkpoint for reasons
that have nothing to do with your patch -- other changes made since then.
Comparing against the checkpoint's own original code isolates your patch as
the only variable.

This is for testing an actual code change you've already written to
`swe_task_agent.py` -- not for exploring "what if" in the abstract. Write
the patch first (the normal editor tool), form a concrete goal (what you
expect to change and why), then test it here before deciding whether to
keep it.

## The workflow

1. **Choose the checkpoint.** Read a specific task's own chat_history.json
   and its result.json/verifier output directly to find a specific round
   worth testing against -- typically wherever you suspect a real decision
   was made that your patch should change. No dedicated tool for this: the
   trajectory and eval results are already fully readable with bash/editor,
   and deciding what's salient in them is exactly the judgment call worth
   making yourself rather than working from a pre-narrowed list.
2. **Write the patch, state a goal.** Edit `swe_task_agent.py` as you
   normally would. Then call `test_patch` with a `goal` describing what you
   expect this patch to change at the checkpoint, and why. This is required,
   not optional -- it's what you compare the actual outcome against once
   the run finishes, and it's what a memory note about the result should
   cite.
3. **The replay and test runs.** One `test_patch` call does the whole
   thing: reconstructs the checkpoint's own original code, replays the
   trajectory prefix for real up to that round, then runs `replicates_per_arm`
   (default 3) replicates of the unmodified original code and
   `replicates_per_arm` replicates of your patch applied on top of it, all
   to full completion. Add `regression_checkpoints` (explicit) and/or
   `n_random_regression` (a count to sample automatically) to also sanity-
   check the patch against points elsewhere in the lineage that weren't
   chosen because of a known problem -- catching a patch that fixes the one
   thing you were looking at while quietly breaking something else.
4. **Review and decide.** `test_patch` returns the full report: every
   replicate's outcome, at every checkpoint, both arms. Compare it against
   the goal you stated in step 2 -- did reality match what you expected, or
   not, and were there surprises at the regression checkpoints? Keep the
   patch, revert it, or iterate. Either way, this is exactly the moment to
   write a memory note recording which it was and why -- see meta_agent.py's
   own memory guidance for the "verified via patch-test, cite the report"
   convention, and note any real surprise even if you keep the patch.

## Why this blocks synchronously

`test_patch` is one tool call that blocks for the whole run -- on the order
of 20 minutes with a handful of checkpoints and replicates. This is
deliberate, not a bug: it means you get the full picture in one call instead
of a queue-now-read-later split across sessions, at the cost of that call
taking a while. Your own process already runs under a multi-hour ceiling,
and `max_tool_calls` counts calls, not wall-clock, so a single long call
doesn't cost you anything except real time -- worth it for getting a
complete, reviewable answer in one shot rather than managing a multi-step
handoff yourself. If a request runs long enough that the internal wait
times out, `test_patch` tells you the request id so you can check back
rather than assuming it failed.

## What it reports, and what it doesn't

- `patch_applies`: per checkpoint, whether your patch actually applied
  cleanly onto that checkpoint's own code (`git apply --check`). A patch
  written against the current parent's code may not apply to a much older
  ancestor's version if it depends on something added since -- that
  checkpoint is reported as skipped, not silently misapplied or treated as
  a failure of the patch itself.
- Per checkpoint, per arm, every replicate's real reward and a short
  outcome summary -- real numbers from real runs, not a probe or an
  estimate.
- What it does NOT do: judge which arm is "better" for you, or decide
  whether the difference between control and treatment is meaningful versus
  noise. Multiple replicates per arm exist so you can see that for
  yourself -- that judgment, and the decision to keep or revert, is still
  yours.
