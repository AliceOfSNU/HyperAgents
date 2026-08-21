---
name: branching
description: Find which specific choice at a specific decision point in a task-agent trajectory actually mattered, by sampling several real continuations from that exact point and comparing outcomes.
---

# Branching

## What this is for

Reading one trajectory tells you what happened once. It doesn't tell you
whether a specific choice at a specific round was the reason -- a different
sample from the same state might have gone the same way regardless, or
might have diverged completely. This skill answers that question directly:
replay a trajectory's prefix into a fresh container (same task, same
environment, functionally the same state -- see "How replay actually
works" below for the honest limits of that), then sample several
independent continuations from that exact point with real temperature.
Where they diverge is the decision that mattered; what happens after tells
you which direction was better.

This is deliberately not something you run just to "see what happens."
It costs real tool-call budget across every sampled branch, and results
are not available in this session (see "This is deferred" below) -- use it
when you've found a round that looks genuinely pivotal (an error just
appeared, a budget decision is imminent, an instruction was ambiguous
enough that reasonable agents could go two ways) and you want real
evidence before turning that observation into a code or prompt change.
Reading a handful of trajectories start-to-finish costs nothing extra;
reach for this once you already suspect where the leverage is and want to
confirm it rather than as a first-pass exploration tool.

## How to invoke it

```
python skills/branching/request_branch.py \
  --source-chat-history <path to a chat_history.json, as YOU see it> \
  --branch-round <N> \
  --k <number of sibling branches, default 3, hard cap 5> \
  --note "<why this round -- what you think is being decided, and what you expect each direction to produce>"
```

`--branch-round` is the last round (1-indexed, matching chat_history.json's
own "round" field) to keep as shared prefix -- branches diverge starting
the round after this one. Pick a round right where you think the
interesting choice happens, not several rounds earlier (padding the prefix
just adds shared, uninteresting rounds that dilute what you're trying to
isolate) and not after it (the choice you wanted to observe will already
be baked into every branch identically).

`--note` is required (the script rejects a request without one) -- write
it like you're leaving a note for the next generation, because you are.
It's the only place your actual intent survives; the branch trajectories
themselves show what happened, not why anyone thought this round was
worth spending budget on. A future generation reading a report with no
context has to re-derive your reasoning from scratch, or worse, guess
wrong about what was actually being tested.

The script validates the round exists and is a real fork point (has at
least one tool call, not the trajectory's own trailing "Done." summary),
then appends a request record. It does not run anything -- see below.

## This is deferred

Your own container has no way to launch new task containers (no docker
access, by design -- see the fixed no-memory-baseline / withmem branches'
own history for why that boundary exists). Requesting a branch just
queues it. The host processes pending requests after your generation's own
meta-agent session ends, using full docker/Pier access -- same trust
boundary as domains/deep_swe/harness.py's own evaluation runs, nothing new
granted to your own sandbox. You will not see results this turn.

Results become readable in `skills/branching/reports/<request_id>/` --
check there at the start of a session (yours or a later generation's) if
you're following up on a request from before. Each report directory
contains:

- `report.json` -- the original request (source trajectory, branch round,
  k, your note) plus, for every branch that ran: its own trajectory
  directory path, whether the probe judge kept it for a full rollout, and
  whatever outcome signal is available (committed or not, self-written
  tests if any, an exception if the branch errored).
- `branch_<i>/chat_history.json` and `chat_history.md` -- each surviving
  branch's own full trajectory, in the exact same format as any other task
  -agent run. Read these exactly like you'd read any trajectory this
  session already showed you how to read: rounds 1 through --branch-round
  are the (replayed, not fresh) shared prefix; everything after is what
  that branch actually sampled and did.

Branches the judge discarded (see below) don't get a directory of their
own -- their short probe continuation is summarized inline in report.json
instead, since it never ran far enough to be worth a full trajectory file.

## What happens host-side (context, not something you control per-request)

1. **Replay.** The recorded prefix's tool calls are re-executed against a
   fresh container from the source task's own image, and the message
   history is reconstructed by splicing the original recorded text back in
   (not re-generating it). This is close to but not exactly the original
   state -- see "How replay actually works" below.
2. **Probe.** k branches each get 1-2 new rounds sampled with real
   temperature, cheaply (small tool-call budget). This is meant to reveal
   whether the branches meaningfully diverge at all, not to reach a real
   outcome.
3. **Judge.** An LLM compares the k probes and identifies which ones
   represent genuinely different strategies, not just superficially
   different phrasing of the same action. Branches judged redundant with
   an already-kept one are dropped here -- cheaply, before they'd cost
   anything more.
4. **Full rollout.** Surviving branches continue to a real budget (agent
   -declared done, or the cap -- see below), so their outcome is
   informative, not just their first move.
5. **Report.** Written as described above.

## How replay actually works, and where it's honest about not being exact

- **Message history**: exact. agent/llm.py's get_response_from_llm only
  ever stores the final visible text + tool_calls, never hidden reasoning
  tokens -- so splicing recorded text back in reproduces exactly what a
  later round in the *original* run would have seen. Nothing is lost by
  replaying instead of continuing live.
- **Environment/filesystem state**: functionally equivalent, not
  byte-identical. The recorded commands are genuinely re-executed against
  a fresh container of the same image, not faked -- but re-execution can
  diverge from the original run in the same ways any two runs of the same
  commands can (timestamps, hash-map/directory iteration order, anything
  environment-dependent). For typical exploration/edit/git commands this
  gap is negligible; for a branch point that depended on something
  timing- or ordering-sensitive, treat that as a real caveat on the
  result, not a bug in this skill.
- **New sampling**: genuinely stochastic (real temperature, and even at
  temperature 0 the underlying model's MoE routing isn't guaranteed
  reproducible across calls). This is the whole point for the probe/full
  -rollout phases -- it's what makes divergence possible at all -- but it
  also means a single branch's outcome is one sample of what that
  direction tends to produce, not a guarantee. Two branches sampled from
  the *same* choice can still land on different outcomes; report.json
  doesn't hide this, so read individual results with that in mind rather
  than over-trusting an n=1 comparison. If a finding matters enough to act
  on, the cheap way to raise confidence is another --k pass at the same
  round rather than trusting one run.

## Cost cap

Enforced host-side regardless of what a request asks for -- k is clamped
to 5 even if you request more, probe rounds are capped at 2, and full
-rollout budget is capped per-branch and in total across all surviving
branches for one request. If the request would have run more branches or
more budget than the cap allows, report.json says so explicitly (which
branches got dropped for budget, not silently omitted). Requesting a
smaller k or a narrower branch point costs less and is more likely to
survive the cap intact -- the cap is a safety backstop, not a budget to
plan around.
