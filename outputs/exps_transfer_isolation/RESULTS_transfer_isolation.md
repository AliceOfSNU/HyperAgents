# Transfer-isolation ablation: transferring task-agent only vs meta-agent only

Reviewer 3fLX asked for (a) numerical snippets from the "transfer only the task agent" and
"transfer only the meta agent" experiments, and (b) a description of how task-agent code,
meta-agent code, and persistent memory are separated/reset during the intervention.

## Setup

Target domain: **imo_grading** (Olympiad-level math grading, withheld from source self-improvement).
Each run is an **imp@50** measurement: `--run_baseline no_selfimprove` (the meta agent is held
**fixed** and runs from a pristine `/DONOTTOUCH` copy each iteration, generating task-agent
modifications over 50 iterations). Paper-default staged eval (screen 10 → full 100 samples).
Score = repo's `get_saved_score` (`overall_accuracy` on the val split; imp@50 = running-best,
initial ≈ 0). 5 seeds per condition; seed N transfers source-run N's selected agent
(the same lineage used by `hyp_wo_selfimp_transfer_N`).

## Results (imp@50, 5 seeds)

Score = `get_saved_score` (`overall_accuracy`). imp@50 = running-best − initial (initial ≈ 0).
95% CI = percentile bootstrap (10k resamples) over the 5 seeds. The loop selects parents on the
**val** split (primary metric); **train** is the in-loop screening set. Reporting both.

**VALIDATION set** (primary):

| Condition | per-seed imp@50 | mean | median | 95% CI |
|---|---|---|---|---|
| NEITHER — initial task + initial meta       | 0.00, 0.01, 0.00, 0.00, 0.13 | 0.028 | 0.00 | [0.000, 0.080] |
| TASK-ONLY — transferred task + initial meta | 0.51, 0.16, 0.01, 0.00, 0.24 | **0.184** | **0.16** | [0.036, 0.356] |
| META-ONLY — initial task + transferred meta | 0.57, 0.59, 0.55, 0.53, 0.61 | **0.570** | **0.57** | [0.546, 0.594] |
| BOTH — transferred task + transferred meta  | 0.60, 0.60, 0.57, 0.60, 0.60 | 0.594 | 0.60 | [0.582, 0.600] |

**TRAIN set** (in-loop screening):

| Condition | per-seed imp@50 | mean | median | 95% CI |
|---|---|---|---|---|
| NEITHER   | 0.00, 0.01, 0.02, 0.00, 0.19 | 0.044 | 0.01 | [0.002, 0.118] |
| TASK-ONLY | 0.59, 0.23, 0.01, 0.00, 0.24 | **0.214** | **0.23** | [0.050, 0.402] |
| META-ONLY | 0.66, 0.67, 0.64, 0.64, 0.69 | **0.660** | **0.66** | [0.644, 0.676] |
| BOTH      | 0.67, 0.64, 0.66, 0.67, 0.67 | 0.662 | 0.67 | [0.650, 0.670] |

**TEST set** (held-out; best-by-val agent per run scored on the 100-sample test set via
`domains.run_eval`. Select-on-val, report-on-test. Initial-agent test acc = 0.000):

| Condition | per-seed test imp@50 | mean | median | 95% CI |
|---|---|---|---|---|
| NEITHER   | 0.00, 0.01, 0.00, 0.00, 0.13 | 0.028 | 0.00 | [0.000, 0.080] |
| TASK-ONLY | 0.56, 0.10, 0.00, 0.00, 0.19 | **0.170** | **0.10** | [0.020, 0.374] |
| META-ONLY | 0.66, 0.63, 0.57, 0.55, 0.62 | **0.606** | **0.62** | [0.570, 0.640] |
| BOTH      | 0.63, 0.63, 0.60, 0.54, 0.63 | 0.606 | 0.63 | [0.570, 0.630] |

(NEITHER = "DGM-H w/o self-improve", BOTH = "DGM-H w/o self-improve + transfer" test evals taken
from `analysis/plot_testevals.py` group `transfer_hyp`/imo_grading — the paper's recorded test-set
numbers; BOTH median 0.63 matches the reported imp@50 ≈ 0.63. TASK-ONLY/META-ONLY are my new runs,
best-by-val agent scored on test via `domains.run_eval`.)

Test confirms the train/val picture with non-overlapping CIs where it matters:
- META-ONLY (0.606) **exactly matches BOTH (0.606)** on held-out test → transferring only the meta
  agent fully recovers the transfer benefit.
- META-ONLY [0.570, 0.640] ≫ TASK-ONLY [0.020, 0.374]; TASK-ONLY sits between NEITHER (0.028) and
  META-ONLY, closer to NEITHER (median 0.10).

Consistent on both splits: META-ONLY ≈ BOTH (overlapping CIs → indistinguishable), and
META-ONLY ≫ TASK-ONLY (cleanly separated CIs, val [0.546,0.594] vs [0.036,0.356]).
Caveat: with n=5 seeds the bootstrap CI is coarse; TASK-ONLY is bimodal (2 seeds ≈ 0, 3 seeds
0.16–0.51), so median is the more robust summary for it.

**Takeaway:** transferring **only the meta agent** recovers essentially all of the full-transfer
benefit (0.570 vs 0.594), whereas transferring **only the task agent** yields little
(0.184, median 0.16 — close to the 0.028 no-transfer baseline; one seed is an outlier at 0.51).
The transferred **meta agent** — not the task agent — carries the transferable improvement capability.

## How the components are separated / reset

The transferred agent is a chain of diff patches (the selected source agent's lineage). At setup,
`process_meta_patch_files` filters each patch with `filter_patch_by_files`:
- **transfer task only** (`--reset_meta_agent`): strips every hunk touching `meta_agent.py`
  → the fixed meta agent is the **initial** meta agent; `task_agent.py` is the transferred one.
- **transfer meta only** (`--reset_task_agent`): strips every hunk touching `task_agent.py`
  → the fixed meta agent is the **transferred** meta agent; `task_agent.py` is the initial one.

Under `no_selfimprove`, the meta agent used to *generate* each modification is always re-instantiated
from a pristine `/DONOTTOUCH` copy of the (filtered) agent, so it never drifts — matching the
"fixed meta agent" definition of imp@50.

**Honest scope note:** the reset is at the level of the two agent code files (`task_agent.py` /
`meta_agent.py`). Shared scaffolding carried in the same lineage patches — persistent memory
(`IMPROVEMENTS_LOG.md`), summaries (`CHANGES_SUMMARY.md`), and helper utilities
(`utils/response_parser.py`, `agent/tools/analyze_eval.py`) — is **not** stripped by either flag and
is present in both conditions. So the isolation is "agent-code-level," not a total wipe of everything
the source run produced. The near-baseline TASK-ONLY result indicates that retaining this shared
scaffolding (including memory) without the transferred meta agent is not sufficient for improvement.

## Reproduction

Per run (seed N, condition):
```
generate_loop.py --domains imo_grading --run_baseline no_selfimprove \
  {--reset_meta_agent | --reset_task_agent} \
  --meta_patch_files outputs_os/exps_transfer/hyp_wo_selfimp_transfer_N/meta_patch_files/model_patch_*.diff \
  --max_generation 50 --output_dir_parent outputs/exps_transfer_isolation/ \
  --run_id {task_only_N | meta_only_N}
```
Models: task/grading = `openai/o4-mini`, meta = `anthropic/claude-sonnet-4-5` (public-API equivalents
of the internal aliases). Shared initial baseline evaluated at `num_samples 10`.
