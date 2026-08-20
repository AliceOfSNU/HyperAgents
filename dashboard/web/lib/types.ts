// Mirrors the JSON shapes written by dashboard/scripts/export.py exactly.
// Any field that's missing in older/pre-fix runs is optional and should be
// rendered as "not available for this run", not treated as an error.

export type RunStatus = "completed" | "running" | "stalled_or_crashed";

export interface RunIndexEntry {
  run_id: string;
  status: RunStatus;
  current_genid: number | null;
  max_generation: number | null;
  max_score: number | null;
  avg_score: number | null;
  pending_pr: PendingPr | null;
}

export interface RunsIndex {
  generated_at: string;
  runs: RunIndexEntry[];
}

export interface PendingPr {
  genid: number;
  number: number;
  url: string;
}

export interface PrInfo {
  number: number | null;
  url: string | null;
  approved: boolean | null;
}

export interface TreeNode {
  genid: number | "initial";
  parent_genid: number | "initial" | null;
  score: number | null;
}

export interface AgentSummary {
  genid: number;
  parent_genid: number | "initial" | null;
  has_diff: boolean;
  node_utility: number | null;
  pr: PrInfo | null;
}

export interface PromotionHistoryEntry {
  checkpoint_genid: number;
  incumbent_before: number | "initial";
  incumbent_after: number | "initial";
  promoted: boolean;
}

export interface EpochInfo {
  incumbent_genid: number | "initial";
  last_checkpoint_genid: number;
  promotion_history: PromotionHistoryEntry[];
}

export interface ErrorSnippet {
  genid: number | null;
  source: string;
  snippet: string;
}

export interface RunDetail {
  run_id: string;
  domain: string;
  status: RunStatus;
  current_genid: number | null;
  max_generation: number | null;
  max_score: number | null;
  avg_score: number | null;
  errors: ErrorSnippet[];
  pending_pr: PendingPr | null;
  epoch: EpochInfo;
  tree: TreeNode[];
  agents: AgentSummary[];
  exported_at: string;
}

export interface ScoreItem {
  index: number;
  content: string;
  weight: number;
  score: number;
  reasoning: string;
}

export interface PerTaskEval {
  task_id: string;
  real_score: number;
  evaluator_score: number;
  combined: number;
  task_agent_ok: boolean;
  real_items?: ScoreItem[];
  evaluator_items?: ScoreItem[];
}

export interface AnchorBreakdownItem {
  index: number;
  content: string;
  weight: number | null;
  predicted: number;
  real: number;
  abs_error: number;
  evaluator_reasoning: string;
}

export interface AnchorBreakdownTask {
  task_id: string;
  items: AnchorBreakdownItem[];
}

export interface EvaluatorAnchorBreakdown {
  checkpoint_genid: number;
  anchor_score: number;
  breakdown: AnchorBreakdownTask[];
}

export interface AgentDetail {
  run_id: string;
  genid: number;
  parent_genid: number | "initial" | null;
  has_diff: boolean;
  pr: PrInfo | null;
  node_utility: number | null;
  real_score_avg: number | null;
  evaluator_score_avg: number | null;
  incumbent_evaluator_genid: number | "initial" | null;
  per_task: PerTaskEval[];
  evaluator_anchor_breakdown: EvaluatorAnchorBreakdown | null;
  log_local_path: string | null;
  log_drive_link: string | null;
}
