"use client";

import { use, useState } from "react";
import { PageShell, Card, SectionLabel } from "@/components/PageShell";
import { ScoreTag } from "@/components/ScoreTag";
import { useDriveJson } from "@/lib/useDriveJson";
import type { AgentDetail, PerTaskEval, ScoreItem } from "@/lib/types";

export default function AgentDetailPage({
  params,
}: {
  params: Promise<{ runId: string; genid: string }>;
}) {
  const { runId, genid } = use(params);
  const { data, loading, error } = useDriveJson<AgentDetail>(`agent_${runId}_${genid}.json`);

  return (
    <PageShell crumbs={[{ label: runId, href: `/runs/${runId}` }, { label: `gen ${genid}` }]}>
      {loading && !data && <p style={{ color: "var(--text-muted)" }}>Loading…</p>}
      {error && !loading && (
        <Card>
          <p style={{ color: "var(--status-critical)" }}>{error}</p>
        </Card>
      )}
      {data && (
        <div className="flex flex-col gap-6">
          <h1 style={{ fontFamily: "var(--font-display)" }} className="text-3xl font-semibold">
            Generation <span className="mono">{data.genid}</span>
          </h1>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <StatTile label="Node utility" value={<ScoreTag score={data.node_utility} />} />
            <StatTile label="Real score avg" value={<ScoreTag score={data.real_score_avg} />} />
            <StatTile label="Evaluator score avg" value={<ScoreTag score={data.evaluator_score_avg} />} />
            <StatTile label="Parent" value={<span className="mono">{data.parent_genid ?? "—"}</span>} />
          </div>

          <Card>
            <SectionLabel>Basic info</SectionLabel>
            <dl className="grid grid-cols-2 gap-y-2 text-sm">
              <dt style={{ color: "var(--text-muted)" }}>Code changed this generation</dt>
              <dd>{data.has_diff ? "Yes" : "No (identical to parent)"}</dd>
              <dt style={{ color: "var(--text-muted)" }}>Evaluator used to score task agent</dt>
              <dd className="mono">gen {data.incumbent_evaluator_genid ?? "—"}</dd>
              <dt style={{ color: "var(--text-muted)" }}>Pull request</dt>
              <dd>
                {data.pr ? (
                  <a href={data.pr.url ?? "#"} target="_blank" rel="noopener noreferrer">
                    #{data.pr.number} on GitHub {data.pr.approved ? "(merged)" : "(open)"} →
                  </a>
                ) : (
                  <span style={{ color: "var(--text-muted)" }}>No PR (no code changes to review)</span>
                )}
              </dd>
              <dt style={{ color: "var(--text-muted)" }}>Output log</dt>
              <dd>
                {data.log_drive_link ? (
                  <a href={data.log_drive_link} target="_blank" rel="noopener noreferrer">
                    View raw log →
                  </a>
                ) : (
                  <span style={{ color: "var(--text-muted)" }}>
                    Not uploaded yet (logs sync every ~10 min)
                  </span>
                )}
              </dd>
            </dl>
          </Card>

          <Card>
            <SectionLabel>Task-agent evaluation results</SectionLabel>
            {data.per_task.length === 0 ? (
              <p className="text-sm" style={{ color: "var(--text-muted)" }}>No evaluation data.</p>
            ) : (
              <div className="flex flex-col gap-2">
                {data.per_task.map((t) => (
                  <TaskRow key={t.task_id} task={t} />
                ))}
              </div>
            )}
          </Card>

          <Card>
            <SectionLabel>Evaluator vs. anchor breakdown</SectionLabel>
            {!data.evaluator_anchor_breakdown ? (
              <p className="text-sm" style={{ color: "var(--text-muted)" }}>
                This generation&rsquo;s evaluator was never scored against the anchor set (only
                incumbent/challenger evaluators get checked, at every 3rd-generation checkpoint).
              </p>
            ) : (
              <AnchorBreakdown breakdown={data.evaluator_anchor_breakdown} />
            )}
          </Card>
        </div>
      )}
    </PageShell>
  );
}

function StatTile({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <Card>
      <p className="text-xs mb-1" style={{ color: "var(--text-muted)" }}>
        {label}
      </p>
      <p className="text-lg font-medium">{value}</p>
    </Card>
  );
}

function TaskRow({ task }: { task: PerTaskEval }) {
  const [open, setOpen] = useState(false);
  const hasItems = (task.real_items?.length ?? 0) > 0 || (task.evaluator_items?.length ?? 0) > 0;
  return (
    <div className="rounded-lg border" style={{ borderColor: "var(--border)" }}>
      <button
        onClick={() => hasItems && setOpen((o) => !o)}
        className="w-full flex items-center justify-between gap-3 px-3 py-2 text-left text-sm"
        style={{ cursor: hasItems ? "pointer" : "default" }}
      >
        <span className="mono">{task.task_id}</span>
        <span className="flex items-center gap-4">
          {!task.task_agent_ok && (
            <span style={{ color: "var(--status-critical)" }} className="text-xs font-medium">
              task agent failed
            </span>
          )}
          <span className="flex items-center gap-1" style={{ color: "var(--text-muted)" }}>
            real <ScoreTag score={task.real_score} />
          </span>
          <span className="flex items-center gap-1" style={{ color: "var(--text-muted)" }}>
            evaluator <ScoreTag score={task.evaluator_score} />
          </span>
          {hasItems && (
            <span style={{ color: "var(--text-muted)" }}>{open ? "▲" : "▼"}</span>
          )}
        </span>
      </button>
      {open && hasItems && (
        <div className="border-t px-3 py-3" style={{ borderColor: "var(--border)" }}>
          <ItemComparison realItems={task.real_items ?? []} evaluatorItems={task.evaluator_items ?? []} />
        </div>
      )}
    </div>
  );
}

function ItemComparison({
  realItems,
  evaluatorItems,
}: {
  realItems: ScoreItem[];
  evaluatorItems: ScoreItem[];
}) {
  const evalByIndex = new Map(evaluatorItems.map((it) => [it.index, it]));
  return (
    <div className="flex flex-col gap-3">
      {realItems.map((real) => {
        const ev = evalByIndex.get(real.index);
        return (
          <div key={real.index} className="text-xs">
            <p className="mb-1" style={{ color: "var(--text-secondary)" }}>
              {real.content}
            </p>
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded p-2" style={{ background: "var(--surface-2)" }}>
                <p className="mb-1 font-medium">
                  Real score: <ScoreTag score={real.score} />
                </p>
                <p style={{ color: "var(--text-secondary)" }}>{real.reasoning}</p>
              </div>
              <div className="rounded p-2" style={{ background: "var(--surface-2)" }}>
                <p className="mb-1 font-medium">
                  Evaluator score: <ScoreTag score={ev?.score ?? null} />
                </p>
                <p style={{ color: "var(--text-secondary)" }}>
                  {ev?.reasoning ?? "No evaluator comment recorded."}
                </p>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

function AnchorBreakdown({
  breakdown,
}: {
  breakdown: NonNullable<AgentDetail["evaluator_anchor_breakdown"]>;
}) {
  const totalItems = breakdown.breakdown.reduce((acc, t) => acc + t.items.length, 0);
  return (
    <div className="flex flex-col gap-4">
      <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
        Checkpoint at gen <span className="mono">{breakdown.checkpoint_genid}</span> &middot; pooled
        anchor score <ScoreTag score={breakdown.anchor_score} /> &middot;{" "}
        <span className="mono">{breakdown.breakdown.length}</span> tasks &times; rubric items ={" "}
        <span className="mono">{totalItems}</span> comparisons
      </p>
      <div className="flex flex-col gap-2 max-h-[32rem] overflow-y-auto">
        {breakdown.breakdown.map((t) => (
          <div key={t.task_id} className="rounded-lg border p-2" style={{ borderColor: "var(--border)" }}>
            <p className="mono text-xs font-medium mb-2">{t.task_id}</p>
            <div className="overflow-x-auto">
            <table className="w-full text-xs" style={{ minWidth: 420 }}>
              <thead>
                <tr className="text-left" style={{ color: "var(--text-muted)" }}>
                  <th className="font-normal pb-1">Criterion</th>
                  <th className="font-normal pb-1">Predicted</th>
                  <th className="font-normal pb-1">Real (anchor)</th>
                  <th className="font-normal pb-1">|error|</th>
                </tr>
              </thead>
              <tbody>
                {t.items.map((it) => (
                  <tr key={it.index} style={{ borderTop: "1px solid var(--border)" }}>
                    <td className="py-1 pr-2" style={{ color: "var(--text-secondary)" }}>
                      {it.content}
                    </td>
                    <td className="py-1">
                      <ScoreTag score={it.predicted} />
                    </td>
                    <td className="py-1">
                      <ScoreTag score={it.real} />
                    </td>
                    <td className="py-1 mono">{it.abs_error.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
