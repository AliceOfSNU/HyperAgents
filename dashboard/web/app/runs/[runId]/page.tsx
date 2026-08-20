"use client";

import { use } from "react";
import Link from "next/link";
import { PageShell, Card, SectionLabel } from "@/components/PageShell";
import { StatusBadge } from "@/components/StatusBadge";
import { ScoreTag } from "@/components/ScoreTag";
import { EvolutionTree } from "@/components/EvolutionTree";
import { useDriveJson } from "@/lib/useDriveJson";
import type { RunDetail } from "@/lib/types";

export default function RunDetailPage({ params }: { params: Promise<{ runId: string }> }) {
  const { runId } = use(params);
  const { data, loading, error } = useDriveJson<RunDetail>(`run_${runId}.json`);

  return (
    <PageShell crumbs={[{ label: runId }]}>
      {loading && !data && <p style={{ color: "var(--text-muted)" }}>Loading…</p>}
      {error && !loading && (
        <Card>
          <p style={{ color: "var(--status-critical)" }}>{error}</p>
        </Card>
      )}
      {data && (
        <div className="flex flex-col gap-6">
          <div className="flex items-center gap-3 flex-wrap">
            <h1 className="mono text-2xl font-semibold break-all">{data.run_id}</h1>
            <StatusBadge status={data.status} />
          </div>

          {data.pending_pr && (
            <Card style={{ borderColor: "var(--status-warning)" }}>
              <div className="flex items-center justify-between flex-wrap gap-3">
                <div>
                  <p className="font-medium" style={{ color: "var(--status-warning)" }}>
                    PR waiting on your review
                  </p>
                  <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
                    Generation {data.pending_pr.genid} &middot; #{data.pending_pr.number}
                  </p>
                </div>
                <a
                  href={data.pending_pr.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="rounded-lg px-4 py-2 text-sm font-medium"
                  style={{ background: "var(--status-warning)", color: "#1a1400" }}
                >
                  Review on GitHub →
                </a>
              </div>
            </Card>
          )}

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <StatTile label="Generation" value={`${data.current_genid ?? "—"} / ${data.max_generation ?? "—"}`} />
            <StatTile label="Max score" value={<ScoreTag score={data.max_score} />} />
            <StatTile label="Avg score" value={<ScoreTag score={data.avg_score} />} />
            <StatTile
              label="Incumbent evaluator"
              value={<span className="mono">gen {data.epoch.incumbent_genid}</span>}
            />
          </div>

          <Card>
            <SectionLabel>Evolution tree</SectionLabel>
            <EvolutionTree runId={data.run_id} tree={data.tree} agents={data.agents} epoch={data.epoch} />
          </Card>

          <Card>
            <SectionLabel>Epoch &amp; evaluator promotion</SectionLabel>
            <p className="text-sm mb-3" style={{ color: "var(--text-secondary)" }}>
              Last checkpoint: gen <span className="mono">{data.epoch.last_checkpoint_genid}</span> &middot;
              current incumbent evaluator: gen{" "}
              <span className="mono">{data.epoch.incumbent_genid}</span>
            </p>
            {data.epoch.promotion_history.length === 0 ? (
              <p className="text-sm" style={{ color: "var(--text-muted)" }}>
                No evaluator-promotion checkpoint has completed yet.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left" style={{ color: "var(--text-muted)" }}>
                      <th className="font-normal pb-2 pr-4">Checkpoint</th>
                      <th className="font-normal pb-2 pr-4">Before</th>
                      <th className="font-normal pb-2 pr-4">After</th>
                      <th className="font-normal pb-2 pr-4">Swapped?</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.epoch.promotion_history.map((p) => (
                      <tr key={p.checkpoint_genid} style={{ borderTop: "1px solid var(--border)" }}>
                        <td className="py-2 pr-4 mono whitespace-nowrap">gen {p.checkpoint_genid}</td>
                        <td className="py-2 pr-4 mono whitespace-nowrap">{p.incumbent_before}</td>
                        <td className="py-2 pr-4 mono whitespace-nowrap">{p.incumbent_after}</td>
                        <td className="py-2 pr-4 whitespace-nowrap">
                          {p.promoted ? (
                            <span style={{ color: "var(--status-good)" }} className="font-medium">
                              evaluator swapped
                            </span>
                          ) : (
                            <span style={{ color: "var(--text-muted)" }}>kept incumbent</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          <Card>
            <SectionLabel>Agents</SectionLabel>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left" style={{ color: "var(--text-muted)" }}>
                    <th className="font-normal pb-2 pr-4">Gen</th>
                    <th className="font-normal pb-2 pr-4">Parent</th>
                    <th className="font-normal pb-2 pr-4">Code changed?</th>
                    <th className="font-normal pb-2 pr-4">Score</th>
                    <th className="font-normal pb-2 pr-4">PR</th>
                  </tr>
                </thead>
                <tbody>
                  {data.agents.map((a) => (
                    <tr key={a.genid} style={{ borderTop: "1px solid var(--border)" }}>
                      <td className="py-2 pr-4 whitespace-nowrap">
                        <Link href={`/runs/${data.run_id}/agents/${a.genid}`} className="mono font-medium">
                          gen {a.genid}
                        </Link>
                      </td>
                      <td className="py-2 pr-4 mono whitespace-nowrap" style={{ color: "var(--text-secondary)" }}>
                        {a.parent_genid ?? "—"}
                      </td>
                      <td className="py-2 pr-4 whitespace-nowrap" style={{ color: "var(--text-secondary)" }}>
                        {a.has_diff ? "yes" : "no"}
                      </td>
                      <td className="py-2 pr-4 whitespace-nowrap">
                        <ScoreTag score={a.node_utility} />
                      </td>
                      <td className="py-2 pr-4 whitespace-nowrap">
                        {a.pr ? (
                          <a href={a.pr.url ?? "#"} target="_blank" rel="noopener noreferrer">
                            #{a.pr.number} {a.pr.approved ? "(merged)" : "(open)"}
                          </a>
                        ) : (
                          <span style={{ color: "var(--text-muted)" }}>—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {data.errors.length > 0 && (
            <Card style={{ borderColor: "var(--status-critical)" }}>
              <SectionLabel>Errors in run log</SectionLabel>
              <div className="flex flex-col gap-3">
                {data.errors.map((e, i) => (
                  <div key={i}>
                    <p className="text-xs mono mb-1" style={{ color: "var(--text-muted)" }}>
                      {e.genid !== null ? `gen ${e.genid} — ` : ""}
                      {e.source}
                    </p>
                    <pre
                      className="mono text-xs overflow-x-auto rounded-lg p-3"
                      style={{ background: "var(--surface-2)", color: "var(--status-critical)" }}
                    >
                      {e.snippet}
                    </pre>
                  </div>
                ))}
              </div>
            </Card>
          )}
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
