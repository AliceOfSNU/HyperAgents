"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import type { AgentSummary, EpochInfo, TreeNode } from "@/lib/types";

const ROW_H = 76;
const COL_W = 88;
const NODE_R = 15;
const PAD = 36;

function seqFill(score: number | null): string {
  if (score === null) return "var(--surface-2)";
  if (score < 2) return "var(--seq-1)";
  if (score < 5) return "var(--seq-2)";
  if (score < 8) return "var(--seq-3)";
  if (score < 12) return "var(--seq-4)";
  return "var(--seq-5)";
}

interface LaidOutNode extends TreeNode {
  x: number;
  y: number;
  hasDiff: boolean;
  pendingPrUrl: string | null;
}

function layout(tree: TreeNode[], agents: AgentSummary[]): LaidOutNode[] {
  const byId = new Map(tree.map((n) => [String(n.genid), n]));
  const depth = new Map<string, number>();
  const agentByGenid = new Map(agents.map((a) => [String(a.genid), a]));

  const depthOf = (key: string): number => {
    if (depth.has(key)) return depth.get(key)!;
    const node = byId.get(key);
    if (!node || node.parent_genid === null) {
      depth.set(key, 0);
      return 0;
    }
    const d = 1 + depthOf(String(node.parent_genid));
    depth.set(key, d);
    return d;
  };

  // Order by genid (numeric genids after "initial") so time flows left->right.
  const ordered = [...tree].sort((a, b) => {
    if (a.genid === "initial") return -1;
    if (b.genid === "initial") return 1;
    return (a.genid as number) - (b.genid as number);
  });

  return ordered.map((n, i) => {
    const key = String(n.genid);
    const agent = agentByGenid.get(key);
    return {
      ...n,
      x: PAD + i * COL_W,
      y: PAD + depthOf(key) * ROW_H,
      hasDiff: agent?.has_diff ?? false,
      // Only a truly unresolved PR (approved is still null -- the gate is
      // still polling) counts as "pending review"; a merged or
      // closed-without-merge PR is resolved history, not something to flag.
      pendingPrUrl: agent?.pr && agent.pr.approved === null ? agent.pr.url : null,
    };
  });
}

export function EvolutionTree({
  runId,
  tree,
  agents,
  epoch,
}: {
  runId: string;
  tree: TreeNode[];
  agents: AgentSummary[];
  epoch: EpochInfo;
}) {
  const nodes = useMemo(() => layout(tree, agents), [tree, agents]);
  const [hovered, setHovered] = useState<string | null>(null);

  if (nodes.length === 0) {
    return <p className="text-sm" style={{ color: "var(--text-muted)" }}>No generations yet.</p>;
  }

  const byId = new Map(nodes.map((n) => [String(n.genid), n]));
  const width = Math.max(...nodes.map((n) => n.x)) + PAD + NODE_R;
  const height = Math.max(...nodes.map((n) => n.y)) + PAD + NODE_R;
  const incumbentKey = String(epoch.incumbent_genid);

  return (
    <div className="relative overflow-x-auto">
      <svg width={width} height={height} role="img" aria-label="Generation lineage tree">
        <g>
          {nodes.map((n) => {
            if (n.parent_genid === null) return null;
            const parent = byId.get(String(n.parent_genid));
            if (!parent) return null;
            const midY = (parent.y + n.y) / 2;
            return (
              <path
                key={`edge-${n.genid}`}
                d={`M ${parent.x} ${parent.y} C ${parent.x} ${midY}, ${n.x} ${midY}, ${n.x} ${n.y}`}
                fill="none"
                stroke="var(--border)"
                strokeWidth={2}
              />
            );
          })}
        </g>
        <g>
          {nodes.map((n) => {
            const key = String(n.genid);
            const isIncumbent = key === incumbentKey;
            const label = n.genid === "initial" ? "0" : String(n.genid);
            return (
              <g
                key={key}
                transform={`translate(${n.x},${n.y})`}
                onMouseEnter={() => setHovered(key)}
                onMouseLeave={() => setHovered((h) => (h === key ? null : h))}
                style={{ cursor: n.genid === "initial" ? "default" : "pointer" }}
              >
                {n.genid !== "initial" ? (
                  <Link href={`/runs/${runId}/agents/${n.genid}`} aria-label={`Generation ${n.genid} detail`}>
                    <circle
                      r={NODE_R}
                      fill={seqFill(n.score)}
                      stroke={n.hasDiff ? "var(--brand)" : "var(--border)"}
                      strokeWidth={n.hasDiff ? 3 : 1.5}
                    />
                  </Link>
                ) : (
                  <circle r={NODE_R} fill="var(--surface-2)" stroke="var(--border)" strokeWidth={1.5} />
                )}
                {isIncumbent && (
                  <circle r={NODE_R + 5} fill="none" stroke="var(--running)" strokeWidth={2} strokeDasharray="3 3" />
                )}
                {n.pendingPrUrl && (
                  <circle cx={NODE_R - 3} cy={-(NODE_R - 3)} r={4} fill="var(--status-warning)" stroke="var(--surface-0)" strokeWidth={1} />
                )}
                <text
                  y={NODE_R + 16}
                  textAnchor="middle"
                  className="mono"
                  fontSize={11}
                  fill="var(--text-secondary)"
                >
                  {label}
                </text>
              </g>
            );
          })}
        </g>
      </svg>

      {hovered && byId.get(hovered) && (
        <div
          className="pointer-events-none absolute z-10 rounded-lg border px-3 py-2 text-xs shadow-lg"
          style={{
            left: Math.min(byId.get(hovered)!.x + 20, width - 200),
            top: byId.get(hovered)!.y - 10,
            background: "var(--surface-1)",
            borderColor: "var(--border)",
            color: "var(--text-primary)",
            minWidth: 150,
          }}
        >
          {(() => {
            const n = byId.get(hovered)!;
            return (
              <>
                <div className="font-medium mono">gen {n.genid === "initial" ? "initial" : n.genid}</div>
                <div style={{ color: "var(--text-secondary)" }}>
                  score: <span className="mono">{n.score === null ? "—" : n.score.toFixed(3)}</span>
                </div>
                {n.genid !== "initial" && (
                  <div style={{ color: "var(--text-secondary)" }}>{n.hasDiff ? "code changed" : "no diff"}</div>
                )}
                {n.pendingPrUrl && <div style={{ color: "var(--status-warning)" }}>PR pending review</div>}
                {String(n.genid) === incumbentKey && (
                  <div style={{ color: "var(--running)" }}>incumbent evaluator</div>
                )}
              </>
            );
          })()}
        </div>
      )}

      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1.5 text-xs" style={{ color: "var(--text-muted)" }}>
        <Legend swatch={<span className="inline-block h-3 w-3 rounded-full border-2" style={{ borderColor: "var(--brand)" }} />} label="code changed this generation" />
        <Legend swatch={<span className="inline-block h-3 w-3 rounded-full border" style={{ borderColor: "var(--running)", borderStyle: "dashed" }} />} label="current incumbent evaluator" />
        <Legend swatch={<span className="inline-block h-2 w-2 rounded-full" style={{ background: "var(--status-warning)" }} />} label="PR pending review" />
        <Legend
          swatch={
            <span className="inline-flex gap-0.5">
              {["var(--seq-1)", "var(--seq-2)", "var(--seq-3)", "var(--seq-4)", "var(--seq-5)"].map((c) => (
                <span key={c} className="inline-block h-3 w-3 rounded-sm" style={{ background: c }} />
              ))}
            </span>
          }
          label="node utility, low -> high"
        />
      </div>
    </div>
  );
}

function Legend({ swatch, label }: { swatch: React.ReactNode; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      {swatch}
      {label}
    </span>
  );
}
