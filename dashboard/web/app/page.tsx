"use client";

import Link from "next/link";
import { PageShell, Card } from "@/components/PageShell";
import { StatusBadge } from "@/components/StatusBadge";
import { ScoreTag } from "@/components/ScoreTag";
import { useDriveJson } from "@/lib/useDriveJson";
import type { RunsIndex } from "@/lib/types";

export default function HomePage() {
  const { data, loading, error } = useDriveJson<RunsIndex>("runs_index.json");

  return (
    <PageShell>
      <div className="flex items-baseline justify-between mb-6">
        <h1 style={{ fontFamily: "var(--font-display)" }} className="text-3xl font-semibold">
          Runs
        </h1>
        {data && (
          <span className="text-xs" style={{ color: "var(--text-muted)" }}>
            updated {new Date(data.generated_at).toLocaleString()}
          </span>
        )}
      </div>

      {loading && !data && <p style={{ color: "var(--text-muted)" }}>Loading…</p>}

      {error && !loading && (
        <Card>
          <p style={{ color: "var(--status-critical)" }} className="font-medium mb-1">
            {error.includes("not set on the server") ? "Dashboard not configured yet" : "Couldn't load runs"}
          </p>
          <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
            {error.includes("not set on the server")
              ? "Set DRIVE_FOLDER_ID, GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET and GOOGLE_OAUTH_REFRESH_TOKEN (no NEXT_PUBLIC_ prefix -- these stay server-side) in Vercel project settings, then redeploy."
              : "Make sure dashboard/scripts/run_loop.py is running and has uploaded at least once."}
          </p>
        </Card>
      )}

      {data && data.runs.length === 0 && (
        <Card>
          <p style={{ color: "var(--text-secondary)" }}>
            No runs exported yet. They&rsquo;ll appear here once <code className="mono">run_loop.py</code> finds an{" "}
            <code className="mono">outputs/generate_*/</code> directory.
          </p>
        </Card>
      )}

      {data && data.runs.length > 0 && (
        <div className="flex flex-col gap-3">
          {data.runs.map((run) => (
            <Link key={run.run_id} href={`/runs/${run.run_id}`} className="no-underline">
              <Card className="transition-colors hover:!border-[var(--brand)] cursor-pointer">
                <div className="flex items-center justify-between gap-4 flex-wrap">
                  <div className="flex items-center gap-3">
                    <span className="mono text-sm font-medium">{run.run_id}</span>
                    <StatusBadge status={run.status} />
                    {run.pending_pr && (
                      <span
                        className="text-xs font-medium rounded-full px-2 py-0.5"
                        style={{ color: "var(--status-warning)", background: "var(--status-warning-soft)" }}
                      >
                        PR waiting on you
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-5 text-sm" style={{ color: "var(--text-secondary)" }}>
                    <span>
                      gen{" "}
                      <span className="mono">
                        {run.current_genid ?? "—"}/{run.max_generation ?? "—"}
                      </span>
                    </span>
                    <span className="flex items-center gap-1.5">
                      max <ScoreTag score={run.max_score} label="max score in archive" />
                    </span>
                    <span className="flex items-center gap-1.5">
                      avg <ScoreTag score={run.avg_score} label="avg score in archive" />
                    </span>
                  </div>
                </div>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </PageShell>
  );
}
