import type { RunStatus } from "@/lib/types";

const STYLES: Record<RunStatus, { label: string; fg: string; bg: string; dot?: boolean }> = {
  completed: { label: "Completed", fg: "var(--status-good)", bg: "var(--status-good-soft)" },
  running: { label: "Running", fg: "var(--running)", bg: "var(--running-soft)", dot: true },
  stalled_or_crashed: {
    label: "Stalled / crashed",
    fg: "var(--status-critical)",
    bg: "var(--status-critical-soft)",
  },
};

export function StatusBadge({ status }: { status: RunStatus }) {
  const s = STYLES[status];
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium"
      style={{ color: s.fg, background: s.bg }}
    >
      {s.dot && (
        <span
          className="h-1.5 w-1.5 rounded-full animate-pulse"
          style={{ background: s.fg }}
          aria-hidden
        />
      )}
      {s.label}
    </span>
  );
}
