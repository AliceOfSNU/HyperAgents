// Sequential-hue score chip: same one-hue ramp used by the tree nodes, so a
// score always reads the same visual "temperature" everywhere it appears.
function seqStep(score: number | null): string {
  if (score === null) return "var(--surface-2)";
  if (score < 2) return "var(--seq-1)";
  if (score < 5) return "var(--seq-2)";
  if (score < 8) return "var(--seq-3)";
  if (score < 12) return "var(--seq-4)";
  return "var(--seq-5)";
}

function textOn(step: string): string {
  // seq-1/seq-2 are light fills even in dark mode's low steps; seq-4/5 need
  // light text. Simple heuristic keyed to the same ramp used above.
  return step === "var(--seq-4)" || step === "var(--seq-5)" ? "#fff" : "var(--text-primary)";
}

export function ScoreTag({ score, label }: { score: number | null; label?: string }) {
  const bg = seqStep(score);
  return (
    <span
      className="mono inline-flex items-center rounded px-1.5 py-0.5 text-xs font-medium tabular-nums"
      style={{ background: bg, color: textOn(bg) }}
      title={label}
    >
      {score === null ? "—" : score.toFixed(2)}
    </span>
  );
}
