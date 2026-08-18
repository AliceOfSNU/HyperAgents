import Link from "next/link";

export function PageShell({
  children,
  crumbs,
}: {
  children: React.ReactNode;
  crumbs?: { label: string; href?: string }[];
}) {
  return (
    <div className="min-h-screen flex flex-col">
      <header
        className="border-b px-6 py-4 flex items-baseline gap-4"
        style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}
      >
        <Link href="/" className="flex items-baseline gap-2 no-underline">
          <span
            style={{ fontFamily: "var(--font-display)", color: "var(--text-primary)" }}
            className="text-lg font-semibold italic"
          >
            RSI
          </span>
          <span className="text-sm" style={{ color: "var(--text-muted)" }}>
            dashboard
          </span>
        </Link>
        {crumbs && crumbs.length > 0 && (
          <nav className="flex items-center gap-2 text-sm" style={{ color: "var(--text-muted)" }}>
            {crumbs.map((c, i) => (
              <span key={i} className="flex items-center gap-2">
                <span aria-hidden>/</span>
                {c.href ? (
                  <Link href={c.href}>{c.label}</Link>
                ) : (
                  <span style={{ color: "var(--text-primary)" }}>{c.label}</span>
                )}
              </span>
            ))}
          </nav>
        )}
      </header>
      <main className="flex-1 px-6 py-8 max-w-6xl w-full mx-auto">{children}</main>
    </div>
  );
}

export function Card({
  children,
  className = "",
  style,
}: {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
}) {
  return (
    <div
      className={`rounded-xl border p-5 ${className}`}
      style={{ borderColor: "var(--border)", background: "var(--surface-1)", ...style }}
    >
      {children}
    </div>
  );
}

export function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h2
      className="text-xs font-semibold uppercase tracking-wider mb-3"
      style={{ color: "var(--text-muted)" }}
    >
      {children}
    </h2>
  );
}
