/** Small shared primitives, so every panel in the app looks like the same product. */
import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";

export function SectionHeading({
  icon: Icon, eyebrow, title, children,
}: { icon?: LucideIcon; eyebrow?: string; title: string; children?: ReactNode }) {
  return (
    <div className="mb-6">
      {eyebrow && (
        <div className="mb-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--color-accent)]">
          {Icon && <Icon size={14} strokeWidth={2.4} />}
          {eyebrow}
        </div>
      )}
      <h2 className="font-[family-name:var(--font-display)] text-2xl leading-tight text-[var(--color-ink)] sm:text-[1.75rem]">
        {title}
      </h2>
      {children && (
        <p className="mt-2 max-w-2xl text-[15px] leading-relaxed text-[var(--color-ink-soft)]">
          {children}
        </p>
      )}
    </div>
  );
}

/** A short explanation of what a number means, shown next to the number itself. */
export function Note({ children }: { children: ReactNode }) {
  return (
    <p className="mt-3 border-l-2 border-[var(--color-accent)] bg-[var(--color-accent-soft)] px-3 py-2 text-[13px] leading-relaxed text-[var(--color-ink-soft)]">
      {children}
    </p>
  );
}

export function Stat({
  label, value, sub, tone = "default",
}: { label: string; value: string; sub?: string; tone?: "default" | "low" | "medium" | "high" }) {
  const toneColor =
    tone === "low" ? "var(--color-low)"
      : tone === "medium" ? "var(--color-medium)"
        : tone === "high" ? "var(--color-high)"
          : "var(--color-ink)";
  return (
    <div className="card p-4">
      <div className="text-[11px] font-semibold uppercase tracking-[0.12em] text-[var(--color-ink-faint)]">
        {label}
      </div>
      <div className="mt-1.5 font-[family-name:var(--font-display)] text-2xl tabular-nums"
        style={{ color: toneColor }}>
        {value}
      </div>
      {sub && <div className="mt-1 text-[12.5px] leading-snug text-[var(--color-ink-soft)]">{sub}</div>}
    </div>
  );
}

export function Table({ head, rows, highlight }: {
  head: string[]; rows: (string | number)[][]; highlight?: number;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] border-collapse text-[13.5px]">
        <thead>
          <tr className="border-b border-[var(--color-line)]">
            {head.map((h, i) => (
              <th key={h} className={`px-3 py-2.5 font-semibold text-[var(--color-ink-faint)] ${i === 0 ? "text-left" : "text-right"}`}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, ri) => (
            <tr key={ri}
              className={`border-b border-[var(--color-line)] last:border-0 ${ri === highlight ? "bg-[var(--color-accent-soft)] font-semibold" : ""}`}>
              {r.map((c, ci) => (
                <td key={ci} className={`px-3 py-2.5 tabular-nums ${ci === 0 ? "text-left font-medium" : "text-right"}`}>
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export { money, moneyExact } from "../lib/currency";
export const pct = (n: number, d = 1) => `${(n * 100).toFixed(d)}%`;
