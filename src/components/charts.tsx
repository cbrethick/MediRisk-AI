/**
 * Hand-rolled SVG charts.
 *
 * No charting library. These are small, the data shapes are fixed, and writing
 * the SVG directly keeps the bundle tiny and lets every chart share one visual
 * language: the same axis weight, the same type scale, the same accent colour.
 */
import type { ReactNode } from "react";

const AXIS = "var(--color-line)";
const INK = "var(--color-ink-faint)";

export function Frame({ title, caption, children, height = 240 }: {
  title: string; caption?: ReactNode; children: ReactNode; height?: number;
}) {
  return (
    <figure className="card p-5">
      <figcaption className="mb-3">
        <h4 className="text-[14px] font-semibold text-[var(--color-ink)]">{title}</h4>
        {caption && (
          <p className="mt-1 text-[12.5px] leading-relaxed text-[var(--color-ink-soft)]">
            {caption}
          </p>
        )}
      </figcaption>
      <div style={{ height }} className="w-full">{children}</div>
    </figure>
  );
}

/* ------------------------------------------------------- precision / recall */

export function CurveChart({ series, xLabel, yLabel, diagonal = false }: {
  series: { name: string; color: string; x: number[]; y: number[]; dash?: boolean }[];
  xLabel: string; yLabel: string; diagonal?: boolean;
}) {
  const W = 420, H = 240, P = { l: 42, r: 12, t: 10, b: 34 };
  const sx = (v: number) => P.l + v * (W - P.l - P.r);
  const sy = (v: number) => H - P.b - v * (H - P.t - P.b);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-full w-full" role="img"
      aria-label={`${yLabel} against ${xLabel}`}>
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={P.l} x2={W - P.r} y1={sy(g)} y2={sy(g)} stroke={AXIS} strokeWidth={1} />
          <text x={P.l - 6} y={sy(g) + 3.5} textAnchor="end" fontSize={9.5} fill={INK}>
            {g.toFixed(2)}
          </text>
          <text x={sx(g)} y={H - P.b + 14} textAnchor="middle" fontSize={9.5} fill={INK}>
            {g.toFixed(2)}
          </text>
        </g>
      ))}
      {diagonal && (
        <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke={INK}
          strokeWidth={1} strokeDasharray="4 4" opacity={0.6} />
      )}
      {series.map((s) => (
        <polyline
          key={s.name} fill="none" stroke={s.color} strokeWidth={2}
          strokeDasharray={s.dash ? "5 4" : undefined}
          strokeLinejoin="round" strokeLinecap="round"
          points={s.x.map((v, i) => `${sx(v)},${sy(s.y[i])}`).join(" ")}
        />
      ))}
      <text x={(W + P.l) / 2} y={H - 3} textAnchor="middle" fontSize={10.5} fill={INK}>
        {xLabel}
      </text>
      <text x={11} y={H / 2} textAnchor="middle" fontSize={10.5} fill={INK}
        transform={`rotate(-90 11 ${H / 2})`}>
        {yLabel}
      </text>
    </svg>
  );
}

export function Legend({ items }: { items: { name: string; color: string; note?: string }[] }) {
  return (
    <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5">
      {items.map((i) => (
        <span key={i.name} className="inline-flex items-center gap-1.5 text-[12px] text-[var(--color-ink-soft)]">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: i.color }} />
          {i.name}{i.note && <span className="text-[var(--color-ink-faint)]">{i.note}</span>}
        </span>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------ confusion grid */

export function ConfusionMatrix({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
  const rowTotals = matrix.map((r) => r.reduce((a, b) => a + b, 0));
  const max = Math.max(...matrix.flat());
  return (
    <div className="overflow-x-auto">
      <table className="border-collapse text-[12.5px]">
        <thead>
          <tr>
            <th className="p-2" />
            <th colSpan={labels.length} className="pb-2 text-[11px] font-semibold uppercase tracking-[0.1em] text-[var(--color-ink-faint)]">
              Model predicted
            </th>
          </tr>
          <tr>
            <th className="p-2" />
            {labels.map((l) => (
              <th key={l} className="px-3 py-1.5 text-[12px] font-semibold text-[var(--color-ink-soft)]">{l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th className="whitespace-nowrap py-1.5 pr-3 text-right text-[12px] font-semibold text-[var(--color-ink-soft)]">
                Actually {labels[i]}
              </th>
              {row.map((v, j) => {
                const on = i === j;
                return (
                  <td key={j} className="p-0.5">
                    <div
                      className="grid h-[62px] w-[86px] place-items-center rounded-lg text-center"
                      style={{
                        background: on
                          ? `color-mix(in srgb, var(--color-accent) ${12 + 58 * (v / max)}%, white)`
                          : `color-mix(in srgb, var(--color-high) ${6 + 40 * (v / max)}%, white)`,
                        color: on && v / max > 0.6 ? "white" : "var(--color-ink)",
                      }}
                    >
                      <span>
                        <span className="block text-[16px] font-semibold tabular-nums">{v}</span>
                        <span className="block text-[10.5px] opacity-75 tabular-nums">
                          {((v / rowTotals[i]) * 100).toFixed(0)}%
                        </span>
                      </span>
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* --------------------------------------------------------------- bar charts */

export function BarChart({ data, format, color = "var(--color-accent)", max: forcedMax }: {
  data: { label: string; value: number; secondary?: number }[];
  format: (n: number) => string;
  color?: string;
  max?: number;
}) {
  const max = forcedMax ?? Math.max(...data.map((d) => d.value));
  return (
    <div className="space-y-1.5">
      {data.map((d) => (
        <div key={d.label} className="flex items-center gap-3">
          <span className="w-[38%] shrink-0 truncate text-right text-[12.5px] font-medium" title={d.label}>
            {d.label}
          </span>
          <span className="h-5 flex-1 overflow-hidden rounded bg-[var(--color-canvas)]">
            <span className="block h-full rounded grow"
              style={{ width: `${(d.value / max) * 100}%`, background: color }} />
          </span>
          <span className="w-16 shrink-0 text-right text-[12px] tabular-nums text-[var(--color-ink-soft)]">
            {format(d.value)}
          </span>
        </div>
      ))}
    </div>
  );
}

/** Paired bars: predicted against actual, for the decile-lift chart. */
export function LiftChart({ data }: {
  data: { decile: number; predictedMean: number; actualMean: number }[];
}) {
  const W = 460, H = 250, P = { l: 48, r: 10, t: 12, b: 36 };
  const max = Math.max(...data.flatMap((d) => [d.predictedMean, d.actualMean])) * 1.08;
  const bw = (W - P.l - P.r) / data.length;
  const sy = (v: number) => H - P.b - (v / max) * (H - P.t - P.b);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-full w-full" role="img"
      aria-label="Predicted and actual mean cost by predicted-cost decile">
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={P.l} x2={W - P.r} y1={sy(g * max)} y2={sy(g * max)} stroke={AXIS} />
          <text x={P.l - 6} y={sy(g * max) + 3.5} textAnchor="end" fontSize={9.5} fill={INK}>
            ${Math.round((g * max) / 1000)}k
          </text>
        </g>
      ))}
      {data.map((d, i) => {
        const x = P.l + i * bw;
        return (
          <g key={d.decile}>
            <rect x={x + bw * 0.14} y={sy(d.predictedMean)} width={bw * 0.32}
              height={H - P.b - sy(d.predictedMean)} rx={2} fill="var(--color-accent)" opacity={0.85} />
            <rect x={x + bw * 0.5} y={sy(d.actualMean)} width={bw * 0.32}
              height={H - P.b - sy(d.actualMean)} rx={2} fill="var(--color-ink)" opacity={0.7} />
            <text x={x + bw / 2} y={H - P.b + 13} textAnchor="middle" fontSize={9.5} fill={INK}>
              {d.decile}
            </text>
          </g>
        );
      })}
      <text x={(W + P.l) / 2} y={H - 3} textAnchor="middle" fontSize={10.5} fill={INK}>
        Decile of predicted cost (1 = cheapest tenth, 10 = costliest tenth)
      </text>
    </svg>
  );
}
