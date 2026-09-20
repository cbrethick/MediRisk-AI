/**
 * The result screen.
 *
 * It shows four things, in this order, because that is the order a person
 * needs them in:
 *   1. The tier, in words, with the confidence behind it.
 *   2. The expected annual cost, with an honest statement of the error bar.
 *   3. Why: the exact contribution of every answer, biggest first.
 *   4. What the number is not: the limits, stated plainly rather than buried.
 */
import {
  AlertTriangle, ArrowLeft, Coins, Gauge, Info, ListTree, RotateCcw, TrendingDown,
  TrendingUp,
} from "lucide-react";
import type { ModelSpec, Prediction } from "../lib/model";
import { FEATURE_LABELS } from "../lib/schema";
import { money, pct } from "./ui";
import { getRate } from "../lib/currency";

const TONE = [
  { name: "Low", color: "var(--color-low)", soft: "var(--color-low-soft)" },
  { name: "Medium", color: "var(--color-medium)", soft: "var(--color-medium-soft)" },
  { name: "High", color: "var(--color-high)", soft: "var(--color-high-soft)" },
];

export default function Result({
  prediction, spec, metrics, onRestart, onResearch,
}: {
  prediction: Prediction;
  spec: ModelSpec;
  metrics: any;
  onRestart: () => void;
  onResearch: () => void;
}) {
  const t = TONE[prediction.tierIndex];
  const cuts = spec.tierCuts;
  const mae = metrics?.regression?.tweedieGlm?.mae ?? 0;
  const perClass = metrics?.classification?.perClass?.[t.name];

  const top = prediction.contributions.slice(0, 9);
  const maxAbs = Math.max(...top.map((c) => Math.abs(c.value)), 0.001);

  return (
    <div className="mx-auto max-w-3xl px-4 py-10 sm:px-6">
      <button
        onClick={onRestart}
        className="mb-6 inline-flex items-center gap-2 text-[13.5px] font-medium text-[var(--color-ink-soft)] hover:text-[var(--color-ink)]"
      >
        <ArrowLeft size={15} /> Change my answers
      </button>

      {/* ---------------------------------------------------------- the tier */}
      <section className="card overflow-hidden rise">
        <div className="px-6 py-7" style={{ background: t.soft }}>
          <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em]"
            style={{ color: t.color }}>
            <Gauge size={14} strokeWidth={2.4} /> Risk tier
          </div>
          <h1 className="mt-2 font-[family-name:var(--font-display)] text-[2.6rem] leading-none"
            style={{ color: t.color }}>
            {t.name}
          </h1>
          <p className="mt-3 max-w-xl text-[14.5px] leading-relaxed text-[var(--color-ink-soft)]">
            {prediction.tierIndex === 0 && (
              <>Your profile looks like the third of adults whose yearly healthcare
                spending falls below {money(cuts[0])}.</>
            )}
            {prediction.tierIndex === 1 && (
              <>Your profile looks like the middle third of adults, who spend roughly{" "}
                {money(cuts[0])} to {money(cuts[1])} a year.</>
            )}
            {prediction.tierIndex === 2 && (
              <>Your profile looks like the third of adults whose yearly healthcare
                spending exceeds {money(cuts[1])}.</>
            )}
          </p>
        </div>

        <div className="px-6 py-5">
          <p className="mb-3 text-[12.5px] font-semibold uppercase tracking-[0.1em] text-[var(--color-ink-faint)]">
            How confident the model is
          </p>
          <div className="space-y-2.5">
            {spec.tierModel.classes.map((cls, i) => (
              <div key={cls} className="flex items-center gap-3">
                <span className="w-16 shrink-0 text-[13px] font-medium">{cls}</span>
                <span className="h-2.5 flex-1 overflow-hidden rounded-full bg-[var(--color-canvas)]">
                  <span
                    className="block h-full rounded-full grow"
                    style={{
                      width: `${prediction.tierProbabilities[i] * 100}%`,
                      background: TONE[i].color,
                      opacity: i === prediction.tierIndex ? 1 : 0.35,
                    }}
                  />
                </span>
                <span className="w-12 shrink-0 text-right text-[13px] tabular-nums">
                  {pct(prediction.tierProbabilities[i], 0)}
                </span>
              </div>
            ))}
          </div>
          {perClass && (
            <p className="mt-4 flex items-start gap-2 rounded-lg bg-[var(--color-canvas)] px-3 py-2.5 text-[13px] leading-relaxed text-[var(--color-ink-soft)]">
              <Info size={14} className="mt-0.5 shrink-0 text-[var(--color-accent)]" />
              <span>
                On held-out people, when this model says “{t.name}” it is right{" "}
                <strong>{pct(perClass.precision, 0)}</strong> of the time, and it catches{" "}
                <strong>{pct(perClass.recall, 0)}</strong> of everyone who genuinely
                belongs in this tier. Those are real numbers from{" "}
                {metrics.dataset.nTest.toLocaleString()} people the model never saw
                during training.
              </span>
            </p>
          )}
        </div>
      </section>

      {/* ------------------------------------------------------- expected cost */}
      <section className="card mt-5 p-6 rise">
        <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--color-accent)]">
          <Coins size={14} strokeWidth={2.4} /> Expected annual cost
        </div>
        <div className="mt-2 flex flex-wrap items-baseline gap-3">
          <span className="font-[family-name:var(--font-display)] text-[2.4rem] leading-none tabular-nums">
            {money(prediction.expectedCost)}
          </span>
          <span className="text-[14px] text-[var(--color-ink-faint)]">per year</span>
        </div>
        <p className="mt-3 text-[14.5px] leading-relaxed text-[var(--color-ink-soft)]">
          This is an <strong>average over many people like you</strong>, not a forecast of
          your own bill. Among a thousand people with this exact profile, a few will
          spend nothing and one or two might spend six figures — this figure is what
          they cost on average, which is the number an insurer needs in order to price
          a pool.
        </p>
        <p className="mt-3 flex items-start gap-2 rounded-lg bg-[var(--color-canvas)] px-3 py-2.5 text-[13px] leading-relaxed text-[var(--color-ink-soft)]">
          <AlertTriangle size={14} className="mt-0.5 shrink-0 text-[var(--color-medium)]" />
          <span>
            Typical error is about <strong>{money(mae)}</strong> per person. Individual
            medical spending is genuinely close to unpredictable — the model explains
            roughly {pct(metrics?.regression?.tweedieGlm?.r2 ?? 0, 0)} of the variation
            between people. It earns its keep by ranking groups, not by nailing
            individuals, and the Research page shows exactly how well it ranks.
          </span>
        </p>
        <p className="mt-3 rounded-lg bg-[var(--color-canvas)] px-3 py-2.5 text-[13px] leading-relaxed text-[var(--color-ink-soft)]">
          <strong>About the rupee figure.</strong> The survey behind this model records{" "}
          <em>US</em> healthcare spending in dollars, converted here at ₹{getRate()} to
          the dollar. A US cost shown in rupees is still a US cost — the same
          treatment is several times cheaper in India — so read this as American
          prices in a familiar unit, not as an Indian hospital bill.
        </p>
        <p className="mt-3 text-[13px] text-[var(--color-ink-soft)]">
          Probability you have any healthcare spending at all this year:{" "}
          <strong className="tabular-nums">{pct(prediction.anySpendProbability, 0)}</strong>
        </p>
      </section>

      {/* ------------------------------------------------------- explanation */}
      <section className="card mt-5 p-6 rise">
        <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--color-accent)]">
          <ListTree size={14} strokeWidth={2.4} /> Why you got this result
        </div>
        <h2 className="mt-2 font-[family-name:var(--font-display)] text-xl">
          Every answer, and exactly what it did
        </h2>
        <p className="mt-2 text-[14px] leading-relaxed text-[var(--color-ink-soft)]">
          Each bar is one of your answers, compared against a reference adult — the
          median age, the most common answer to every other question. Bars to the
          right pushed you towards High risk; bars to the left pulled you away from it.
        </p>

        <div className="mt-5 space-y-2">
          {top.map((c) => {
            const w = (Math.abs(c.value) / maxAbs) * 50;
            const up = c.value > 0;
            return (
              <div key={c.feature} className="flex items-center gap-3">
                <span className="w-[42%] shrink-0 truncate text-right text-[13px] font-medium"
                  title={FEATURE_LABELS[c.feature] ?? c.feature}>
                  {FEATURE_LABELS[c.feature] ?? c.feature}
                </span>
                <span className="relative h-6 flex-1">
                  <span className="absolute inset-y-0 left-1/2 w-px bg-[var(--color-line)]" />
                  <span
                    className="absolute top-1/2 h-4 -translate-y-1/2 rounded-sm grow"
                    style={{
                      width: `${w}%`,
                      left: up ? "50%" : `${50 - w}%`,
                      background: up ? "var(--color-high)" : "var(--color-low)",
                      opacity: 0.85,
                    }}
                  />
                </span>
                <span className="flex w-14 shrink-0 items-center justify-end gap-1 text-[12px] tabular-nums"
                  style={{ color: up ? "var(--color-high)" : "var(--color-low)" }}>
                  {up ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                  {c.value > 0 ? "+" : ""}{c.value.toFixed(2)}
                </span>
              </div>
            );
          })}
        </div>

        <p className="mt-5 border-l-2 border-[var(--color-accent)] bg-[var(--color-accent-soft)] px-3 py-2.5 text-[13px] leading-relaxed text-[var(--color-ink-soft)]">
          These numbers are <strong>exact, not estimated</strong>. The deployed model is
          linear, so a feature's contribution is its coefficient times how far your
          answer sits from the reference — the same quantity SHAP would compute, with
          no sampling involved. They add up precisely: {prediction.baselineLogit.toFixed(2)}{" "}
          (the reference person) {prediction.personLogit - prediction.baselineLogit >= 0 ? "+" : "−"}{" "}
          {Math.abs(prediction.personLogit - prediction.baselineLogit).toFixed(2)} (your
          answers) = {prediction.personLogit.toFixed(2)}, your High-risk score.
        </p>
      </section>

      {/* ------------------------------------------------------------ caveats */}
      <section className="card mt-5 border-[var(--color-medium)]/30 bg-[var(--color-medium-soft)] p-6">
        <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--color-medium)]">
          <AlertTriangle size={14} strokeWidth={2.4} /> What this is not
        </div>
        <ul className="mt-3 space-y-2 text-[13.5px] leading-relaxed text-[var(--color-ink-soft)]">
          <li><strong>Not medical advice.</strong> It predicts spending, not health. A low tier does not mean you are well, and a high tier does not mean you are ill.</li>
          <li><strong>Not a quote.</strong> No insurer has seen this and no price is being offered.</li>
          <li><strong>US prices, 2024.</strong> Trained on a US survey, so the dollar figures reflect US healthcare pricing in that year and transfer poorly elsewhere.</li>
          <li><strong>A student project.</strong> Built to demonstrate a modelling pipeline end to end, on public data, for coursework.</li>
        </ul>
      </section>

      <div className="mt-6 flex flex-wrap gap-3">
        <button onClick={onRestart}
          className="inline-flex items-center gap-2 rounded-xl border border-[var(--color-line)] bg-white px-4 py-2.5 text-[14px] font-medium">
          <RotateCcw size={15} /> Start over
        </button>
        <button onClick={onResearch}
          className="inline-flex items-center gap-2 rounded-xl bg-[var(--color-ink)] px-5 py-2.5 text-[14px] font-semibold text-white hover:bg-black">
          Read how the model was built
        </button>
      </div>
    </div>
  );
}
