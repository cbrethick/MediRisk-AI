/**
 * The assessment form.
 *
 * Design rules this screen follows:
 *  - One topic per step, with a visible progress rail, so the form never feels
 *    like a wall of inputs.
 *  - Every question states WHY it is being asked, inline, before it is answered.
 *  - Every question can be skipped. The model has explicit missingness
 *    indicators, so "prefer not to say" is a real answer rather than a silent
 *    zero, and the interface says so.
 *  - Nothing is sent anywhere. The model runs in this tab.
 */
import { useMemo, useState } from "react";
import {
  ArrowLeft, ArrowRight, Calculator, Check, CircleHelp, Info, Lock, Scale,
  ShieldCheck, Sparkles, Stethoscope, X,
} from "lucide-react";
import {
  CONDITIONS, CONDITION_KEYS, STEPS, emptyAnswers, type Question,
} from "../lib/schema";
import { bmiFrom, type Answers } from "../lib/model";

/* ------------------------------------------------------------------ helpers */

function WhyNote({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-1.5 text-[12.5px] font-medium text-[var(--color-accent)] hover:text-[var(--color-accent-deep)]"
        aria-expanded={open}
      >
        <CircleHelp size={13} strokeWidth={2.2} />
        {open ? "Hide" : "Why we ask this"}
      </button>
      {open && (
        <p className="mt-2 rounded-lg bg-[var(--color-canvas)] px-3 py-2.5 text-[13px] leading-relaxed text-[var(--color-ink-soft)] rise">
          {text}
        </p>
      )}
    </div>
  );
}

function SkipButton({ active, onClick }: { active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`rounded-full border px-3 py-1 text-[12px] font-medium transition ${
        active
          ? "border-[var(--color-ink-faint)] bg-[var(--color-ink)] text-white"
          : "border-[var(--color-line)] text-[var(--color-ink-faint)] hover:border-[var(--color-ink-faint)]"
      }`}
    >
      {active ? "Skipped" : "Prefer not to say"}
    </button>
  );
}

/* ---------------------------------------------------------------- BMI helper */

function BmiCalculator({ value, onChange }: {
  value: number | null; onChange: (v: number | null) => void;
}) {
  const [cm, setCm] = useState(170);
  const [kg, setKg] = useState(70);
  const computed = useMemo(() => bmiFrom(kg, cm), [kg, cm]);
  const band =
    computed < 18.5 ? { t: "Underweight", c: "var(--color-medium)" }
      : computed < 25 ? { t: "Healthy range", c: "var(--color-low)" }
        : computed < 30 ? { t: "Overweight", c: "var(--color-medium)" }
          : { t: "Obese (BMI 30+)", c: "var(--color-high)" };

  return (
    <div className="rounded-xl border border-[var(--color-line)] bg-[var(--color-canvas)] p-4">
      <div className="grid gap-4 sm:grid-cols-2">
        {([["Height", cm, setCm, "cm", 120, 215], ["Weight", kg, setKg, "kg", 35, 200]] as const).map(
          ([label, val, set, unit, lo, hi]) => (
            <label key={label} className="block">
              <span className="text-[12.5px] font-semibold text-[var(--color-ink-soft)]">
                {label}
              </span>
              <div className="mt-1.5 flex items-center gap-2">
                <input
                  type="number" min={lo} max={hi} value={val}
                  onChange={(e) => set(Number(e.target.value))}
                  className="w-24 rounded-lg border border-[var(--color-line)] bg-white px-3 py-2 text-[15px] tabular-nums"
                />
                <span className="text-[13px] text-[var(--color-ink-faint)]">{unit}</span>
              </div>
            </label>
          ),
        )}
      </div>
      <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-[var(--color-line)] pt-3">
        <div>
          <span className="text-[12.5px] text-[var(--color-ink-faint)]">Your BMI</span>
          <div className="flex items-baseline gap-2">
            <span className="font-[family-name:var(--font-display)] text-2xl tabular-nums">
              {computed.toFixed(1)}
            </span>
            <span className="text-[13px] font-medium" style={{ color: band.c }}>{band.t}</span>
          </div>
        </div>
        <button
          type="button"
          onClick={() => onChange(Number(computed.toFixed(1)))}
          className="inline-flex items-center gap-2 rounded-lg bg-[var(--color-accent)] px-4 py-2 text-[13.5px] font-semibold text-white hover:bg-[var(--color-accent-deep)]"
        >
          <Calculator size={15} />
          {value === null ? "Use this BMI" : "Update"}
        </button>
      </div>
      {value !== null && (
        <p className="mt-3 flex items-center gap-1.5 text-[12.5px] font-medium text-[var(--color-low)]">
          <Check size={14} strokeWidth={2.5} /> Using BMI {value.toFixed(1)}
        </p>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ question */

function QuestionCard({ q, value, onChange }: {
  q: Question; value: number | null; onChange: (v: number | null) => void;
}) {
  const Icon = q.icon;
  return (
    <div className="card p-5 rise">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-[var(--color-accent-soft)] text-[var(--color-accent)]">
          <Icon size={18} strokeWidth={2} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h3 className="text-[15.5px] font-semibold text-[var(--color-ink)]">{q.label}</h3>
            {q.optional && (
              <SkipButton active={value === null} onClick={() => onChange(null)} />
            )}
          </div>

          <div className="mt-4">
            {q.kind === "choice" && (
              <div className="flex flex-wrap gap-2">
                {q.choices!.map((c) => {
                  const on = value === c.value;
                  return (
                    <button
                      key={c.value}
                      type="button"
                      onClick={() => onChange(on ? null : c.value)}
                      aria-pressed={on}
                      className={`rounded-xl border px-3.5 py-2.5 text-left transition ${
                        on
                          ? "border-[var(--color-accent)] bg-[var(--color-accent-soft)] ring-1 ring-[var(--color-accent)]"
                          : "border-[var(--color-line)] bg-white hover:border-[var(--color-ink-faint)]"
                      }`}
                    >
                      <span className={`block text-[14px] font-medium ${on ? "text-[var(--color-accent-deep)]" : "text-[var(--color-ink)]"}`}>
                        {c.label}
                      </span>
                      {c.hint && (
                        <span className="mt-0.5 block text-[12px] text-[var(--color-ink-faint)]">
                          {c.hint}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            )}

            {q.kind === "slider" && (
              <div>
                <div className="flex items-baseline gap-2">
                  <span className="font-[family-name:var(--font-display)] text-3xl tabular-nums text-[var(--color-ink)]">
                    {value ?? "—"}
                  </span>
                  <span className="text-[13px] text-[var(--color-ink-faint)]">{q.unit}</span>
                </div>
                <input
                  type="range" min={q.min} max={q.max} step={q.step}
                  value={value ?? Math.round(((q.min! + q.max!) / 2))}
                  onChange={(e) => onChange(Number(e.target.value))}
                  aria-label={q.label}
                  className="mt-3"
                />
                <div className="mt-1 flex justify-between text-[11.5px] text-[var(--color-ink-faint)]">
                  <span>{q.min}</span><span>{q.max}</span>
                </div>
              </div>
            )}

            {q.kind === "number" && q.key === "bmi" && (
              <BmiCalculator value={value} onChange={onChange} />
            )}
          </div>

          <WhyNote text={q.why} />
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------- wizard */

export default function Wizard({ onComplete }: { onComplete: (a: Answers) => void }) {
  const [answers, setAnswers] = useState<Answers>(emptyAnswers);
  const [step, setStep] = useState(0);
  const total = STEPS.length;
  const current = STEPS[step];
  const StepIcon = current.icon;

  const set = (k: string, v: number | null) => setAnswers((a) => ({ ...a, [k]: v }));

  const answeredCount = Object.entries(answers).filter(
    ([k, v]) => v !== null && !CONDITION_KEYS.includes(k),
  ).length;
  const totalQuestions = STEPS.reduce((n, s) => n + s.questions.length, 0);

  return (
    <div className="mx-auto max-w-3xl px-4 py-10 sm:px-6">
      {/* progress rail */}
      <div className="mb-8">
        <div className="flex items-center gap-2">
          {STEPS.map((s, i) => (
            <button
              key={s.id}
              type="button"
              onClick={() => setStep(i)}
              className="group flex-1"
              aria-label={`Go to step ${i + 1}: ${s.title}`}
            >
              <span
                className={`block h-1.5 rounded-full transition-colors ${
                  i <= step ? "bg-[var(--color-accent)]" : "bg-[var(--color-line)]"
                }`}
              />
              <span
                className={`mt-2 hidden text-[12px] font-medium sm:block ${
                  i === step ? "text-[var(--color-ink)]" : "text-[var(--color-ink-faint)]"
                }`}
              >
                {s.title}
              </span>
            </button>
          ))}
        </div>
        <p className="mt-3 text-[12.5px] text-[var(--color-ink-faint)] sm:hidden">
          Step {step + 1} of {total} · {current.title}
        </p>
      </div>

      <header className="mb-6">
        <div className="mb-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--color-accent)]">
          <StepIcon size={14} strokeWidth={2.4} />
          Step {step + 1} of {total}
        </div>
        <h1 className="font-[family-name:var(--font-display)] text-[1.9rem] leading-tight">
          {current.title}
        </h1>
        <p className="mt-2 text-[15px] leading-relaxed text-[var(--color-ink-soft)]">
          {current.blurb}
        </p>
      </header>

      <div className="space-y-4">
        {current.questions.map((q) => (
          <QuestionCard key={q.key} q={q} value={answers[q.key] ?? null}
            onChange={(v) => set(q.key, v)} />
        ))}

        {/* the conditions grid only appears on the health step */}
        {current.id === "health" && (
          <div className="card p-5 rise">
            <div className="flex items-start gap-3">
              <span className="mt-0.5 grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-[var(--color-accent-soft)] text-[var(--color-accent)]">
                <Stethoscope size={18} strokeWidth={2} />
              </span>
              <div className="min-w-0 flex-1">
                <h3 className="text-[15.5px] font-semibold">
                  Has a doctor ever told you that you have any of these?
                </h3>
                <p className="mt-1 text-[13px] text-[var(--color-ink-soft)]">
                  Tap the ones that apply. Anything you leave untapped is treated as “no”.
                </p>
                <div className="mt-4 grid gap-2 sm:grid-cols-2">
                  {CONDITIONS.map((c) => {
                    const on = answers[c.key] === 1;
                    const CIcon = c.icon;
                    return (
                      <button
                        key={c.key}
                        type="button"
                        aria-pressed={on}
                        onClick={() => set(c.key, on ? 0 : 1)}
                        className={`flex items-center gap-3 rounded-xl border px-3 py-2.5 text-left transition ${
                          on
                            ? "border-[var(--color-high)] bg-[var(--color-high-soft)]"
                            : "border-[var(--color-line)] bg-white hover:border-[var(--color-ink-faint)]"
                        }`}
                      >
                        <span
                          className="grid h-8 w-8 shrink-0 place-items-center rounded-lg"
                          style={{
                            background: on ? "var(--color-high)" : "var(--color-canvas)",
                            color: on ? "#fff" : "var(--color-ink-faint)",
                          }}
                        >
                          <CIcon size={16} strokeWidth={2.1} />
                        </span>
                        <span className="min-w-0">
                          <span className="block truncate text-[13.5px] font-medium">{c.label}</span>
                          <span className="block truncate text-[11.5px] text-[var(--color-ink-faint)]">
                            {c.note}
                          </span>
                        </span>
                        {on && <Check size={15} className="ml-auto shrink-0 text-[var(--color-high)]" strokeWidth={2.6} />}
                      </button>
                    );
                  })}
                </div>
                <p className="mt-3 rounded-lg bg-[var(--color-canvas)] px-3 py-2.5 text-[13px] leading-relaxed text-[var(--color-ink-soft)]">
                  The model uses each condition on its own <em>and</em> the total count,
                  because someone managing four conditions at once costs more than four
                  separate people with one condition each.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* navigation */}
      <div className="mt-8 flex items-center justify-between gap-3">
        <button
          type="button"
          onClick={() => setStep((s) => Math.max(0, s - 1))}
          disabled={step === 0}
          className="inline-flex items-center gap-2 rounded-xl border border-[var(--color-line)] bg-white px-4 py-2.5 text-[14px] font-medium disabled:opacity-40"
        >
          <ArrowLeft size={16} /> Back
        </button>

        <span className="text-[12.5px] text-[var(--color-ink-faint)]">
          {answeredCount} of {totalQuestions} answered
        </span>

        {step < total - 1 ? (
          <button
            type="button"
            onClick={() => setStep((s) => s + 1)}
            className="inline-flex items-center gap-2 rounded-xl bg-[var(--color-ink)] px-5 py-2.5 text-[14px] font-semibold text-white hover:bg-black"
          >
            Continue <ArrowRight size={16} />
          </button>
        ) : (
          <button
            type="button"
            onClick={() => onComplete(answers)}
            className="inline-flex items-center gap-2 rounded-xl bg-[var(--color-accent)] px-5 py-2.5 text-[14px] font-semibold text-white hover:bg-[var(--color-accent-deep)]"
          >
            <Sparkles size={16} /> See my result
          </button>
        )}
      </div>

      <p className="mt-6 flex items-start gap-2 text-[12.5px] leading-relaxed text-[var(--color-ink-faint)]">
        <Lock size={14} className="mt-0.5 shrink-0" />
        Everything stays in this browser tab. The model is a 14 KB file of
        coefficients that runs locally, so nothing you type is uploaded, stored
        or sent to a server.
      </p>
    </div>
  );
}
