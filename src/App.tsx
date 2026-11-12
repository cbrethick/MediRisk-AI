/**
 * App shell: a landing page, the assessment, the result, and the research
 * write-up. State is kept here because there are only four screens and a
 * router would be more machinery than the app needs.
 */
import { useEffect, useMemo, useState } from "react";
import {
  Activity, ArrowRight, BookOpen, Github, Lock, ScrollText, ShieldCheck,
  Sparkles, Stethoscope, Zap,
} from "lucide-react";
import Wizard from "./components/Wizard";
import Result from "./components/Result";
import Research from "./components/Research";
import {
  fillSkipped, loadModel, predict, withEngineered, type Answers, type ModelSpec, type Prediction,
} from "./lib/model";
import { CONDITION_KEYS } from "./lib/schema";
import { getCode, getRate, money, setCurrency } from "./lib/currency";

type Cur = "INR" | "USD";
const CUR_KEY = "medirisk-currency";
const RATE_KEY = "medirisk-usd-inr";

type View = "home" | "assess" | "result" | "research";

export default function App() {
  const [view, setView] = useState<View>("home");
  const [spec, setSpec] = useState<ModelSpec | null>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [currency, setCur] = useState<Cur>("INR");
  const [rate, setRateState] = useState(88);
  const [defaultRate, setDefaultRate] = useState(88);

  // Display currency only: predictions are computed in USD and never change.
  const chooseCurrency = (c: Cur) => {
    setCurrency({ code: c });
    setCur(c);
    try { localStorage.setItem(CUR_KEY, c); } catch { /* storage unavailable */ }
  };

  // Rupees per dollar. Changes only how dollars are shown in rupees.
  const chooseRate = (r: number) => {
    setCurrency({ usdRate: r });
    setRateState(r);
    try { localStorage.setItem(RATE_KEY, String(r)); } catch { /* storage unavailable */ }
  };

  useEffect(() => {
    Promise.all([
      loadModel(),
      fetch(`${import.meta.env.BASE_URL}metrics.json`).then((r) => r.json()),
    ])
      .then(([s, m]) => {
        setCurrency(s.currency);
        let saved: string | null = null;
        try { saved = localStorage.getItem(CUR_KEY); } catch { /* ignore */ }
        if (saved === "INR" || saved === "USD") setCurrency({ code: saved });
        setDefaultRate(getRate());
        let savedRate = NaN;
        try { savedRate = Number(localStorage.getItem(RATE_KEY)); } catch { /* ignore */ }
        if (savedRate >= 1 && savedRate <= 1000) setCurrency({ usdRate: savedRate });
        setRateState(getRate());
        setCur(getCode());
        setSpec(s); setMetrics(m);
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => { window.scrollTo({ top: 0 }); }, [view]);

  const run = (raw: Answers) => {
    if (!spec) return;
    setPrediction(predict(withEngineered(fillSkipped(raw, spec), CONDITION_KEYS), spec));
    setView("result");
  };

  return (
    <div className="min-h-screen">
      <Nav view={view} setView={setView} currency={currency} onCurrency={chooseCurrency}
        rate={rate} defaultRate={defaultRate} onRate={chooseRate} />

      {error && (
        <div className="mx-auto mt-10 max-w-lg card border-[var(--color-high)] p-5 text-[14px]">
          <strong>Could not load the model.</strong>
          <p className="mt-1 text-[var(--color-ink-soft)]">{error}</p>
          <p className="mt-2 text-[13px] text-[var(--color-ink-soft)]">
            Run <code className="rounded bg-[var(--color-canvas)] px-1.5 py-0.5">python3 ml/train.py</code>{" "}
            then <code className="rounded bg-[var(--color-canvas)] px-1.5 py-0.5">python3 ml/export_web_model.py</code>{" "}
            to generate <code>public/model.json</code>.
          </p>
        </div>
      )}

      {!error && !spec && (
        <div className="grid min-h-[60vh] place-items-center text-[14px] text-[var(--color-ink-faint)]">
          Loading the model…
        </div>
      )}

      {spec && metrics && (
        <>
          {view === "home" && <Home metrics={metrics} onStart={() => setView("assess")}
            onResearch={() => setView("research")} />}
          {view === "assess" && <Wizard onComplete={run} />}
          {view === "result" && prediction && (
            <Result prediction={prediction} spec={spec} metrics={metrics}
              onRestart={() => setView("assess")} onResearch={() => setView("research")} />
          )}
          {view === "research" && <Research m={metrics} />}
        </>
      )}

      <Footer />
    </div>
  );
}

/* ---------------------------------------------------------------------- nav */

function Nav({ view, setView, currency, onCurrency, rate, defaultRate, onRate }: {
  view: View; setView: (v: View) => void; currency: Cur; onCurrency: (c: Cur) => void;
  rate: number; defaultRate: number; onRate: (r: number) => void;
}) {
  const tabs: { id: View; label: string }[] = [
    { id: "home", label: "Overview" },
    { id: "assess", label: "Assessment" },
    { id: "research", label: "Research" },
  ];
  return (
    <header className="sticky top-0 z-30 border-b border-[var(--color-line)] bg-[var(--color-surface)]/85 backdrop-blur">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 py-3 sm:px-6">
        <button onClick={() => setView("home")} className="flex items-center gap-2.5">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-[var(--color-accent)] text-white">
            <Activity size={17} strokeWidth={2.4} />
          </span>
          <span className="text-left">
            <span className="block text-[14.5px] font-semibold leading-tight">Medical Risk AI</span>
            <span className="block text-[11px] leading-tight text-[var(--color-ink-faint)]">
              MEPS 2024 · cost &amp; risk tier
            </span>
          </span>
        </button>
        <nav className="flex flex-wrap items-center gap-1">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setView(t.id)}
              className={`rounded-lg px-3 py-1.5 text-[13.5px] font-medium transition ${
                view === t.id || (view === "result" && t.id === "assess")
                  ? "bg-[var(--color-accent-soft)] text-[var(--color-accent-deep)]"
                  : "text-[var(--color-ink-soft)] hover:bg-[var(--color-canvas)]"
              }`}
            >
              {t.label}
            </button>
          ))}
          <span className="ml-2 inline-flex rounded-lg border border-[var(--color-line)] bg-white p-0.5"
            role="group" aria-label="Currency">
            {(["INR", "USD"] as const).map((c) => (
              <button key={c} onClick={() => onCurrency(c)} aria-pressed={currency === c}
                title={c === "INR" ? "Show amounts in Indian rupees" : "Show amounts in US dollars"}
                className={`rounded-md px-2.5 py-1 text-[13px] font-semibold transition ${
                  currency === c
                    ? "bg-[var(--color-accent)] text-white"
                    : "text-[var(--color-ink-soft)] hover:bg-[var(--color-canvas)]"
                }`}>
                {c === "INR" ? "₹" : "$"}
              </button>
            ))}
          </span>
          {currency === "INR" && (
            <RateEditor rate={rate} defaultRate={defaultRate} onRate={onRate} />
          )}
        </nav>
      </div>
    </header>
  );
}

/**
 * Exchange-rate box: "$1 = Rs [88]". Typing a new rate (e.g. today's 95)
 * re-prices every rupee amount in the app immediately. It never changes a
 * prediction, which is always computed in US dollars.
 */
function RateEditor({ rate, defaultRate, onRate }: {
  rate: number; defaultRate: number; onRate: (r: number) => void;
}) {
  const [draft, setDraft] = useState(String(rate));
  useEffect(() => { setDraft(String(rate)); }, [rate]);
  const n = Number(draft);
  const valid = draft.trim() !== "" && n >= 1 && n <= 1000;
  return (
    <label className="ml-1 inline-flex items-center gap-1 rounded-lg border border-[var(--color-line)] bg-white px-2 py-1 text-[12.5px] text-[var(--color-ink-soft)]"
      title="Rupees per US dollar. Change it to today's rate; every rupee amount updates.">
      <span className="hidden sm:inline">$1 =</span> ₹
      <input
        type="number" inputMode="decimal" min={1} max={1000} step="0.01"
        value={draft}
        onChange={(e) => {
          setDraft(e.target.value);
          const v = Number(e.target.value);
          if (e.target.value.trim() !== "" && v >= 1 && v <= 1000) onRate(v);
        }}
        onBlur={() => { if (!valid) setDraft(String(rate)); }}
        aria-label="Rupees per US dollar"
        className={`w-14 rounded border px-1 py-0.5 text-right font-semibold tabular-nums text-[var(--color-ink)] focus:outline-none ${
          valid ? "border-transparent focus:border-[var(--color-accent)]" : "border-[var(--color-high)]"
        }`}
      />
      {rate !== defaultRate && (
        <button type="button" onClick={() => onRate(defaultRate)}
          className="ml-0.5 text-[11.5px] font-medium text-[var(--color-accent)] hover:underline"
          title={`Reset to ₹${defaultRate}`}>
          reset
        </button>
      )}
    </label>
  );
}

/* --------------------------------------------------------------------- home */

function Home({ metrics, onStart, onResearch }: {
  metrics: any; onStart: () => void; onResearch: () => void;
}) {
  const d = metrics.dataset;
  const cls = metrics.classification;
  const deployed = cls.models.find((m: any) => m.model === cls.deployed);

  return (
    <main className="mx-auto max-w-4xl px-4 sm:px-6">
      <section className="py-16 sm:py-20">
        <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-[var(--color-line)] bg-white px-3 py-1 text-[12px] font-medium text-[var(--color-ink-soft)]">
          <Zap size={13} className="text-[var(--color-accent)]" />
          Trained on {d.rowsAdults.toLocaleString()} real survey respondents
        </div>
        <h1 className="font-[family-name:var(--font-display)] text-[2.5rem] leading-[1.08] sm:text-[3.4rem]">
          What will a year of healthcare cost —<br className="hidden sm:block" />
          and who is most at risk?
        </h1>
        <p className="mt-5 max-w-2xl text-[16.5px] leading-relaxed text-[var(--color-ink-soft)]">
          Answer around a dozen questions about your age, health and circumstances.
          A model trained on the 2024 US Medical Expenditure Panel Survey estimates
          your expected annual healthcare cost, places you in a risk tier, and shows
          you exactly which of your answers moved the result and by how much.
        </p>

        <div className="mt-8 flex flex-wrap gap-3">
          <button onClick={onStart}
            className="inline-flex items-center gap-2 rounded-xl bg-[var(--color-accent)] px-6 py-3 text-[15px] font-semibold text-white transition hover:bg-[var(--color-accent-deep)]">
            <Sparkles size={17} /> Start the assessment
          </button>
          <button onClick={onResearch}
            className="inline-flex items-center gap-2 rounded-xl border border-[var(--color-line)] bg-white px-6 py-3 text-[15px] font-semibold transition hover:border-[var(--color-ink-faint)]">
            <ScrollText size={17} /> Read the research
          </button>
        </div>

        <p className="mt-5 flex items-center gap-2 text-[13px] text-[var(--color-ink-faint)]">
          <Lock size={14} /> Runs entirely in your browser. Nothing you enter is uploaded.
        </p>
      </section>

      <section className="grid gap-4 pb-6 sm:grid-cols-3">
        {[
          { i: Stethoscope, t: "Two predictions", b: `An expected annual cost and a Low / Medium / High tier. Typical cost error is about ${money(metrics.regression.tweedieGlm.mae)}, and the tier is right about ${Math.round(deployed.accuracy * 100)}% of the time on people the model never saw.` },
          { i: BookOpen, t: "Every answer explained", b: "The deployed model is linear, so each answer's effect is exact rather than estimated — the contributions add up precisely to your score, and the app shows the arithmetic." },
          { i: ShieldCheck, t: "Audited, and honest about it", b: "Race is excluded from the model and then used to check it. The research page reports where the model works less well, including where that is uncomfortable." },
        ].map((c) => (
          <div key={c.t} className="card p-5">
            <span className="grid h-9 w-9 place-items-center rounded-lg bg-[var(--color-accent-soft)] text-[var(--color-accent)]">
              <c.i size={18} strokeWidth={2.1} />
            </span>
            <h3 className="mt-3 text-[14.5px] font-semibold">{c.t}</h3>
            <p className="mt-1.5 text-[13px] leading-relaxed text-[var(--color-ink-soft)]">{c.b}</p>
          </div>
        ))}
      </section>

      <section className="card my-10 p-6">
        <h2 className="font-[family-name:var(--font-display)] text-xl">How it works, in four steps</h2>
        <ol className="mt-4 space-y-3">
          {[
            ["Clean the survey", `MEPS codes "refused" and "not applicable" as negative numbers. All of them become missing values before anything is fitted, and ${pctS(d.zeroSpendShare)} of adults who spent nothing are kept rather than dropped.`],
            ["Build the features", `${d.nFeaturesSource} inputs become ${d.nFeaturesEncoded} columns, including a smoking-and-obesity interaction and a count of chronic conditions.`],
            ["Fit and compare", `Linear, forest and boosted models compete under ${d.nFolds}-fold cross-validation. The winner is measured once on ${d.nTest.toLocaleString()} held-out people.`],
            ["Ship the coefficients", "The chosen model is exported as a 14 KB JSON file that this page evaluates directly, verified to reproduce scikit-learn to the last decimal place."],
          ].map(([t, b], i) => (
            <li key={t} className="flex gap-3">
              <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-[var(--color-ink)] text-[12px] font-semibold text-white">
                {i + 1}
              </span>
              <span>
                <strong className="text-[14px]">{t}</strong>
                <span className="mt-0.5 block text-[13px] leading-relaxed text-[var(--color-ink-soft)]">{b}</span>
              </span>
            </li>
          ))}
        </ol>
        <button onClick={onStart}
          className="mt-6 inline-flex items-center gap-2 text-[14px] font-semibold text-[var(--color-accent)] hover:text-[var(--color-accent-deep)]">
          Try it <ArrowRight size={15} />
        </button>
      </section>
    </main>
  );
}

const pctS = (n: number) => `${(n * 100).toFixed(0)}%`;

function Footer() {
  return (
    <footer className="border-t border-[var(--color-line)] bg-[var(--color-surface)]">
      <div className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
        <p className="text-[13px] leading-relaxed text-[var(--color-ink-soft)]">
          <strong>Not medical advice and not an insurance quote.</strong> A student
          project built on public data from the US Agency for Healthcare Research and
          Quality — Medical Expenditure Panel Survey, HC-256, 2024 Full Year
          Consolidated Data File. Predictions describe averages over groups of similar
          people, never an individual's actual future costs.
        </p>
      </div>
    </footer>
  );
}
