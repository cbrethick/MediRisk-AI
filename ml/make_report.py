"""
Generate docs/04-RESULTS.md and the figures it embeds, straight from
ml/artifacts/metrics.json.

The results document is NOT written by hand. Hand-written numbers rot the
moment anyone retrains, and a report that disagrees with the model is worse
than no report. Re-run train.py and then this, and the document is current.

    python3 ml/make_report.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import ARTIFACTS, DOC_ASSETS, ROOT, USD_TO_INR

M = json.loads((ARTIFACTS / "metrics.json").read_text())
D, REG, CLS, EXP = M["dataset"], M["regression"], M["classification"], M["explainability"]
TIERS = CLS["tierNames"]

INK, ACCENT, LINE = "#0f1b2a", "#0d7d75", "#e3e9f1"
TIER_C = ["#2f8f5b", "#b57314", "#c0392f"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": "#9aa8bb", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#46586e", "ytick.color": "#46586e",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.grid": True,
    "grid.color": LINE, "grid.linewidth": 0.8,
})

# All stored metrics are in USD; rupees are a display conversion only.
def money(usd: float) -> str:
    return f"\u20b9{usd * USD_TO_INR:,.0f}"

n3 = lambda v: f"{v:.3f}"


def save(fig, name: str) -> str:
    fig.tight_layout()
    fig.savefig(DOC_ASSETS / name, bbox_inches="tight")
    plt.close(fig)
    return f"assets/{name}"


# ---------------------------------------------------------------- figures ---
def fig_confusion() -> str:
    cm = np.array(CLS["confusionMatrix"])
    pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(pct, cmap="BuGn", vmin=0, vmax=1)
    ax.set_xticks(range(3), TIERS); ax.set_yticks(range(3), TIERS)
    ax.set_xlabel("Predicted tier"); ax.set_ylabel("Actual tier")
    ax.set_title("Confusion matrix, held-out test set", fontsize=10, pad=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center",
                    color="white" if pct[i, j] > 0.55 else INK, fontsize=9)
    ax.grid(False)
    return save(fig, "confusion_matrix.png")


def fig_pr() -> str:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for t, c in zip(TIERS, TIER_C):
        p = CLS["prCurves"][t]
        ax.plot(p["recall"], p["precision"], color=c, lw=1.8,
                label=f"{t}  (AP {p['ap']:.3f}, AUC {p['rocAuc']:.3f})")
    ax.axhline(1 / 3, color="#7d8da3", ls="--", lw=1, label="Random (0.333)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–recall by risk tier (one-vs-rest)", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=7.5, frameon=False)
    return save(fig, "precision_recall.png")


def fig_roc_cal() -> str:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    r = M["part1"]["roc"]
    a1.plot(r["fpr"], r["tpr"], color=ACCENT, lw=1.8)
    a1.plot([0, 1], [0, 1], ls="--", lw=1, color="#7d8da3")
    a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate")
    a1.set_title("Part 1 — ROC: any spending?", fontsize=10)

    cal = M["part1"]["calibration"]
    a2.plot([c["predicted"] for c in cal], [c["observed"] for c in cal],
            "o-", color=ACCENT, lw=1.6, ms=4)
    a2.plot([0, 1], [0, 1], ls="--", lw=1, color="#7d8da3")
    a2.set_xlabel("Predicted probability"); a2.set_ylabel("Observed frequency")
    a2.set_title("Part 1 — calibration", fontsize=10)
    return save(fig, "part1_roc_calibration.png")


def fig_lift() -> str:
    lift = REG["decileLift"]
    x = np.arange(len(lift))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(x - 0.2, [l["predictedMean"] for l in lift], 0.38, color=ACCENT,
           label="Predicted mean")
    ax.bar(x + 0.2, [l["actualMean"] for l in lift], 0.38, color=INK, alpha=0.75,
           label="Actual mean")
    ax.set_xticks(x, [l["decile"] for l in lift])
    ax.set_xlabel("Decile of predicted cost (1 = cheapest tenth)")
    ax.set_ylabel("Annual cost (INR)")
    ax.yaxis.set_major_formatter(
        lambda v, _: f"\u20b9{v * USD_TO_INR / 100000:.1f}L")   # lakh
    ax.set_title(f"Decile lift — top decile costs "
                 f"{REG['topBottomDecileRatio']:.1f}x the bottom",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    return save(fig, "decile_lift.png")


def fig_importance() -> str:
    g = EXP["groupedImportance"][:14][::-1]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.barh([r["feature"] for r in g], [r["meanAbsContribution"] for r in g],
            color=ACCENT)
    ax.set_xlabel("Mean |contribution| to the High-risk score")
    ax.set_title("Feature importance, exact linear attribution", fontsize=10)
    return save(fig, "feature_importance.png")


def fig_fairness() -> str:
    groups = M["fairness"]["race"]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    names = [g["group"] for g in groups]
    ax.bar(names, [g["highRecall"] for g in groups], color=ACCENT)
    ax.axhline(CLS["perClass"]["High"]["recall"], color="#c0392f", ls="--", lw=1.2,
               label=f"Overall ({CLS['perClass']['High']['recall']:.2f})")
    ax.set_ylabel("High-tier recall"); ax.set_ylim(0, 1)
    ax.set_title("High-tier recall by race group (race is not a model input)",
                 fontsize=10)
    ax.tick_params(axis="x", labelrotation=18, labelsize=7.5)
    ax.legend(fontsize=8, frameon=False)
    return save(fig, "fairness_race.png")


figs = {
    "confusion": fig_confusion(), "pr": fig_pr(), "roc": fig_roc_cal(),
    "lift": fig_lift(), "importance": fig_importance(), "fairness": fig_fairness(),
}


# ------------------------------------------------------------- markdown ---
def table(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


dep_cls = next(m for m in CLS["models"] if m["model"] == CLS["deployed"])
md = f"""# 4 · Results and validation

> Generated automatically by `ml/make_report.py` from `ml/artifacts/metrics.json`.
> Training run: **{M['generatedAt']}**. Do not edit by hand — re-run the pipeline.

Every figure on this page comes from **{D['nTest']:,} held-out people** who took no
part in training, in feature engineering, or in choosing a model.

> **A note on currency.** MEPS records **US** healthcare spending in US dollars,
> and the models are trained and evaluated in dollars. Every money figure below is
> converted for display at **1 USD = {USD_TO_INR:.0f} INR**, set in `ml/config.py`.
>
> Converting a US cost into rupees does **not** turn it into an Indian cost. The
> same treatment costs several times more in the US than in India, so read these
> as *"what this care costs in the American system, shown in rupees"* — a familiar
> unit, not a prediction of an Indian hospital bill. Re-training on Indian claims
> data would be a different project, and the pipeline here would transfer to it.

---

## 4.1 How validation was set up

| Choice | Value | Why |
|---|---|---|
| Test split | {D['testSize']:.0%} ({D['nTest']:,} people) | Touched exactly once, at the very end |
| Cross-validation | {D['nFolds']}-fold stratified | Used for all model selection, on the training split only |
| Random seed | {D['seed']} | Fixed everywhere, so the run reproduces |
| Stratification | By risk tier | Keeps tier balance identical across splits |
| Tier cut points | Learned on train only | So the test set does not even influence the labels |
| Uncertainty | 400-sample bootstrap | Reported as 95% intervals on headline metrics |

**Leakage guard.** Total expenditure defines the tier label, so it can never be an
input. The training script asserts this and aborts if the target or the survey
weight ever appears in the feature matrix.

---

## 4.2 Task B — risk stratification

Three equal-sized tiers, so a random guesser scores about 0.333 macro-F1.

- **Low** — under {money(CLS['tierCuts'][0])} a year
- **Medium** — {money(CLS['tierCuts'][0])} to {money(CLS['tierCuts'][1])}
- **High** — over {money(CLS['tierCuts'][1])}

### Model comparison

{table(["Model", "CV macro-F1", "Test macro-F1", "Accuracy", "Balanced acc.", "High recall", "High precision", "Macro ROC-AUC"],
       [[m["model"] + (" **(deployed)**" if m["model"] == CLS["deployed"] else ""),
         f"{m['cvMacroF1']:.4f} ± {m['cvMacroF1Std']:.4f}", n3(m["macroF1"]),
         n3(m["accuracy"]), n3(m["balancedAccuracy"]), n3(m["highRecall"]),
         n3(m["highPrecision"]), n3(m["macroRocAuc"])] for m in CLS["models"]])}

The three real models sit within about 0.01 macro-F1 of one another, comfortably
inside the fold-to-fold spread. When the scores tie, the tiebreak has to be
something other than the third decimal: the logistic regression is deployed
because it serialises to 14 KB, runs in the browser, and produces **exact**
per-person attributions instead of approximate ones.

**95% bootstrap interval on the deployed macro-F1:**
[{n3(CLS['macroF1CI95'][0])}, {n3(CLS['macroF1CI95'][1])}]

### Precision and recall per tier

{table(["Tier", "Precision", "Recall", "F1", "Support"],
       [[t, n3(CLS["perClass"][t]["precision"]), n3(CLS["perClass"][t]["recall"]),
         n3(CLS["perClass"][t]["f1"]), CLS["perClass"][t]["support"]] for t in TIERS])}

*Precision* — when the model says this tier, how often it is right.
*Recall* — of everyone truly in this tier, how many it found.

The **Medium** tier is the weakest, and that is structural rather than a tuning
failure: it is bounded by two arbitrary cut points with genuinely similar people
on either side. A person spending {money(CLS['tierCuts'][0] - 50)} and one spending
{money(CLS['tierCuts'][0] + 50)} are indistinguishable in every way that matters,
yet the labels put them in different classes. Low and High are separated by real
differences, and the model finds them.

### Confusion matrix

![Confusion matrix]({figs['confusion']})

{table(["", *[f"Predicted {t}" for t in TIERS]],
       [[f"**Actually {TIERS[i]}**", *CLS["confusionMatrix"][i]] for i in range(3)])}

The two errors are not equally expensive. A Low person called High is a customer
who walks to a competitor. A High person called Low is an unpriced loss on the
books. The second is the one to minimise.

### Precision–recall curves

![Precision-recall curves]({figs['pr']})

{table(["Tier", "Average precision", "ROC-AUC", "Prevalence"],
       [[t, n3(CLS["prCurves"][t]["ap"]), n3(CLS["prCurves"][t]["rocAuc"]),
         f"{CLS['prCurves'][t]['prevalence']:.3f}"] for t in TIERS])}

### Choosing the High-tier threshold

Missing a high-cost person is the costliest error, so the decision threshold is a
business choice rather than a default. Lowering it below 0.5 buys recall with
precision:

{table(["Threshold on P(High)", "Recall", "Precision", "Share of people flagged"],
       [[t["threshold"], n3(t["recall"]), n3(t["precision"]), f"{t['flaggedShare']:.1%}"]
        for t in CLS["highTierThresholds"]])}

---

## 4.3 Task A — expected annual cost

{table(["Model", "MAE", "RMSE", "R²", "Median error", "Mean predicted"],
       [[name, money(v["mae"]), money(v["rmse"]), f"{v['r2']:.4f}", money(v["medae"]),
         money(v["meanPredicted"])]
        for name, v in [
            (f"**Tweedie GLM (deployed, power={REG['tweediePower']})**", REG["tweedieGlm"]),
            ("XGBoost, Tweedie objective", REG["xgboostTweedie"]),
            ("Two-part log, mean-calibrated", REG["twoPartCalibrated"]),
            ("OLS on raw dollars", REG["olsRawDollars"]),
            ("Predict the training mean", REG["meanBaseline"]),
            ("Single-stage log + Duan smearing", REG["singleStageLogSmearing"]),
        ]])}

Actual mean cost in the test set: **{money(REG['tweedieGlm']['meanActual'])}**.
95% bootstrap interval on the deployed MAE:
[{money(REG['tweedieGlm']['maeCI95'][0])}, {money(REG['tweedieGlm']['maeCI95'][1])}].

### Reading these numbers honestly

R² of about {REG['tweedieGlm']['r2']:.2f} is low, and no amount of tuning fixes it.
Individual annual medical spending is close to unpredictable from demographics,
because most of the variance comes from *events* — an accident, a new diagnosis —
that no survey question can anticipate. This is a well-known ceiling in health
economics, not a defect in this pipeline. The model earns its place by ranking
groups, which is what pricing actually needs.

### The negative result worth reporting

The brief prescribes training on log(charges) and back-transforming with Duan's
smearing estimator. That was implemented, and it **failed**: MAE
{money(REG['singleStageLogSmearing']['mae'])}, R²
{REG['singleStageLogSmearing']['r2']:.2f} — far worse than guessing the average for
everybody.

The cause is that smearing multiplies by the mean of exp(residual). With residuals
this long-tailed, that mean is dominated by a handful of enormous ones: the
out-of-fold factor came out at **{M['part2']['smearing']:.2f}**, so every prediction
was inflated roughly threefold.

Two fixes were tried:

1. **Mean-preserving calibration.** One scalar, fitted out-of-fold, so total
   predicted spend equals total observed spend. Factor: {REG['calibrationFactor']:.3f}.
   This rescues the two-part model to MAE {money(REG['twoPartCalibrated']['mae'])}.
2. **Stop back-transforming at all.** A Tweedie distribution with power between 1
   and 2 is a compound Poisson–Gamma: an atom of probability at exactly zero plus
   a continuous right-skewed part. That is a literal description of annual medical
   spending, which is why general insurers price on it. With a log link it models
   expected dollars directly. The power was chosen by cross-validated MAE:
   **{REG['tweediePower']}**.

The second fix won and is deployed.

### Decile lift — the metric that matters for pricing

![Decile lift]({figs['lift']})

{table(["Decile", "Predicted mean", "Actual mean", "People"],
       [[l["decile"], money(l["predictedMean"]), money(l["actualMean"]), l["n"]]
        for l in REG["decileLift"]])}

Sort everyone by predicted cost and cut into ten equal groups: the top tenth really
did spend **{REG['topBottomDecileRatio']:.1f}×** what the bottom tenth spent. A model
with R² of {REG['tweedieGlm']['r2']:.2f} that separates groups this cleanly is useful
for pricing a pool, even though it cannot tell any individual what they will spend.

---

## 4.4 Two-part model, Part 1 — does this person spend anything?

{M['part1']['positiveRate']:.1%} of held-out adults had some spending during the year.

{table(["Model", "CV ROC-AUC", "Test ROC-AUC", "PR-AUC", "Accuracy", "F1", "Brier"],
       [[m["model"] + (" **(deployed)**" if m["model"] == M["part1"]["deployed"] else ""),
         f"{m['cvRocAuc']:.4f} ± {m['cvRocAucStd']:.4f}", n3(m["rocAuc"]),
         n3(m["prAuc"]), n3(m["accuracy"]), n3(m["f1"]), n3(m["brier"])]
        for m in M["part1"]["models"]])}

![ROC and calibration]({figs['roc']})

The **Brier score** is included because the two-part model multiplies by this
probability. A classifier that ranks well but is badly calibrated would still
corrupt the dollar figure downstream, and ROC-AUC alone would not reveal it.

### Part 2 — how much, given that they spend anything?

Fitted on {M['part2']['nSpendersTrain']:,} training spenders, evaluated on
{M['part2']['nSpendersTest']:,} test spenders, in log space.

{table(["Model", "CV R² (log)", "Test R² (log)", "MAE (log)", "RMSE (log)"],
       [[m["model"] + (" **(deployed)**" if m["model"] == M["part2"]["deployed"] else ""),
         f"{m['cvR2Log']:.4f} ± {m['cvR2LogStd']:.4f}", n3(m["r2Log"]),
         n3(m["maeLog"]), n3(m["rmseLog"])] for m in M["part2"]["models"]])}

Conditional on spending anything at all, the model explains about
{max(m['r2Log'] for m in M['part2']['models']):.0%} of the variation in log-spend —
much better than the dollar-scale R², because the log scale removes the tail that
dominates everything else.

---

## 4.5 Explainability

{EXP['method']}

![Feature importance]({figs['importance']})

{table(["Source feature", "Mean |contribution|", "TreeSHAP cross-check"],
       [[r["feature"], f"{r['meanAbsContribution']:.4f}", f"{r['treeShap']:.4f}"]
        for r in EXP["groupedImportance"][:14]])}

**Why these attributions are exact.** SHAP for a tree ensemble is an estimate. For
a linear model it is closed-form: the contribution of feature *j* is its
coefficient times how far that person's standardised answer sits from a fixed
reference person, and the contributions sum to the person's score minus the
reference score with nothing left over. The training run checks that identity
numerically on every run — largest discrepancy across the whole test set:
**{EXP['additivityMaxError']:.2e}**, which is floating-point rounding.

**Why a reference person.** The one-hot dummies for region and the other
categorical answers are collinear, so their raw coefficients are identified only
up to a shared constant, which L2 regularisation splits arbitrarily. Differencing
against a fixed reference cancels that constant exactly — and it makes the
explanation legible: *compared with a typical adult, here is what your answers
did.*

**Agreement with TreeSHAP:** Spearman rank correlation
{EXP['rankAgreementWithTreeShap']:.3f} between the exact linear attribution and
TreeSHAP on the XGBoost challenger. The two model families broadly read the same
signal, which is reassurance that the deployed model is not leaning on an artefact.

---

## 4.6 Fairness audit

Race is **not** an input to any model here. That is necessary but not sufficient —
a model can still perform unevenly by leaning on correlated features — so the
finished model is measured separately within each group.

![Fairness by race]({figs['fairness']})

"""

for key, title in (("race", "Race and ethnicity"), ("sex", "Sex"),
                   ("povertyCategory", "Income band")):
    md += f"\n### {title}\n\n" + table(
        ["Group", "People", "High-tier recall", "Macro-F1", "Mean cost error",
         "Actual mean cost", "Predicted mean cost"],
        [[g["group"], g["n"], n3(g["highRecall"]), n3(g["macroF1"]), money(g["mae"]),
          money(g["meanActual"]), money(g["meanPredicted"])]
         for g in M["fairness"][key]]) + "\n"

rr = [g["highRecall"] for g in M["fairness"]["race"]]
md += f"""
### What the audit found

High-tier recall ranges from **{min(rr):.2f}** to **{max(rr):.2f}** across race
groups. The model finds high-cost people in some groups noticeably more reliably
than in others. Some groups have only a couple of hundred people in the test set,
so their individual figures are unstable — but the spread is wide enough that it
should not be waved away.

Dollar errors differ by group too, largely because mean spending itself differs: a
group that spends more gives the model more room to be wrong. Before anything like
this went near a real pricing decision, these gaps would need to be the subject of
the conversation rather than a footnote to it.

---

## 4.7 Summary

| Question | Answer |
|---|---|
| Can we predict an individual's annual cost? | Only weakly. R² {REG['tweedieGlm']['r2']:.2f}, typical miss {money(REG['tweedieGlm']['mae'])}. |
| Can we rank people by cost? | Yes. The top predicted decile really costs {REG['topBottomDecileRatio']:.1f}× the bottom. |
| Can we sort people into risk tiers? | Reasonably. Macro-F1 {n3(dep_cls['macroF1'])} against 0.333 for chance. |
| Do we catch the high-cost people? | {CLS['perClass']['High']['recall']:.0%} of them, at {CLS['perClass']['High']['precision']:.0%} precision — tunable via the threshold table above. |
| Can we explain a decision? | Exactly, to {EXP['additivityMaxError']:.0e} of floating-point error. |
| Does it work equally well for everyone? | No. See the fairness audit above. |
"""


out = ROOT / "docs" / "04-RESULTS.md"
out.write_text(md)
print(f"Wrote {out} ({len(md):,} chars) and {len(figs)} figures in {DOC_ASSETS}")
