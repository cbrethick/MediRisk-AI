# 3 · Models and training parameters

Everything here is set in `ml/config.py` and is the single source of truth for
both the training scripts and this document.

## 3.1 The two tasks

| | Task A | Task B |
|---|---|---|
| Question | What will this person cost in a year? | Which risk tier are they in? |
| Type | Regression | 3-class classification |
| Target | `TOTEXP24` (USD) | Low / Medium / High |
| Selection metric | Cross-validated MAE | Cross-validated macro-F1 |
| Reported as | MAE, RMSE, R², median error, decile lift | macro-F1, per-class precision/recall, confusion matrix, ROC-AUC, PR-AUC |

## 3.2 Creating the risk tier

MEPS has no risk-tier column, so one is constructed. **Option A** from the brief:
tertiles of total expenditure.

| Tier | Boundary (USD) | Boundary (INR at 88) |
|---|---|---|
| Low | under $965 | under ₹84,920 |
| Medium | $965 – $6,398 | ₹84,920 – ₹5.63 L |
| High | over $6,398 | over ₹5.63 L |

Two safeguards, both enforced in code:

1. **The cut points are computed on the training split only** and then frozen, so
   the test set does not influence even the definition of the labels.
2. **Expenditure is never an input to the classifier.** Since expenditure
   *defines* the label, feeding it in would produce near-perfect accuracy that
   means nothing. `train.py` carries an assertion that aborts the run if the
   target or the survey weight ever appears in the feature list.

## 3.3 Shared training parameters

| Parameter | Value | Reason |
|---|---|---|
| `SEED` | 42 | Fixed in every split, model and bootstrap, so runs reproduce |
| `TEST_SIZE` | 0.20 → 3,097 people | Held out, touched once at the end |
| `N_FOLDS` | 5, stratified | All model selection happens here, on the training split |
| `MIN_AGE` | 18 | BMI and smoking are adult-only survey items |
| Stratification | By risk tier | Identical tier balance across train and test |
| Bootstrap | 400 resamples | 95% percentile intervals on headline metrics |

## 3.4 Hyper-parameters

### XGBoost (`XGB_COMMON`) — used for every boosted model

| Parameter | Value | Reason |
|---|---|---|
| `n_estimators` | 400 | |
| `max_depth` | 4 | Shallow. With 60 columns and 12k rows, deeper trees memorise |
| `learning_rate` | 0.05 | Low, paired with many trees |
| `subsample` | 0.85 | Row sampling, for variance reduction |
| `colsample_bytree` | 0.85 | Column sampling, same reason |
| `min_child_weight` | 5 | Stops leaves forming on a handful of outlier spenders |
| `reg_lambda` | 1.5 | L2 on leaf weights |
| `tree_method` | `hist` | |
| `random_state` | 42 | |

### Random Forest (`RF_COMMON`)

| Parameter | Value | Reason |
|---|---|---|
| `n_estimators` | 400 | |
| `max_depth` | 14 | |
| `min_samples_leaf` | 8 | Prevents single-person leaves |
| `class_weight` | `balanced_subsample` | Classification only |

### Linear models

| Model | Parameters |
|---|---|
| Logistic Regression | `C=1.0`, `max_iter=2000–3000`, `class_weight="balanced"` for the 3-class task |
| Ridge | `alpha=1.0` |
| Tweedie GLM | `power` chosen by CV from {1.1, 1.3, 1.5, 1.7, 1.9} → **1.3**, `link="log"`, `alpha=1e-3`, `max_iter=3000` |

All linear models are fitted on standardised features.

## 3.5 Task A — the modelling problem, and four attempts at it

The target is zero for 13.4% of adults and skewed 9.8 to the right for the rest.
Four approaches were built and measured against the same held-out people.

### Attempt 1 — OLS on raw dollars

The naive baseline. Included to show why it is not enough: it is unbiased on
average but predicts negative costs for some people and is dominated by the
handful of million-dollar patients.

### Attempt 2 — the two-part model (the brief's recipe)

The standard health-economics structure:

- **Part 1** — logistic regression for `P(spend > 0)`
- **Part 2** — regression on `log1p(spend)` among people who spent something
- **Recombine** — `E[cost] = P(spend>0) × E[cost | spend>0]`

The brief prescribes Duan's smearing estimator for the back-transform. It was
implemented, and **it failed** — see [04-RESULTS.md §4.3](04-RESULTS.md). Briefly:
smearing multiplies by the mean of `exp(residual)`, which with residuals this
long-tailed is dominated by a few enormous ones, giving a correction factor of
2.96 and roughly threefold over-prediction of everybody.

The fix that works is a **mean-preserving calibration factor**: one scalar, fitted
on out-of-fold training predictions so that total predicted spend equals total
observed spend. That is the property an insurer needs — the book has to balance.

> One detail that matters more than it looks: the calibration factor, and the
> smearing factor it replaced, must be computed from **out-of-fold** residuals. An
> in-sample residual from a model that has partly memorised the training set
> understates the spread, and for a boosted model it produces a wildly wrong
> factor.

### Attempt 3 — single-stage log + smearing

Kept in the results table as a **cautionary result** rather than deleted. It
scores worse than predicting the average for everybody, and showing that is more
useful than hiding it.

### Attempt 4 — Tweedie GLM (deployed)

A Tweedie distribution with power between 1 and 2 is a compound Poisson–Gamma: an
atom of probability at exactly zero, plus a continuous right-skewed part. That is
a literal description of annual medical spending, which is why general insurers
price on it.

With a log link it models `E[cost]` on the dollar scale **directly** — no
back-transform, so no smearing problem can arise. The variance power was selected
by cross-validated MAE.

## 3.6 Task B — classification

Four models compete: a stratified random baseline, logistic regression, random
forest and XGBoost, all with balanced class weights, selected on cross-validated
macro-F1.

Macro-F1 rather than accuracy, because it weights all three tiers equally instead
of letting the easy ones carry the score.

## 3.7 Which model is deployed, and why it is not the winner

The three real classifiers land within about **0.01 macro-F1** of one another —
comfortably inside the fold-to-fold spread. The same is true of the regression
models.

When scores tie, the tiebreak has to be something other than the third decimal.
The **linear models are deployed**:

| Reason | Detail |
|---|---|
| **Size** | 14 KB of coefficients versus multiple megabytes of trees. The whole model ships with the web page. |
| **Exact explanations** | For a linear model, SHAP is closed-form rather than estimated. Contributions sum to the prediction with nothing left over, verified numerically every run to 2.2 × 10⁻¹⁵. |
| **Auditability** | A regulator can read 60 coefficients. Nobody audits 400 trees. |
| **No accuracy sacrifice worth the name** | The gap is inside the noise, and it is reported in full rather than glossed over. |

The tree models are not thrown away. They are trained, measured and reported in
every comparison table, and XGBoost is kept as a TreeSHAP cross-check on the
deployed model's explanations.

### What is actually deployed

| Model | Type | Purpose |
|---|---|---|
| `tierModel` | Multinomial logistic, 3 classes | Risk tier + probabilities |
| `costModel` | Tweedie GLM, log link, power 1.3 | Expected annual cost |
| `anySpendModel` | Logistic | P(any spending at all) |

Exported by `ml/export_web_model.py` to `medical-risk-ai/public/model.json`, which
also runs a parity check: the exported coefficients are re-scored by hand and
compared against scikit-learn. Current agreement — tier probabilities **0.00e+00**,
expected cost **$0.000000**. If that ever drifts, the web app is lying to the user
and the export fails loudly.
