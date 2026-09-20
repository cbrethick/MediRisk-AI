"""
M2 + M3 -- model training, model selection and validation.

Two modelling tasks are solved on the same encoded feature matrix:

  TASK A  Charge prediction (regression).
          TOTEXP24 is zero for 13.4% of adults and violently right-skewed for
          the rest (skew ~ 9.8), so a single least-squares model on dollars is
          the wrong shape. We fit the standard health-economics TWO-PART MODEL:
              Part 1  P(spend > 0)              -- binary classifier
              Part 2  E[log1p(spend) | spend>0] -- regressor
          and recombine as  E[cost] = P(spend>0) x exp(mu + smearing) - 1,
          using Duan's smearing estimator so the back-transform is unbiased.
          A single-stage regressor on log1p(spend) is also fitted as a
          reference point, so the report can show the two-part model earns
          its extra complexity.

  TASK B  Risk stratification (3-class classification) into Low / Medium /
          High. MEPS has no tier column, so we create one -- Option A from the
          brief, tertiles of TOTEXP24. The cut points are computed on the
          TRAINING SPLIT ONLY and then frozen.

          *** TARGET LEAKAGE GUARD ***
          TOTEXP24 defines the label, so it can never be an input. The feature
          matrix is built from FEATURE_COLUMNS only, and an assertion below
          fails loudly if the target or the survey weight ever appears in it.

Every model is compared against a baseline, scored with stratified 5-fold
cross-validation on the training split, and then measured exactly once on a
held-out 20% test split.

    python3 ml/train.py
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, TweedieRegressor)
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, classification_report, confusion_matrix, f1_score,
    mean_absolute_error, precision_recall_curve, r2_score, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, cross_val_score, train_test_split)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from config import (
    ARTIFACTS, FEATURE_COLUMNS, N_FOLDS, RF_COMMON, SEED, TARGET, TEST_SIZE,
    TIER_NAMES, TIER_QUANTILES, WEIGHT, XGB_COMMON,
)
from data_prep import build_dataset
from encoder import MepsEncoder

warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED)


# =============================================================================
# helpers
# =============================================================================
def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def dollar_metrics(y_true, y_pred) -> dict:
    """Regression metrics in the units a pricing analyst actually cares about."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        # Median absolute error is reported alongside MAE because a handful of
        # million-dollar patients dominate the mean and hide typical accuracy.
        "medae": float(np.median(np.abs(y_true - y_pred))),
        "meanPredicted": float(y_pred.mean()),
        "meanActual": float(y_true.mean()),
    }


def log_metrics(y_true_log, y_pred_log) -> dict:
    return {
        "maeLog": float(mean_absolute_error(y_true_log, y_pred_log)),
        "rmseLog": rmse(y_true_log, y_pred_log),
        "r2Log": float(r2_score(y_true_log, y_pred_log)),
    }


def bootstrap_ci(y_true, y_pred, fn, n=400, alpha=0.05) -> list[float]:
    """Percentile bootstrap confidence interval for any scalar metric."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    idx = np.arange(len(y_true))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        try:
            vals.append(fn(y_true[s], y_pred[s]))
        except Exception:
            pass
    return [float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))]


# =============================================================================
# 1. Data, split, tier labels
# =============================================================================
print("=" * 78)
print("STEP 1  Load and split")
print("=" * 78)
data = build_dataset(verbose=False)
print(f"Adults with a 2024 expenditure record: {len(data):,}")

assert TARGET not in FEATURE_COLUMNS and WEIGHT not in FEATURE_COLUMNS, \
    "LEAKAGE: the target or survey weight is in the feature list."

y_cost = data[TARGET].to_numpy(float)

# Provisional tiers on the full data are used ONLY to stratify the split so the
# test set has the same tier balance; the tiers that are actually modelled are
# re-derived from the training split below.
prov = pd.qcut(y_cost, [0, *TIER_QUANTILES, 1], labels=[0, 1, 2])

train_df, test_df = train_test_split(
    data, test_size=TEST_SIZE, random_state=SEED, stratify=prov
)
print(f"Train: {len(train_df):,}   Test (held out): {len(test_df):,}")

# ---- frozen tier cut points, learned on TRAIN ONLY --------------------------
cuts = np.quantile(train_df[TARGET].to_numpy(float), TIER_QUANTILES)
print(f"Tier cut points from training split: Low < ${cuts[0]:,.0f} "
      f"<= Medium < ${cuts[1]:,.0f} <= High")


def to_tier(cost: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(cost, float), cuts, right=False)


y_tier_tr, y_tier_te = to_tier(train_df[TARGET]), to_tier(test_df[TARGET])

# ---- encode -----------------------------------------------------------------
enc = MepsEncoder().fit(train_df[FEATURE_COLUMNS])
X_tr = enc.transform(train_df[FEATURE_COLUMNS])
X_te = enc.transform(test_df[FEATURE_COLUMNS])
feat_names = enc.feature_names_
print(f"Encoded matrix: {X_tr.shape[1]} columns "
      f"({len(FEATURE_COLUMNS)} source features + missingness indicators + one-hots)")

# One shared standardiser for every linear model (trees ignore it).
scaler_c = StandardScaler().fit(X_tr)

cost_tr, cost_te = train_df[TARGET].to_numpy(float), test_df[TARGET].to_numpy(float)
any_tr, any_te = (cost_tr > 0).astype(int), (cost_te > 0).astype(int)

results: dict = {
    "generatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
    "dataset": {
        "source": "MEPS HC-256, 2024 Full Year Consolidated (AHRQ)",
        "rowsRaw": 19140,
        "rowsAdults": int(len(data)),
        "nTrain": int(len(train_df)),
        "nTest": int(len(test_df)),
        "nFeaturesSource": len(FEATURE_COLUMNS),
        "nFeaturesEncoded": int(X_tr.shape[1]),
        "zeroSpendShare": float((cost_tr == 0).mean()),
        "targetSkew": float(pd.Series(y_cost).skew()),
        "tierCuts": [float(c) for c in cuts],
        "seed": SEED,
        "testSize": TEST_SIZE,
        "nFolds": N_FOLDS,
    },
}

# =============================================================================
# 2. TASK A -- Part 1: does this person spend anything at all?
# =============================================================================
print("\n" + "=" * 78)
print("STEP 2  Two-part model, Part 1: P(any spending)")
print("=" * 78)

part1_candidates = {
    "Majority baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, random_state=SEED),
    "Random Forest": RandomForestClassifier(**RF_COMMON),
    "XGBoost": XGBClassifier(eval_metric="logloss", **XGB_COMMON),
}

scaler_p1 = StandardScaler().fit(X_tr)
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
part1_rows = []
part1_fitted = {}

for name, model in part1_candidates.items():
    Xt = scaler_p1.transform(X_tr) if "Logistic" in name else X_tr
    Xv = scaler_p1.transform(X_te) if "Logistic" in name else X_te
    cv_auc = cross_val_score(model, Xt, any_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
    model.fit(Xt, any_tr)
    p = model.predict_proba(Xv)[:, 1]
    pred = (p >= 0.5).astype(int)
    row = {
        "model": name,
        "cvRocAuc": float(cv_auc.mean()), "cvRocAucStd": float(cv_auc.std()),
        "rocAuc": float(roc_auc_score(any_te, p)),
        "prAuc": float(average_precision_score(any_te, p)),
        "accuracy": float(accuracy_score(any_te, pred)),
        "f1": float(f1_score(any_te, pred)),
        "brier": float(brier_score_loss(any_te, p)),
    }
    part1_rows.append(row)
    part1_fitted[name] = (model, p)
    print(f"  {name:<22s} CV AUC {row['cvRocAuc']:.4f}+-{row['cvRocAucStd']:.4f}  "
          f"test AUC {row['rocAuc']:.4f}  PR-AUC {row['prAuc']:.4f}  "
          f"Brier {row['brier']:.4f}")

best_p1_name = max(
    (r for r in part1_rows if r["model"] != "Majority baseline"),
    key=lambda r: r["cvRocAuc"],
)["model"]

# ---------------------------------------------------------------------------
# DEPLOYMENT CHOICE (see docs/MODELS.md).
# The tree models win by a hair, but the gap is a few thousandths of AUC while
# the cost is a multi-megabyte model that cannot ship to a browser and cannot
# be explained exactly. We therefore deploy the regularised LINEAR model, which
# (a) is within a whisker of the best CV score, (b) serialises to ~10 KB, and
# (c) yields EXACT per-feature contributions -- for a linear model the SHAP
# value of feature j is exactly coef_j * (x_j - mean_j) / std_j, no sampling
# and no approximation. Every number below is measured on that deployed model,
# so the report describes the thing that actually runs.
DEPLOYED_P1 = "Logistic Regression"
part1_model, p_any_te = part1_fitted[DEPLOYED_P1]
print(f"  -> best by CV: {best_p1_name}   |   deployed: {DEPLOYED_P1}")

fpr, tpr, _ = roc_curve(any_te, p_any_te)
prec1, rec1, _ = precision_recall_curve(any_te, p_any_te)
step = max(1, len(fpr) // 120)
results["part1"] = {
    "description": "Binary classifier for whether a person incurs ANY 2024 spending.",
    "positiveRate": float(any_te.mean()),
    "bestByCv": best_p1_name,
    "deployed": DEPLOYED_P1,
    "models": part1_rows,
    "roc": {"fpr": fpr[::step].round(4).tolist(), "tpr": tpr[::step].round(4).tolist()},
    "pr": {"recall": rec1[::max(1, len(rec1)//120)].round(4).tolist(),
           "precision": prec1[::max(1, len(prec1)//120)].round(4).tolist()},
    "calibration": [],
}
# reliability curve (10 equal-width probability bins)
bins = np.linspace(0, 1, 11)
bid = np.clip(np.digitize(p_any_te, bins) - 1, 0, 9)
for b in range(10):
    m = bid == b
    if m.sum() >= 20:
        results["part1"]["calibration"].append(
            {"predicted": float(p_any_te[m].mean()),
             "observed": float(any_te[m].mean()), "n": int(m.sum())}
        )

# =============================================================================
# 3. TASK A -- Part 2: how much, given that they spend anything?
# =============================================================================
print("\n" + "=" * 78)
print("STEP 3  Two-part model, Part 2: E[log1p(cost) | cost > 0]")
print("=" * 78)

spend_tr, spend_te = cost_tr > 0, cost_te > 0
Xs_tr, Xs_te = X_tr[spend_tr], X_te[spend_te]
ylog_tr = np.log1p(cost_tr[spend_tr])
ylog_te = np.log1p(cost_te[spend_te])
print(f"  Spenders: {spend_tr.sum():,} train / {spend_te.sum():,} test")

scaler_p2 = StandardScaler().fit(Xs_tr)
part2_candidates = {
    "Mean baseline": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=SEED),
    "Random Forest": RandomForestRegressor(**RF_COMMON),
    "XGBoost": XGBRegressor(**XGB_COMMON),
}
part2_rows, part2_fitted = [], {}
for name, model in part2_candidates.items():
    lin = name in ("Linear Regression", "Ridge Regression")
    Xt = scaler_p2.transform(Xs_tr) if lin else Xs_tr
    Xv = scaler_p2.transform(Xs_te) if lin else Xs_te
    cv_r2 = cross_val_score(model, Xt, ylog_tr, cv=N_FOLDS, scoring="r2", n_jobs=-1)
    model.fit(Xt, ylog_tr)
    pl = model.predict(Xv)
    row = {"model": name, "cvR2Log": float(cv_r2.mean()),
           "cvR2LogStd": float(cv_r2.std()), **log_metrics(ylog_te, pl)}
    part2_rows.append(row)
    part2_fitted[name] = (model, pl)
    print(f"  {name:<22s} CV R2(log) {row['cvR2Log']:.4f}+-{row['cvR2LogStd']:.4f}  "
          f"test R2(log) {row['r2Log']:.4f}  MAE(log) {row['maeLog']:.4f}")

best_p2_name = max(
    (r for r in part2_rows if r["model"] != "Mean baseline"), key=lambda r: r["cvR2Log"]
)["model"]
DEPLOYED_P2 = "Ridge Regression"
part2_model, pred_log_te = part2_fitted[DEPLOYED_P2]
lin2 = True
print(f"  -> best by CV: {best_p2_name}   |   deployed: {DEPLOYED_P2}")

# Duan's smearing estimator. exp(mean of logs) is biased low, so we multiply by
# the mean of exp(residuals). CRITICAL DETAIL: the residuals must be OUT-OF-FOLD.
# Using in-sample residuals from a model that has partly memorised the training
# set understates the spread, and for a boosted model it produces a wildly wrong
# factor. We therefore take cross-validated predictions to compute it.
oof_log = cross_val_predict(
    type(part2_model)(**part2_model.get_params()),
    scaler_p2.transform(Xs_tr), ylog_tr, cv=N_FOLDS, n_jobs=-1,
)
smearing = float(np.mean(np.exp(ylog_tr - oof_log)))
print(f"  Duan smearing factor (out-of-fold): {smearing:.4f}")

results["part2"] = {
    "description": "Regressor on log1p(cost) among people with cost > 0.",
    "nSpendersTrain": int(spend_tr.sum()), "nSpendersTest": int(spend_te.sum()),
    "bestByCv": best_p2_name, "deployed": DEPLOYED_P2,
    "smearing": smearing, "smearingNote": "computed from out-of-fold residuals",
    "models": part2_rows,
}

# =============================================================================
# 4. TASK A -- expected annual cost, and why we changed the back-transform
# =============================================================================
print("\n" + "=" * 78)
print("STEP 4  Expected annual cost in dollars")
print("=" * 78)

# --- 4a. the brief's recipe: two-part model, recombined ----------------------
# Duan's smearing estimator is the textbook back-transform, and we implemented
# it -- but it FAILS on this data. exp() of a long-tailed residual distribution
# is dominated by a handful of enormous residuals, so the correction factor
# blows up (~2.5x out-of-fold) and the model systematically over-predicts.
# We therefore report smearing for completeness but deploy a mean-preserving
# calibration instead: a single scalar k fitted on OUT-OF-FOLD training
# predictions so that total predicted spend equals total observed spend. That
# is the property an insurer actually needs -- the book has to balance.
oof_p_any = cross_val_predict(
    LogisticRegression(max_iter=2000, C=1.0, random_state=SEED),
    scaler_p1.transform(X_tr), any_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
oof_mu_all = cross_val_predict(
    Ridge(alpha=1.0, random_state=SEED),
    scaler_p2.transform(X_tr), np.log1p(cost_tr), cv=N_FOLDS, n_jobs=-1)
oof_raw = oof_p_any * np.expm1(oof_mu_all)
k_cal = float(cost_tr.mean() / oof_raw.mean())
print(f"  Duan smearing factor (out-of-fold, unstable) : {smearing:.4f}")
print(f"  Mean-preserving calibration factor k         : {k_cal:.4f}")

cond_cost = np.expm1(part2_model.predict(scaler_p2.transform(X_te)))
two_part_pred = p_any_te * cond_cost * k_cal
two_part = dollar_metrics(cost_te, two_part_pred)
two_part["maeCI95"] = bootstrap_ci(cost_te, two_part_pred, mean_absolute_error)
print(f"\n  Two-part log (calibrated)  MAE ${two_part['mae']:,.0f}  "
      f"RMSE ${two_part['rmse']:,.0f}  R2 {two_part['r2']:.4f}")

# --- 4b. Tweedie GLM: the actuarial standard --------------------------------
# A Tweedie distribution with 1 < p < 2 is a compound Poisson-Gamma: an atom of
# probability at exactly zero plus a continuous right-skewed part. That is a
# literal description of annual medical spending, which is why general insurers
# price on it. With a log link it models E[cost] on the dollar scale directly,
# so there is NO back-transform and no smearing problem at all.
print("\n  Selecting the Tweedie variance power by cross-validated MAE:")
best_pw, best_mae = None, np.inf
for pw in (1.1, 1.3, 1.5, 1.7, 1.9):
    sc_mae = -cross_val_score(
        TweedieRegressor(power=pw, link="log", alpha=1e-3, max_iter=3000),
        scaler_c.transform(X_tr), cost_tr, cv=N_FOLDS,
        scoring="neg_mean_absolute_error", n_jobs=-1).mean()
    print(f"    power={pw}  CV MAE ${sc_mae:,.0f}")
    if sc_mae < best_mae:
        best_pw, best_mae = pw, sc_mae
print(f"  -> Tweedie power = {best_pw}")

tweedie = TweedieRegressor(power=best_pw, link="log", alpha=1e-3,
                           max_iter=3000).fit(scaler_c.transform(X_tr), cost_tr)
tweedie_pred = tweedie.predict(scaler_c.transform(X_te))
tweedie_m = dollar_metrics(cost_te, tweedie_pred)
tweedie_m["maeCI95"] = bootstrap_ci(cost_te, tweedie_pred, mean_absolute_error)
tweedie_m["cvMae"] = float(best_mae)
print(f"  Tweedie GLM (DEPLOYED)     MAE ${tweedie_m['mae']:,.0f}  "
      f"RMSE ${tweedie_m['rmse']:,.0f}  R2 {tweedie_m['r2']:.4f}  "
      f"MedAE ${tweedie_m['medae']:,.0f}")

# --- 4c. challengers and baselines ------------------------------------------
xgb_tw = XGBRegressor(objective="reg:tweedie", tweedie_variance_power=best_pw,
                      **XGB_COMMON).fit(X_tr, cost_tr)
xgb_tw_m = dollar_metrics(cost_te, xgb_tw.predict(X_te))
print(f"  XGBoost Tweedie            MAE ${xgb_tw_m['mae']:,.0f}  "
      f"RMSE ${xgb_tw_m['rmse']:,.0f}  R2 {xgb_tw_m['r2']:.4f}")

single = XGBRegressor(**XGB_COMMON)
single_oof = cross_val_predict(XGBRegressor(**XGB_COMMON), X_tr, np.log1p(cost_tr),
                               cv=N_FOLDS, n_jobs=-1)
single_smear = float(np.mean(np.exp(np.log1p(cost_tr) - single_oof)))
single.fit(X_tr, np.log1p(cost_tr))
single_m = dollar_metrics(cost_te, np.expm1(single.predict(X_te)) * single_smear)
print(f"  Single-stage log + smearing MAE ${single_m['mae']:,.0f}  "
      f"RMSE ${single_m['rmse']:,.0f}  R2 {single_m['r2']:.4f}   "
      f"<- smearing blow-up, shown as a cautionary result")

raw_lin = LinearRegression().fit(scaler_c.transform(X_tr), cost_tr)
raw_m = dollar_metrics(cost_te, raw_lin.predict(scaler_c.transform(X_te)))
print(f"  OLS on raw dollars         MAE ${raw_m['mae']:,.0f}  "
      f"RMSE ${raw_m['rmse']:,.0f}  R2 {raw_m['r2']:.4f}")

mean_m = dollar_metrics(cost_te, np.full_like(cost_te, cost_tr.mean()))
print(f"  Train-mean baseline        MAE ${mean_m['mae']:,.0f}  "
      f"RMSE ${mean_m['rmse']:,.0f}  R2 {mean_m['r2']:.4f}")

cost_pred = tweedie_pred          # the deployed prediction, used from here on

# --- 4d. decile lift ---------------------------------------------------------
# R^2 is a poor summary for a task this noisy. What a pricing team actually
# needs is rank ordering: if we sort people by predicted cost, do the top
# deciles really cost more? Decile lift answers exactly that.
order = np.argsort(cost_pred)
lift = [{"decile": i + 1,
         "predictedMean": float(cost_pred[d].mean()),
         "actualMean": float(cost_te[d].mean()), "n": int(len(d))}
        for i, d in enumerate(np.array_split(order, 10))]
print("\n  Decile lift (sorted by predicted cost):")
for r in lift:
    print(f"    decile {r['decile']:>2d}  predicted ${r['predictedMean']:>8,.0f}  "
          f"actual ${r['actualMean']:>9,.0f}")
top_bottom = lift[-1]["actualMean"] / max(lift[0]["actualMean"], 1)
print(f"  Top decile costs {top_bottom:.1f}x the bottom decile in reality.")

results["regression"] = {
    "deployed": f"Tweedie GLM (log link, power={best_pw})",
    "tweediePower": best_pw,
    "calibrationFactor": k_cal,
    "tweedieGlm": tweedie_m, "twoPartCalibrated": two_part,
    "xgboostTweedie": xgb_tw_m, "singleStageLogSmearing": single_m,
    "olsRawDollars": raw_m, "meanBaseline": mean_m,
    "decileLift": lift, "topBottomDecileRatio": float(top_bottom),
    "scatter": [{"actual": float(a), "predicted": float(pp)}
                for a, pp in zip(cost_te[::7], cost_pred[::7])][:900],
}

# =============================================================================
# 5. TASK B -- risk tier classification
# =============================================================================
print("\n" + "=" * 78)
print("STEP 5  Risk stratification: Low / Medium / High")
print("=" * 78)
print(f"  Train tier counts: {np.bincount(y_tier_tr).tolist()}")

tier_candidates = {
    "Stratified baseline": DummyClassifier(strategy="stratified", random_state=SEED),
    "Logistic Regression": LogisticRegression(
        max_iter=3000, C=1.0, class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(class_weight="balanced_subsample", **RF_COMMON),
    "XGBoost": XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss", **XGB_COMMON),
}
tier_rows, tier_fitted = [], {}
for name, model in tier_candidates.items():
    lin = "Logistic" in name
    Xt = scaler_c.transform(X_tr) if lin else X_tr
    Xv = scaler_c.transform(X_te) if lin else X_te
    cv_f1 = cross_val_score(model, Xt, y_tier_tr, cv=cv, scoring="f1_macro", n_jobs=-1)
    model.fit(Xt, y_tier_tr)
    proba = model.predict_proba(Xv)
    pred = proba.argmax(1)
    rep = classification_report(y_tier_te, pred, target_names=TIER_NAMES,
                                output_dict=True, zero_division=0)
    row = {
        "model": name,
        "cvMacroF1": float(cv_f1.mean()), "cvMacroF1Std": float(cv_f1.std()),
        "macroF1": float(f1_score(y_tier_te, pred, average="macro")),
        "accuracy": float(accuracy_score(y_tier_te, pred)),
        "balancedAccuracy": float(balanced_accuracy_score(y_tier_te, pred)),
        "highRecall": float(rep["High"]["recall"]),
        "highPrecision": float(rep["High"]["precision"]),
        "macroRocAuc": float(roc_auc_score(y_tier_te, proba, multi_class="ovr",
                                           average="macro")),
    }
    tier_rows.append(row)
    tier_fitted[name] = (model, proba, pred)
    print(f"  {name:<22s} CV macroF1 {row['cvMacroF1']:.4f}+-{row['cvMacroF1Std']:.4f}  "
          f"test macroF1 {row['macroF1']:.4f}  High-recall {row['highRecall']:.4f}  "
          f"AUC {row['macroRocAuc']:.4f}")

best_tier_name = max(
    (r for r in tier_rows if r["model"] != "Stratified baseline"),
    key=lambda r: r["cvMacroF1"],
)["model"]
DEPLOYED_TIER = "Logistic Regression"
tier_model, tier_proba, tier_pred = tier_fitted[DEPLOYED_TIER]
print(f"  -> best by CV: {best_tier_name}   |   deployed: {DEPLOYED_TIER}")

cm = confusion_matrix(y_tier_te, tier_pred)
rep = classification_report(y_tier_te, tier_pred, target_names=TIER_NAMES,
                            output_dict=True, zero_division=0)
print("\n  Confusion matrix (rows = actual, cols = predicted)")
print("            " + "".join(f"{n:>10s}" for n in TIER_NAMES))
for i, n in enumerate(TIER_NAMES):
    print(f"  {n:<10s}" + "".join(f"{v:>10d}" for v in cm[i]))
print()
for n in TIER_NAMES:
    r = rep[n]
    print(f"  {n:<8s} precision {r['precision']:.3f}  recall {r['recall']:.3f}  "
          f"F1 {r['f1-score']:.3f}  support {int(r['support'])}")

# per-class one-vs-rest PR curves
pr_curves = {}
for i, n in enumerate(TIER_NAMES):
    yb = (y_tier_te == i).astype(int)
    pr, rc, _ = precision_recall_curve(yb, tier_proba[:, i])
    s = max(1, len(pr) // 120)
    pr_curves[n] = {
        "precision": pr[::s].round(4).tolist(), "recall": rc[::s].round(4).tolist(),
        "ap": float(average_precision_score(yb, tier_proba[:, i])),
        "rocAuc": float(roc_auc_score(yb, tier_proba[:, i])),
        "prevalence": float(yb.mean()),
    }

# Operating point: missing a High-tier person is the costliest error, so we also
# report what happens if we simply lower the threshold on P(High).
thresholds = []
for t in np.arange(0.15, 0.65, 0.05):
    flag = tier_proba[:, 2] >= t
    tp = int(((y_tier_te == 2) & flag).sum())
    fp = int(((y_tier_te != 2) & flag).sum())
    fn = int(((y_tier_te == 2) & ~flag).sum())
    thresholds.append({
        "threshold": round(float(t), 2),
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "flaggedShare": float(flag.mean()),
    })

results["classification"] = {
    "tierNames": TIER_NAMES,
    "tierCuts": [float(c) for c in cuts],
    "bestByCv": best_tier_name,
    "deployed": DEPLOYED_TIER,
    "models": tier_rows,
    "confusionMatrix": cm.tolist(),
    "perClass": {n: {"precision": rep[n]["precision"], "recall": rep[n]["recall"],
                     "f1": rep[n]["f1-score"], "support": int(rep[n]["support"])}
                 for n in TIER_NAMES},
    "macroF1CI95": bootstrap_ci(y_tier_te, tier_pred,
                                lambda a, b: f1_score(a, b, average="macro")),
    "prCurves": pr_curves,
    "highTierThresholds": thresholds,
}

# =============================================================================
# 6. Explainability (M4) -- global importance
# =============================================================================
print("\n" + "=" * 78)
print("STEP 6  Explainability")
print("=" * 78)

import shap  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

# --- (a) EXACT attributions for the deployed linear tier model ---------------
# For a linear/logistic model on standardised inputs, the SHAP value of feature
# j for person i is exactly  coef_j * (x_ij - mean_j) / std_j. There is nothing
# to estimate: the attributions sum precisely to the logit minus its baseline.
# This is why the web app can reproduce the explanation the model actually used
# rather than an approximation of it.
# Attributions are measured RELATIVE TO A REFERENCE PERSON, not relative to a
# feature value of zero. This matters: the one-hot dummies for region (and the
# other categoricals) are collinear, so their raw coefficients are identified
# only up to a constant and L2 regularisation splits that constant arbitrarily.
# Differencing against a fixed reference cancels the constant exactly.
#
# The reference is the "typical adult": the training median of every continuous
# feature and the training mode of every categorical one. The identity
#     sum_j contribution_j = logit(person) - logit(reference)
# then holds exactly, which is what makes the web app's waterfall honest.
ref_raw = pd.DataFrame([{
    **{c: float(np.nanmedian(train_df[c].astype(float))) for c in FEATURE_COLUMNS},
}])
for c in ("sex", "region", "insurance", "poverty_category", "marital_status",
          "smoking_status"):
    ref_raw[c] = float(train_df[c].mode().iloc[0])
z_ref = scaler_c.transform(enc.transform(ref_raw[FEATURE_COLUMNS]))[0]

Xc_te = scaler_c.transform(X_te)
coef_high = tier_model.coef_[2]                     # logit for the High tier
contrib_high = (Xc_te - z_ref) * coef_high          # (n_test, n_features)
exact_imp = np.abs(contrib_high).mean(axis=0)

# Verify the additivity identity rather than asserting it in prose.
recon = contrib_high.sum(axis=1) + float(tier_model.decision_function(
    z_ref.reshape(1, -1))[0][2])
max_err = float(np.max(np.abs(recon - tier_model.decision_function(Xc_te)[:, 2])))
print(f"  Additivity check: max |sum(contributions) + baseline - logit| = {max_err:.2e}")

# --- (b) TreeSHAP on the XGBoost challenger, as a cross-check ---------------
xgb_tier = tier_fitted["XGBoost"][0]
sv = shap.TreeExplainer(xgb_tier).shap_values(X_te[:1500])
arr = np.abs(np.asarray(sv))
tree_imp = arr.mean(axis=(0, 2)) if arr.ndim == 3 else arr.mean(axis=0)

# --- (c) permutation importance on the deployed model, model-agnostic -------
perm = permutation_importance(
    tier_model, Xc_te, y_tier_te, n_repeats=8, random_state=SEED,
    scoring="f1_macro", n_jobs=-1,
)

order = np.argsort(-exact_imp)[:24]
global_imp = [{
    "feature": feat_names[i],
    "meanAbsContribution": float(exact_imp[i]),
    "signedMeanContribution": float(contrib_high[:, i].mean()),
    "coefficientHigh": float(coef_high[i]),
    "treeShap": float(tree_imp[i]),
    "permutationDrop": float(perm.importances_mean[i]),
} for i in order]

# One-hot dummies are split across many columns, which makes a per-column chart
# misleading. Roll every encoded column back up to the source feature a person
# actually answers in the form.
def source_of(name: str) -> str:
    return name.split("=")[0].split("__isna")[0]

agg: dict[str, float] = {}
agg_tree: dict[str, float] = {}
for i, nm in enumerate(feat_names):
    agg[source_of(nm)] = agg.get(source_of(nm), 0.0) + float(exact_imp[i])
    agg_tree[source_of(nm)] = agg_tree.get(source_of(nm), 0.0) + float(tree_imp[i])
grouped = sorted(
    ({"feature": k, "meanAbsContribution": v, "treeShap": agg_tree.get(k, 0.0)}
     for k, v in agg.items()),
    key=lambda r: -r["meanAbsContribution"])

print("  Importance by source feature (what the user actually enters):")
for r in grouped[:12]:
    print(f"    {r['feature']:<26s} linear {r['meanAbsContribution']:.4f}   "
          f"treeSHAP {r['treeShap']:.4f}")
print()
print("  Top drivers of the High-risk logit (deployed model, exact attribution)")
print(f"    {'feature':<30s}{'|contrib|':>11s}{'coef':>9s}{'treeSHAP':>10s}{'perm':>9s}")
for r in global_imp[:14]:
    print(f"    {r['feature']:<30s}{r['meanAbsContribution']:>11.4f}"
          f"{r['coefficientHigh']:>9.3f}{r['treeShap']:>10.4f}{r['permutationDrop']:>9.4f}")

# Rank agreement between the exact linear attribution and TreeSHAP tells us
# whether the two model families are reading the same signal.
from scipy.stats import spearmanr  # noqa: E402
rank_agreement = float(spearmanr(exact_imp, tree_imp).statistic)
print(f"  Spearman rank agreement, linear attribution vs TreeSHAP: {rank_agreement:.3f}")

results["explainability"] = {
    "method": ("Exact linear SHAP on the deployed logistic model "
               "(coef_j * standardised x_j), cross-checked against TreeSHAP on "
               "the XGBoost challenger and permutation importance."),
    "rankAgreementWithTreeShap": rank_agreement,
    "globalImportance": global_imp,
    "groupedImportance": grouped,
    "additivityMaxError": max_err,
    "baselineLogitHigh": float(tier_model.decision_function(
        z_ref.reshape(1, -1))[0][2]),
}

# =============================================================================
# 7. Fairness audit
# =============================================================================
print("\n" + "=" * 78)
print("STEP 7  Fairness audit (race is NOT a model input; audited anyway)")
print("=" * 78)
race_labels = {1: "Hispanic", 2: "White non-Hispanic", 3: "Black non-Hispanic",
               4: "Asian non-Hispanic", 5: "Other / multiple"}
sex_labels = {1: "Male", 2: "Female"}
pov_labels = {1: "Poor / negative", 2: "Near poor", 3: "Low income",
              4: "Middle income", 5: "High income"}

fairness = {}
for col, labels, key in (("audit_RACETHX", race_labels, "race"),
                         ("audit_SEX", sex_labels, "sex"),
                         ("audit_POVCAT24", pov_labels, "povertyCategory")):
    groups = []
    g = test_df[col].to_numpy()
    for code, label in labels.items():
        m = g == code
        if m.sum() < 60:
            continue
        groups.append({
            "group": label, "n": int(m.sum()),
            "highRecall": float(((tier_pred[m] == 2) & (y_tier_te[m] == 2)).sum()
                                / max((y_tier_te[m] == 2).sum(), 1)),
            "macroF1": float(f1_score(y_tier_te[m], tier_pred[m],
                                      average="macro", zero_division=0)),
            "mae": float(mean_absolute_error(cost_te[m], cost_pred[m])),
            "meanActual": float(cost_te[m].mean()),
            "meanPredicted": float(cost_pred[m].mean()),
        })
    fairness[key] = groups
    print(f"  {key}:")
    for gr in groups:
        print(f"    {gr['group']:<22s} n={gr['n']:<5d} High-recall {gr['highRecall']:.3f}  "
              f"macroF1 {gr['macroF1']:.3f}  MAE ${gr['mae']:,.0f}")
results["fairness"] = fairness

# =============================================================================
# 8. Persist
# =============================================================================
joblib.dump({
    "encoder": enc, "scaler_p1": scaler_p1, "scaler_p2": scaler_p2,
    "scaler_c": scaler_c, "part1": part1_model, "part2": part2_model,
    "tier": tier_model, "tweedie": tweedie, "smearing": smearing,
    "calibration": k_cal, "cuts": cuts,
    "featureNames": feat_names, "part1Linear": "Logistic" in best_p1_name,
    "part2Linear": lin2, "tierLinear": "Logistic" in best_tier_name,
}, ARTIFACTS / "models.joblib")

(ARTIFACTS / "metrics.json").write_text(json.dumps(results, indent=2))
print(f"\nSaved models  -> {ARTIFACTS / 'models.joblib'}")
print(f"Saved metrics -> {ARTIFACTS / 'metrics.json'}")
