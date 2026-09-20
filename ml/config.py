"""
Central configuration for the Medical Insurance Risk Stratification project.

Everything that a reviewer might want to change (paths, the feature list, the
risk-tier cut points, model hyper-parameters, the random seed) lives here so
that the training scripts stay readable and every run is reproducible.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent   # repo root; the Vite app lives here too
RAW_DTA = ROOT / "h256.dta"                 # MEPS HC-256 Full Year Consolidated, 2024
ARTIFACTS = ROOT / "ml" / "artifacts"       # models, metrics, intermediate data
DOC_ASSETS = ROOT / "docs" / "assets"       # PNG figures used by the markdown reports
WEB_PUBLIC = ROOT / "public"                # JSON consumed by the web app

for _p in (ARTIFACTS, DOC_ASSETS, WEB_PUBLIC):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20      # held-out test set, never touched during model selection
N_FOLDS = 5           # stratified K-fold used for cross-validated model selection

# ----------------------------------------------------------------------------
# Population filter
# ----------------------------------------------------------------------------
MIN_AGE = 18          # BMI and smoking are only collected for adults in MEPS

# MEPS uses negative reserved codes for "inapplicable / refused / don't know /
# not ascertained / cannot be computed". They are NOT real values and must be
# converted to NaN before any modelling happens.
MEPS_MISSING_CODES = [-1, -7, -8, -9, -15]

# ----------------------------------------------------------------------------
# Raw MEPS columns we pull out of the 1,615-column file
# ----------------------------------------------------------------------------
TARGET = "TOTEXP24"        # total annual healthcare expenditure, USD
WEIGHT = "PERWT24F"        # person-level survey weight (NOT a feature)
ID = "DUPERSID"

RAW_COLUMNS = [
    ID, WEIGHT, TARGET,
    "AGELAST", "SEX", "RACETHX", "MARRY24X", "EDUCYR", "REGION24",
    "ADBMI42", "OFTSMK53", "ADDAYEXER42", "RTHLTH31",
    "INSCOV24", "POVCAT24",
    "HIBPDX", "DIABDX_M18", "CHDDX", "CANCERDX",
    "ASTHDX", "ARTHDX", "CHOLDX", "STRKDX",
]

# The eight "ever diagnosed" chronic-condition flags. MEPS codes them 1 = Yes,
# 2 = No; we recode to 1/0 so that the sign of a coefficient is readable.
CONDITION_COLS = [
    "HIBPDX", "DIABDX_M18", "CHDDX", "CANCERDX",
    "ASTHDX", "ARTHDX", "CHOLDX", "STRKDX",
]

CONDITION_LABELS = {
    "HIBPDX": "High blood pressure",
    "DIABDX_M18": "Diabetes",
    "CHDDX": "Coronary heart disease",
    "CANCERDX": "Cancer",
    "ASTHDX": "Asthma",
    "ARTHDX": "Arthritis",
    "CHOLDX": "High cholesterol",
    "STRKDX": "Stroke",
}

# ----------------------------------------------------------------------------
# Model feature list (post feature-engineering)
# ----------------------------------------------------------------------------
# NOTE ON FAIRNESS: race/ethnicity (RACETHX) is deliberately EXCLUDED from the
# model inputs. It is loaded only so that we can audit the finished model for
# disparate error rates across race groups. Pricing on race is both unlawful in
# most jurisdictions and ethically indefensible.
PROTECTED_AUDIT_COLS = ["RACETHX", "SEX", "POVCAT24"]

NUMERIC_FEATURES = [
    "age",
    "bmi",
    "education_years",
    "exercise_days",
    "chronic_count",
]

ORDINAL_FEATURES = [
    "self_rated_health",   # 1 = excellent ... 5 = poor
]

CATEGORICAL_FEATURES = [
    "sex",                 # 1 male, 2 female
    "region",              # 1 NE, 2 MW, 3 S, 4 W
    "insurance",           # 1 private, 2 public only, 3 uninsured
    "poverty_category",    # 1 poor ... 5 high income
    "marital_status",
    "smoking_status",      # 1 daily, 2 some days, 3 not at all
]

ENGINEERED_BINARY = [
    "is_obese",            # BMI >= 30
    "is_smoker",           # smokes daily or some days
    "smoker_and_obese",    # the interaction the brief calls out explicitly
    "is_senior",           # age >= 65 (Medicare eligibility changes spending)
    "multimorbid",         # 3 or more chronic conditions
]

FEATURE_COLUMNS = (
    NUMERIC_FEATURES + ORDINAL_FEATURES + CATEGORICAL_FEATURES
    + ENGINEERED_BINARY + CONDITION_COLS
)

# ----------------------------------------------------------------------------
# Risk tiers -- Option A (data-driven tertiles of TOTEXP24)
# ----------------------------------------------------------------------------
# The cut points are LEARNED FROM THE TRAINING SPLIT ONLY and then frozen, so
# that no information about the test set leaks into the labels.
TIER_NAMES = ["Low", "Medium", "High"]
TIER_QUANTILES = [1 / 3, 2 / 3]

# ----------------------------------------------------------------------------
# Hyper-parameter grids
# ----------------------------------------------------------------------------
XGB_COMMON = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    reg_lambda=1.5,
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
)

RF_COMMON = dict(
    n_estimators=400,
    max_depth=14,
    min_samples_leaf=8,
    random_state=SEED,
    n_jobs=-1,
)

# ----------------------------------------------------------------------------
# Currency
# ----------------------------------------------------------------------------
# MEPS records US healthcare spending in US dollars. The models are trained,
# evaluated and stored in USD -- changing the display currency must never change
# the model. Conversion happens only at presentation time, here and in the web
# app's src/lib/currency.ts, which reads the same two constants from model.json.
#
# IMPORTANT CAVEAT, repeated wherever a rupee figure appears: converting a US
# cost to rupees does NOT make it an Indian cost. US healthcare is several times
# more expensive than Indian healthcare for the same treatment, so these figures
# are "what this care costs in the US, expressed in INR" -- useful for reading
# the numbers in a familiar unit, not a prediction of an Indian medical bill.
DISPLAY_CURRENCY = "INR"
USD_TO_INR = 88.0            # update here and re-run export_web_model.py
