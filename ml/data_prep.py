"""
M1 -- Data pipeline.

Turns the raw MEPS HC-256 Stata file into a model-ready table.

Steps, in order:
  1. Read only the columns we need out of the 1,615-column file.
  2. Filter to adults (age >= 18), because BMI and smoking are adult-only items.
  3. Replace MEPS reserved negative codes (-1, -7, -8, -9, -15) with NaN.
  4. Recode condition flags from MEPS 1=Yes/2=No to a readable 1/0.
  5. Engineer the interaction / count features the brief asks for.
  6. Return a tidy DataFrame plus the survey weight and audit columns.

Run directly to print a data-quality report:
    python3 ml/data_prep.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from config import (
    ARTIFACTS, CONDITION_COLS, FEATURE_COLUMNS, MEPS_MISSING_CODES, MIN_AGE,
    PROTECTED_AUDIT_COLS, RAW_COLUMNS, RAW_DTA, TARGET, WEIGHT,
)

warnings.filterwarnings("ignore")


def load_raw() -> pd.DataFrame:
    """Read the subset of MEPS columns we actually model on."""
    if not RAW_DTA.exists():
        raise FileNotFoundError(
            f"Could not find {RAW_DTA}. Download the HC-256 Stata file from "
            "https://meps.ahrq.gov and place h256.dta next to this project."
        )
    return pd.read_stata(RAW_DTA, convert_categoricals=False, columns=RAW_COLUMNS)


def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    """MEPS negative reserved codes are not data. Convert them to NaN."""
    out = df.copy()
    for col in out.columns:
        if col in (TARGET, WEIGHT, "DUPERSID"):
            continue          # expenditure and weights are genuinely non-negative
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].mask(out[col].isin(MEPS_MISSING_CODES))
            # Anything else negative in these fields is also a reserved code.
            out[col] = out[col].mask(out[col] < 0)
    return out


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Rename MEPS codes to human names and build the derived features."""
    d = pd.DataFrame(index=df.index)

    d["age"] = df["AGELAST"]
    d["sex"] = df["SEX"]
    d["bmi"] = df["ADBMI42"]
    d["education_years"] = df["EDUCYR"]
    d["exercise_days"] = df["ADDAYEXER42"]
    d["self_rated_health"] = df["RTHLTH31"]
    d["region"] = df["REGION24"]
    d["insurance"] = df["INSCOV24"]
    d["poverty_category"] = df["POVCAT24"]
    d["marital_status"] = df["MARRY24X"]
    d["smoking_status"] = df["OFTSMK53"]          # 1 daily, 2 some days, 3 not at all

    # Conditions: MEPS 1 = Yes, 2 = No  ->  1 / 0
    for c in CONDITION_COLS:
        d[c] = df[c].map({1: 1, 2: 0})

    # ---- engineered features -------------------------------------------------
    d["chronic_count"] = d[CONDITION_COLS].sum(axis=1, min_count=1)
    d["is_obese"] = (d["bmi"] >= 30).astype("float").where(d["bmi"].notna())
    d["is_smoker"] = d["smoking_status"].isin([1, 2]).astype("float").where(
        d["smoking_status"].notna()
    )
    # The brief calls this out by name: smoking WITH high BMI costs far more
    # than either factor on its own, and a purely additive model cannot see it.
    d["smoker_and_obese"] = (d["is_smoker"] * d["is_obese"])
    d["is_senior"] = (d["age"] >= 65).astype("float")
    d["multimorbid"] = (d["chronic_count"] >= 3).astype("float").where(
        d["chronic_count"].notna()
    )

    # carried through for evaluation, NOT used as model inputs
    d[TARGET] = df[TARGET]
    d[WEIGHT] = df[WEIGHT]
    for c in PROTECTED_AUDIT_COLS:
        if c in df.columns:
            d[f"audit_{c}"] = df[c]
    d["DUPERSID"] = df["DUPERSID"]
    return d


def build_dataset(verbose: bool = True) -> pd.DataFrame:
    raw = load_raw()
    n_all = len(raw)

    adults = raw[raw["AGELAST"] >= MIN_AGE].copy()
    n_adult = len(adults)

    cleaned = clean_missing(adults)
    data = engineer(cleaned)

    # A person with no expenditure record at all cannot be used for either task.
    data = data[data[TARGET].notna()]

    if verbose:
        print(f"Rows in raw file              : {n_all:,}")
        print(f"Adults (age >= {MIN_AGE})           : {n_adult:,}")
        print(f"Usable rows (target present)  : {len(data):,}")
        print("\nMissingness of model features (% of usable rows):")
        miss = (data[FEATURE_COLUMNS].isna().mean() * 100).sort_values(ascending=False)
        for k, v in miss.items():
            print(f"  {k:<20s} {v:6.2f}%")
        t = data[TARGET]
        print(f"\nTarget TOTEXP24: zero for {100 * (t == 0).mean():.1f}% of adults")
        print(f"  mean ${t.mean():,.0f}   median ${t.median():,.0f}   "
              f"p95 ${t.quantile(0.95):,.0f}   max ${t.max():,.0f}")
        print(f"  skew {t.skew():.2f}  ->  log transform is required")
    return data


if __name__ == "__main__":
    df = build_dataset()
    out = ARTIFACTS / "dataset.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved -> {out}")
