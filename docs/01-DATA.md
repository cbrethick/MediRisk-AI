# 1 · The data

## What it is

**MEPS HC-256 — 2024 Full Year Consolidated Data File**, published by the US
Agency for Healthcare Research and Quality (AHRQ).

This is real survey data, not a simulation and not the 1,338-row Kaggle toy
dataset. It is a nationally representative sample of the US civilian
non-institutionalised population, collected across Rounds 3–5 of Panel 28 and
Rounds 1–3 of Panel 29, with **actual recorded expenditure** rather than modelled
premiums.

| | |
|---|---|
| File | `h256.dta` (Stata format, ~57 MB uncompressed) |
| Rows | 19,140 people (one row per person) |
| Columns | 1,615 |
| Reference year | Calendar year 2024 |
| Download | <https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-256> |

Stata format was chosen over XLSX: roughly ten times smaller, and it preserves
variable labels.

## Verifying the variable names

The project brief listed variable names from general MEPS naming conventions
rather than from the 2024 codebook, and **three of them were wrong for this
file**. They were found by searching the actual column list rather than trusting
the list:

| Brief said | Actually in HC-256 | Notes |
|---|---|---|
| `BMINDX53` | **`ADBMI42`** | Adult BMI, from the self-administered questionnaire |
| `ADSMOK42` | **`OFTSMK53`** | "How often do you smoke now": 1 daily, 2 some days, 3 not at all |
| `DIABDX` | **`DIABDX_M18`** | Diabetes, adults 18+ |

The rest verified as written. This is worth stating in any write-up: MEPS renames
variables between years, so names must always be checked against the file rather
than assumed.

## Columns actually used

Out of 1,615 columns, 23 are loaded.

| Column | Type | Meaning | Role |
|---|---|---|---|
| `DUPERSID` | string | Person identifier | Dropped before training |
| `TOTEXP24` | float | Total 2024 healthcare expenditure, USD | **Target** |
| `PERWT24F` | float | Person-level survey weight | Not a feature (see below) |
| `AGELAST` | int | Age at last interview, 0–85 | Feature |
| `SEX` | 1/2 | Male / female | Feature + fairness audit |
| `RACETHX` | 1–5 | Race and ethnicity | **Audit only, never a feature** |
| `MARRY24X` | 1–6 | Marital status | Feature |
| `EDUCYR` | 0–17 | Years of education | Feature |
| `REGION24` | 1–4 | Northeast / Midwest / South / West | Feature |
| `ADBMI42` | float | Body mass index | Feature |
| `OFTSMK53` | 1–3 | Current smoking frequency | Feature |
| `ADDAYEXER42` | 0–7 | Days a week of moderate activity | Feature |
| `RTHLTH31` | 1–5 | Self-rated health, excellent → poor | Feature |
| `INSCOV24` | 1–3 | Private / public only / uninsured | Feature |
| `POVCAT24` | 1–5 | Family income vs the federal poverty line | Feature + audit |
| `HIBPDX` | 1/2 | Ever diagnosed: high blood pressure | Feature |
| `DIABDX_M18` | 1/2 | Diabetes | Feature |
| `CHDDX` | 1/2 | Coronary heart disease | Feature |
| `CANCERDX` | 1/2 | Cancer | Feature |
| `ASTHDX` | 1/2 | Asthma | Feature |
| `ARTHDX` | 1/2 | Arthritis | Feature |
| `CHOLDX` | 1/2 | High cholesterol | Feature |
| `STRKDX` | 1/2 | Stroke | Feature |

## The four problems in the raw file

### 1. Negative numbers that are not numbers

MEPS encodes non-answers as negative reserved codes:

| Code | Meaning |
|---|---|
| −1 | Inapplicable |
| −7 | Refused |
| −8 | Don't know |
| −9 | Not ascertained |
| −15 | Cannot be computed |

Left alone, a model cheerfully learns that a BMI of −15 predicts something. Every
reserved code is converted to `NaN` in `clean_missing()` before anything else
happens, and any other negative value in a feature column is treated the same way.

`ADBMI42` illustrates the scale of this: its raw minimum is −15 and its raw mean
is 15.25, which is not a BMI at all — it is a mixture of real values and reserved
codes. After cleaning, the median is 27.46.

### 2. A target that is zero for one adult in seven

**13.4%** of adults spent nothing at all during 2024, and the rest are skewed
**9.82** to the right (a symmetric distribution would be 0).

| Statistic | USD | INR at 88 |
|---|---|---|
| Mean | $10,728 | ₹9.44 L |
| Median | $2,741 | ₹2.41 L |
| 95th percentile | $45,343 | ₹39.9 L |
| Maximum | $847,517 | ₹7.46 Cr |

Ordinary least squares on raw dollars is the wrong shape for a distribution like
this. See [03-MODELS.md](03-MODELS.md) for the two approaches that were built and
compared, and [04-RESULTS.md](04-RESULTS.md) for the one that failed.

### 3. Adult-only questions

BMI, smoking and physical activity come from a self-administered questionnaire
given only to adults. The population is therefore filtered to **age ≥ 18**, which
takes 19,140 rows down to **15,484**. This is a scope limit, stated rather than
hidden: nothing in this project applies to children.

### 4. Heavy missingness in the most useful features

After cleaning, among adults:

| Feature | Missing | Why |
|---|---|---|
| `bmi` | 30.1% | Self-administered questionnaire not returned |
| `exercise_days` | 28.8% | Same questionnaire |
| `smoking_status` | 2.1% | |
| `self_rated_health` | 1.8% | |
| Chronic conditions | 0.6–1.3% | |
| Age, sex, insurance, income | 0.0% | |

Dropping rows with any missing value would throw away roughly a third of the
sample, and not at random — people who skip the questionnaire differ
systematically from people who return it. Instead, every feature carries a
companion `__isna` indicator, so the model can use *the fact of non-response*
as information. See [02-FEATURES.md](02-FEATURES.md).

## Survey weights

`PERWT24F` is a person-level weight that makes the sample representative of the
US population. It is **loaded but not used for fitting**.

That is a deliberate choice with a trade-off. Weights are correct for producing
population totals; for a predictive model they re-weight the loss towards
over-sampled groups without improving prediction for any individual. The
consequence, stated in the limitations: **these models describe the MEPS sample,
not the US population.** Population-level claims would require weighted
estimation.

## Terms of use

MEPS data are free and public, with no account needed. By using them you agree to
use them only for statistical reporting and analysis, and AHRQ asks to be cited.
No attempt is made anywhere in this project to identify any individual
respondent.

> Agency for Healthcare Research and Quality. *Medical Expenditure Panel Survey,
> HC-256: 2024 Full Year Consolidated Data File.* Rockville, MD: AHRQ, August 2026.
