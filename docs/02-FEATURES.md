# 2 · Features and encoding

25 source features become 60 encoded columns.

## 2.1 The feature list

### Continuous and ordinal (6)

| Feature | Source column | Range |
|---|---|---|
| `age` | `AGELAST` | 18–85 |
| `bmi` | `ADBMI42` | ~12–50 |
| `education_years` | `EDUCYR` | 0–17 |
| `exercise_days` | `ADDAYEXER42` | 0–7 |
| `chronic_count` | derived | 0–8 |
| `self_rated_health` | `RTHLTH31` | 1 excellent – 5 poor |

`self_rated_health` is kept as a single ordinal number rather than five dummies,
because its levels are genuinely ordered and treating them as ordered spends one
coefficient instead of four.

### Categorical (6, one-hot encoded)

| Feature | Source | Levels |
|---|---|---|
| `sex` | `SEX` | 2 |
| `region` | `REGION24` | 4 |
| `insurance` | `INSCOV24` | 3 |
| `poverty_category` | `POVCAT24` | 5 |
| `marital_status` | `MARRY24X` | 6 |
| `smoking_status` | `OFTSMK53` | 3 |

### Chronic conditions (8, binary)

`HIBPDX`, `DIABDX_M18`, `CHDDX`, `CANCERDX`, `ASTHDX`, `ARTHDX`, `CHOLDX`,
`STRKDX` — recoded from the MEPS convention of 1 = Yes / 2 = No to a plain 1 / 0,
so that the sign of a coefficient reads the way a person expects.

### Engineered (5)

| Feature | Definition | Why it has to exist |
|---|---|---|
| `smoker_and_obese` | `is_smoker × is_obese` | **The interaction the brief names.** Two separate coefficients can never express "worse together than apart" — a linear model with a smoking term and a BMI term is structurally incapable of it. |
| `chronic_count` | Sum of the 8 flags | Managing four conditions at once costs more than four people with one condition each. The count captures the compounding that the individual flags miss. |
| `is_obese` | `bmi ≥ 30` | Clinical risk turns at a threshold rather than rising smoothly, and the WHO obesity cut-off is where most of the published risk literature sits. |
| `is_senior` | `age ≥ 65` | Medicare eligibility changes both how much care is used and who records paying for it — a discontinuity, not a slope. |
| `multimorbid` | `chronic_count ≥ 3` | A threshold widely used in health services research. |

## 2.2 The variable that is deliberately absent

**Race and ethnicity (`RACETHX`) is never given to any model.**

It is loaded from the survey, and it is used — but only to *audit* the finished
model for unequal error rates across race groups. Pricing on race is unlawful in
most jurisdictions and indefensible regardless.

Excluding the variable is necessary but not sufficient. A model can still perform
unevenly across groups by leaning on correlated features such as region and income
band. That is precisely why the audit exists, and why its results are reported in
[04-RESULTS.md §4.6](04-RESULTS.md) including where they are uncomfortable.

## 2.3 Encoding

Encoding is done by a small hand-written class (`ml/encoder.py`) rather than by
scikit-learn's `ColumnTransformer`. The reason is deployment: the finished model
has to run **inside a browser**, so every transformation must be replayable in
TypeScript from a JSON spec. A class whose entire state is four dictionaries
serialises trivially; a pickled `ColumnTransformer` does not.

The rules are deliberately boring:

| Input type | Becomes |
|---|---|
| Continuous / ordinal | `[value imputed with the training median, is-missing indicator]` |
| Binary | `[value imputed with the training mode, is-missing indicator]` |
| Categorical | One-hot over a frozen category list; missing or unseen → all zeros |

All imputation statistics are computed **on the training split only** and then
frozen, so nothing about the test set leaks into the encoding.

### Why the missingness indicators matter

BMI is missing for 30% of adults, and not at random — people who do not return the
self-administered questionnaire differ systematically from people who do. Imputing
the median and saying nothing would tell the model "this person has a median BMI",
which is false. The `bmi__isna` column lets the model learn what non-response
itself predicts, which is information that would otherwise be destroyed.

This is also what lets the web form offer "prefer not to say" as a genuine answer
rather than a polite fiction: a skipped question is handled at prediction time
exactly as it was handled during training.

### The resulting matrix

```
6 continuous/ordinal  × 2  = 12   (value + indicator)
13 binary             × 2  = 26   (8 conditions + 5 engineered)
23 one-hot levels          = 23   (2+4+3+5+6+3)
                            ----
                             60 columns
```

Linear models are additionally standardised (zero mean, unit variance) using
training-split statistics. Tree models are given the unstandardised matrix, since
scaling is irrelevant to them.

## 2.4 One consequence worth knowing about

One-hot encoding *without* dropping a reference level makes the dummies within a
group collinear: they sum to 1 whenever the category is observed. L2
regularisation keeps this harmless for prediction, but it means the coefficients
inside a group are identified only up to a shared constant, which the regulariser
splits arbitrarily.

That does not affect accuracy, but it wrecks naive explanations — region appeared
as the single largest "driver" purely as an artefact. The fix is to state every
attribution relative to a fixed reference person, which cancels the constant
exactly. See [04-RESULTS.md §4.5](04-RESULTS.md).
