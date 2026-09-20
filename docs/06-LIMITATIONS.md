# 6 · Limitations

Stated in full, because a model's limits are part of its specification rather
than an apology for it.

## 6.1 Statistical

### Individual spending is close to unpredictable

R² of about **0.09** means roughly 91% of the variation between individuals is
not explained by anything in this model. That is not a tuning failure. The
dominant driver of what a person spends in a year is *events* — an accident, a
new diagnosis, a hospitalisation — which no demographic survey question can
anticipate. Published models on MEPS-type data sit in the same range.

What the model *can* do is rank: the costliest predicted decile really spends
**21.6×** the cheapest. That is the property insurance pricing needs, and it is
invisible in R².

### The Medium tier is genuinely hard

Macro-F1 0.571 overall, but the Medium tier scores 0.454. This is structural. The
tier is bounded by two arbitrary percentile cuts, and a person spending ₹84,000 a
year and one spending ₹85,000 are indistinguishable in every way that matters —
yet the labels put them in different classes. Low and High are separated by real
differences, and the model finds them.

### Trained unweighted

MEPS uses a complex survey design with person-level weights (`PERWT24F`). Those
weights are correct for producing population totals; they were **not** used for
fitting, because re-weighting a predictive loss towards over-sampled groups does
not improve prediction for any individual.

The consequence: **these models describe the MEPS sample, not the US population.**
Any population-level claim would need weighted estimation.

### One year, no external validation

Trained and tested on 2024 alone. The held-out split is a random sample of the
same survey, so it measures generalisation to *new people*, not to a new year, a
new population or a new health system. A temporal validation — train on 2023,
test on 2024 — would be the natural next step and is not done here.

## 6.2 Data

### US prices, 2024

Costs reflect the US healthcare system in one calendar year. They do not transfer
to other countries, and without inflation adjustment they do not transfer to other
years.

This is the substantive caveat behind the rupee figures. Converting a US cost to
INR at ₹88 changes the *unit*, not the *system*: the same treatment costs several
times less in India. These are American prices shown in rupees, not a prediction
of an Indian hospital bill. Re-training on Indian claims data would be a different
project — the pipeline would transfer, the coefficients would not.

### Adults only

Age 18 and over, because BMI, smoking and physical activity are adult-only survey
items. Nothing here applies to children.

### Self-reported

Conditions, smoking, BMI and self-rated health are what respondents *said*, not
what was measured. Self-report is known to under-state smoking and body weight,
so the coefficients on those features are probably attenuated towards zero.

### Missingness is heavy and not random

BMI is missing for 30% of adults and physical activity for 29%, both because the
self-administered questionnaire was not returned. The model handles this with
explicit indicators rather than by dropping rows, which is the better of the
available options — but people who skip the questionnaire differ systematically
from people who return it, and no encoding fixes that.

## 6.3 Fairness

**The audit found real gaps.** High-tier recall ranges from about 0.49 to about
0.72 across race groups: the model identifies high-cost people considerably more
reliably in some groups than others. Race is not an input, which means these gaps
come through correlated features such as region, income band and insurance type.

Some of the groups have only a couple of hundred people in the test set, so their
individual figures are unstable. But the spread is wide enough that it should not
be waved away, and "we excluded the variable" is not an answer to it.

Before anything resembling this went near a real pricing decision, these gaps
would need to be the subject of the conversation rather than a footnote to it.

## 6.4 Conceptual

### Association, not causation

Every relationship here is correlational. Nothing in this model supports a claim
that *changing* a feature would change a person's costs. A coefficient on smoking
is not an estimate of what quitting would save.

### Spending is not health

The target is **expenditure**, which measures healthcare *used*, not health. Two
distortions follow directly:

- Someone who cannot afford care, or has no coverage, shows low spending while
  being in poor health. The model will call them low-risk.
- Someone well-insured and cautious may spend a lot while being perfectly healthy.

This is a known bias in cost-based risk models and a documented source of harm
when such models are used to allocate care rather than price it.

### Optimising cost can conflict with health

A risk tier built from spending is a financial instrument. Used for pricing a
pool, that is what it is for. Used to decide who gets care, it would
systematically disadvantage exactly the people who have historically been unable
to access it.

## 6.5 Scope

This is a student project built on public data for coursework. It is **not
medical advice**, **not an insurance quote**, and has not been clinically
validated, externally audited or reviewed by anyone qualified to price insurance.
