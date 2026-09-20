# Medical Insurance Risk Stratification & Charge Prediction

Predicting annual healthcare cost and sorting people into risk tiers, from the
**2024 US Medical Expenditure Panel Survey** — with an explanation of every
individual prediction that is exact rather than approximate.

**Department of Data Science and Business Systems**
SRM Institute of Science and Technology, Chennai

| Name | Registration number |
|---|---|
| Rethick C B | RA2411056010016 |
| Anumitha | RA2411056010017 |
| Dheepak S | RA2411056010018 |

---

## What this is

Three deliverables, all built on real survey data rather than a simulated
dataset:

1. **A machine-learning pipeline** — cleans a 1,615-column government survey
   file, engineers features, trains and cross-validates seven models across two
   tasks, and reports the results honestly, including the approach that failed.
2. **A web application** — a guided questionnaire that runs the trained model
   *in the browser* and shows exactly what each answer did to the result.
3. **A written report** — this `docs/` folder, with the results section
   generated automatically from the training run so it can never disagree with
   the model.

## Headline results

Measured on **3,097 held-out people** who took no part in training or model
selection. Money is shown in rupees at ₹88 to the dollar (see the note below).

| | Result |
|---|---|
| Risk tier macro-F1 | **0.571** (chance = 0.333) |
| High-tier recall / precision | **63.3% / 60.1%**, tunable by threshold |
| Cost model MAE | **₹9.45 L** per person per year |
| Cost model R² | **0.092** — low, and honestly so |
| Decile lift | The costliest predicted tenth really spends **21.6×** the cheapest tenth |
| Explanation error | **2.2 × 10⁻¹⁵** — floating-point rounding |

The low R² is not a tuning failure. Individual annual medical spending is close
to unpredictable from demographics, because most of the variance comes from
events no survey question can anticipate. The model earns its place by **ranking
groups**, which is what insurance pricing actually needs — and the decile-lift
figure is the evidence for that.

> **On currency.** MEPS records *US* healthcare spending in US dollars, and every
> model is trained and evaluated in dollars. Rupees are a display conversion only,
> at a rate set in one place (`USD_TO_INR` in `ml/config.py`). A US cost shown in
> rupees is still a US cost — the same treatment is several times cheaper in India
> — so read these as American prices in a familiar unit, not as Indian hospital
> bills.

## Running it

### 1. Get the data

Download **MEPS HC-256, 2024 Full Year Consolidated, Stata format** from
<https://meps.ahrq.gov> and put `h256.dta` in the project root. It is free and
needs no account.

### 2. Train

```bash
pip install -r requirements.txt

python3 ml/data_prep.py         # clean the survey, print a data-quality report
python3 ml/train.py             # train and validate everything  (~40 s)
python3 ml/make_report.py       # write docs/04-RESULTS.md and its figures
python3 ml/export_web_model.py  # export model.json for the web app
```

`train.py` prints its full working to the terminal and writes
`ml/artifacts/metrics.json`, which is the single source of truth for both the
report and the website.

### 3. Run the app

```bash
npm install
npm run dev          # http://localhost:5173
```

The app is a static site — see [DEPLOY.md](DEPLOY.md) to put it on Vercel.

## Project structure

```
MediRisk-AI/                     <- repo root IS the web app, so Vercel needs no config
├── index.html
├── package.json
├── vite.config.ts
├── public/
│   ├── model.json               14 KB of coefficients — the deployed model
│   └── metrics.json             feeds the in-app research page
├── src/
│   ├── App.tsx                  screens: overview / assessment / result / research
│   ├── lib/model.ts             in-browser inference, exact parity with sklearn
│   ├── lib/schema.ts            the questionnaire
│   ├── lib/currency.ts          USD -> INR, display only
│   └── components/              Wizard, Result, Research, charts, ui
│
├── h256.dta                     MEPS raw data — download separately, gitignored
├── requirements.txt
├── ml/
│   ├── config.py                every tunable: paths, features, seed, hyper-parameters
│   ├── data_prep.py             M1 — cleaning and feature engineering
│   ├── encoder.py               feature encoder, serialisable to JSON
│   ├── train.py                 M2 + M3 + M4 — training, validation, fairness audit
│   ├── make_report.py           generates docs/04-RESULTS.md and its figures
│   ├── export_web_model.py      M5 bridge — exports the model, with a parity check
│   └── artifacts/metrics.json   single source of truth for the report and the site
│
├── docs/
│   ├── 01-DATA.md               the dataset, and the four problems in it
│   ├── 02-FEATURES.md           25 features -> 60 columns, and why
│   ├── 03-MODELS.md             every model and every training parameter
│   ├── 04-RESULTS.md            GENERATED. metrics, validation, fairness
│   ├── 05-WEB-APP.md            how the browser runs the model
│   └── 06-LIMITATIONS.md        what this cannot do
└── DEPLOY.md                    Vercel deployment guide
```

## Module map

| Module | Task | Where |
|---|---|---|
| M1 | Data pipeline | `ml/data_prep.py`, `ml/encoder.py` |
| M2 | Charge prediction | `ml/train.py` steps 2–4 |
| M3 | Risk stratification | `ml/train.py` step 5 |
| M4 | Explainability | `ml/train.py` step 6 |
| M5 | Interface | `src/`, `public/` |

## Five decisions worth defending

1. **Race is excluded from the model, then used to audit it.** Pricing on race is
   indefensible. But excluding a variable does not guarantee an even-handed
   model, so the finished model is measured within each race group — and the gaps
   it found are reported rather than buried. ([§4.6](docs/04-RESULTS.md))

2. **The prescribed back-transform failed, and the failure is in the report.**
   Duan's smearing estimator blew up on this data. The results table keeps that
   run as a cautionary result, next to the Tweedie GLM that replaced it.
   ([§4.3](docs/04-RESULTS.md))

3. **The deployed model is not the most accurate one.** The linear models are
   within noise of the tree models, and they ship in 14 KB, run in a browser and
   explain themselves exactly. That trade was made deliberately and is documented.
   ([§3.7](docs/03-MODELS.md))

4. **Missingness is a feature, not a nuisance.** BMI is missing for 30% of adults,
   and not at random. Every feature carries a "was this missing?" indicator, which
   is also what makes "prefer not to say" a real option in the form.
   ([§2.3](docs/02-FEATURES.md))

5. **The report is generated, not written.** `docs/04-RESULTS.md` is produced from
   the training run's own output. Retrain and it updates itself, so it cannot
   quietly drift away from the model it describes.

## Data source

Agency for Healthcare Research and Quality. *Medical Expenditure Panel Survey,
HC-256: 2024 Full Year Consolidated Data File.* Rockville, MD: AHRQ, August 2026.

MEPS data are free and public. Use is limited to statistical reporting and
analysis, and AHRQ asks to be cited as the source. No attempt is made anywhere in
this project to identify any individual respondent.

---

**Not medical advice and not an insurance quote.** This is a student project.
Predictions describe averages over groups of similar people, never an
individual's actual future costs.
