/**
 * In-browser inference for the trained MEPS models.
 *
 * This file is a faithful re-implementation of `ml/encoder.py` and of the three
 * deployed scikit-learn models. Nothing here is mocked or hard-coded: every
 * number comes from `public/model.json`, which `ml/export_web_model.py` writes
 * straight out of the fitted estimators. A parity test in that script confirms
 * the two implementations agree to 0.00e+00.
 *
 * The three models are:
 *   anySpendModel  logistic  ->  P(any healthcare spending this year)
 *   costModel      Tweedie GLM with a log link  ->  E[annual cost in USD]
 *   tierModel      multinomial logistic  ->  P(Low), P(Medium), P(High)
 */

export interface ModelSpec {
  encoder: {
    continuous: string[];
    binary: string[];
    categorical: string[];
    medians: Record<string, number>;
    modes: Record<string, number>;
    categories: Record<string, number[]>;
    featureNames: string[];
  };
  scaler: { mean: number[]; scale: number[] };
  reference: { raw: Record<string, number>; z: number[] };
  tierModel: { classes: string[]; coef: number[][]; intercept: number[] };
  anySpendModel: { coef: number[]; intercept: number };
  costModel: { power: number; coef: number[]; intercept: number };
  currency?: { code: string; usdRate: number };
  tierCuts: number[];
  featureNames: string[];
  conditionLabels: Record<string, string>;
}

/** What the form collects. `null` means "prefer not to say", which is a first-class
 *  answer: the model was trained with explicit missingness indicators, so an
 *  unanswered question is handled the way it was handled in training. */
export type Answers = Record<string, number | null>;

export interface Contribution {
  feature: string;   // the source feature, e.g. "bmi"
  value: number;     // signed effect on the High-risk logit, vs the reference person
}

export interface Prediction {
  tierProbabilities: number[];
  tierIndex: number;
  tierName: string;
  expectedCost: number;
  anySpendProbability: number;
  contributions: Contribution[];
  baselineLogit: number;
  personLogit: number;
}

let spec: ModelSpec | null = null;

export async function loadModel(): Promise<ModelSpec> {
  if (spec) return spec;
  const res = await fetch(`${import.meta.env.BASE_URL}model.json`);
  if (!res.ok) throw new Error(`Could not load model.json (${res.status})`);
  spec = (await res.json()) as ModelSpec;
  return spec;
}

/**
 * Encode one person into the model's feature vector.
 * Mirrors MepsEncoder.transform exactly:
 *   continuous/binary -> [imputed value, is-missing indicator]
 *   categorical       -> one-hot over the frozen category list; an unknown or
 *                        missing category is an all-zero block
 */
function encode(a: Answers, m: ModelSpec): number[] {
  const e = m.encoder;
  const out: number[] = [];
  for (const c of e.continuous) {
    const v = a[c];
    const missing = v === null || v === undefined || Number.isNaN(v);
    out.push(missing ? e.medians[c] : (v as number), missing ? 1 : 0);
  }
  for (const c of e.binary) {
    const v = a[c];
    const missing = v === null || v === undefined || Number.isNaN(v);
    out.push(missing ? e.modes[c] : (v as number), missing ? 1 : 0);
  }
  for (const c of e.categorical) {
    const v = a[c];
    for (const cat of e.categories[c]) out.push(v === cat ? 1 : 0);
  }
  return out;
}

const dot = (a: number[], b: number[]) => a.reduce((s, x, i) => s + x * b[i], 0);

/** Derive the engineered features from the raw answers, exactly as data_prep.py does. */
export function withEngineered(raw: Answers, conditionKeys: string[]): Answers {
  const a: Answers = { ...raw };

  const known = conditionKeys.filter((k) => a[k] !== null && a[k] !== undefined);
  a.chronic_count = known.length
    ? known.reduce((s, k) => s + (a[k] as number), 0)
    : null;

  const bmi = a.bmi;
  a.is_obese = bmi === null || bmi === undefined ? null : bmi >= 30 ? 1 : 0;

  const smoke = a.smoking_status;
  a.is_smoker =
    smoke === null || smoke === undefined ? null : smoke === 1 || smoke === 2 ? 1 : 0;

  a.smoker_and_obese =
    a.is_smoker === null || a.is_obese === null
      ? null
      : (a.is_smoker as number) * (a.is_obese as number);

  const age = a.age;
  a.is_senior = age === null || age === undefined ? null : age >= 65 ? 1 : 0;

  a.multimorbid =
    a.chronic_count === null ? null : (a.chronic_count as number) >= 3 ? 1 : 0;

  return a;
}

export function predict(answers: Answers, m: ModelSpec): Prediction {
  const x = encode(answers, m);
  // standardise
  const z = x.map((v, i) => (v - m.scaler.mean[i]) / m.scaler.scale[i]);

  // --- risk tier: softmax over three linear scores -------------------------
  const logits = m.tierModel.coef.map((c, k) => dot(z, c) + m.tierModel.intercept[k]);
  const mx = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - mx));
  const sum = exps.reduce((s, v) => s + v, 0);
  const probs = exps.map((v) => v / sum);
  const tierIndex = probs.indexOf(Math.max(...probs));

  // --- expected annual cost: Tweedie GLM, log link --------------------------
  const expectedCost = Math.exp(dot(z, m.costModel.coef) + m.costModel.intercept);

  // --- probability of any spending at all -----------------------------------
  const anyLogit = dot(z, m.anySpendModel.coef) + m.anySpendModel.intercept;
  const anySpendProbability = 1 / (1 + Math.exp(-anyLogit));

  // --- exact attribution against the reference person -----------------------
  // For a linear model this IS the SHAP value, with no sampling:
  //   contribution_j = coef_j * (z_j - z_ref_j)
  // and the contributions sum precisely to (person logit - reference logit).
  const coefHigh = m.tierModel.coef[2];
  const perColumn = z.map((v, i) => coefHigh[i] * (v - m.reference.z[i]));

  // Roll one-hot columns and missingness indicators back up to the source
  // feature the person actually answered, so the explanation names a question
  // rather than a matrix column.
  const grouped = new Map<string, number>();
  m.featureNames.forEach((name, i) => {
    const src = name.split("=")[0].replace("__isna", "");
    grouped.set(src, (grouped.get(src) ?? 0) + perColumn[i]);
  });

  const contributions = [...grouped.entries()]
    .map(([feature, value]) => ({ feature, value }))
    .filter((c) => Math.abs(c.value) > 1e-9)
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  const baselineLogit =
    dot(m.reference.z, coefHigh) + m.tierModel.intercept[2];

  return {
    tierProbabilities: probs,
    tierIndex,
    tierName: m.tierModel.classes[tierIndex],
    expectedCost,
    anySpendProbability,
    contributions,
    baselineLogit,
    personLogit: logits[2],
  };
}

export const bmiFrom = (kg: number, cm: number) => kg / (cm / 100) ** 2;
