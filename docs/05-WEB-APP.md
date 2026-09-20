# 5 · The web application

## 5.1 The core claim

**The app performs real inference.** It does not call a server, and it does not
replay canned answers. It re-implements the arithmetic of the trained models in
TypeScript, using coefficients exported straight from the fitted estimators.

This is verifiable rather than asserted. `ml/export_web_model.py` ends by scoring
200 real people twice — once through scikit-learn, once by hand through the
exported JSON exactly as the browser will — and comparing:

```
Tier probability parity   max abs diff: 0.00e+00
Expected-cost parity      max abs diff: $0.000000
```

If that ever drifts, the export fails loudly, because a web app that silently
disagrees with its own model is worse than no web app.

## 5.2 Why it can run in a browser at all

Because every deployed model is linear. A logistic model or a log-link GLM is a
dot product followed by one scalar function, which a browser evaluates in
microseconds:

| Model | Arithmetic |
|---|---|
| Risk tier | `softmax(W · z + b)` — 3 × 60 coefficients |
| Expected cost | `exp(w · z + b)` — 60 coefficients |
| Any spending | `sigmoid(w · z + b)` — 60 coefficients |

The whole thing is **14 KB of JSON**. A 400-tree forest would be several
megabytes, would need a tree-walking interpreter, and would still only give
*approximate* explanations. See [03-MODELS.md §3.7](03-MODELS.md) for that trade.

## 5.3 What `model.json` contains

| Key | What it is |
|---|---|
| `encoder` | Category lists, training medians and modes, feature names |
| `scaler` | Per-column mean and scale from the training split |
| `reference` | The "typical adult" every explanation is measured against |
| `tierModel` | 3 × 60 coefficients + 3 intercepts |
| `costModel` | 60 coefficients + intercept + Tweedie power |
| `anySpendModel` | 60 coefficients + intercept |
| `tierCuts` | The two tertile boundaries, in USD |
| `currency` | Display code and USD rate — presentation only |

`src/lib/model.ts` mirrors `ml/encoder.py` line for line. The two files are meant
to be read side by side.

## 5.4 The three screens

### Overview
What the tool does, what it is trained on, and the four-step summary of how it
works. Every number on it is read from `metrics.json`.

### Assessment
A four-step questionnaire, roughly a dozen questions.

Design rules it follows:

- **One topic per step**, with a progress rail, so it never becomes a wall of
  inputs.
- **Every question states why it is being asked**, inline, expandable, before it
  is answered. Nobody is asked for information without being told what it is for.
- **Every question can be skipped.** Because the model was trained with explicit
  missingness indicators, "prefer not to say" is handled at prediction time
  exactly as it was during training — it is a real answer, not a silent zero.
- **BMI is calculated, not demanded.** Height and weight go in; BMI comes out,
  with its band named.
- **Nothing leaves the tab.** Stated on the screen, and true.

### Result
Four things, in the order a person needs them:

1. **The tier**, in words, with the model's confidence across all three — and the
   real held-out precision and recall for that tier, so the confidence figure can
   be judged.
2. **The expected annual cost**, with the typical error stated next to it rather
   than hidden, and an explicit note that this is an average over similar people
   rather than a forecast of one person's bill.
3. **Why** — every answer's exact contribution, largest first, as a signed bar
   chart against the reference person, with the arithmetic shown.
4. **What it is not** — the limits, stated plainly.

### Research
The full write-up, rendered from `metrics.json` at runtime: model comparison
tables, the confusion matrix, per-tier precision and recall, PR curves,
calibration, decile lift, feature importance, the fairness audit and the
limitations. Retrain, re-export, and this page updates itself.

## 5.5 Accessibility and presentation

- Risk tiers are distinguished by **colour and by word**, never colour alone.
- All interactive elements have visible focus rings; toggle buttons carry
  `aria-pressed`; charts carry `role="img"` and a text label.
- Animations are inside a `prefers-reduced-motion` guard.
- Layout works down to phone width; tables and charts scroll horizontally inside
  their own containers rather than forcing the page to.
- Charts are hand-written SVG — no charting library, so the bundle stays small and
  every chart shares one visual language.

## 5.6 Stack

React 19, TypeScript, Vite 8, Tailwind CSS 4, `lucide-react` for icons. Three
runtime dependencies in total.

```bash
cd medical-risk-ai
npm install
npm run dev      # http://localhost:3000
npm run build    # static bundle in dist/
npm run lint     # tsc --noEmit
```

The built app is fully static — `dist/` can be served from anywhere, including
GitHub Pages, with no backend at all.
