# Deploying to Vercel

The repo root **is** the Vite app, so Vercel detects everything automatically.
There is no `vercel.json`, no build configuration to write, and no backend.

The site is fully static: the model is `public/model.json`, a 14 KB file of
coefficients that the browser evaluates directly. Nothing runs on a server, so
there are no environment variables, no secrets and no serverless functions.

---

## Option A — the dashboard (easiest, ~2 minutes)

1. Go to **<https://vercel.com/new>** and sign in with GitHub.
2. If this is your first time, click **Add GitHub Account** / **Install** and give
   Vercel access to the `MediRisk-AI` repository.
3. Find **`cbrethick/MediRisk-AI`** in the list and click **Import**.
4. Vercel will show these settings already filled in. **Change nothing:**

   | Field | Value |
   |---|---|
   | Framework Preset | `Vite` |
   | Root Directory | `./` |
   | Build Command | `npm run build` |
   | Output Directory | `dist` |
   | Install Command | `npm install` |

   If the Framework Preset says "Other", set it to **Vite** manually.
5. Click **Deploy**. The build takes roughly 30–60 seconds.
6. You get a URL like `https://medirisk-ai.vercel.app`.

Every push to `main` from then on redeploys automatically. Pull requests get
their own preview URL.

---

## Option B — the CLI

```bash
npm i -g vercel

cd MediRisk-AI
vercel login          # opens the browser
vercel                # preview deployment
vercel --prod         # production deployment
```

On the first run it asks a few questions. The answers are:

```
? Set up and deploy?                  yes
? Which scope?                        <your account>
? Link to existing project?           no
? What's your project's name?         medirisk-ai
? In which directory is your code?    ./
```

It then detects Vite and fills in the build settings itself.

---

## Before you deploy: check it builds

Vercel runs exactly these two commands, so run them locally first. If they pass
here, the deployment will pass there.

```bash
npm install
npm run build        # must end with "built in ..."
npm run preview      # serves dist/ at http://localhost:3000 — click through it
```

---

## Things that actually go wrong

### The page loads but says "Could not load the model"

`public/model.json` is missing from the commit. It is required at runtime and
**must be committed** — check `.gitignore` is not excluding it:

```bash
git check-ignore -v public/model.json    # should print nothing
git ls-files public/                     # should list model.json and metrics.json
```

If they are missing, regenerate and commit them:

```bash
python3 ml/export_web_model.py
git add -f public/model.json public/metrics.json
git commit -m "Add model artefacts"
git push
```

### Build fails with a TypeScript error

Vercel runs `vite build`, which does not typecheck — so a type error usually
means something else broke. Reproduce it locally:

```bash
npm run lint         # tsc --noEmit
```

### Build fails on `npm install`

The lockfile and `package.json` disagree. Delete and regenerate:

```bash
rm -rf node_modules package-lock.json
npm install
git add package-lock.json && git commit -m "Refresh lockfile" && git push
```

### The deployed site shows old numbers

The model and metrics are baked into the commit. After retraining you have to
re-export and push:

```bash
python3 ml/train.py
python3 ml/make_report.py
python3 ml/export_web_model.py
git add -A && git commit -m "Retrain" && git push
```

---

## Adding a custom domain

Project → **Settings** → **Domains** → add your domain, then set the DNS records
Vercel shows you. HTTPS is issued automatically.

---

## What is *not* deployed

`ml/` and `docs/` are committed to the repository but are not part of the built
site — Vercel only publishes `dist/`. The Python pipeline is there so the work is
reproducible and reviewable, not because the site needs it.

`h256.dta` (57 MB of MEPS raw data) is **gitignored on purpose**. GitHub warns
above 50 MB and rejects above 100 MB, and the file is freely downloadable from
AHRQ. Anyone reproducing the training downloads it themselves — see the README.
