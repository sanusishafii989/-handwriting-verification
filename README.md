# AI-Enabled Decision Support System for Examination Integrity

> **Handwriting Verification Using Siamese Neural Networks**
>
> Author: Sanusi Shafii

A full-stack, production-quality Next.js 14 web UI that runs a trained
Siamese neural network directly in the browser (TensorFlow.js) to verify
whether two handwritten samples were produced by the same person.

The system is designed as a decision-support tool for examination
integrity — operators upload two handwritten images and receive a
similarity score, a verdict ("Same Writer" / "Different Writer"), a
confidence margin, and an auditable JSON export of each result.

---

## Architecture

Model (epoch 69, best validation weights):

| Layer        | Detail                                              |
|--------------|-----------------------------------------------------|
| Backbone     | Xception (ImageNet pretrained, frozen)              |
| Input        | 224 × 224, grayscale (tiled to 3-channel for Xcep.) |
| Embedding    | 64-dimensional, L2-normalized                       |
| Distance     | L1 (Manhattan) distance between twin embeddings     |
| Head         | Dense + Sigmoid → similarity score in `[0, 1]`      |
| Threshold    | `t = 0.4866`  (tuned on validation split)           |

Validation-set performance:

| Metric         | Value       |
|----------------|-------------|
| Accuracy       | **80.00%**  |
| Precision      | **0.8448**  |
| Recall         | **0.7350**  |
| F1-Score       | **0.7861**  |
| ROC-AUC        | **0.8914**  |

Confusion matrix at t = 0.4866:

|                | Pred Same | Pred Diff |
|----------------|-----------|-----------|
| **Actual Same**| TP = 219  | FN = 79   |
| **Actual Diff**| FP = 40   | TN = 270  |

---

## Tech Stack

- **Next.js 14** (App Router) + **React 18** + **TypeScript 5**
- **Tailwind CSS 3** with a bespoke Navy / Cyan / Gold palette and
  glassmorphism card language
- **TensorFlow.js 4** for client-side model inference (no backend needed)
- **Recharts 2** for the dashboard (ROC, confusion, similarity histogram)
- **lucide-react** for icons
- **next-themes** for dark / light mode (dark mode is the default)

---

## Project Structure

```
.
├── app/
│   ├── layout.tsx              # Root layout + ThemeProvider + Header/Footer
│   ├── globals.css             # Tailwind + Inter, glass utilities, slider CSS
│   ├── page.tsx                # Landing page (/)
│   ├── verify/
│   │   └── page.tsx            # Verification workbench (/verify)
│   └── dashboard/
│       └── page.tsx            # Metrics + charts dashboard (/dashboard)
├── components/
│   ├── layout/
│   │   ├── Header.tsx
│   │   └── Footer.tsx
│   ├── theme/
│   │   └── ThemeToggle.tsx
│   └── ui/
│       ├── Button.tsx          # href-aware (Next link or <button>)
│       ├── Card.tsx            # Glass card w/ Header/Content/Footer
│       ├── UploadArea.tsx      # Drag-and-drop sample upload + preview
│       ├── ResultDisplay.tsx   # Verdict, score, progress bar, stats
│       ├── ThresholdSlider.tsx # 0.0000–1.0000 threshold + live preview
│       └── ModelStatus.tsx     # Loaded / loading / error + arch summary
├── lib/
│   ├── types.ts                # Types, DEFAULT_THRESHOLD, MODEL_VERSION
│   ├── modelLoader.ts          # TF.js LayersModel loader (/public/model)
│   ├── preprocess.ts           # ImageData → [1,224,224,1] tensor
│   └── similarity.ts           # compareImages, export JSON, history CRUD
├── public/
│   └── model/                  # ⚠️ TF.js converted model goes here
│       ├── .gitkeep
│       └── README.txt
├── tailwind.config.ts
├── tsconfig.json
├── next.config.js
└── package.json
```

---

## 1. Install dependencies

```bash
npm install
```

---

## 2. ⚠️ Convert the Keras model to TensorFlow.js format

**The UI loads a TF.js `LayersModel` — it cannot read `.keras` or `.h5`
directly.** A `.gitkeep` and a README.txt are provided at
`public/model/` to remind you.

From a Python shell with the project root as your working directory:

```bash
# 1. Install tensorflowjs
pip install tensorflowjs

# 2. Convert Keras model → TF.js LayersModel
tensorflowjs_converter \
  --input_format=keras \
  ./model.keras \
  ./public/model

# Or, if you prefer the legacy H5 file:
tensorflowjs_converter \
  --input_format=keras \
  ./model.h5 \
  ./public/model
```

After conversion, `public/model/` should contain:

```
public/model/
├── model.json
└── group1-shard1of<N>.bin   # one or more weight shards
```

Until these files exist, the "Model Status" card on `/verify` and
`/dashboard` will render a friendly "Load Failed" notice pointing to
these instructions.

---

## 3. Run the application

```bash
# Development server (http://localhost:3000)
npm run dev

# Production build
npm run build
npm start
```

---

## Using the App

### `/` — Landing page
Project overview, author, tagline, and two CTAs: **Get Started**
routes to `/verify`, **View Dashboard** routes to `/dashboard`.

### `/verify` — Verification workbench
1. Upload Sample A (query) and Sample B (reference) via drag & drop or
   click — PNG, JPG, JPEG up to 10 MB each.
2. Click **Compare Handwriting**. The model:
   - Decodes both files onto a 224×224 canvas (fit-cover with white fill)
   - Produces `[1, 224, 224, 1]` grayscale tensors normalized to `[0, 1]`
   - Tiles each tensor to 3 channels for the Xception input
   - Runs the Siamese twin to produce a sigmoid similarity score
   - Applies the threshold `t = 0.4866` to yield a verdict
3. Use the **Threshold Slider** to tune sensitivity; the verdict bar
   updates live as you drag, with a gold warning if adjusting from the
   default flips the verdict.
4. **Export JSON** downloads a timestamped `.json` file with the score,
   threshold, verdict, confidence, inference time, file names, model
   version, and architecture summary.
5. **Recent Comparisons** keeps the last 20 results in `localStorage`
   (capped) for quick per-session review.
6. **Clear** wipes both samples and the current result.

### `/dashboard` — Performance dashboard
- 6 top-level metric cards (Accuracy / Precision / Recall / F1 /
  ROC-AUC / Optimal Threshold).
- **Confusion Matrix**: styled heatmap table + stacked bar chart of
  Actual × Predicted counts.
- **ROC Curve**: Recharts LineChart across 14 representative operating
  points with the AUC badge and a gold no-skill baseline.
- **Similarity Score Distribution**: bimodal AreaChart comparing
  same-writer vs different-writer score histograms.
- Live Model Status card at the top of the page.

---

## Defaults & Configuration

- **Default threshold**: `DEFAULT_THRESHOLD = 0.4866` in
  [lib/types.ts](lib/types.ts). The UI slider always starts here and
  can be reset via the "Reset" button in the threshold card.
- **Model version**: `MODEL_VERSION = 'siamese-xception-v1.0-epoch69'`
  in [lib/types.ts](lib/types.ts). Bump this string whenever a new
  checkpoint is deployed so that exported JSON files remain auditable.
- **History cap**: last 20 comparisons (configurable in
  `lib/similarity.ts` → `HISTORY_MAX`).
- **Model path**: `/model/model.json` (Next.js resolves this from
  `public/model/`). Change `MODEL_PATH` in [lib/modelLoader.ts] if you
  host the model on a CDN.

---

## Design System

- Colors
  - Deep Navy `#023047`
  - Dark Background `#000814`
  - Medical Cyan `#219EBC`
  - Gold Accent `#FFB703`
- Typography: **Inter** (Google Fonts, weights 300–900)
- Surfaces: Glassmorphism (`backdrop-blur`, soft 1 px light border,
  `rounded-2xl`), subtle cyan/gold glows on primary actions.
- Themes: Dark mode is the default; `ThemeToggle` (sun/moon) in the
  header swaps to a light-mode gradient via `next-themes`.
- Responsive: Mobile-first. Breakpoints at `sm` (640), `lg` (1024),
  and `xl` (1280) are used to split the verification workbench into a
  two-column layout at `xl` and keep a single-column flow below.

---

## License / Usage

Research project artifact — © Sanusi Shafii.
