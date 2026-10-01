# Cardio3D: coronary artery risk on an interactive 3D heart

Cardio3D predicts whether a patient has coronary artery disease (CAD) and whether the three main coronary
arteries (LAD, LCX, RCA) are narrowed by 50% or more. It uses routine clinical, ECG, laboratory and echo
measurements. The predicted probabilities colour the arteries of a 3D heart built from real BodyParts3D
anatomy, and a dashboard explains which measurements drove each prediction (SHAP).

Built for Track A (Cardiovascular Risk Visualization & Prediction) on the UCI *Extension of Z-Alizadeh Sani*
dataset: 303 patients, 55 input features.

> **Decision support and educational use only.** Predictions are not a substitute for coronary angiography,
> CT angiography or a clinician's judgement. A warning is visible at the top and bottom of the screen at all times.

* Full write-up: [`docs/Project_Documentation.pdf`](docs/Project_Documentation.pdf) (5 pages)
* Demo video script and checklist: [`docs/Demo_Video_Script.md`](docs/Demo_Video_Script.md)
* Third-party credits: [`NOTICE.md`](NOTICE.md)

---

## Run it (about 3 minutes)

Requires **Python 3.11 or newer**. The trained models are included, so there is nothing to train. The page
loads everything from this server (no CDN, no web fonts), so it works offline.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.lock       # exact tested versions (requirements.txt is a looser alternative)
uvicorn cardio.serve:app --port 8000
```

Open <http://localhost:8000>. Choose an example patient or edit any value on the left; the heart and the
explanation update about a third of a second after you stop typing. `make run`, `make test`, `make train`
do the same things if you have `make`.

If the page says the model could not be loaded, check that `artifacts/model_bundle.joblib` exists and that
`pip install` finished without errors (the bundle needs `pyarrow`).

### Using the viewer

| Do this | What happens |
|---|---|
| Drag / scroll | Rotate / zoom. While dragging, the scene renders at lower resolution to stay smooth |
| Front, Back, Left | Standard camera views |
| Hover over an artery or heart part | Tooltip with the anatomical name (and the vessel's probability) |
| Click an artery | Selects that vessel; the explanation switches to it |
| Click the ventricles, an atrium, the aorta... | Region card: supplying arteries, their probabilities, related echo findings |
| Click empty space | Back to overall CAD |
| Skeleton / Heartbeat | Hide the ribs and spine / toggle the beat animation |
| Prediction / Performance / Importance tabs | Patient explanation / cross-validated metrics, ROC and calibration / global SHAP importance |

On software-only graphics (or with `?lite=1` in the URL) the page switches to a lite mode: no antialiasing,
lower resolution, no idle animation.

### Retrain, test, rebuild

```bash
python -m cardio.train --data "data/extention of Z-Alizadeh sani dataset.xlsx"   # about 5 min on one core
python -m pytest -q tests                                                         # 12 tests, about 1 min
cd tests/e2e && npm install && npm test        # browser test; start the app first (needs Node 20+; tested with Node 22)
python scripts/optimize_glb.py                 # optional: shrink the skeleton in thorax.glb
```

The seed is fixed (42), but exact numbers can shift slightly between library versions; always quote the
metrics that ship with the artifacts you run. `python -m cardio.demo_data` makes a synthetic stand-in dataset
for software testing only (the UI then shows a red SYNTHETIC banner).

---

## Results

Nested, repeated cross-validation over all 303 patients: every number comes from patients the model that
predicted them never saw. Model choice happened inside the folds, probabilities are Platt-calibrated, and each
target has its own decision threshold chosen on other patients' predictions. Intervals are patient-level
bootstrap (500 resamples).

| Target | Model | Threshold | ROC-AUC (95% CI) | Accuracy | Precision | Recall | Specificity | F1 | Brier |
|---|---|---|---|---|---|---|---|---|---|
| CAD | logistic l2 | 0.76 | 0.899 (0.86-0.93) | 0.831 | 0.929 | 0.826 | 0.843 | 0.874 | 0.112 |
| LAD | random forest | 0.52 | 0.811 (0.76-0.86) | 0.730 | 0.784 | 0.746 | 0.709 | 0.764 | 0.171 |
| LCX | gradient boosting | 0.37 | 0.717 (0.66-0.76) | 0.656 | 0.551 | 0.689 | 0.634 | 0.612 | 0.210 |
| RCA | logistic l1 | 0.33 | 0.691 (0.64-0.74) | 0.617 | 0.495 | 0.757 | 0.533 | 0.598 | 0.213 |

(Means over 3 x 5 outer folds; full intervals for every metric are in `artifacts/metrics.json`.)

* **CAD and LAD are informative; LCX and RCA are weak** (AUC about 0.7). We tried L1/L2 logistic regression,
  feature selection, forests and boosting and none lifted them much, which points to the data, not the model.
* At a naive 0.5 cut-off, LCX and RCA recall would be 0.41 and 0.34. The per-target
  thresholds fix that at the cost of precision. The UI shows each cut-off beside the probability.
* Choosing the best of five models on all patients and reporting that score would give CAD AUC
  0.935; the nested estimate above (0.899) is the honest one.

## How the brief is covered

| Requirement | Where |
|---|---|
| 1a-b Predict CAD and LAD / LCX / RCA | `cardio/train.py`: four calibrated binary classifiers |
| 1c Use all clinical feature groups | All 55 remaining columns |
| 1d No LAD, LCX, RCA, Cath inputs | `LEAKAGE_KEYS` in `data.py`, asserted in `prepare()`, tested on data and on fitted pipelines |
| 1e Accuracy, precision, recall, F1, ROC-AUC | `artifacts/metrics.json` / `.md`, Performance tab (plus specificity, Brier, calibration error, CIs) |
| 2a Interactive 3D torso and heart | `static/app.js` + `static/models/thorax.glb` (BodyParts3D), Three.js |
| 2b Artery colours from probabilities | `paintVessel()`: green to red, every segment of each artery |
| 2c Rotate, zoom, select regions | OrbitControls, vessel picking, region cards, hover names |
| 3a Probabilities beside the 3D view | Right panel and vessel cards (with each target's cut-off) |
| 3b SHAP explanation | `cardio/explain.py`; plain-language summary, per-measurement bars, global importance |
| 3c Measurements with relative contribution | Value, unit, glossary tooltip and signed % share per feature |
| 4a Open meshes | BodyParts3D 4.0 (CC BY 4.0), see `NOTICE.md` |
| 5 Visible disclaimer | Header, footer, this README, PDF footer |

## Project layout

| Path | Purpose |
|---|---|
| `cardio/data.py` | Loading, cleaning, targets, leakage guard, feature glossary and schema |
| `cardio/train.py` | Nested CV, calibration, thresholds, bootstrap, final models, bundle |
| `cardio/explain.py` | SHAP helpers shared by training and serving |
| `cardio/serve.py` | FastAPI backend and static file server |
| `static/` | `index.html`, `app.js` (viewer + dashboard), `anatomy.js` (anatomy config), `models/thorax.glb`, `vendor/three/` |
| `artifacts/` | Trained bundle and metrics |
| `scripts/` | `build_thorax_glb.py` (BodyParts3D to GLB), `optimize_glb.py` (decimate bones) |
| `tests/` | 12 Python tests; `tests/e2e/` headless-browser test |
| `docs/` | Documentation PDF, demo script, figures |

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/schema` | Features, units, groups, glossary, disclaimer |
| `GET /api/samples` | Eight example patients with recorded outcomes and out-of-fold estimates |
| `GET /api/metrics` | Metrics with intervals, ROC and calibration points, thresholds |
| `GET /api/importance` | Global SHAP importance per target |
| `GET /api/model-info` | Data hash, library versions, protocol, version-mismatch note |
| `POST /api/predict` | `{"features": {...}}` returns four probabilities, cut-offs, SHAP, summaries, warnings |

Blank inputs are imputed (and reported). Values outside the training range raise a warning.
`CARDIO_ARTIFACTS=/path` serves a different bundle.

## Extending it

* **Features:** add columns to the data and retrain; the form, schema, explanations and importance regenerate.
* **Models:** add one entry to `candidates()` in `train.py`.
* **Anatomy:** add a group to `static/anatomy.js` (and a region entry to make it selectable).

## Limitations

* One centre, 303 patients, no external validation; intervals are wide (e.g. RCA AUC 0.64 to 0.74).
* The four models are independent, so CAD can disagree with the vessel results. Probabilities are calibrated
  against this dataset, not a general population.
* The dataset has no lesion location, so each artery carries one whole-vessel probability.
* Display bands (Low < 33%, Moderate < 66%, High) are not clinical categories and differ from the cut-offs.
* The anatomy is a generic adult thorax. Browser testing used headless Chromium with software WebGL
  (about 90 to 120 ms per frame on one CPU core); physical GPUs and other browsers were not tested.
  The `Dockerfile` has not been built.
