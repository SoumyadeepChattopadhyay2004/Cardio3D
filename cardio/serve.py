"""FastAPI backend: prediction + SHAP explanation API and the static 3D dashboard.

    uvicorn cardio.serve:app --port 8000        then open http://localhost:8000
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import explain
from .data import TARGETS
from .train import apply_platt

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
DISCLAIMER = ("Decision support / educational use only. Predictions are not a substitute for "
              "formal diagnostic imaging (e.g. coronary angiography or CT angiography) or "
              "clinical judgement.")

app = FastAPI(title="Cardio3D - CAD & coronary stenosis risk", version="2.0.0")
_lock, _state = threading.Lock(), {}


def risk_band(p: float) -> str:
    """Display bands only; they are not clinical categories."""
    return "Low" if p < 0.33 else "Moderate" if p < 0.66 else "High"


def _load() -> dict:
    with _lock:
        if _state:
            return _state
        path = Path(os.environ.get("CARDIO_ARTIFACTS", ROOT / "artifacts")) / "model_bundle.joblib"
        if not path.exists():
            raise HTTPException(503, "Model not trained yet. Run `python -m cardio.train --data ...`.")
        b = joblib.load(path)
        num, cat = explain.split_columns(b["columns"], b["background"])
        b["explainers"] = {t: explain.make_explainer(b["pipelines"][t], b["background"], num, cat) for t in TARGETS}
        b["num_cols"], b["cat_cols"] = num, cat
        trained = b["meta"]["versions"]["scikit-learn"]
        b["version_note"] = (None if trained == sklearn.__version__ else
                             f"Model trained with scikit-learn {trained}, running {sklearn.__version__}. "
                             "Retrain (python -m cardio.train) for an exact match.")
        _state.update(b)
        return _state


def _plain_summary(contribs: list[dict], labels: dict[str, str]) -> dict:
    """Top three measurements pushing risk up and down, ready to show as a sentence."""
    fmt = lambda c: {"feature": c["feature"], "label": labels.get(c["feature"], c["feature"]),  # noqa: E731
                     "value": c["value"], "unit": c["unit"]}
    return {"raises": [fmt(c) for c in contribs if c["contribution"] > 0][:3],
            "lowers": [fmt(c) for c in contribs if c["contribution"] < 0][:3]}


class PredictRequest(BaseModel):
    features: dict[str, Any]


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": bool(_state)}


@app.get("/api/schema")
def schema():
    b = _load()
    return {"features": b["schema"], "targets": TARGETS, "disclaimer": DISCLAIMER,
            "excluded_from_inputs": b["excluded"], "is_synthetic": b["is_synthetic"], "data_file": b["data_file"]}


@app.get("/api/samples")
def samples():
    return _load()["samples"]


@app.get("/api/metrics")
def metrics():
    b = _load()
    return {"metrics": b["metrics"], "thresholds": b["thresholds"], "is_synthetic": b["is_synthetic"],
            "protocol": b["meta"]["protocol"]}


@app.get("/api/importance")
def importance():
    b = _load()
    labels = {f["name"]: f["label"] for f in b["schema"]}
    return {t: [{**r, "label": labels.get(r["feature"], r["feature"])} for r in rows]
            for t, rows in b["importance"].items()}


@app.get("/api/model-info")
def model_info():
    b = _load()
    return {**b["meta"], "models": {t: b["metrics"][t]["model"] for t in TARGETS},
            "runtime_scikit_learn": sklearn.__version__, "version_note": b["version_note"],
            "is_synthetic": b["is_synthetic"]}


@app.post("/api/predict")
def predict(req: PredictRequest):
    b = _load()
    unknown = set(req.features) - set(b["columns"])
    if unknown:
        raise HTTPException(422, f"Unknown features: {sorted(unknown)}")
    spec = {f["name"]: f for f in b["schema"]}
    row, warnings = {}, []
    for c in b["columns"]:
        v = req.features.get(c)
        if v is None or v == "":
            row[c] = np.nan
        elif c in b["num_cols"]:
            try:
                row[c] = float(v)
            except (TypeError, ValueError):
                raise HTTPException(422, f"'{c}' must be numeric, got {v!r}")
            s = spec[c]
            if s["type"] == "number" and not (s["min"] <= row[c] <= s["max"]):
                warnings.append(f"{c}={row[c]:g} is outside the training range "
                                f"[{s['min']:g}, {s['max']:g}]; treat this prediction with extra caution.")
        else:
            row[c] = str(v)
    X = pd.DataFrame([row], columns=b["columns"])
    missing = [c for c in b["columns"] if pd.isna(row[c])]
    if missing:
        warnings.append(f"{len(missing)} feature(s) left blank and imputed with training medians/modes.")
    if b["version_note"]:
        warnings.append(b["version_note"])
    labels = {f["name"]: f["label"] for f in b["schema"]}

    out = {}
    for t in TARGETS:
        pipe = b["pipelines"][t]
        raw = float(pipe.predict_proba(X)[0, 1])
        p = float(apply_platt(b["calibrators"][t], [raw])[0])
        ex, units, owners = b["explainers"][t]
        sv, base = explain.shap_matrix(ex, pipe.named_steps["pre"].transform(X))
        agg = explain.by_feature(sv, owners).iloc[0]
        total = float(np.abs(agg).sum()) or 1.0
        contribs = [{"feature": f, "label": labels.get(f, f), "about": spec[f].get("about", ""),
                     "value": None if pd.isna(row[f]) else row[f], "unit": spec[f]["unit"],
                     "contribution": float(agg[f]), "share": float(agg[f] / total * 100)} for f in agg.index]
        contribs.sort(key=lambda d: -abs(d["contribution"]))
        out[t] = {"probability": p, "raw_score": raw, "label": int(p >= b["thresholds"][t]), "risk_band": risk_band(p),
                  "threshold": b["thresholds"][t], "model": b["metrics"][t]["model"],
                  "summary": _plain_summary(contribs, labels),
                  "explanation": {"units": units, "base_value": base, "contributions": contribs}}
    return {"predictions": out, "warnings": warnings, "disclaimer": DISCLAIMER}


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
