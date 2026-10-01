"""Train and evaluate CAD + LAD/LCX/RCA classifiers with nested, repeated cross-validation.

    python -m cardio.train --data data/<dataset file> [--out artifacts] [--seed 42]

Protocol (per target; the four targets are handled independently)
------------------------------------------------------------------
1. **Outer loop, for evaluation.** Repeated stratified 5-fold CV (3 repeats) over *all* patients.
   Every patient receives one out-of-fold (OOF) prediction per repeat from a model that never saw them.
2. **Inner loop, for model selection.** Inside each outer training fold, five candidate models are
   ranked by 5-fold ROC-AUC and the winner is refit on that outer training fold. Because selection
   happens inside the outer loop, the reported scores are not inflated by choosing the best model on
   the same data.
3. **Calibration.** Raw scores are mapped to probabilities with Platt scaling. For evaluation the
   mapping is cross-fitted (fitted on other patients' OOF scores), so calibration is also honest.
4. **Uncertainty.** Patient-level bootstrap (500 resamples) gives 95% intervals for every metric.
5. **Final model.** The same selection procedure is run once on all patients, the winner is refit on
   everything, and a Platt calibrator fitted on all OOF scores is stored next to it.

Preprocessing (imputation, scaling, encoding) lives inside each pipeline, so nothing leaks across folds.
LAD, LCX, RCA, Cath and CAD are excluded from every model's inputs (see ``data.prepare``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import explain
from .data import TARGETS, build_schema, load_raw, prepare

THRESHOLD = 0.5
CFG = {"repeats": 3, "outer": 5, "inner": 5, "boot": 500}      # --fast shrinks these for smoke tests
SHAP_SAMPLE = 150


# --------------------------------------------------------------------------- models
def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    parts = [("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num)]
    if cat:
        parts.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)),
        ]), cat))
    return ColumnTransformer(parts, verbose_feature_names_out=False)


def candidates(seed: int) -> dict:
    """Five SHAP-compatible candidates. Add a model here and it is compared, calibrated and explained."""
    return {
        "logistic_l2": LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced", random_state=seed),
        "logistic_l1": LogisticRegression(C=0.1, penalty="l1", solver="liblinear", max_iter=5000,
                                          class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                                                random_state=seed, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                            random_state=seed, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=2,
                                                        subsample=0.8, random_state=seed),
    }


def pipeline(X: pd.DataFrame, clf) -> Pipeline:
    return Pipeline([("pre", make_preprocessor(X)), ("clf", clone(clf))])


def select_model(X: pd.DataFrame, y: pd.Series, seed: int) -> tuple[str, dict]:
    """Rank candidates by inner 5-fold ROC-AUC; return (winner, {name: mean AUC})."""
    cv = StratifiedKFold(CFG["inner"], shuffle=True, random_state=seed)
    scores = {n: float(cross_val_score(pipeline(X, c), X, y, cv=cv, scoring="roc_auc").mean())
              for n, c in candidates(seed).items()}
    return max(scores, key=scores.get), scores


# --------------------------------------------------------------------------- calibration & metrics
def _logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_platt(y, p) -> dict:
    lr = LogisticRegression(C=1e4).fit(_logit(p), y)
    return {"coef": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0])}


def apply_platt(cal: dict, p):
    return 1 / (1 + np.exp(-(cal["coef"] * _logit(np.asarray(p)).ravel() + cal["intercept"])))


def crossfit_platt(y, p, seed, k=5):
    out = np.zeros_like(p, dtype=float)
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(p, y):
        out[te] = apply_platt(fit_platt(y[tr], p[tr]), p[te])
    return out


def ece(y, p, bins=10) -> float:
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def point_metrics(y, p, thr=THRESHOLD) -> dict:
    yh = p >= thr
    tp, tn = int((yh & (y == 1)).sum()), int((~yh & (y == 0)).sum())
    fp, fn = int((yh & (y == 0)).sum()), int((~yh & (y == 1)).sum())
    div = lambda a, b: a / b if b else 0.0  # noqa: E731
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec, "specificity": div(tn, tn + fp),
            "f1": div(2 * prec * rec, prec + rec), "roc_auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "ece": ece(y, p), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def youden_threshold(y, p) -> float:
    """Threshold maximising sensitivity + specificity - 1 (searched on a 0.01 grid)."""
    grid = np.linspace(0.05, 0.95, 91)
    pos, neg = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
    j = [((p >= g) & (y == 1)).sum() / pos + ((p < g) & (y == 0)).sum() / neg for g in grid]
    return float(grid[int(np.argmax(j))])


def crossfit_threshold(y, p, seed, k=5):
    """Per-patient threshold chosen on the *other* patients, so tuned-threshold metrics stay honest."""
    out = np.zeros_like(p, dtype=float)
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed + 7).split(p, y):
        out[te] = youden_threshold(y[tr], p[tr])
    return out


def mean_over_repeats(y, P, idx=None, thr=THRESHOLD) -> dict:
    """Mean of per-repeat metrics. ``thr`` is a scalar or a (repeats, n) array of per-patient thresholds."""
    idx = slice(None) if idx is None else idx
    rows = [point_metrics(y[idx], P[r][idx], thr[r][idx] if np.ndim(thr) == 2 else thr) for r in range(P.shape[0])]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def bootstrap_ci(y, P, seed, thr=THRESHOLD) -> dict:
    rng, n, draws = np.random.default_rng(seed), len(y), []
    while len(draws) < CFG["boot"]:
        idx = rng.integers(0, n, n)
        if 0 < y[idx].sum() < n:
            draws.append(mean_over_repeats(y, P, idx, thr))
    return {k: [float(np.percentile([d[k] for d in draws], 2.5)),
                float(np.percentile([d[k] for d in draws], 97.5))]
            for k in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "brier", "ece")}


def curves(y, p) -> dict:
    fpr, tpr, _ = roc_curve(y, p)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 60).astype(int))
    edges = np.linspace(0, 1, 9)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, 7)
    cal = [(float(p[idx == b].mean()), float(y[idx == b].mean()), int((idx == b).sum())) for b in range(8) if (idx == b).any()]
    return {"roc": {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()},
            "calibration": {"mean_pred": [c[0] for c in cal], "frac_pos": [c[1] for c in cal], "n": [c[2] for c in cal]}}


def _py(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else float(v)
    return v


# --------------------------------------------------------------------------- main
def train(data_path: str, out_dir: str = "artifacts", seed: int = 42, fast: bool = False) -> dict:
    if fast:
        CFG.update(repeats=1, outer=3, inner=3, boot=50)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    t0 = time.time()
    df = load_raw(data_path)
    X, targets, excluded = prepare(df)
    X = X.reset_index(drop=True)
    print(f"Loaded {len(X)} patients, {X.shape[1]} input features. Excluded from inputs: {excluded}", flush=True)

    Y = {t: targets[t].fillna(-1).astype(int).to_numpy() for t in TARGETS}
    metrics, final_pipes, calibrators, oof_cal, importance = {}, {}, {}, {}, {}
    num, cat = explain.split_columns(list(X.columns), X)
    background = X.sample(min(100, len(X)), random_state=seed)
    shap_rows = X.sample(min(SHAP_SAMPLE, len(X)), random_state=seed + 1)

    for t in TARGETS:
        y = Y[t]
        ok = y >= 0
        assert ok.all(), f"Missing labels for {t}"
        raw = np.zeros((CFG["repeats"], len(y)))
        chosen = Counter()
        for rep in range(CFG["repeats"]):
            outer = StratifiedKFold(CFG["outer"], shuffle=True, random_state=seed + rep)
            for k, (tr, te) in enumerate(outer.split(X, y)):
                name, _ = select_model(X.iloc[tr], pd.Series(y[tr]), seed + 100 * rep + k)
                chosen[name] += 1
                print(f"  {t} rep {rep + 1}/{CFG['repeats']} fold {k + 1}/{CFG['outer']}: {name}", flush=True)
                pipe = pipeline(X.iloc[tr], candidates(seed)[name]).fit(X.iloc[tr], y[tr])
                raw[rep, te] = pipe.predict_proba(X.iloc[te])[:, 1]
        cal = np.vstack([crossfit_platt(y, raw[r], seed + r) for r in range(CFG["repeats"])])
        thr_cf = np.vstack([crossfit_threshold(y, cal[r], seed + r) for r in range(CFG["repeats"])])
        m = mean_over_repeats(y, cal, thr=thr_cf)           # headline: per-target threshold, chosen out-of-sample
        m05 = mean_over_repeats(y, cal, thr=THRESHOLD)      # for reference: fixed 0.5
        ci = bootstrap_ci(y, cal, seed, thr_cf)
        final_thr = youden_threshold(np.tile(y, CFG["repeats"]), cal.ravel())
        m_raw = mean_over_repeats(y, raw)
        pooled = cal.mean(axis=0)
        thr_table = [{"threshold": th, **{k: v for k, v in mean_over_repeats(y, cal, thr=th).items()
                                          if k in ("recall", "specificity", "precision")}} for th in (0.3, 0.4, 0.5, 0.6, 0.7)]

        best, sel_scores = select_model(X, pd.Series(y), seed)
        final = pipeline(X, candidates(seed)[best]).fit(X, y)
        calibrators[t] = fit_platt(np.tile(y, CFG["repeats"]), raw.ravel())
        final_pipes[t], oof_cal[t] = final, pooled
        importance[t] = explain.global_importance(final, background, shap_rows, num, cat)

        metrics[t] = {
            "model": best, "model_scores_all_data_cv": sel_scores,
            "selection_frequency": {k: v / sum(chosen.values()) for k, v in sorted(chosen.items())},
            "n": int(len(y)), "prevalence": float(y.mean()), "threshold": final_thr,
            "metrics_at_0.5": {k: m05[k] for k in ("accuracy", "precision", "recall", "specificity", "f1")},
            "metrics": {k: {"mean": m[k], "ci": ci[k]} for k in ci},
            "confusion_matrix": [[round(m["tn"], 1), round(m["fp"], 1)], [round(m["fn"], 1), round(m["tp"], 1)]],
            "raw_brier": m_raw["brier"], "raw_ece": m_raw["ece"],
            "thresholds": thr_table, **curves(y, pooled),
        }
        mm = metrics[t]["metrics"]
        print(f"[{t}] final={best:<18} AUC {mm['roc_auc']['mean']:.3f} "
              f"[{mm['roc_auc']['ci'][0]:.3f}, {mm['roc_auc']['ci'][1]:.3f}]  acc {mm['accuracy']['mean']:.3f}  "
              f"rec {mm['recall']['mean']:.3f} spec {mm['specificity']['mean']:.3f} thr {final_thr:.2f}  Brier {m_raw['brier']:.3f}->{m['brier']:.3f}  "
              f"ECE {m_raw['ece']:.3f}->{m['ece']:.3f}  picks {dict(chosen)}  ({time.time() - t0:.0f}s)", flush=True)

    # Example patients with their honest out-of-fold CAD probability.
    p, yc = oof_cal["CAD"], Y["CAD"]
    rng = np.random.default_rng(seed)
    picks = [int(np.argmax(np.where(yc == 1, p, -1))), int(np.argmin(np.where(yc == 0, p, 2))),
             int(np.argmax(np.where(yc == 0, p, -1))), int(np.argmin(np.where(yc == 1, p, 2)))]
    picks += [int(i) for i in rng.permutation(len(X)) if int(i) not in picks][:4]
    samples = [{"id": i, "features": {c: _py(v) for c, v in X.loc[i].items()},
                "truth": {t: _py(Y[t][i]) for t in TARGETS},
                "oof": {t: float(oof_cal[t][i]) for t in TARGETS}} for i in picks]

    raw_bytes = Path(data_path).read_bytes()
    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {"outer": f"{CFG['repeats']}x repeated stratified {CFG['outer']}-fold", "inner": f"{CFG['inner']}-fold ROC-AUC selection",
                     "calibration": "Platt scaling (cross-fitted for evaluation)", "bootstrap": CFG["boot"],
                     "operating_threshold": "per target, Youden J on out-of-fold predictions (cross-fitted for evaluation)",
                     "candidates": list(candidates(seed)), "seed": seed},
        "data": {"file": Path(data_path).name, "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                 "patients": int(len(X)), "features": int(X.shape[1])},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "train_seconds": round(time.time() - t0),
    }
    bundle = {"pipelines": final_pipes, "calibrators": calibrators, "metrics": metrics, "meta": meta,
              "importance": importance, "schema": build_schema(X), "columns": list(X.columns),
              "excluded": excluded, "samples": samples, "background": background,
              "thresholds": {t: metrics[t]["threshold"] for t in TARGETS},
              "data_file": Path(data_path).name, "is_synthetic": "synthetic" in Path(data_path).name.lower()}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / "model_bundle.joblib", compress=3)
    (out / "metrics.json").write_text(json.dumps({"meta": meta, "targets": metrics}, indent=2), encoding="utf-8")
    rows = ["| Target | Model | Threshold | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC (95% CI) | Brier | ECE |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, r in metrics.items():
        g = r["metrics"]
        rows.append(f"| {t} | {r['model']} | {r['threshold']:.2f} | {g['accuracy']['mean']:.3f} | {g['precision']['mean']:.3f} | {g['recall']['mean']:.3f} | "
                    f"{g['specificity']['mean']:.3f} | {g['f1']['mean']:.3f} | {g['roc_auc']['mean']:.3f} "
                    f"({g['roc_auc']['ci'][0]:.2f} to {g['roc_auc']['ci'][1]:.2f}) | {g['brier']['mean']:.3f} | {g['ece']['mean']:.3f} |")
    (out / "metrics.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nSaved bundle + metrics to {out}/ in {time.time() - t0:.0f}s", flush=True)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true", help="1 repeat, 3 folds: for smoke tests only")
    a = ap.parse_args()
    train(a.data, a.out, a.seed, a.fast)
