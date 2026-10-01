"""SHAP helpers shared by training (global importance) and serving (per-patient explanations).

A fitted pipeline is ``preprocess -> classifier``. SHAP runs on the preprocessed matrix; values of
one-hot columns are then summed back onto the original measurement, so every input feature
appears exactly once.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

PROB_UNIT_MODELS = (RandomForestClassifier, ExtraTreesClassifier)


def split_columns(columns: list[str], reference: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Numeric columns first, then text columns (the ColumnTransformer's output order)."""
    num = [c for c in columns if pd.api.types.is_numeric_dtype(reference[c])]
    return num, [c for c in columns if c not in num]


def owners(pre, num: list[str], cat: list[str]) -> list[str]:
    """Map every transformed column back to the original input feature it came from."""
    out = list(num)
    if cat:
        oh = pre.named_transformers_["cat"].named_steps["oh"]
        drops = oh.drop_idx_ if oh.drop_idx_ is not None else [None] * len(cat)
        for col, cats, d in zip(cat, oh.categories_, drops):
            out += [col] * (len(cats) - (0 if d is None else 1))
    return out


def make_explainer(pipe, background: pd.DataFrame, num: list[str], cat: list[str]):
    """Return (explainer, units, owners) for a fitted pipeline."""
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    if isinstance(clf, LogisticRegression):
        ex, units = shap.LinearExplainer(clf, pre.transform(background)), "log-odds"
    else:
        ex = shap.TreeExplainer(clf)
        units = "probability" if isinstance(clf, PROB_UNIT_MODELS) else "log-odds"
    return ex, units, owners(pre, num, cat)


def shap_matrix(ex, X) -> tuple[np.ndarray, float]:
    """SHAP values for the positive class, shape (n_rows, n_transformed_columns), plus base value."""
    sv = ex.shap_values(X)
    if isinstance(sv, list):
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    return sv, float(np.ravel(ex.expected_value)[-1])


def by_feature(sv: np.ndarray, col_owners: list[str]) -> pd.DataFrame:
    """Sum transformed columns that belong to the same original feature."""
    return pd.DataFrame(sv, columns=col_owners).T.groupby(level=0, sort=False).sum().T


def global_importance(pipe, background: pd.DataFrame, sample: pd.DataFrame, num, cat) -> list[dict]:
    """Mean absolute SHAP per measurement over ``sample``, as shares of the total (sorted)."""
    ex, units, own = make_explainer(pipe, background, num, cat)
    sv, _ = shap_matrix(ex, pipe.named_steps["pre"].transform(sample))
    mean_abs = by_feature(sv, own).abs().mean()
    total = float(mean_abs.sum()) or 1.0
    rows = [{"feature": f, "mean_abs": float(v), "share": float(v / total * 100)} for f, v in mean_abs.items()]
    return sorted(rows, key=lambda r: -r["mean_abs"])
