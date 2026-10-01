"""Fast checks. They train on a synthetic dataset in --fast mode, so they never touch the real data."""
import json
import os
import re
import struct
from pathlib import Path

import numpy as np
import pytest
import shap
from fastapi.testclient import TestClient

from cardio import demo_data, explain
from cardio.data import FEATURE_INFO, LEAKAGE_KEYS, norm, prepare
from cardio.train import apply_platt, candidates, ece, fit_platt, make_preprocessor, train

ROOT = Path(__file__).resolve().parent.parent
TARGETS = {"CAD", "LAD", "LCX", "RCA"}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    d = tmp_path_factory.mktemp("art")
    csv = d / "demo_synthetic.csv"
    demo_data.make().to_csv(csv, index=False)
    metrics = train(str(csv), str(d / "artifacts"), fast=True)
    return d / "artifacts", metrics


@pytest.fixture(scope="module")
def client(trained):
    os.environ["CARDIO_ARTIFACTS"] = str(trained[0])
    from cardio import serve
    serve._state.clear()
    return TestClient(serve.app)


# ---------------------------------------------------------------- data & leakage
def test_no_target_leakage():
    X, targets, excluded = prepare(demo_data.make())
    assert not ({norm(c) for c in X.columns} & LEAKAGE_KEYS)
    assert {"LAD", "LCX", "RCA", "Cath"} <= set(excluded)
    assert set(targets) == TARGETS


def test_every_feature_has_a_glossary_entry():
    X, _, _ = prepare(demo_data.make())
    assert {norm(c) for c in X.columns} <= set(FEATURE_INFO)


def test_trained_models_never_see_leakage_columns(trained):
    import joblib
    b = joblib.load(trained[0] / "model_bundle.joblib")
    assert not ({norm(c) for c in b["columns"]} & LEAKAGE_KEYS)
    for pipe in b["pipelines"].values():
        assert not ({norm(c) for c in pipe.feature_names_in_} & LEAKAGE_KEYS)


# ---------------------------------------------------------------- evaluation
def test_metrics_structure_and_bounds(trained):
    _, m = trained
    assert set(m) == TARGETS
    for r in m.values():
        for k in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "brier", "ece"):
            lo, hi = r["metrics"][k]["ci"]
            assert lo <= hi
            assert 0 <= r["metrics"][k]["mean"] <= 1
        assert len(r["roc"]["fpr"]) == len(r["roc"]["tpr"]) >= 2
        assert len(r["thresholds"]) == 5
        assert abs(sum(r["selection_frequency"].values()) - 1) < 1e-9


def test_calibration_helpers():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < p ** 2).astype(int)           # scores are over-confident on purpose
    cal = fit_platt(y, p)
    q = apply_platt(cal, p)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)    # monotone: ranking (AUC) is preserved
    assert ece(y, q) < ece(y, p)
    assert ece(np.array([0, 1]), np.array([0.0, 1.0])) == 0


# ---------------------------------------------------------------- explanations
def test_shap_additivity_all_model_families(trained):
    """base + sum(SHAP) must reproduce the model output for every candidate family."""
    import joblib
    b = joblib.load(trained[0] / "model_bundle.joblib")
    Xb, cols = b["background"], b["columns"]
    num, cat = explain.split_columns(cols, Xb)
    pre = make_preprocessor(Xb[cols]).fit(Xb[cols])
    Xt = pre.transform(Xb)
    y = np.arange(len(Xt)) % 2
    row = pre.transform(Xb.iloc[[0]])
    assert len(explain.owners(pre, num, cat)) == Xt.shape[1]
    for name, clf in candidates(0).items():
        clf.fit(Xt, y)
        linear = name.startswith("logistic")
        ex = shap.LinearExplainer(clf, Xt) if linear else shap.TreeExplainer(clf)
        sv, base = explain.shap_matrix(ex, row)
        out, proba = base + sv.sum(), clf.predict_proba(row)[0, 1]
        if name in ("random_forest", "extra_trees"):
            assert abs(out - proba) < 1e-3, name
        else:
            assert abs(1 / (1 + np.exp(-out)) - proba) < 1e-3, name


# ---------------------------------------------------------------- API
def test_api_predict_and_explain(client):
    sch = client.get("/api/schema").json()
    assert "not a substitute" in sch["disclaimer"]
    feats = {f["name"]: f["default"] for f in sch["features"]}
    r = client.post("/api/predict", json={"features": feats}).json()
    assert set(r["predictions"]) == TARGETS
    for p in r["predictions"].values():
        assert 0 <= p["probability"] <= 1
        assert len(p["explanation"]["contributions"]) == len(feats)
        assert p["summary"]["raises"] is not None and p["summary"]["lowers"] is not None
        shares = [abs(c["share"]) for c in p["explanation"]["contributions"]]
        assert abs(sum(shares) - 100) < 1e-6
    assert client.get("/api/metrics").json()["metrics"]["CAD"]["metrics"]["roc_auc"]["mean"] > 0.5
    smp = client.get("/api/samples").json()
    assert len(smp) == 8 and set(smp[0]["oof"]) == TARGETS


def test_importance_and_model_info(client):
    imp = client.get("/api/importance").json()
    assert set(imp) == TARGETS
    rows = imp["CAD"]
    assert abs(sum(r["share"] for r in rows) - 100) < 1e-6
    assert rows == sorted(rows, key=lambda r: -r["mean_abs"])
    info = client.get("/api/model-info").json()
    assert set(info["models"]) == TARGETS and len(info["data"]["sha256"]) == 64


def test_bad_input(client):
    assert client.post("/api/predict", json={"features": {"nope": 1}}).status_code == 422
    assert client.post("/api/predict", json={"features": {"Age": "abc"}}).status_code == 422


def test_blank_input_and_range_warning(client):
    r = client.post("/api/predict", json={"features": {"Age": 200}}).json()
    assert any("outside the training range" in w for w in r["warnings"])
    assert any("left blank" in w for w in r["warnings"])


def test_prediction_changes_with_input(client):
    sch = client.get("/api/schema").json()
    feats = {f["name"]: f["default"] for f in sch["features"]}
    a = client.post("/api/predict", json={"features": feats}).json()["predictions"]["CAD"]["probability"]
    feats["Age"] = 85
    feats["Typical Chest Pain"] = 1
    b = client.post("/api/predict", json={"features": feats}).json()["predictions"]["CAD"]["probability"]
    assert a != b


# ---------------------------------------------------------------- 3D asset consistency
def _glb_node_names(path: Path) -> list[str]:
    raw = path.read_bytes()
    assert raw[:4] == b"glTF"
    length, kind = struct.unpack("<II", raw[12:20])
    assert kind == 0x4E4F534A                              # first chunk is JSON
    return [n.get("name", "") for n in json.loads(raw[20:20 + length])["nodes"]]


def test_glb_contains_everything_the_viewer_needs():
    names = _glb_node_names(ROOT / "static" / "models" / "thorax.glb")
    groups = {n.split("|")[0] for n in names}
    anatomy = (ROOT / "static" / "anatomy.js").read_text(encoding="utf-8")
    declared = set(re.findall(r"^\s+(\w+): \{ kind:", anatomy, re.M))
    assert declared <= groups, declared - groups
    assert {"lad", "lcx", "rca"} <= groups                  # the three predicted vessels
    for part in re.findall(r"'atria\|(FJ\d+)'", anatomy):  # individually selectable parts
        assert any(n.startswith(f"atria|{part}|") for n in names)
    assert sum(n.startswith("lad|") for n in names) > 1     # several segments per vessel
