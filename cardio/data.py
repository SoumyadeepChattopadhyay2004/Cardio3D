"""Dataset loading, cleaning, target construction and leakage control.

Built for the "Extension of Z-Alizadeh Sani" dataset (UCI), but column matching is
case/space/punctuation-insensitive so small naming differences between file versions
do not break the pipeline.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ["CAD", "LAD", "LCX", "RCA"]

# Columns that must NEVER be model inputs (brief, requirement 1d): the vessel labels,
# the catheterisation outcome and the overall CAD label all encode the answer.
LEAKAGE_KEYS = {"lad", "lcx", "rca", "cath", "cad"}
ID_KEYS = {"id", "patientid", "unnamed0", "index"}

POS = {"1", "1.0", "y", "yes", "true", "t", "cad", "stenosis", "stenotic",
       "positive", "pos", "abnormal", "present", "disease"}
NEG = {"0", "0.0", "n", "no", "false", "f", "normal", "negative", "neg",
       "none", "absent", "healthy"}

# Form grouping (keys are normalised column names).
GROUP_ORDER = ["Demographics", "History & risk factors", "Clinical examination",
               "ECG", "Laboratory", "Echocardiography", "Other"]
_G = {
    "Demographics": "age weight length sex bmi",
    "History & risk factors": "dm htn currentsmoker exsmoker fh obesity crf cva airwaydisease "
                              "thyroiddisease chf dlp",
    "Clinical examination": "bp pr edema weakperipheralpulse lungrales systolicmurmur "
                            "diastolicmurmur typicalchestpain dyspnea functionclass atypical "
                            "nonanginal exertionalcp lowthang",
    "ECG": "qwave stelevation stdepression tinversion lvh poorrprogression bbb",
    "Laboratory": "fbs cr tg ldl hdl bun esr hb k na wbc lymph neut plt",
    "Echocardiography": "eftte regionrwma vhd",
}
GROUPS = {k: g for g, keys in _G.items() for k in keys.split()}

UNITS = {"age": "years", "weight": "kg", "length": "cm", "bmi": "kg/m²", "bp": "mmHg",
         "pr": "bpm", "fbs": "mg/dL", "cr": "mg/dL", "tg": "mg/dL", "ldl": "mg/dL",
         "hdl": "mg/dL", "bun": "mg/dL", "esr": "mm/h", "hb": "g/dL", "k": "mEq/L",
         "na": "mEq/L", "wbc": "cells/µL", "lymph": "%", "neut": "%", "plt": "1000/µL",
         "eftte": "%"}


# Plain-language glossary shown as tooltips in the UI (keys are normalised column names).
FEATURE_INFO = {
    "age": ("Age", "Patient age. Coronary disease becomes more common with age."),
    "weight": ("Body weight", "Weight in kilograms."),
    "length": ("Height", "Body height in centimetres (called Length in the dataset)."),
    "sex": ("Sex", "Recorded sex. The dataset spells female as 'Fmale'."),
    "bmi": ("Body mass index", "Weight relative to height; high values indicate overweight or obesity."),
    "dm": ("Diabetes mellitus", "1 if the patient has diabetes."),
    "htn": ("Hypertension", "1 if the patient has high blood pressure."),
    "currentsmoker": ("Current smoker", "1 if the patient currently smokes."),
    "exsmoker": ("Ex-smoker", "1 if the patient used to smoke."),
    "fh": ("Family history", "1 if a close relative had coronary artery disease."),
    "obesity": ("Obesity", "Y if clinically obese (BMI above 25 in this dataset)."),
    "crf": ("Chronic renal failure", "Y if the patient has long-term kidney failure."),
    "cva": ("Stroke history", "Y if the patient has had a cerebrovascular accident."),
    "airwaydisease": ("Airway disease", "Y if the patient has asthma, COPD or similar."),
    "thyroiddisease": ("Thyroid disease", "Y if the patient has a thyroid disorder."),
    "chf": ("Congestive heart failure", "Y if the patient has heart failure."),
    "dlp": ("Dyslipidemia", "Y if blood fats (cholesterol or triglycerides) are abnormal."),
    "bp": ("Blood pressure", "Blood pressure in mmHg, as recorded in the dataset."),
    "pr": ("Pulse rate", "Heart rate in beats per minute."),
    "edema": ("Oedema", "1 if there is fluid swelling, typically in the legs."),
    "weakperipheralpulse": ("Weak peripheral pulse", "Y if pulses in the limbs are weak."),
    "lungrales": ("Lung rales", "Y if crackling sounds are heard in the lungs, a sign of fluid."),
    "systolicmurmur": ("Systolic murmur", "Y if a heart murmur is heard while the heart contracts."),
    "diastolicmurmur": ("Diastolic murmur", "Y if a heart murmur is heard while the heart relaxes."),
    "typicalchestpain": ("Typical chest pain", "1 if the pain matches classic angina (pressure on exertion, relieved by rest)."),
    "dyspnea": ("Shortness of breath", "Y if the patient reports breathlessness."),
    "functionclass": ("NYHA function class", "0 to 4 scale of how much symptoms limit daily activity."),
    "atypical": ("Atypical chest pain", "Y if the chest pain does not fit the classic pattern."),
    "nonanginal": ("Non-anginal chest pain", "Y if the chest pain is unlikely to come from the heart."),
    "exertionalcp": ("Exertional chest pain", "Y if chest pain is triggered by exertion."),
    "lowthang": ("Low-threshold angina", "Y if angina appears at low levels of effort."),
    "qwave": ("Q wave on ECG", "1 if a pathological Q wave is present, often a sign of old heart-muscle damage."),
    "stelevation": ("ST elevation on ECG", "1 if the ST segment is raised, a sign of acute injury."),
    "stdepression": ("ST depression on ECG", "1 if the ST segment is lowered, a sign of reduced blood supply."),
    "tinversion": ("T-wave inversion on ECG", "1 if T waves point downwards, which can signal ischaemia."),
    "lvh": ("Left ventricular hypertrophy", "Y if the left ventricle wall is thickened (ECG finding)."),
    "poorrprogression": ("Poor R-wave progression", "Y if R waves fail to grow across the chest leads."),
    "bbb": ("Bundle branch block", "Conduction delay on the ECG: N (none), LBBB (left) or RBBB (right)."),
    "fbs": ("Fasting blood sugar", "Blood glucose after fasting, mg/dL."),
    "cr": ("Creatinine", "Kidney function marker, mg/dL."),
    "tg": ("Triglycerides", "Blood fat, mg/dL."),
    "ldl": ("LDL cholesterol", "'Bad' cholesterol, mg/dL."),
    "hdl": ("HDL cholesterol", "'Good' cholesterol, mg/dL. Higher is protective."),
    "bun": ("Blood urea nitrogen", "Kidney function marker, mg/dL."),
    "esr": ("Erythrocyte sedimentation rate", "General inflammation marker, mm/h."),
    "hb": ("Haemoglobin", "Oxygen-carrying protein in the blood, g/dL."),
    "k": ("Potassium", "Blood potassium, mEq/L."),
    "na": ("Sodium", "Blood sodium, mEq/L."),
    "wbc": ("White blood cell count", "Cells per microlitre of blood."),
    "lymph": ("Lymphocytes", "Share of white cells that are lymphocytes, %."),
    "neut": ("Neutrophils", "Share of white cells that are neutrophils, %."),
    "plt": ("Platelet count", "Thousands per microlitre of blood."),
    "eftte": ("Ejection fraction (echo)", "Share of blood the left ventricle pumps out per beat, %. Low values mean weaker pumping."),
    "regionrwma": ("Wall-motion abnormality region", "Echo finding: 0 means normal wall motion, 1 to 4 code the affected region as recorded in the dataset."),
    "vhd": ("Valvular heart disease", "Severity of any valve disease (none, mild, moderate, severe)."),
}


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Download the 'Extension of Z-Alizadeh Sani Dataset' "
            "from the UCI repository into data/, or run `python -m cardio.demo_data` for a "
            "synthetic stand-in (for testing the software only).")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).str.match(r"^Unnamed")]
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].astype("string").str.strip()
            s = s.mask(s.str.lower().isin(["", "nan", "na", "n/a", "null", "?"]))
            num = pd.to_numeric(s, errors="coerce")
            if s.notna().sum() and num.notna().sum() >= 0.95 * s.notna().sum():
                df[c] = num.astype(float)
            else:
                df[c] = s.astype(object).where(s.notna(), np.nan)
    return df.reset_index(drop=True)


def binarize(s: pd.Series, name: str) -> pd.Series:
    """Convert a label column to float {0,1,NaN}."""
    if pd.api.types.is_numeric_dtype(s):
        if set(np.unique(s.dropna())) <= {0, 1}:
            return s.astype(float)
        return (s >= 50).astype(float).where(s.notna())  # % diameter stenosis; CAD if >=50%
    low = s.astype(str).str.strip().str.lower()
    unknown = set(low[s.notna()]) - POS - NEG
    if unknown:
        raise ValueError(f"Unrecognised values {sorted(unknown)} in label column '{name}'.")
    out = low.map(lambda v: 1.0 if v in POS else 0.0 if v in NEG else np.nan)
    return out.where(s.notna())


def prepare(df: pd.DataFrame):
    """Return (X features, {target: 0/1 series}, list of excluded columns)."""
    df = clean(df)
    keymap = {norm(c): c for c in df.columns}
    cad_col = keymap.get("cad") or keymap.get("cath")
    if cad_col is None:
        raise ValueError("No overall CAD label found (expected a 'CAD' or 'Cath' column).")
    targets = {"CAD": binarize(df[cad_col], cad_col)}
    for v in ("LAD", "LCX", "RCA"):
        if v.lower() not in keymap:
            raise ValueError(f"Missing vessel label column '{v}'.")
        targets[v] = binarize(df[keymap[v.lower()]], v)
    excluded = [c for c in df.columns if norm(c) in LEAKAGE_KEYS | ID_KEYS]
    X = df.drop(columns=excluded).dropna(axis=1, how="all")
    leaked = {norm(c) for c in X.columns} & LEAKAGE_KEYS
    assert not leaked, f"Target leakage: {leaked}"
    return X, targets, excluded


def build_schema(X: pd.DataFrame) -> list[dict]:
    """Describe every input feature so the UI can build its form dynamically."""
    feats = []
    for c in X.columns:
        s, k = X[c], norm(c)
        label, about = FEATURE_INFO.get(k, (c, ""))
        item = {"name": c, "key": k, "group": GROUPS.get(k, "Other"), "unit": UNITS.get(k, ""),
                "label": label, "about": about}
        if pd.api.types.is_numeric_dtype(s):
            u = sorted(s.dropna().unique().tolist())
            if len(u) <= 6 and all(float(v).is_integer() for v in u):
                item.update(type="choice", options=[int(v) for v in u],
                            default=int(s.mode().iloc[0]))
            else:
                item.update(type="number", min=float(s.min()), max=float(s.max()),
                            default=float(round(s.median(), 2)))
        else:
            item.update(type="choice", options=sorted(s.dropna().astype(str).unique().tolist()),
                        default=str(s.mode().iloc[0]))
        feats.append(item)
    feats.sort(key=lambda f: GROUP_ORDER.index(f["group"]))
    return feats
