"""Generate a SYNTHETIC stand-in dataset with the same column layout as the
Extension of Z-Alizadeh Sani dataset.

It exists only so the software can be run and tested without the real data.
Metrics obtained on it are meaningless for clinical purposes. The output file name
contains "synthetic" so the app displays a visible warning.

    python -m cardio.demo_data            ->  data/demo_synthetic.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(n: int = 303, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    yn = lambda p: np.where(rng.random(n) < p, "Y", "N")
    age = rng.integers(30, 86, n)
    sex = rng.choice(["Male", "Female"], n, p=[0.55, 0.45])
    height = np.where(sex == "Male", rng.normal(171, 7, n), rng.normal(158, 6, n)).round()
    weight = np.clip(rng.normal(27, 4, n) * (height / 100) ** 2, 45, 120).round()
    bmi = (weight / (height / 100) ** 2).round(2)
    dm, htn = rng.random(n) < 0.25 + 0.004 * (age - 30), rng.random(n) < 0.3 + 0.006 * (age - 30)
    smoker = rng.random(n) < 0.2
    typ_cp = rng.random(n) < 0.35
    ldl = rng.normal(105, 35, n).clip(40, 250).round()
    hdl = rng.normal(40, 10, n).clip(15, 90).round(1)
    ef = rng.normal(47, 8, n).clip(15, 65).round()
    st_el, t_inv = rng.random(n) < 0.06, rng.random(n) < 0.25
    q_wave, rwma = rng.random(n) < 0.08, rng.choice([0, 1, 2, 3, 4], n, p=[.6, .15, .12, .08, .05])

    base = (0.04 * (age - 58) + 0.7 * dm + 0.5 * htn + 0.6 * smoker + 1.0 * typ_cp
            + 0.01 * (ldl - 105) - 0.02 * (hdl - 40) - 0.05 * (ef - 47) + 0.4 * (rwma > 0))
    lad = base + 0.9 * st_el + 0.6 * t_inv + rng.normal(0, 1, n) > 1.0
    lcx = base + 0.5 * q_wave + rng.normal(0, 1, n) > 1.5
    rca = base - 0.3 * (sex == "Female") + 0.4 * (rwma >= 3) + rng.normal(0, 1, n) > 1.4
    cad = lad | lcx | rca

    df = pd.DataFrame({
        "Age": age, "Weight": weight, "Length": height, "Sex": sex, "BMI": bmi,
        "DM": dm.astype(int), "HTN": htn.astype(int), "Current Smoker": smoker.astype(int),
        "EX-Smoker": (rng.random(n) < 0.05).astype(int), "FH": (rng.random(n) < 0.12).astype(int),
        "Obesity": np.where(bmi > 30, "Y", "N"), "CRF": yn(0.03), "CVA": yn(0.03),
        "Airway disease": yn(0.08), "Thyroid Disease": yn(0.03), "CHF": yn(0.02), "DLP": yn(0.4),
        "BP": rng.normal(130, 18, n).clip(90, 190).round(), "PR": rng.normal(75, 10, n).clip(50, 110).round(),
        "Edema": rng.integers(0, 2, n) * (rng.random(n) < 0.1), "Weak Peripheral Pulse": yn(0.02),
        "Lung rales": yn(0.06), "Systolic Murmur": yn(0.2), "Diastolic Murmur": yn(0.03),
        "Typical Chest Pain": typ_cp.astype(int), "Dyspnea": yn(0.4),
        "Function Class": rng.choice([0, 1, 2, 3, 4], n, p=[.5, .2, .18, .09, .03]),
        "Atypical": yn(0.3), "Nonanginal": yn(0.15), "Exertional CP": yn(0.02),
        "LowTH Ang": yn(0.05), "Q Wave": q_wave.astype(int), "St Elevation": st_el.astype(int),
        "St Depression": (rng.random(n) < 0.3).astype(int), "Tinversion": t_inv.astype(int),
        "LVH": yn(0.1), "Poor R Progression": yn(0.08),
        "BBB": rng.choice(["N", "LBBB", "RBBB"], n, p=[.9, .05, .05]),
        "FBS": rng.normal(110, 40, n).clip(60, 350).round(), "CR": rng.normal(1.1, 0.3, n).clip(0.5, 3).round(1),
        "TG": rng.normal(150, 80, n).clip(40, 600).round(), "LDL": ldl, "HDL": hdl,
        "BUN": rng.normal(17, 6, n).clip(5, 60).round(), "ESR": rng.normal(20, 14, n).clip(1, 90).round(),
        "HB": rng.normal(13, 1.6, n).clip(8, 17).round(1), "K": rng.normal(4.2, 0.4, n).clip(3, 6).round(1),
        "Na": rng.normal(141, 3, n).clip(128, 150).round(), "WBC": rng.normal(7500, 2000, n).clip(3000, 16000).round(-2),
        "Lymph": rng.normal(32, 9, n).clip(5, 60).round(), "Neut": rng.normal(60, 10, n).clip(30, 90).round(),
        "PLT": rng.normal(230, 55, n).clip(90, 500).round(), "EF-TTE": ef, "Region RWMA": rwma,
        "VHD": rng.choice(["N", "mild", "Moderate", "Severe"], n, p=[.55, .3, .12, .03]),
        "LAD": lad.astype(int), "LCX": lcx.astype(int), "RCA": rca.astype(int),
        "Cath": np.where(cad, "Cad", "Normal"),
    })
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/demo_synthetic.csv")
    out = Path(ap.parse_args().out)
    out.parent.mkdir(parents=True, exist_ok=True)
    make().to_csv(out, index=False)
    print(f"Wrote SYNTHETIC demo data to {out}")
