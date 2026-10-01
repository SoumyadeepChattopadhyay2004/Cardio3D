"""Build static/models/thorax.glb (heart, great vessels, coronary arteries, ribcage) from BodyParts3D.

Source geometry: BodyParts3D 4.0, (c) The Database Center for Life Science, CC BY 4.0,
as re-packed by https://github.com/ashemag/human-atlas (public/models/atlas.json + body-N.bin).
Usage:  python scripts/build_thorax_glb.py <dir containing atlas.json and body-8/9/11/12/13/14.bin>
Axes of the result: +x = patient's left, +y = up, +z = anterior; ribcage ~3.2 units tall.
"""
import json
import re
import sys
from pathlib import Path
import numpy as np
import trimesh

src = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out = Path(__file__).resolve().parent.parent / "static" / "models" / "thorax.glb"
atlas = json.loads((src / "atlas.json").read_text())
chunks = {n: (src / f"body-{n}.bin").read_bytes() for n in (8, 9, 11, 12, 13, 14)}
ORD = "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth".split()


def group_of(p):
    n, pid = p["name"].lower(), p["id"]
    i = int(pid[2:]) if pid[2:].isdigit() else -1
    if pid == "FJ2428": return "vent"
    if pid in ("FJ2438", "FJ2439"): return "atria"
    if pid in ("FJ3413", "FJ3411", "FJ3417", "FJ3483", "FJ3479"): return "aorta"
    if pid in ("FJ2966", "FJ2924", "FJ3019"): return "pulm"
    if "pulmonary vein" in n and p["chunk"] == 11: return "pvein"
    if pid == "FJ3645": return "vena"          # superior vena cava (the inferior one leaves the chest and is omitted)
    if 2631 <= i <= 2648: return "lad"
    if 2649 <= i <= 2654: return "lcx"
    if 2667 <= i <= 2677 or 2692 <= i <= 2700 or 2714 <= i <= 2723: return "rca"
    if i == 2737: return "lm"
    if p["system"] == "skeletal":
        if re.search(r" rib$", n): return "rib"
        if "costal cartilage" in n: return "cart"
        if pid in ("FJ3178", "FJ3290", "FJ3153"): return "sternum"
        if "clavicle" in n: return "clav"
        if re.fullmatch(r"(%s) thoracic vertebra" % "|".join(ORD), n): return "vert"
        if re.fullmatch(r"intervertebral disk of (%s) thoracic vertebra" % "|".join(ORD), n): return "disc"
    return None


def load(p):
    b, vc = chunks[p["chunk"]], p["vertexCount"]
    pos = np.frombuffer(b, "<f4", vc * 3, p["positions"]).reshape(-1, 3).astype(np.float64)
    nor = np.frombuffer(b, "<i2", vc * 3, p["normals"]).reshape(-1, 3).astype(np.float64) / 32767
    idx = np.frombuffer(b, "<u4", p["indexCount"], p["indices"]).reshape(-1, 3).astype(np.int64)
    return pos, nor, idx


parts = [(group_of(p), p) for p in atlas["parts"] if p["chunk"] in chunks]
parts = [(g, p) for g, p in parts if g]
data = {p["id"]: load(p) for _, p in parts}
rib = np.vstack([data[p["id"]][0] for g, p in parts if g == "rib"])
ymin, ymax = rib[:, 1].min(), rib[:, 1].max()
s = 3.2 / (ymax - ymin)
c = np.array([0.0, (ymin + ymax) / 2, (rib[:, 2].min() + rib[:, 2].max()) / 2])
INFLATE = {"lad": 0.014, "lcx": 0.014, "rca": 0.014, "lm": 0.014}   # thicken 3 mm arteries so they read at this scale

scene = trimesh.Scene()
for g, p in parts:
    pos, nor, idx = data[p["id"]]
    pos = (pos - c) * s + np.array([0, 0.325, 0])
    if g in INFLATE:
        nn = nor / np.maximum(np.linalg.norm(nor, axis=1, keepdims=True), 1e-9)
        pos = pos + nn * INFLATE[g]
    m = trimesh.Trimesh(pos, idx, vertex_normals=nor, process=False)
    scene.add_geometry(m, node_name=f"{g}|{p['id']}|{p['name']}".replace(" ", "_"), geom_name=p["id"])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(scene.export(file_type="glb"))
cnt = {}
for g, _ in parts: cnt[g] = cnt.get(g, 0) + 1
print("parts per group:", cnt)
print("GLB:", out, round(out.stat().st_size / 1e6, 2), "MB; scale", round(s, 3))
for grp in ("vent", "aorta", "pulm", "vena", "lad", "lcx", "rca", "rib", "sternum"):
    v = np.vstack([(data[p["id"]][0] - c) * s + [0, 0.325, 0] for g, p in parts if g == grp])
    print(grp, np.round(v.min(0), 2), np.round(v.max(0), 2))
