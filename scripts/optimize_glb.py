"""Reduce the triangle count of the skeleton in thorax.glb so the scene stays light on integrated GPUs.

    python scripts/optimize_glb.py [input.glb] [output.glb] [--ratio 0.3]

Heart, great vessels and coronary arteries are copied unchanged (they carry the predictions).
Bone groups (rib, cart, sternum, clav, vert, disc) are decimated with quadric edge collapse.
Requires: trimesh, fast-simplification.
"""
import sys
from pathlib import Path

import trimesh

BONES = {"rib", "cart", "sternum", "clav", "vert", "disc"}
args = [a for a in sys.argv[1:] if not a.startswith("--")]
ratio = float(sys.argv[sys.argv.index("--ratio") + 1]) if "--ratio" in sys.argv else 0.3
if "--ratio" in sys.argv:
    args.remove(sys.argv[sys.argv.index("--ratio") + 1])
root = Path(__file__).resolve().parent.parent / "static" / "models"
src = Path(args[0]) if args else root / "thorax.glb"
dst = Path(args[1]) if len(args) > 1 else src

scene, out = trimesh.load(src), trimesh.Scene()
before = after = 0
for node in scene.graph.nodes_geometry:
    transform, gname = scene.graph[node]
    mesh = scene.geometry[gname].copy()
    before += len(mesh.faces)
    if node.split("|")[0] in BONES and len(mesh.faces) > 400:
        mesh = mesh.simplify_quadric_decimation(percent=1 - ratio)
    after += len(mesh.faces)
    out.add_geometry(mesh, node_name=node, geom_name=gname, transform=transform)
dst.write_bytes(out.export(file_type="glb"))
print(f"{before:,} -> {after:,} triangles; wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
