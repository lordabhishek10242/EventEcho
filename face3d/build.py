"""Build the face GLB.

    python3 -m face3d.build --out web/face3d/face.glb --preview out/preview.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import head as headmod
from .glb import GlbBuilder, inspect


def assemble(params: dict | None = None, quality: float = 1.0):
    nu = int(168 * quality)
    nv = int(128 * quality)
    parts = []
    P, N, C, tris = headmod.build_head(params, nu, nv)
    parts.append({"name": "face", "positions": P, "normals": N, "colours": C, "tris": tris})
    parts += headmod.build_eyes(params, max(20, int(34 * quality)))
    parts += headmod.build_ears(params, max(16, int(26 * quality)))
    parts += headmod.build_hair(params, int(nu * 0.90), int(nv * 0.90))
    parts += headmod.build_neck_and_torso(params, max(28, int(72 * quality)),
                                          max(22, int(56 * quality)))
    return parts


def write_glb(parts, path: str) -> dict:
    builder = GlbBuilder()
    for part in parts:
        material = builder.add_material(
            part["name"], roughness=0.72 if part["name"] != "eye" else 0.18,
            use_vertex_color=True, double_sided=part.get("double_sided", False),
        )
        tris = part["tris"]
        if "mask" in part:
            keep = part["mask"]
            tris = tris[keep[tris].all(axis=1)]
        builder.add_mesh(part["name"], part["positions"], part["normals"], tris,
                         material, colors=part["colours"])
    data = builder.to_bytes()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return inspect(data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="web/face3d/face.glb")
    ap.add_argument("--preview", default=None)
    ap.add_argument("--params", default=None, help="JSON file of parameter overrides")
    ap.add_argument("--quality", type=float, default=1.0)
    args = ap.parse_args(argv)

    params = None
    if args.params:
        with open(args.params) as fh:
            params = json.load(fh)

    parts = assemble(params, args.quality)
    info = write_glb(parts, args.out)
    print(json.dumps(info, indent=2))
    if info["bytes"] > 2_000_000:
        print("ERROR: over the 2 MB limit", file=sys.stderr)
        return 1

    if args.preview:
        from .render import sheet
        os.makedirs(os.path.dirname(os.path.abspath(args.preview)), exist_ok=True)
        sheet(parts, args.preview)
        print(f"preview -> {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
