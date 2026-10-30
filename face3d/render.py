"""Offline preview renderer: z-buffered software rasteriser to PNG.

Build-time tool only (needs numpy + Pillow); the shipped service has no
dependency on it.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .head import vertex_normals


def camera(azimuth_deg: float, elevation_deg: float = 0.0):
    """Returns (right, up, eye): `eye` points from the subject toward the camera."""
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    eye = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    right = np.cross(np.array([0.0, 1.0, 0.0]), eye)
    right /= np.linalg.norm(right)
    up = np.cross(eye, right)
    return right, up, eye


def render(parts, width=420, height=520, azimuth=0.0, elevation=4.0,
           background=(0.09, 0.10, 0.12), fov_scale=1.0) -> Image.Image:
    right, up, eye = camera(azimuth, elevation)
    head_parts = [p for p in parts if p["name"] in ("face", "hair", "ear", "eye")]
    if not head_parts:
        head_parts = parts
    head_pos = np.concatenate([p["positions"] for p in head_parts])
    centre = head_pos.mean(axis=0)
    centre[1] = 0.0
    extent = np.abs(head_pos - centre).max() * 2.25 / fov_scale

    zbuf = np.full((height, width), -np.inf)
    img = np.zeros((height, width, 3), dtype=np.float64)
    img[:] = np.array(background)

    key = np.array([-0.45, 0.62, 0.65])
    key /= np.linalg.norm(key)
    fill = np.array([0.75, 0.10, 0.35])
    fill /= np.linalg.norm(fill)

    for part in parts:
        P = part["positions"] - centre
        N = part["normals"]
        C = np.clip(part["colours"], 0, 1)
        tris = part["tris"]
        if "mask" in part:
            keep = part["mask"]
            tris = tris[keep[tris].all(axis=1)]
        if len(tris) == 0:
            continue

        # Screen space.
        sx = P @ right
        sy = P @ up
        depth = P @ eye
        px = (sx / extent + 0.5) * width
        py = (0.5 - sy / extent) * height

        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        p0 = np.stack([px[a], py[a]], axis=1)
        p1 = np.stack([px[b], py[b]], axis=1)
        p2 = np.stack([px[c], py[c]], axis=1)

        area = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
        facing = (N[a] + N[b] + N[c]) @ eye
        sel = (np.abs(area) > 1e-9)
        if not part.get("double_sided"):
            sel &= facing > 0.02
        if not sel.any():
            continue

        for t in np.nonzero(sel)[0]:
            i0, i1, i2 = a[t], b[t], c[t]
            x0, y0 = p0[t]
            x1, y1 = p1[t]
            x2, y2 = p2[t]
            minx = max(int(np.floor(min(x0, x1, x2))), 0)
            maxx = min(int(np.ceil(max(x0, x1, x2))), width - 1)
            miny = max(int(np.floor(min(y0, y1, y2))), 0)
            maxy = min(int(np.ceil(max(y0, y1, y2))), height - 1)
            if minx > maxx or miny > maxy:
                continue
            ys, xs = np.mgrid[miny:maxy + 1, minx:maxx + 1]
            det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(det) < 1e-9:
                continue
            w0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / det
            w1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / det
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not inside.any():
                continue
            z = w0 * depth[i0] + w1 * depth[i1] + w2 * depth[i2]
            current = zbuf[miny:maxy + 1, minx:maxx + 1]
            update = inside & (z > current)
            if not update.any():
                continue
            n = (w0[..., None] * N[i0] + w1[..., None] * N[i1] + w2[..., None] * N[i2])
            n_len = np.linalg.norm(n, axis=-1)
            n_len[n_len == 0] = 1.0
            n = n / n_len[..., None]
            flip = np.where((n @ eye) < 0.0, -1.0, 1.0)
            n = n * flip[..., None]
            albedo = (w0[..., None] * C[i0] + w1[..., None] * C[i1] + w2[..., None] * C[i2])

            lam = np.clip(n @ key, 0, 1) * 0.85
            fill_amt = np.clip(n @ fill, 0, 1) * 0.22
            ambient = 0.30 + 0.14 * np.clip(n[:, :, 1], 0, 1)
            rim = np.clip(1.0 - np.abs(n @ eye), 0, 1) ** 3 * 0.10
            shade = (ambient + lam + fill_amt + rim)[..., None]
            colour = albedo * shade

            sub = img[miny:maxy + 1, minx:maxx + 1]
            sub[update] = colour[update]
            current[update] = z[update]

    img = np.clip(img, 0, 1) ** (1 / 2.2)
    return Image.fromarray((img * 255).astype(np.uint8))


def sheet(parts, path, size=(420, 520)):
    views = [("front", 0.0, 3.0), ("three-quarter", 38.0, 4.0), ("profile", 88.0, 3.0)]
    tiles = [render(parts, size[0], size[1], az, el) for _n, az, el in views]
    out = Image.new("RGB", (size[0] * len(tiles), size[1]))
    for i, tile in enumerate(tiles):
        out.paste(tile, (i * size[0], 0))
    out.save(path)
    return path
