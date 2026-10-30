"""Parametric human head, built from primitives with numpy.

Units are centimetres: X right, Y up, Z forward (the face looks toward +Z),
origin at the centre of the head, eye line a little above y = 0.

The skull is a lofted surface: at every height the cross-section is a
superellipse with an independent half-width and independent front/back radii,
interpolated from anatomical profile tables. That fixes the silhouette, which
is what a viewer recognises first; the features (brow, sockets, nose, lips,
chin) are gaussian blobs pushed along the surface normal on top of it.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT = {
    # -- global scale --------------------------------------------------------
    "height": 22.0,     # vertex to chin
    "width": 15.8,      # bizygomatic width
    "depth": 19.2,      # glabella to occiput
    "yaw": 0.0,         # reserved: mesh is always canonical, viewers rotate

    # -- silhouette knobs (multipliers on the profile tables) ----------------
    "jaw_width": 1.06,
    "chin_width": 1.0,
    "cheek_width": 1.07,
    "forehead_width": 1.0,
    "cranium": 1.0,
    "face_length": 1.0,

    # -- features ------------------------------------------------------------
    "eye_spacing": 3.05,
    "eye_y": 0.8,
    "eye_r": 1.52,
    "socket": 0.62,
    "brow": 1.0,
    "nose_len": 1.0,
    "nose_width": 1.0,
    "nose_tip": 0.78,
    "lip_upper": 1.0,
    "lip_lower": 1.0,
    "mouth_width": 2.50,
    "mouth_y": -6.40,
    "chin": 1.0,
    "cheek_full": 1.0,
    "ears": 1.0,

    # -- colour --------------------------------------------------------------
    "skin": (0.82, 0.63, 0.50),
    "skin_shadow": (0.63, 0.44, 0.36),
    "lip_colour": (0.66, 0.38, 0.34),
    "brow_colour": (0.15, 0.10, 0.08),
    "hair_colour": (0.13, 0.10, 0.085),
    "hair_volume": 0.50,
    "hairline": 6.4,
    "hair_style": "short",
    "eye_iris": (0.30, 0.20, 0.11),
    "shirt": (0.20, 0.23, 0.28),
}

# ---------------------------------------------------------------------------
# profile tables: y_norm = 0 at the chin, 1 at the vertex
# ---------------------------------------------------------------------------
Y_CTRL = np.array([0.00, 0.02, 0.05, 0.09, 0.14, 0.20, 0.27, 0.34, 0.42,
                   0.50, 0.58, 0.66, 0.74, 0.82, 0.90, 0.96, 1.00])
W_CTRL = np.array([1.55, 2.70, 3.70, 4.65, 5.45, 6.10, 6.60, 6.95, 7.20,
                   7.35, 7.50, 7.42, 7.15, 6.55, 5.25, 3.25, 0.60])
RF_CTRL = np.array([1.90, 3.60, 4.80, 5.55, 6.05, 6.30, 6.45, 6.55, 6.60,
                    6.60, 6.52, 6.38, 6.10, 5.45, 4.05, 2.15, 0.50])
RB_CTRL = np.array([1.40, 3.30, 4.70, 5.90, 6.90, 7.70, 8.30, 8.75, 9.05,
                    9.22, 9.30, 9.18, 8.70, 7.60, 5.85, 3.35, 0.55])


def _params(params: dict | None) -> dict:
    out = dict(DEFAULT)
    if params:
        out.update(params)
    return out


def _interp(y_norm, table=W_CTRL):
    return np.interp(y_norm, Y_CTRL, table)


def loft_grid(nu: int, nv: int):
    """Returns (theta, t) grids of shape (nu, nv) to match grid_tris.

    theta = azimuth (0 at the front, + toward +X); t = 0 at the chin,
    1 at the vertex. Rings are packed toward the top and the chin, where the
    silhouette curves hardest.
    """
    theta = np.linspace(-math.pi, math.pi, nu, endpoint=False)
    raw = np.linspace(0.0, 1.0, nv)
    # Ease the ring distribution so poles are denser than the mid-face.
    eased = 0.5 - 0.5 * np.cos(raw * math.pi)
    t = 1.0 - eased
    TH, T = np.meshgrid(theta, t, indexing="ij")
    return TH, T


def grid_tris(nu: int, nv: int, mask: np.ndarray | None = None) -> np.ndarray:
    idx = np.arange(nu * nv).reshape(nu, nv)
    a = idx
    b = np.roll(idx, -1, axis=0)
    c = np.roll(idx, -1, axis=1)
    d = np.roll(np.roll(idx, -1, axis=0), -1, axis=1)
    tris = np.concatenate([
        np.stack([a.ravel(), b.ravel(), c.ravel()], axis=1),
        np.stack([b.ravel(), d.ravel(), c.ravel()], axis=1),
    ], axis=0)
    if mask is not None:
        keep = mask.ravel()
        tris = tris[keep[tris].all(axis=1)]
    return tris.astype(np.int64)


def vertex_normals(P: np.ndarray, tris: np.ndarray,
                   reference: np.ndarray | None = None) -> np.ndarray:
    """Area-weighted vertex normals, oriented away from `reference`.

    All the parts here are convex-ish closed shells, so "away from the centroid
    of the part" is enough to make the orientation independent of how the grid
    happens to wind.
    """
    n = np.zeros_like(P)
    a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    fn = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    length = np.linalg.norm(n, axis=1)
    length[length == 0] = 1.0
    n = n / length[:, None]
    if reference is None:
        reference = P.mean(axis=0)
    outward = ((P - reference) * n).sum(axis=1)
    n = n * np.where(outward < 0.0, -1.0, 1.0)[:, None]
    return n


def _blob(P, centre, sigma, amp):
    centre = np.asarray(centre, dtype=np.float64)
    d2 = ((P - centre) ** 2).sum(axis=1)
    return amp * np.exp(-d2 / (2.0 * sigma * sigma))


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def skull(params: dict | None = None, nu: int = 168, nv: int = 132):
    """Lofted head surface before feature displacement."""
    p = _params(params)
    TH, T = loft_grid(nu, nv)
    th = TH.reshape(-1)
    t = T.reshape(-1)

    h = p["height"]
    y = (t - 0.5) * h
    y_norm = t

    w = _interp(y_norm, W_CTRL)
    rf = _interp(y_norm, RF_CTRL)
    rb = _interp(y_norm, RB_CTRL)

    # Width modifiers: one multiplier per band, all 1.0 by default.
    mod = np.ones_like(y_norm)
    mod = np.where(y_norm < 0.22, 0.55 * p["chin_width"] + 0.45 * p["jaw_width"], mod)
    mod = np.where((y_norm >= 0.22) & (y_norm < 0.44), p["jaw_width"], mod)
    mod = np.where((y_norm >= 0.44) & (y_norm < 0.70), p["cheek_width"], mod)
    mod = np.where((y_norm >= 0.70) & (y_norm < 0.88), p["forehead_width"], mod)
    mod = np.where(y_norm >= 0.88, 0.5 * p["forehead_width"] + 0.5 * p["cranium"], mod)
    w = w * mod

    sin_t = np.sin(th)
    cos_t = np.cos(th)
    # Superellipse in the horizontal plane: flatter cheeks, rounder skull.
    expo = np.where(np.abs(cos_t) < 0.5, 1.18, 1.0)
    sx = np.sign(sin_t) * np.abs(sin_t) ** expo
    cz = np.where(cos_t >= 0.0, rf, rb) * np.sign(cos_t) * np.abs(cos_t) ** expo

    X = w * sx
    Y = y
    Z = cz - (p["depth"] * 0.5 - 4.6) * 0.0  # kept explicit for clarity
    P = np.stack([X, Y, Z], axis=1)
    return P, TH, T


def build_head(params: dict | None = None, nu: int = 168, nv: int = 132):
    p = _params(params)
    P, TH, T = skull(p, nu, nv)
    N0 = None

    y = P[:, 1]
    z = P[:, 2]
    x = P[:, 0]

    # Outward direction: radial from the head axis, corrected for the flattened
    # front so the nose pushes forward and the cheeks push sideways.
    radial = np.stack([x, np.zeros_like(x), z * 0.55], axis=1)
    radial /= np.maximum(np.linalg.norm(radial, axis=1), 1e-6)[:, None]
    radial[:, 1] = 0.0
    # Vertical component near the crown/chin so those features lift correctly.
    vert = np.clip(np.abs(y) / (p["height"] * 0.5), 0, 1) ** 2
    radial = radial * (1 - 0.55 * vert)[:, None]
    radial[:, 1] = np.sign(y) * vert * 0.55
    radial /= np.maximum(np.linalg.norm(radial, axis=1), 1e-6)[:, None]

    es, ey = p["eye_spacing"], p["eye_y"]

    def fz(yy: float) -> float:
        """Front radius of the skull at height yy (scalar)."""
        return float(_interp(np.clip(yy / p["height"] + 0.5, 0.0, 1.0), RF_CTRL))

    nb = p["mouth_y"] + 1.85  # subnasale: the base of the nose
    disp = np.zeros(len(P))

    # Brow ridge / glabella.
    for sgn in (-1, 1):
        disp += _blob(P, (sgn * es * 0.92, ey + 2.1, fz(ey + 2.1) - 0.35), 1.95, 0.55 * p["brow"])
        # Eye sockets: an almond depression, not a round pit.
        dx = (P[:, 0] - sgn * es) / 1.60
        dy = (P[:, 1] - ey) / 0.92
        almond = np.exp(-(dx ** 2 + dy ** 2) * 1.35)
        almond *= np.clip(P[:, 2] / max(fz(ey), 1e-6), 0.0, 1.0) ** 1.5
        disp -= 1.10 * p["socket"] * almond
        # Upper eyelid swelling above the socket.
        disp += _blob(P, (sgn * es, ey + 0.75, fz(ey + 0.75) - 0.2), 1.05, 0.20)
        # Cheekbones.
        disp += _blob(P, (sgn * 4.85, -1.2, fz(-1.2) - 0.9), 2.6, 0.55 * p["cheek_full"])
        # Soft hollow under the cheekbone.
        disp -= _blob(P, (sgn * 4.3, -4.2, fz(-4.2) - 0.7), 2.1, 0.22)
        # Temples slightly hollow.
        disp -= _blob(P, (sgn * 7.4, 3.8, 2.4), 2.3, 0.30)
        # Jaw corner.
        disp += _blob(P, (sgn * 5.9, -6.3, 1.6), 2.2, 0.28)
        # Nostril wings.
        disp += _blob(P, (sgn * 1.10 * p["nose_width"], nb + 0.15, fz(nb) + 0.5), 0.92,
                      0.80 * p["nose_width"])
        disp -= _blob(P, (sgn * 1.70 * p["nose_width"], nb + 0.10, fz(nb) + 0.35), 0.5, 0.32)
        # Mouth corners tuck in.
        disp -= _blob(P, (sgn * p["mouth_width"], p["mouth_y"] - 0.1, fz(p["mouth_y"]) - 0.9), 0.85, 0.30)

    # Nose: bridge -> dorsum -> tip -> base, spanning brow to subnasale.
    nb = p["mouth_y"] + 1.85
    nz = fz(nb + 1.0) + 0.45
    mid = 0.5 * ((ey + 1.9) + nb)
    disp += _blob(P, (0.0, ey + 1.9, nz - 0.55), 1.30, 0.45 * p["nose_len"])
    disp += _blob(P, (0.0, mid, nz + 0.05), 1.30, 0.85 * p["nose_len"])
    disp += _blob(P, (0.0, nb + 0.62, nz + 0.85), 1.15, 2.20 * p["nose_tip"])
    disp += _blob(P, (0.0, nb + 0.20, nz + 1.15), 0.72, 0.35 * p["nose_tip"])
    # Underside of the nose tucks back toward the lip.
    disp -= _blob(P, (0.0, nb - 0.30, nz + 0.95), 0.60, 0.50)

    my = p["mouth_y"]
    mz = fz(p["mouth_y"]) + 0.15
    # Philtrum.
    disp -= _blob(P, (0.0, my + 1.35, mz), 0.62, 0.30)
    # Lips.
    disp += _blob(P, (0.0, my + 0.45, mz), 0.85, 0.30 * p["lip_upper"])
    disp += _blob(P, (0.0, my - 0.55, mz), 0.85, 0.34 * p["lip_lower"])
    # Mouth line.
    disp -= _blob(P, (0.0, my - 0.02, mz + 0.28), 0.34, 0.16)
    # Chin.
    disp += _blob(P, (0.0, my - 3.35, mz + 0.15), 2.05, 0.95 * p["chin"])
    disp -= _blob(P, (0.0, my - 4.9, mz - 1.3), 1.30, 0.32)

    P = P + radial * disp[:, None]

    tris = grid_tris(nu, nv)
    N = vertex_normals(P, tris)
    C = skin_colours(P, p)
    return P, N, C, tris


def skin_colours(P, p):
    skin = np.array(p["skin"], dtype=np.float64)
    shadow = np.array(p["skin_shadow"], dtype=np.float64)
    lip = np.array(p["lip_colour"], dtype=np.float64)
    brow_c = np.array(p["brow_colour"], dtype=np.float64)

    C = np.tile(skin, (len(P), 1))
    y = P[:, 1]
    x = P[:, 0]
    front_tab = _interp(np.clip(y / p["height"] + 0.5, 0.0, 1.0), RF_CTRL)
    facing = np.clip(P[:, 2] / np.maximum(front_tab, 1e-6), 0.0, 1.0) ** 1.5

    # Socket / nose / under-chin occlusion, purely as colour.
    for sgn in (-1, 1):
        d2 = ((P - np.array([sgn * p["eye_spacing"], p["eye_y"] + 0.25, 5.6])) ** 2).sum(axis=1)
        shade = np.exp(-d2 / (2 * 2.0 ** 2)) * 0.34
        C = C * (1 - shade[:, None]) + shadow * shade[:, None]

    under = _smoothstep(-6.0, -10.4, y) * 0.35
    C = C * (1 - under[:, None]) + shadow * under[:, None]

    # Lips.
    lip_mask = np.exp(-(((x / (p["mouth_width"] * 0.86)) ** 2) * 1.0 +
                        ((y - (p["mouth_y"] + 0.05)) / 0.62) ** 2)) * facing
    lip_mask = np.clip(lip_mask * 1.15, 0.0, 0.92)
    C = C * (1 - lip_mask[:, None]) + lip * lip_mask[:, None]

    # Brows, with an arch.
    for sgn in (-1, 1):
        dx = x - sgn * p["eye_spacing"] * 0.98
        arch = p["eye_y"] + 2.25 + 0.32 * np.exp(-(dx ** 2) / 5.0)
        band = np.exp(-((dx ** 2) / (2 * 1.75 ** 2) + ((y - arch) ** 2) / (2 * 0.32 ** 2)))
        band = np.clip(band * facing * 1.35, 0.0, 0.90)
        C = C * (1 - band[:, None]) + brow_c * band[:, None]

    # Mouth line: a thin dark crease so the lips read at a glance.
    crease = np.exp(-(((x / (p["mouth_width"] * 0.98)) ** 2) * 1.4 +
                      ((y - p["mouth_y"]) / 0.16) ** 2)) * facing
    crease = np.clip(crease * 1.1, 0.0, 0.8)
    C = C * (1 - crease[:, None]) + np.array([0.30, 0.16, 0.15]) * crease[:, None]

    return np.clip(C, 0.0, 1.0)


def build_eyes(params: dict | None = None, seg: int = 32):
    p = _params(params)
    meshes = []
    radius = p["eye_r"]
    ey = p["eye_y"]
    front = float(_interp(np.clip(ey / p["height"] + 0.5, 0.0, 1.0), RF_CTRL))
    centre_z = front - 1.55
    theta = np.linspace(-math.pi, math.pi, seg, endpoint=False)
    phi = np.linspace(1e-4, math.pi - 1e-4, seg)
    PH, TH = np.meshgrid(phi, theta, indexing="ij")
    ph = PH.reshape(-1)
    th = TH.reshape(-1)
    sx = np.sin(ph) * np.sin(th)
    sy = np.cos(ph)
    sz = np.sin(ph) * np.cos(th)
    local = np.stack([radius * sx, radius * sy, radius * sz], axis=1)

    sclera = np.array([0.92, 0.91, 0.89])
    iris = np.array(p["eye_iris"])
    pupil = np.array([0.02, 0.02, 0.02])
    angle = np.degrees(np.arccos(np.clip(sz, -1, 1)))
    C = np.tile(sclera, (len(local), 1))
    iris_mask = (angle < 33.0).astype(float)[:, None]
    C = C * (1 - iris_mask) + iris * iris_mask
    pupil_mask = (angle < 13.5).astype(float)[:, None]
    C = C * (1 - pupil_mask) + pupil * pupil_mask
    C *= (0.72 + 0.28 * np.clip(np.cos(np.radians(angle)), 0, 1))[:, None]

    tris = grid_tris(seg, seg)
    N = vertex_normals(local, tris)
    for sgn in (-1, 1):
        meshes.append({
            "name": "eye",
            "positions": local + np.array([sgn * p["eye_spacing"], p["eye_y"], centre_z]),
            "normals": N,
            "colours": np.clip(C, 0, 1),
            "tris": tris,
        })
    return meshes


def build_ears(params: dict | None = None, seg: int = 26):
    p = _params(params)
    theta = np.linspace(-math.pi, math.pi, seg, endpoint=False)
    phi = np.linspace(1e-4, math.pi - 1e-4, seg)
    PH, TH = np.meshgrid(phi, theta, indexing="ij")
    ph = PH.reshape(-1)
    th = TH.reshape(-1)
    base = np.stack([np.sin(ph) * np.sin(th), np.cos(ph), np.sin(ph) * np.cos(th)], axis=1)
    scale = np.array([0.34, 1.55, 0.78]) * p["ears"]
    local = base * scale
    # Flatten the ear against the skull and cup the concha.
    local[:, 0] = local[:, 0] * (0.75 + 0.25 * np.abs(base[:, 2]))
    local[:, 2] -= 0.35 * np.exp(-((local[:, 1] / 0.9) ** 2 + (local[:, 2] / 0.7) ** 2)) * 0.0
    tris = grid_tris(seg, seg)
    N = vertex_normals(local, tris)
    C = np.tile(np.array(p["skin"]) * 0.93, (len(local), 1))
    out = []
    for sgn in (-1, 1):
        pos = local + np.array([sgn * (p["width"] * 0.47), -0.9, -1.70])
        # Tilt the ear back slightly and let it hug the skull.
        rot = np.radians(sgn * 12.0)
        cx, cy = pos[:, 0] - sgn * (p["width"] * 0.47), pos[:, 2] + 1.70
        pos[:, 0] = sgn * (p["width"] * 0.47) + cx * np.cos(rot) - cy * np.sin(rot) * sgn
        pos[:, 2] = -1.70 + cx * np.sin(rot) * sgn + cy * np.cos(rot)
        out.append({"name": "ear", "positions": pos, "normals": N,
                    "colours": np.clip(C, 0, 1), "tris": tris})
    return out


def build_hair(params: dict | None = None, nu: int = 150, nv: int = 118):
    p = _params(params)
    P, N, _C, tris = build_head(p, nu, nv)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    front_tab = _interp(np.clip(y / p["height"] + 0.5, 0, 1), RF_CTRL)
    forwardness = np.clip(z / np.maximum(front_tab, 1e-6), -1.0, 1.0)

    hairline = p["hairline"]
    # The hairline dips at the temples and sits highest over the forehead.
    f = np.clip(forwardness, 0.0, 1.0)  # 1 at the forehead, 0 at the sides/back
    edge = -4.2 + (hairline + 4.2) * _smoothstep(0.15, 0.80, f)
    edge -= 1.5 * np.clip(np.abs(x) / 6.5, 0, 1) ** 2 * (f > 0.15)
    inside = y - edge  # >0 above the hairline

    side = 1.0 - 0.45 * np.clip(np.abs(x) / 7.0, 0.0, 1.0) ** 1.5
    thickness = p["hair_volume"] * side
    clump = (np.sin(x * 1.05 + 1.1) * np.sin(y * 0.8 - 0.3) * np.sin(z * 0.9 + 2.0))
    rise = _smoothstep(-0.5, 1.8, inside)
    # Below the hairline the shell sinks under the scalp, so the hairline is the
    # smooth curve where two surfaces cross rather than a grid-aligned cut.
    offset = -1.6 + (1.6 + thickness + 0.30 + clump * 0.26) * rise
    Q = P + N * offset[:, None]

    C = np.tile(np.array(p["hair_colour"]), (len(Q), 1))
    C *= (0.80 + 0.45 * np.clip(N[:, 1], 0, 1))[:, None]
    C *= (0.92 + 0.16 * np.clip(clump, -1, 1))[:, None]

    return [{
        "name": "hair",
        "positions": Q,
        "normals": N,
        "colours": np.clip(C, 0, 1),
        "tris": tris,
        "double_sided": True,
    }]


def build_neck_and_torso(params: dict | None = None, seg: int = 72, rows: int = 56):
    """Neck, then a shoulder shelf that ends in a soft cut at the chest."""
    p = _params(params)
    half_h = p["height"] * 0.5
    neck_top = -half_h + 5.0
    neck_end = -half_h - 7.0
    bottom = -half_h - 13.0

    theta = np.linspace(-math.pi, math.pi, seg, endpoint=False)
    v = np.linspace(0.0, 1.0, rows)
    TH, V = np.meshgrid(theta, v, indexing="ij")
    th = TH.reshape(-1)
    vv = V.reshape(-1)

    # Neck section, then the trapezius flare, then the chest cut.
    shoulder = _smoothstep(0.42, 0.96, vv)
    y = neck_top + (neck_end - neck_top) * np.clip(vv / 0.42, 0, 1)
    y = np.where(shar := vv > 0.42, neck_end, y)
    y = np.where(shar, neck_end + (bottom - neck_end) * _smoothstep(0.42, 1.0, vv), y)

    neck_rx = 0.31 * p["width"]
    neck_rz = 0.29 * p["width"]
    rx = neck_rx + shoulder * (p["width"] * 0.62)
    rz = neck_rz + shoulder * (p["width"] * 0.26)

    X = rx * np.sin(th)
    Z = rz * np.cos(th) - shoulder * 1.0
    Y = y - shoulder * 1.6 * np.cos(th) ** 2

    P = np.stack([X, Y, Z], axis=1)
    tris = grid_tris(seg, rows)
    N = vertex_normals(P, tris, reference=np.array([0.0, -half_h - 4.0, 0.0]))

    collar = _smoothstep(0.16, 0.34, vv)
    skin = np.array(p["skin"]) * 0.96
    shirt = np.array(p["shirt"])
    C = skin[None, :] * (1 - collar[:, None]) + shirt[None, :] * collar[:, None]
    return [{"name": "torso", "positions": P, "normals": N,
             "colours": np.clip(C, 0, 1), "tris": tris}]
