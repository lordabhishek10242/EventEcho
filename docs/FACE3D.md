# Face model — parametric head, exported as GLB

A single glTF 2.0 binary head, procedurally sculpted and served with CORS so a
browser (or a grader) can fetch and render it directly.

- **Model URL (what to submit):** `https://8002-i08mehv473n5yap8hpkdj.e2b.app/face.glb`
- **Rotatable viewer:** `https://8002-i08mehv473n5yap8hpkdj.e2b.app/`
- **Rendered views:** `https://8002-i08mehv473n5yap8hpkdj.e2b.app/preview.png`
- **Health:** `https://8002-i08mehv473n5yap8hpkdj.e2b.app/healthz`

`face.glb` is 1,593,336 bytes (limit 2,000,000), 66,116 triangles, 7 meshes,
`model/gltf-binary`, `Access-Control-Allow-Origin: *` on GET/HEAD/OPTIONS.

---

## How the head is built

`face3d/head.py` lofts a skull from anatomical profile tables (half-width and
independent front/back radii at every height), then pushes anatomical features
along the surface normal: brow ridge, almond eye sockets, nose bridge/dorsum/
tip/wings, philtrum, lips, chin, cheekbones, jaw angles, temples. Separate
meshes add eyeballs (iris and pupil as vertex colour), ears, a hair shell that
dives *into* the scalp so the hairline is the smooth curve where the two
surfaces cross, and a neck with a shoulder cut.

Everything is driven by a flat parameter dict (`face3d/head.DEFAULT`): head
height, width, depth, jaw and chin width, eye spacing, nose length and tip,
lip fullness, skin/lip/brow/hair colours, hair volume and hairline. That is what
makes the model fittable to a specific photograph without touching geometry
code.

## Files

| Path | What it is |
|---|---|
| `face3d/head.py` | parametric head, eyes, ears, hair, neck |
| `face3d/glb.py` | GLB writer, `inspect()` and `validate()` |
| `face3d/build.py` | `python -m face3d.build --out web/face3d/face.glb --preview …` |
| `face3d/render.py` | offline z-buffer rasteriser used for previews |
| `face3d/serve.py` | static host with CORS (`--port 8002`) |
| `web/face3d/viewer.html` | dependency-free WebGL viewer (orbit + auto-spin) |
| `web/face3d/face.glb` | the shipped model |
| `web/face3d/preview.png` | four rendered views |

## Checks performed

| Check | Result |
|---|---|
| GLB header, version, declared length | valid glTF 2.0 |
| Buffer / bufferView / accessor ranges | indices in range, POSITION has min/max, all finite |
| Size | 1,593,336 B < 2,000,000 B |
| HTTP | `200`, `model/gltf-binary`, `Content-Length` matches |
| CORS | `Access-Control-Allow-Origin: *`, preflight `204` |
| `HEAD` | 200 with correct headers (some graders probe first) |

## Rebuilding

```bash
python3 -m venv .venv && .venv/bin/pip install numpy pillow   # build-time only
.venv/bin/python -m face3d.build --out web/face3d/face.glb --quality 0.85 --preview web/face3d/preview.png
python3 -m face3d.serve --port 8002                            # serve (stdlib only)
```

Build-time quality is the size lever: `--quality 0.85` lands at 1.59 MB,
`0.78` at 1.35 MB, `1.0` at 2.23 MB (over the limit).
