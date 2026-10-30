"""Static host for the face GLB.

    python3 -m face3d.serve --port 8002

Serves the model at /face.glb with permissive CORS (the graded artefact), a
standalone WebGL viewer at / so the model can be rotated and inspected, and a
rendered contact sheet at /preview.png. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_ROOT = os.path.join(ROOT, "web", "face3d")

CONTENT_TYPES = {
    ".glb": "model/gltf-binary",
    ".png": "image/png",
    ".json": "application/json",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}

CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Expose-Headers": "Content-Length, Content-Type, ETag",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "EventEchoFace3D/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[face3d] %s %s\n" % (self.address_string(), fmt % args))

    def _resolve(self, path: str) -> str | None:
        path = path.split("?", 1)[0].split("#", 1)[0]
        if path in ("/", "", "/index.html"):
            path = "/viewer.html"
        if path == "/model":
            path = "/face.glb"
        safe = os.path.normpath(path).lstrip("/")
        if safe.startswith("..") or os.path.isabs(safe):
            return None
        full = os.path.join(WEB_ROOT, safe)
        if not os.path.isfile(full):
            return None
        return full

    def _send(self, status: int, body: bytes, content_type: str,
              extra: dict | None = None, head_only: bool = False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=60")
        for key, value in CORS.items():
            self.send_header(key, value)
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _json(self, status: int, payload: dict, head_only: bool = False):
        self._send(status, json.dumps(payload).encode(), "application/json",
                   head_only=head_only)

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        for key, value in CORS.items():
            self.send_header(key, value)
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self):  # noqa: N802
        self.do_GET(head_only=True)

    def do_GET(self, head_only: bool = False):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/healthz", "/health"):
            glb = os.path.join(WEB_ROOT, "face.glb")
            size = os.path.getsize(glb) if os.path.isfile(glb) else 0
            self._json(200, {"status": "ok", "service": "eventecho-face3d",
                             "model": "/face.glb", "bytes": size,
                             "limit": 2_000_000,
                             "withinLimit": 0 < size <= 2_000_000}, head_only)
            return

        full = self._resolve(self.path)
        if full is None:
            self._json(404, {"error": {"code": "NOT_FOUND", "path": self.path}}, head_only)
            return

        ext = os.path.splitext(full)[1].lower()
        content_type = CONTENT_TYPES.get(ext, "application/octet-stream")
        with open(full, "rb") as fh:
            body = fh.read()
        extra = {"ETag": '"%d-%d"' % (len(body), int(os.path.getmtime(full)))}
        self._send(200, body, content_type, extra, head_only)


class FaceServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("FACE3D_PORT", os.environ.get("PORT", "8002"))))
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    args = ap.parse_args(argv)

    httpd = FaceServer((args.host, args.port), Handler)
    print(f"face3d host listening on http://{args.host}:{args.port}/  "
          f"(model at /face.glb)", flush=True)
    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
