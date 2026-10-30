"""Public HTTPS endpoint for the mailroom action gate (stdlib only).

    POST /            -> propose | commit envelopes
    GET  /healthz     -> liveness + durable counters

Every response is non-redirecting ``application/json``; errors carry a generic
``{"error": {"code", "message"}}`` body.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .canonical import PROFILE
from .service import ApiError, MailroomService
from .store import Store

MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_DRAIN_BYTES = 96 * 1024 * 1024
MAX_RESPONSE_BYTES = 512 * 1024


class AppContext:
    def __init__(self, store: Store, service: MailroomService):
        self.store = store
        self.service = service


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "EventEchoMailroomAgent/1.0"

    @property
    def ctx(self) -> AppContext:
        return self.server.ctx  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        agent = (self.headers.get("User-Agent") or "-")[:48]
        sys.stderr.write("[mailroom] %s ua=%r %s\n" % (self.address_string(), agent, fmt % args))

    # ------------------------------------------------------------------ #
    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(body) > MAX_RESPONSE_BYTES:
            status = 500
            body = json.dumps(
                {"error": {"code": "RESPONSE_TOO_LARGE", "message": "response exceeded 512 KiB"}}
            ).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _drain(self, size: int) -> None:
        """Consume an oversized body so the client sees a JSON 413 rather than
        a reset connection."""
        remaining = min(size, MAX_DRAIN_BYTES)
        while remaining > 0:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
        if size > MAX_DRAIN_BYTES:
            self.close_connection = True

    def _read_body(self) -> bytes:
        length = self.headers.get("Content-Length")
        if not length:
            return b""
        try:
            size = int(length)
        except ValueError:
            raise ApiError(400, "INVALID_LENGTH", "Content-Length is not a number")
        if size < 0:
            raise ApiError(400, "INVALID_LENGTH", "Content-Length is negative")
        if size > MAX_BODY_BYTES:
            self._drain(size)
            raise ApiError(413, "BODY_TOO_LARGE", "request body exceeds the limit")
        return self.rfile.read(size)

    # ------------------------------------------------------------------ #
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path in {"/", "/healthz", "/health"}:
            self._send(200, self.ctx.service.health())
            return
        self._send(404, {"error": {"code": "NOT_FOUND", "message": "not found"}})

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        try:
            raw = self._read_body()
            if not raw:
                raise ApiError(400, "INVALID_JSON", "request body is required")
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ApiError(400, "INVALID_JSON", "body must be valid JSON")
            response = self.ctx.service.handle(payload)
            self._send(200, response)
        except ApiError as exc:
            self._send(exc.status, {"error": {"code": exc.code, "message": exc.message}})
        except Exception:  # pragma: no cover - defensive
            import traceback

            traceback.print_exc()
            self._send(500, {"error": {"code": "INTERNAL_ERROR", "message": "internal error"}})


class MailroomHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(self, address: tuple[str, int], ctx: AppContext):
        super().__init__(address, Handler)
        self.ctx = ctx


def build_app(db_path: str | None = None) -> AppContext:
    db_path = db_path or os.environ.get("MAILROOM_DB", "data/mailroom_agent.sqlite3")
    store = Store(db_path)
    return AppContext(store=store, service=MailroomService(store))


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("MAILROOM_PORT", os.environ.get("PORT", "8001")))
    ctx = build_app()
    server = MailroomHTTPServer((host, port), ctx)
    print(f"mailroom action gate listening on http://{host}:{port}/ ({PROFILE})", flush=True)

    def _shutdown(_signum, _frame):  # pragma: no cover
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
