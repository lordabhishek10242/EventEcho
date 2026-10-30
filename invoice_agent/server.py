"""A2A 1.0 HTTP+JSON surface (standard library only).

Routes (relative to the submitted base URL, default ``/a2a``)::

    GET  {origin}/.well-known/agent-card.json   public Agent Card
    POST {base}/message:send
    GET  {base}/tasks/{id}
    GET  {base}/tasks
    POST {base}/tasks/{id}:cancel
"""

from __future__ import annotations

import json
import os
import re
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlparse

from .ai import build_provider
from .protocol import (
    A2A_MEDIA_TYPE,
    A2A_VERSION,
    MAX_RESPONSE_BYTES,
    agent_card,
    error_body,
)
from .service import ApiError, InvoiceService
from .store import Store

CARD_PATH = "/.well-known/agent-card.json"
_CANCEL_RE = re.compile(r"^/tasks/([^/]+):cancel$")
_TASK_RE = re.compile(r"^/tasks/([^/]+)$")
_MAX_BODY = 32 * 1024 * 1024


class AppContext:
    def __init__(self, store: Store, service: InvoiceService, base_path: str, public_base: str | None):
        self.store = store
        self.service = service
        self.base_path = "/" + base_path.strip("/") if base_path.strip("/") else ""
        self.public_base = public_base


def _strip_base(path: str, base_path: str) -> str:
    if base_path and (path == base_path or path.startswith(base_path + "/")):
        path = path[len(base_path):]
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return path or "/"


def _error(message: str) -> tuple[int, str, dict[str, Any]]:
    return 0, message, {}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "EventEchoInvoiceAgent/1.0"

    # -- infrastructure -------------------------------------------------- #
    @property
    def ctx(self) -> AppContext:
        return self.server.ctx  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib signature
        agent = (self.headers.get("User-Agent") or "-")[:48]
        sys.stderr.write(
            "[invoice-agent] %s ua=%r %s\n" % (self.address_string(), agent, fmt % args)
        )

    def _send(self, status: int, payload: dict[str, Any], media_type: str) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(body) > MAX_RESPONSE_BYTES:
            status = 500
            body = json.dumps(error_body("RESPONSE_TOO_LARGE", "response exceeded 512 KiB")).encode("utf-8")
            media_type = A2A_MEDIA_TYPE
        try:
            self.send_response(status)
            self.send_header("Content-Type", media_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, A2A-Version")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Expose-Headers", "Content-Type")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _read_body(self) -> bytes:
        length = self.headers.get("Content-Length")
        if not length:
            return b""
        try:
            size = int(length)
        except ValueError:
            return b""
        if size <= 0:
            return b""
        return self.rfile.read(min(size, _MAX_BODY))

    def _origin_base(self) -> str:
        if self.ctx.public_base:
            return self.ctx.public_base.rstrip("/") + self.ctx.base_path + "/"
        host = self.headers.get("X-Forwarded-Host") or self.headers.get("Host") or "localhost"
        local = host.split(":")[0] in {"localhost", "127.0.0.1", "::1", "[::1]"}
        proto = self.headers.get("X-Forwarded-Proto") or ("http" if local else "https")
        return f"{proto}://{host}{self.ctx.base_path}/"

    # -- verbs ----------------------------------------------------------- #
    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, A2A-Version")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def _dispatch(self, method: str) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path in {"/healthz", "/health"}:
            stats = self.ctx.store.stats()
            self._send(
                200,
                {
                    "status": "ok",
                    "service": "eventecho-invoice-agent",
                    "protocol": "A2A 1.0 HTTP+JSON",
                    "tasks": stats["tasks"],
                    "cachedDecisions": stats["cached_decisions"],
                },
                "application/json",
            )
            return
        if path.rstrip("/") == CARD_PATH:
            self._send(200, agent_card(self._origin_base()), "application/json")
            return
        if path in {"", "/"}:
            self._send(
                200,
                {
                    "name": "EventEcho Invoice Action Agent",
                    "protocol": "A2A 1.0 HTTP+JSON",
                    "baseUrl": self._origin_base(),
                    "agentCard": CARD_PATH,
                },
                "application/json",
            )
            return

        route = _strip_base(path, self.ctx.base_path)
        principal = self._authenticate()
        if principal is None:
            return
        version = self.headers.get("A2A-Version")
        if version != A2A_VERSION:
            self._send(400, error_body("UNSUPPORTED_VERSION", "A2A-Version 1.0 is required"), A2A_MEDIA_TYPE)
            return
        if method == "POST":
            content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            has_body = (self.headers.get("Content-Length") or "0").strip() not in {"", "0"}
            if has_body and content_type != A2A_MEDIA_TYPE:
                self._send(
                    415,
                    error_body("UNSUPPORTED_MEDIA_TYPE", f"Content-Type must be {A2A_MEDIA_TYPE}"),
                    A2A_MEDIA_TYPE,
                )
                return

        try:
            if route == "/message:send":
                if method != "POST":
                    raise ApiError(405, "METHOD_NOT_ALLOWED", "use POST")
                payload = self._parse_json()
                result = self.ctx.service.send_message(principal, payload)
                self._send(200, result, A2A_MEDIA_TYPE)
                return

            if route == "/tasks":
                if method != "GET":
                    raise ApiError(405, "METHOD_NOT_ALLOWED", "use GET")
                self._send(200, self.ctx.service.list_tasks(principal), A2A_MEDIA_TYPE)
                return

            match = _CANCEL_RE.match(route)
            if match:
                if method != "POST":
                    raise ApiError(405, "METHOD_NOT_ALLOWED", "use POST")
                self._send(200, self.ctx.service.cancel_task(principal, unquote(match.group(1))), A2A_MEDIA_TYPE)
                return

            match = _TASK_RE.match(route)
            if match:
                if method != "GET":
                    raise ApiError(405, "METHOD_NOT_ALLOWED", "use GET")
                self._send(200, self.ctx.service.get_task(principal, unquote(match.group(1))), A2A_MEDIA_TYPE)
                return

            raise ApiError(404, "NOT_FOUND", "not found")
        except ApiError as exc:
            self._send(exc.status, error_body(exc.code, exc.message), A2A_MEDIA_TYPE)
        except json.JSONDecodeError:
            self._send(400, error_body("INVALID_JSON", "body must be valid JSON"), A2A_MEDIA_TYPE)
        except Exception:  # pragma: no cover - defensive
            import traceback

            traceback.print_exc()
            self._send(500, error_body("INTERNAL_ERROR", "internal error"), A2A_MEDIA_TYPE)

    # -- helpers --------------------------------------------------------- #
    def _authenticate(self) -> str | None:
        header = self.headers.get("Authorization") or ""
        parts = header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
            self._send(401, error_body("UNAUTHORIZED", "a Bearer token is required"), A2A_MEDIA_TYPE)
            return None
        return parts[1].strip()

    def _parse_json(self) -> dict[str, Any]:
        raw = self._read_body()
        if not raw:
            raise ApiError(400, "INVALID_JSON", "request body is required")
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ApiError(400, "INVALID_JSON", "body must be valid JSON")
        if not isinstance(parsed, dict):
            raise ApiError(400, "INVALID_REQUEST", "body must be a JSON object")
        return parsed


class AgentHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128

    def __init__(self, address: tuple[str, int], ctx: AppContext):
        super().__init__(address, Handler)
        self.ctx = ctx


def build_app(
    db_path: str | None = None,
    base_path: str | None = None,
    public_base: str | None = None,
) -> AppContext:
    db_path = db_path or os.environ.get("INVOICE_AGENT_DB", "data/invoice_agent.sqlite3")
    base_path = base_path if base_path is not None else os.environ.get("INVOICE_AGENT_BASE_PATH", "/a2a")
    public_base = public_base if public_base is not None else os.environ.get("INVOICE_AGENT_PUBLIC_BASE")
    store = Store(db_path)
    provider = build_provider()
    service = InvoiceService(store, provider)
    return AppContext(store=store, service=service, base_path=base_path, public_base=public_base)


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    ctx = build_app()
    server = AgentHTTPServer((host, port), ctx)
    print(f"invoice agent listening on http://{host}:{port}{ctx.base_path}/", flush=True)
    print(f"agent card: http://{host}:{port}/.well-known/agent-card.json", flush=True)

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
