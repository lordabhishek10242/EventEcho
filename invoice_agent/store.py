"""SQLite persistence: tasks, message idempotency and the decision cache.

Everything that must survive a retry lives here.  Writes go through a single
serialised connection and explicit ``BEGIN IMMEDIATE`` transactions so the
cancel-versus-receipt race resolves to exactly one winner.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Iterable

TERMINAL_STATES = {"TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"}
NONTERMINAL_STATES = (
    "TASK_STATE_SUBMITTED",
    "TASK_STATE_WORKING",
    "TASK_STATE_INPUT_REQUIRED",
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    principal      TEXT NOT NULL,
    id             TEXT NOT NULL,
    context_id     TEXT NOT NULL,
    state          TEXT NOT NULL,
    created_ms     INTEGER NOT NULL,
    updated_ms     INTEGER NOT NULL,
    batch_id       TEXT,
    message_id     TEXT NOT NULL,
    message_hash   TEXT NOT NULL,
    proposals_json TEXT,
    receipts_json  TEXT,
    history_json   TEXT NOT NULL,
    results_json   TEXT,
    metadata_json  TEXT,
    PRIMARY KEY (principal, id)
);
CREATE INDEX IF NOT EXISTS idx_tasks_principal_created ON tasks(principal, created_ms DESC);

CREATE TABLE IF NOT EXISTS idempotency (
    principal  TEXT NOT NULL,
    message_id TEXT NOT NULL,
    msg_hash   TEXT NOT NULL,
    task_id    TEXT NOT NULL,
    kind       TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    PRIMARY KEY (principal, message_id)
);

CREATE TABLE IF NOT EXISTS decision_cache (
    cache_key     TEXT PRIMARY KEY,
    decision_json TEXT NOT NULL,
    created_ms    INTEGER NOT NULL
);
"""


def now_ms() -> int:
    return int(time.time() * 1000)


class Store:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _begin(self) -> None:
        self._conn.execute("BEGIN IMMEDIATE")

    def _commit(self) -> None:
        self._conn.execute("COMMIT")

    def _rollback(self) -> None:
        try:
            self._conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass

    def _row_to_task(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        task = dict(row)
        for key in ("proposals_json", "receipts_json", "history_json", "results_json", "metadata_json"):
            raw = task.pop(key, None)
            task[key.replace("_json", "")] = json.loads(raw) if raw else None
        return task

    # ------------------------------------------------------------------ #
    # idempotency
    # ------------------------------------------------------------------ #
    def lookup_idempotency(self, principal: str, message_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM idempotency WHERE principal=? AND message_id=?",
                (principal, message_id),
            ).fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------ #
    # tasks
    # ------------------------------------------------------------------ #
    def get_task(self, principal: str, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE principal=? AND id=?",
                (principal, task_id),
            ).fetchone()
            return self._row_to_task(row)

    def list_tasks(self, principal: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE principal=? ORDER BY created_ms DESC, id DESC LIMIT ?",
                (principal, limit),
            ).fetchall()
            return [self._row_to_task(row) for row in rows]  # type: ignore[misc]

    def create_task(
        self,
        *,
        principal: str,
        task_id: str,
        context_id: str,
        state: str,
        batch_id: str | None,
        message_id: str,
        message_hash: str,
        proposals: list[dict[str, Any]] | None,
        history: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        kind: str = "batch",
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Create a task and its idempotency record atomically.

        Returns ``(task, existing)``.  ``existing`` is non-None when another
        request won the race for the same ``(principal, messageId)``.
        """
        ts = now_ms()
        with self._lock:
            self._begin()
            try:
                row = self._conn.execute(
                    "SELECT * FROM idempotency WHERE principal=? AND message_id=?",
                    (principal, message_id),
                ).fetchone()
                if row is not None:
                    existing_task = self._conn.execute(
                        "SELECT * FROM tasks WHERE principal=? AND id=?",
                        (principal, row["task_id"]),
                    ).fetchone()
                    self._commit()
                    return None, self._row_to_task(existing_task)
                self._conn.execute(
                    "INSERT INTO tasks (principal, id, context_id, state, created_ms, updated_ms,"
                    " batch_id, message_id, message_hash, proposals_json, receipts_json,"
                    " history_json, results_json, metadata_json)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        principal,
                        task_id,
                        context_id,
                        state,
                        ts,
                        ts,
                        batch_id,
                        message_id,
                        message_hash,
                        json.dumps(proposals, ensure_ascii=False) if proposals is not None else None,
                        None,
                        json.dumps(history, ensure_ascii=False),
                        None,
                        json.dumps(metadata or {}, ensure_ascii=False),
                    ),
                )
                self._conn.execute(
                    "INSERT INTO idempotency (principal, message_id, msg_hash, task_id, kind, created_ms)"
                    " VALUES (?,?,?,?,?,?)",
                    (principal, message_id, message_hash, task_id, kind, ts),
                )
                self._commit()
            except Exception:
                self._rollback()
                raise
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE principal=? AND id=?", (principal, task_id)
            ).fetchone()
            return self._row_to_task(row), None

    def record_message(
        self,
        *,
        principal: str,
        message_id: str,
        message_hash: str,
        task_id: str,
        kind: str,
    ) -> bool:
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO idempotency (principal, message_id, msg_hash, task_id, kind, created_ms)"
                    " VALUES (?,?,?,?,?,?)",
                    (principal, message_id, message_hash, task_id, kind, now_ms()),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def complete_task(
        self,
        *,
        principal: str,
        task_id: str,
        receipts: dict[str, Any],
        results: list[dict[str, Any]],
        history: list[dict[str, Any]],
        message_id: str,
        message_hash: str,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Atomically move a non-terminal task to COMPLETED.

        Returns ``(ok, task)``; ``ok`` is False when another operation (cancel
        or a competing receipt) already made the task terminal.
        """
        ts = now_ms()
        with self._lock:
            self._begin()
            try:
                cur = self._conn.execute(
                    "UPDATE tasks SET state=?, updated_ms=?, receipts_json=?, results_json=?,"
                    " history_json=? WHERE principal=? AND id=? AND state IN (?,?,?)",
                    (
                        "TASK_STATE_COMPLETED",
                        ts,
                        json.dumps(receipts, ensure_ascii=False),
                        json.dumps(results, ensure_ascii=False),
                        json.dumps(history, ensure_ascii=False),
                        principal,
                        task_id,
                        *NONTERMINAL_STATES,
                    ),
                )
                if cur.rowcount != 1:
                    self._commit()
                    row = self._conn.execute(
                        "SELECT * FROM tasks WHERE principal=? AND id=?", (principal, task_id)
                    ).fetchone()
                    return False, self._row_to_task(row)
                try:
                    self._conn.execute(
                        "INSERT INTO idempotency (principal, message_id, msg_hash, task_id, kind, created_ms)"
                        " VALUES (?,?,?,?,?,?)",
                        (principal, message_id, message_hash, task_id, "results", ts),
                    )
                except sqlite3.IntegrityError:
                    pass
                self._commit()
            except Exception:
                self._rollback()
                raise
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE principal=? AND id=?", (principal, task_id)
            ).fetchone()
            return True, self._row_to_task(row)

    def cancel_task(self, *, principal: str, task_id: str) -> tuple[bool, dict[str, Any] | None]:
        """Atomically move a non-terminal task to CANCELED."""
        ts = now_ms()
        with self._lock:
            self._begin()
            try:
                cur = self._conn.execute(
                    "UPDATE tasks SET state=?, updated_ms=? WHERE principal=? AND id=?"
                    " AND state IN (?,?,?)",
                    (
                        "TASK_STATE_CANCELED",
                        ts,
                        principal,
                        task_id,
                        *NONTERMINAL_STATES,
                    ),
                )
                if cur.rowcount != 1:
                    self._commit()
                    row = self._conn.execute(
                        "SELECT * FROM tasks WHERE principal=? AND id=?", (principal, task_id)
                    ).fetchone()
                    return False, self._row_to_task(row)
                self._commit()
            except Exception:
                self._rollback()
                raise
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE principal=? AND id=?", (principal, task_id)
            ).fetchone()
            return True, self._row_to_task(row)

    # ------------------------------------------------------------------ #
    # decision cache
    # ------------------------------------------------------------------ #
    def get_cached_decision(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT decision_json FROM decision_cache WHERE cache_key=?", (cache_key,)
            ).fetchone()
        return json.loads(row["decision_json"]) if row else None

    def put_cached_decision(self, cache_key: str, decision: dict[str, Any]) -> None:
        with self._lock:
            self._begin()
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO decision_cache (cache_key, decision_json, created_ms)"
                    " VALUES (?,?,?)",
                    (cache_key, json.dumps(decision, ensure_ascii=False), now_ms()),
                )
                self._commit()
            except Exception:
                self._rollback()
                raise

    def stats(self) -> dict[str, int]:
        with self._lock:
            tasks = self._conn.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
            cached = self._conn.execute("SELECT COUNT(*) AS n FROM decision_cache").fetchone()["n"]
        return {"tasks": int(tasks), "cached_decisions": int(cached)}
