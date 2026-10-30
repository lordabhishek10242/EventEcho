"""SQLite persistence for the mailroom action gate.

Durable state, in evaluation-scoped tables:

``evaluations``   one row per evaluation: propose hash, input digest, the
                  receipt-verification key supplied with that evaluation, the
                  exact propose response, commit state and outcomes.
``proposals``     the persisted proposal for every dossier of an evaluation.
``receipts``      every verified receipt, keyed by receipt id.
``decision_cache`` dossier content fingerprint -> decision, so repeated
                  Checks/Saves never re-run the model for stable dossiers.

All writes are serialised through one connection with ``BEGIN IMMEDIATE``
transactions, so a propose/commit race cannot leave half-written state.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id       TEXT PRIMARY KEY,
    propose_hash        TEXT NOT NULL,
    input_digest        TEXT NOT NULL,
    verifier_json       TEXT NOT NULL,
    allowed_actions     TEXT NOT NULL,
    status              TEXT NOT NULL,
    response_json       TEXT NOT NULL,
    commit_hash         TEXT,
    commit_response_json TEXT,
    created_ms          INTEGER NOT NULL,
    updated_ms          INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS proposals (
    evaluation_id   TEXT NOT NULL,
    dossier_id      TEXT NOT NULL,
    call_id         TEXT NOT NULL,
    action          TEXT NOT NULL,
    proposal_json   TEXT NOT NULL,
    proposal_digest TEXT NOT NULL,
    PRIMARY KEY (evaluation_id, dossier_id)
);

CREATE TABLE IF NOT EXISTS receipts (
    evaluation_id TEXT NOT NULL,
    receipt_id    TEXT NOT NULL,
    receipt_json  TEXT NOT NULL,
    verified      INTEGER NOT NULL,
    PRIMARY KEY (evaluation_id, receipt_id)
);

CREATE TABLE IF NOT EXISTS decision_cache (
    fingerprint   TEXT PRIMARY KEY,
    decision_json TEXT NOT NULL,
    created_ms    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evaluations_created ON evaluations(created_ms DESC);
"""


def now_ms() -> int:
    return int(time.time() * 1000)


class Store:
    def __init__(self, path: str):
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.executescript(SCHEMA)

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

    # ------------------------------------------------------------------ #
    # evaluations
    # ------------------------------------------------------------------ #
    def get_evaluation(self, evaluation_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM evaluations WHERE evaluation_id=?", (evaluation_id,)
            ).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["verifier"] = json.loads(record.pop("verifier_json") or "{}")
        record["allowed_actions"] = json.loads(record.pop("allowed_actions") or "[]")
        record["response"] = json.loads(record.pop("response_json") or "null")
        record["commit_response"] = json.loads(record.pop("commit_response_json") or "null")
        return record

    def create_evaluation(
        self,
        *,
        evaluation_id: str,
        propose_hash: str,
        input_digest: str,
        verifier: dict[str, Any],
        allowed_actions: list[str],
        response: dict[str, Any],
        proposals: list[dict[str, Any]],
    ) -> None:
        ts = now_ms()
        with self._lock:
            self._begin()
            try:
                self._conn.execute(
                    "INSERT INTO evaluations (evaluation_id, propose_hash, input_digest, verifier_json,"
                    " allowed_actions, status, response_json, created_ms, updated_ms)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        evaluation_id,
                        propose_hash,
                        input_digest,
                        json.dumps(verifier, ensure_ascii=False),
                        json.dumps(allowed_actions, ensure_ascii=False),
                        "awaiting_receipts",
                        json.dumps(response, ensure_ascii=False),
                        ts,
                        ts,
                    ),
                )
                self._conn.executemany(
                    "INSERT INTO proposals (evaluation_id, dossier_id, call_id, action,"
                    " proposal_json, proposal_digest) VALUES (?,?,?,?,?,?)",
                    [
                        (
                            evaluation_id,
                            proposal["dossierId"],
                            proposal["callId"],
                            proposal["action"],
                            json.dumps(proposal, ensure_ascii=False),
                            proposal["_proposalDigest"],
                        )
                        for proposal in proposals
                    ],
                )
                self._commit()
            except Exception:
                self._rollback()
                raise

    def get_proposals(self, evaluation_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM proposals WHERE evaluation_id=?", (evaluation_id,)
            ).fetchall()
        proposals = []
        for row in rows:
            view = json.loads(row["proposal_json"])
            view["_proposalDigest"] = row["proposal_digest"]
            proposals.append(view)
        return proposals

    def complete_evaluation(
        self,
        *,
        evaluation_id: str,
        commit_hash: str,
        commit_response: dict[str, Any],
        receipts: list[dict[str, Any]],
    ) -> bool:
        """Record verified receipts and finish the evaluation atomically."""
        ts = now_ms()
        with self._lock:
            self._begin()
            try:
                cursor = self._conn.execute(
                    "UPDATE evaluations SET status='completed', commit_hash=?,"
                    " commit_response_json=?, updated_ms=? WHERE evaluation_id=? AND status=?",
                    (commit_hash, json.dumps(commit_response, ensure_ascii=False), ts, evaluation_id, "awaiting_receipts"),
                )
                if cursor.rowcount != 1:
                    self._commit()
                    return False
                self._conn.executemany(
                    "INSERT OR REPLACE INTO receipts (evaluation_id, receipt_id, receipt_json, verified)"
                    " VALUES (?,?,?,?)",
                    [
                        (
                            evaluation_id,
                            receipt["receiptId"],
                            json.dumps(receipt, ensure_ascii=False),
                            1,
                        )
                        for receipt in receipts
                    ],
                )
                self._commit()
            except Exception:
                self._rollback()
                raise
            return True

    # ------------------------------------------------------------------ #
    # decision cache
    # ------------------------------------------------------------------ #
    def get_cached_decision(self, fingerprint: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT decision_json FROM decision_cache WHERE fingerprint=?", (fingerprint,)
            ).fetchone()
        return json.loads(row["decision_json"]) if row else None

    def put_cached_decision(self, fingerprint: str, decision: dict[str, Any]) -> None:
        with self._lock:
            self._begin()
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO decision_cache (fingerprint, decision_json, created_ms)"
                    " VALUES (?,?,?)",
                    (fingerprint, json.dumps(decision, ensure_ascii=False), now_ms()),
                )
                self._commit()
            except Exception:
                self._rollback()
                raise

    def stats(self) -> dict[str, int]:
        with self._lock:
            evaluations = self._conn.execute("SELECT COUNT(*) AS n FROM evaluations").fetchone()["n"]
            cached = self._conn.execute("SELECT COUNT(*) AS n FROM decision_cache").fetchone()["n"]
            receipts = self._conn.execute("SELECT COUNT(*) AS n FROM receipts").fetchone()["n"]
        return {
            "evaluations": int(evaluations),
            "cachedDecisions": int(cached),
            "receipts": int(receipts),
        }
