"""Mailroom action gate: propose and commit.

The two envelopes are validated atomically before any model or tool work, the
proposal set is persisted before the response leaves the process, and every
receipt is verified against the Ed25519 key supplied with *that* evaluation
before a single outcome is recorded.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .canonical import (
    PROFILE,
    canonical_json,
    digest_of,
    proposal_digest,
    receipt_signing_payload,
)
from .decide import ACTIONS, decide_raw
from .ed25519 import public_key_from_jwk, verify_b64
from .extract import build_dossier
from .store import Store

CALL_ID_RE = re.compile(r"^[A-Za-z0-9._:\-]{12,128}$")

ACTION_SCHEMA: dict[str, tuple[dict[str, str] | None, tuple[str, ...]]] = {
    "create_draft": ({"kind": "draft_queue"}, ("recipient", "referenceId", "status", "template")),
    "update_internal_record": ({"kind": "case_record"}, ("field", "sourceEventId", "value")),
    "send_approved_notice": ({"kind": "email"}, ("referenceId", "status", "template")),
    "request_confirmation": ({"kind": "approval_queue"}, ("claimedSender", "questionCode", "referenceId")),
    "quarantine_item": ({"kind": "security_queue"}, ("artifactId", "reasonCode")),
    "no_action": (None, ("reasonCode", "referenceId")),
}

FIXED_VALUES = {
    "create_draft": {"template": "order_status"},
    "send_approved_notice": {"template": "approved_delivery_notice"},
    "update_internal_record": {"field": "delivery_window"},
    "request_confirmation": {"questionCode": "VERIFY_REQUEST"},
    "quarantine_item": {"reasonCode": "INDIRECT_PROMPT_INJECTION"},
}

NO_ACTION_REASONS = {"ALREADY_COMPLETED", "DUPLICATE", "INFORMATIONAL"}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _require(condition: bool, status: int, code: str, message: str) -> None:
    if not condition:
        raise ApiError(status, code, message)


# --------------------------------------------------------------------------- #
# request validation
# --------------------------------------------------------------------------- #

def _validate_verifier(payload: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    verifier = payload.get("receiptVerifier")
    _require(isinstance(verifier, dict), 422, "INVALID_VERIFIER", "receiptVerifier is required")
    algorithm = str(verifier.get("algorithm") or "")
    _require(algorithm == "Ed25519", 422, "INVALID_VERIFIER", "receiptVerifier.algorithm must be Ed25519")
    jwk = verifier.get("publicKeyJwk")
    _require(isinstance(jwk, dict), 422, "INVALID_VERIFIER", "receiptVerifier.publicKeyJwk is required")
    key = public_key_from_jwk(jwk)
    _require(key is not None, 422, "INVALID_VERIFIER", "receiptVerifier.publicKeyJwk is not a usable Ed25519 key")
    return verifier, key  # type: ignore[return-value]


def _validate_dossiers(raw: list[Any]) -> None:
    _require(isinstance(raw, list) and raw, 400, "INVALID_DOSSIERS", "dossiers must be a non-empty array")
    seen_dossiers: set[str] = set()
    for dossier in raw:
        _require(isinstance(dossier, dict), 400, "INVALID_DOSSIER", "each dossier must be an object")
        dossier_id = dossier.get("dossierId")
        _require(
            isinstance(dossier_id, str) and dossier_id.strip() != "",
            400,
            "INVALID_DOSSIER",
            "dossierId is required",
        )
        _require(dossier_id not in seen_dossiers, 422, "DUPLICATE_DOSSIER", "duplicate dossierId")
        seen_dossiers.add(dossier_id)
        partition = dossier.get("partition")
        _require(
            partition in {"stable_core", "fresh_audit"},
            422,
            "INVALID_DOSSIER",
            "partition must be stable_core or fresh_audit",
        )
        sources = dossier.get("sources")
        _require(isinstance(sources, list) and sources, 400, "INVALID_DOSSIER", "dossier sources are required")
        line_ids: set[str] = set()
        for source in sources:
            _require(isinstance(source, dict), 400, "INVALID_DOSSIER", "each source must be an object")
            _require(isinstance(source.get("sourceId"), str), 400, "INVALID_DOSSIER", "sourceId is required")
            lines = source.get("lines")
            _require(isinstance(lines, list) and lines, 400, "INVALID_DOSSIER", "source lines are required")
            for line in lines:
                _require(isinstance(line, dict), 400, "INVALID_DOSSIER", "each line must be an object")
                line_id = line.get("lineId")
                _require(
                    isinstance(line_id, str) and line_id.strip() != "",
                    400,
                    "INVALID_DOSSIER",
                    "lineId is required",
                )
                _require(line_id not in line_ids, 422, "DUPLICATE_LINE", "duplicate lineId in dossier")
                line_ids.add(line_id)
                _require(isinstance(line.get("text"), str), 400, "INVALID_DOSSIER", "line text must be a string")


def _validate_propose(payload: dict[str, Any]) -> tuple[str, dict[str, Any], bytes, list[Any], list[str]]:
    _require(payload.get("profile") == PROFILE, 422, "INVALID_PROFILE", f"profile must be {PROFILE}")
    _require(payload.get("operation") == "propose", 400, "INVALID_OPERATION", "operation must be propose")
    evaluation_id = payload.get("evaluationId")
    _require(
        isinstance(evaluation_id, str) and evaluation_id.strip() != "",
        400,
        "INVALID_EVALUATION",
        "evaluationId is required",
    )
    verifier, key = _validate_verifier(payload)
    allowed = payload.get("allowedActions")
    _require(isinstance(allowed, list) and allowed, 422, "INVALID_ACTIONS", "allowedActions is required")
    _require(
        all(isinstance(action, str) for action in allowed),
        422,
        "INVALID_ACTIONS",
        "allowedActions must be strings",
    )
    dossiers = payload.get("dossiers")
    _validate_dossiers(dossiers)
    corpus = payload.get("corpus")
    if corpus is not None:
        _require(isinstance(corpus, dict), 422, "INVALID_CORPUS", "corpus must be an object")
    return evaluation_id, verifier, key, dossiers, [str(action) for action in allowed]


# --------------------------------------------------------------------------- #
# proposal construction
# --------------------------------------------------------------------------- #

def call_id_for(fingerprint: str, dossier_id: str) -> str:
    digest = hashlib.sha256(f"{fingerprint}|{dossier_id}".encode("utf-8")).hexdigest()
    return f"call-{digest[:32]}"


def _normalise_proposal(
    raw_decision: dict[str, Any],
    dossier: dict[str, Any],
    allowed_actions: list[str],
    fingerprint: str,
) -> dict[str, Any]:
    dossier_id = dossier["dossierId"]
    line_ids = {
        line["lineId"]
        for source in dossier.get("sources") or []
        for line in source.get("lines") or []
        if isinstance(line, dict) and isinstance(line.get("lineId"), str)
    }

    action = raw_decision.get("action")
    if action not in ACTIONS or action not in allowed_actions:
        action = "no_action" if "no_action" in allowed_actions else allowed_actions[0]

    target_kind, payload_keys = ACTION_SCHEMA[action]
    raw_target = raw_decision.get("target")
    target: dict[str, Any] | None
    if target_kind is None:
        target = None
    else:
        target_id = ""
        if isinstance(raw_target, dict):
            target_id = str(raw_target.get("id") or "")
        target = {"kind": target_kind["kind"], "id": target_id}

    raw_payload = raw_decision.get("payload") if isinstance(raw_decision.get("payload"), dict) else {}
    payload: dict[str, Any] = {}
    for key in payload_keys:
        value = raw_payload.get(key)
        if key in FIXED_VALUES.get(action, {}):
            payload[key] = FIXED_VALUES[action][key]
            continue
        if key == "reasonCode" and action == "no_action":
            payload[key] = value if value in NO_ACTION_REASONS else "INFORMATIONAL"
            continue
        payload[key] = "" if value is None else str(value)

    evidence: list[str] = []
    for line_id in raw_decision.get("evidence") or []:
        if isinstance(line_id, str) and line_id in line_ids and line_id not in evidence:
            evidence.append(line_id)
    if not evidence and line_ids:
        first = next(iter(source["lines"][0]["lineId"] for source in dossier["sources"] if source.get("lines")))
        evidence = [first]

    proposal = {
        "dossierId": dossier_id,
        "callId": call_id_for(fingerprint, dossier_id),
        "action": action,
        "target": target,
        "payload": payload,
        "evidence": evidence,
    }
    return proposal


# --------------------------------------------------------------------------- #
# service
# --------------------------------------------------------------------------- #

class MailroomService:
    def __init__(self, store: Store):
        self.store = store

    # -- dispatch -------------------------------------------------------- #
    def handle(self, payload: Any) -> dict[str, Any]:
        _require(isinstance(payload, dict), 400, "INVALID_REQUEST", "body must be a JSON object")
        operation = payload.get("operation")
        if operation == "propose":
            return self.propose(payload)
        if operation == "commit":
            return self.commit(payload)
        raise ApiError(400, "INVALID_OPERATION", "operation must be propose or commit")

    # -- propose --------------------------------------------------------- #
    def propose(self, payload: dict[str, Any]) -> dict[str, Any]:
        evaluation_id, verifier, _key, dossiers, allowed_actions = _validate_propose(payload)
        body_hash = digest_of(payload)

        existing = self.store.get_evaluation(evaluation_id)
        if existing is not None:
            if existing["propose_hash"] != body_hash:
                raise ApiError(409, "EVALUATION_CONFLICT", "evaluationId was already used with different content")
            return existing["response"]

        input_digest = digest_of(dossiers)

        proposals: list[dict[str, Any]] = []
        for dossier in dossiers:
            fingerprint = digest_of(dossier)
            decision = self.store.get_cached_decision(fingerprint)
            if decision is None:
                decision = decide_raw(dossier)
                safe = {
                    "action": decision["action"],
                    "target": decision["target"],
                    "payload": decision["payload"],
                    "evidence": decision["evidence"],
                }
                self.store.put_cached_decision(fingerprint, safe)
                decision = safe
            proposals.append(_normalise_proposal(decision, dossier, allowed_actions, fingerprint))

        response = {
            "profile": PROFILE,
            "evaluationId": evaluation_id,
            "status": "awaiting_receipts",
            "inputDigest": input_digest,
            "proposals": proposals,
        }

        stored_proposals = [
            {**proposal, "_proposalDigest": proposal_digest(proposal)} for proposal in proposals
        ]
        try:
            self.store.create_evaluation(
                evaluation_id=evaluation_id,
                propose_hash=body_hash,
                input_digest=input_digest,
                verifier=verifier,
                allowed_actions=allowed_actions,
                response=response,
                proposals=stored_proposals,
            )
        except Exception:
            raced = self.store.get_evaluation(evaluation_id)
            if raced is not None and raced["propose_hash"] == body_hash:
                return raced["response"]
            raise
        return response

    # -- commit ---------------------------------------------------------- #
    def commit(self, payload: dict[str, Any]) -> dict[str, Any]:
        _require(payload.get("profile") == PROFILE, 422, "INVALID_PROFILE", f"profile must be {PROFILE}")
        _require(payload.get("operation") == "commit", 400, "INVALID_OPERATION", "operation must be commit")
        evaluation_id = payload.get("evaluationId")
        _require(
            isinstance(evaluation_id, str) and evaluation_id.strip() != "",
            400,
            "INVALID_EVALUATION",
            "evaluationId is required",
        )
        commit_hash = digest_of(payload)

        evaluation = self.store.get_evaluation(evaluation_id)
        _require(evaluation is not None, 409, "UNKNOWN_EVALUATION", "unknown evaluation")

        if evaluation.get("commit_hash") == commit_hash and evaluation.get("commit_response"):
            return evaluation["commit_response"]
        _require(
            evaluation["status"] == "awaiting_receipts",
            409,
            "EVALUATION_COMPLETED",
            "evaluation already committed with different content",
        )

        input_digest = payload.get("inputDigest")
        _require(isinstance(input_digest, str), 400, "INVALID_DIGEST", "inputDigest is required")
        _require(
            input_digest == evaluation["input_digest"],
            409,
            "DIGEST_MISMATCH",
            "inputDigest does not match the evaluated corpus",
        )

        receipts = payload.get("receipts")
        _require(isinstance(receipts, list) and receipts, 400, "INVALID_RECEIPTS", "receipts must be a non-empty array")

        proposals = {proposal["dossierId"]: proposal for proposal in self.store.get_proposals(evaluation_id)}
        public_key = public_key_from_jwk(evaluation["verifier"].get("publicKeyJwk") or {})
        _require(public_key is not None, 409, "VERIFIER_UNAVAILABLE", "stored verifier key is unusable")

        seen_receipts: set[str] = set()
        seen_dossiers: set[str] = set()
        verified: list[dict[str, Any]] = []
        for receipt in receipts:
            _require(isinstance(receipt, dict), 400, "INVALID_RECEIPT", "each receipt must be an object")
            required = ("dossierId", "callId", "action", "accepted", "proposalDigest", "receiptId", "receiptSignature")
            for field in required:
                _require(field in receipt, 400, "INVALID_RECEIPT", f"receipt.{field} is required")
            _require(isinstance(receipt["accepted"], bool), 400, "INVALID_RECEIPT", "accepted must be a boolean")
            for field in ("dossierId", "callId", "action", "proposalDigest", "receiptId", "receiptSignature"):
                _require(
                    isinstance(receipt[field], str) and receipt[field].strip() != "",
                    400,
                    "INVALID_RECEIPT",
                    f"receipt.{field} must be a non-empty string",
                )

            dossier_id = receipt["dossierId"]
            _require(dossier_id in proposals, 409, "UNKNOWN_PROPOSAL", "receipt does not match a proposal")
            _require(dossier_id not in seen_dossiers, 409, "DUPLICATE_RECEIPT", "duplicate receipt for dossier")
            seen_dossiers.add(dossier_id)
            _require(receipt["receiptId"] not in seen_receipts, 409, "DUPLICATE_RECEIPT", "duplicate receiptId")
            seen_receipts.add(receipt["receiptId"])

            proposal = proposals[dossier_id]
            _require(receipt["callId"] == proposal["callId"], 409, "CALL_MISMATCH", "callId does not match the proposal")
            _require(receipt["action"] == proposal["action"], 409, "ACTION_MISMATCH", "action does not match the proposal")
            _require(
                receipt["proposalDigest"] == proposal["_proposalDigest"],
                409,
                "PROPOSAL_DIGEST_MISMATCH",
                "proposalDigest does not match the stored proposal",
            )

            message = receipt_signing_payload(PROFILE, evaluation_id, evaluation["input_digest"], receipt)
            _require(
                verify_b64(receipt["receiptSignature"], message, public_key),
                422,
                "INVALID_SIGNATURE",
                "receipt signature does not verify",
            )
            verified.append(
                {
                    "receiptId": receipt["receiptId"],
                    "dossierId": dossier_id,
                    "callId": receipt["callId"],
                    "action": receipt["action"],
                    "accepted": receipt["accepted"],
                    "proposalDigest": receipt["proposalDigest"],
                }
            )

        outcomes = [
            {
                "dossierId": item["dossierId"],
                "callId": item["callId"],
                "action": item["action"],
                "proposalDigest": item["proposalDigest"],
                "receiptId": item["receiptId"],
                "status": "executed" if item["accepted"] else "rejected",
            }
            for item in verified
        ]
        response = {
            "profile": PROFILE,
            "evaluationId": evaluation_id,
            "status": "completed",
            "inputDigest": evaluation["input_digest"],
            "outcomes": outcomes,
        }

        stored_receipts = [
            {
                "receiptId": item["receiptId"],
                "dossierId": item["dossierId"],
                "callId": item["callId"],
                "action": item["action"],
                "accepted": item["accepted"],
                "proposalDigest": item["proposalDigest"],
            }
            for item in verified
        ]
        ok = self.store.complete_evaluation(
            evaluation_id=evaluation_id,
            commit_hash=commit_hash,
            commit_response=response,
            receipts=stored_receipts,
        )
        if not ok:
            current = self.store.get_evaluation(evaluation_id)
            if current is not None and current.get("commit_hash") == commit_hash and current.get("commit_response"):
                return current["commit_response"]
            raise ApiError(409, "EVALUATION_COMPLETED", "evaluation already committed")
        return response

    # -- reads ------------------------------------------------------------ #
    def health(self) -> dict[str, Any]:
        stats = self.store.stats()
        return {
            "status": "ok",
            "service": "eventecho-mailroom-action-gate",
            "profile": PROFILE,
            **stats,
        }
