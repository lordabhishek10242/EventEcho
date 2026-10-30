"""Canonical JSON + hashing helpers for the mailroom action gate.

Every digest in the contract is computed over "recursively key-sorted, compact
JSON" encoded as UTF-8: object keys sorted at every level, no insignificant
whitespace, arrays in their original order, JSON primitives spelled the way
``json.dumps`` spells them.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

PROFILE = "ga5-mailroom-action-gate/v2"


def canonical_json(value: Any) -> str:
    """Recursively key-sorted, compact JSON (UTF-8 friendly, non-ASCII kept)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: str | bytes) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def digest_of(value: Any) -> str:
    return sha256_hex(canonical_json(value))


def input_digest(dossiers: Any) -> str:
    """SHA-256 over the dossiers array exactly as received (post JSON parse)."""
    return digest_of(dossiers)


def proposal_digest(proposal: dict[str, Any]) -> str:
    """Digest of the normalised proposal view.

    Keep exactly dossierId, callId, action, target (null when absent), payload
    and evidence; sort the evidence strings; hash the canonical JSON view.
    """
    view = {
        "dossierId": proposal.get("dossierId"),
        "callId": proposal.get("callId"),
        "action": proposal.get("action"),
        "target": proposal.get("target"),
        "payload": proposal.get("payload"),
        "evidence": sorted(proposal.get("evidence") or []),
    }
    return digest_of(view)


def receipt_signing_payload(
    profile: str,
    evaluation_id: str,
    input_digest_hex: str,
    receipt: dict[str, Any],
) -> bytes:
    """Bytes covered by the Ed25519 receipt signature."""
    inner = {key: value for key, value in receipt.items() if key != "receiptSignature"}
    envelope = {
        "profile": profile,
        "evaluationId": evaluation_id,
        "inputDigest": input_digest_hex,
        "receipt": inner,
    }
    return canonical_json(envelope).encode("utf-8")
