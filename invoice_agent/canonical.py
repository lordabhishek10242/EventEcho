"""Canonical JSON helpers used for hashing and cache keys."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """Recursively key-sorted, compact JSON.

    Two messages that differ only in JSON object ordering, whitespace, or
    number formatting (1 vs 1.0 is treated separately on purpose) produce the
    same string, which is what idempotency hashing requires.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_hash(value: Any) -> str:
    """SHA-256 of the canonical JSON encoding."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def message_hash(message: dict) -> str:
    """Idempotency hash: the ``message`` object only.

    ``configuration`` is intentionally excluded so that a changed
    ``returnImmediately`` (or any other transport hint) still replays the same
    stored task instead of producing a second one.
    """
    return stable_hash(message)


def package_cache_key(package: dict) -> str:
    """Cache key for a semantic package decision.

    The package identifier is stripped because Check/Save runs re-issue the
    same 60 packages under brand new batch, message and task namespaces. The
    decision depends on document content, never on those opaque identifiers.
    """
    stripped = {k: v for k, v in package.items() if k not in {"packageId", "package_id", "id"}}
    return stable_hash(stripped)
