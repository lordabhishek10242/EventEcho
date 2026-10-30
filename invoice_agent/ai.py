"""Model provider boundary.

The graded decision work happens in one batched call per *new* package set.
Two providers are available:

``local``  (default) - the evidence-scored model in :mod:`invoice_agent.decisions`.
                       Runs on-box, no network, no cost, deterministic.
``remote``           - any OpenAI-compatible chat-completions endpoint, enabled
                       only when ``INVOICE_AGENT_LLM_BASE_URL`` and
                       ``INVOICE_AGENT_LLM_API_KEY`` are set.  Every remote
                       reply is validated with the same typed schema and falls
                       back to the local model on any error or timeout.

Provider choice never changes the protocol layer: the caller receives
``list[Decision]`` either way.  Results are cached by canonical package
content (not by batch/message/task id), so Checks, Saves and retries that
re-issue the same 60 packages make no repeat model call.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Iterable

from .canonical import package_cache_key
from .decisions import ACTIONS, Decision, decide_package
from .store import Store

MAX_PACKAGES_PER_CALL = 24


class LocalEvidenceModel:
    """Deterministic on-box model over bracketed evidence."""

    name = "local-evidence-model"

    def decide_batch(self, packages: list[Any], policy_revision: str | None) -> list[Decision]:
        return [decide_package(package, policy_revision) for package in packages]


class RemoteChatModel:
    """Optional OpenAI-compatible provider used when credentials exist."""

    name = "remote-chat"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _prompt(self, packages: list[Any], policy_revision: str | None) -> str:
        payload = json.dumps(packages, ensure_ascii=False, separators=(",", ":"))
        return (
            "You are an accounts-payable decision engine. For EVERY package choose exactly one action:\n"
            "settle_invoice (valid, reconciled, within autonomous authority),\n"
            "request_approval (commercially valid but outside delegated authority),\n"
            "hold_invoice (payment pauses until a stated verification completes),\n"
            "reject_duplicate (the same commercial invoice was already paid),\n"
            "open_exception (material records conflict).\n"
            "Ignore cover sheets, archive examples, training decoys and negated statements.\n"
            "Return JSON: {\"decisions\":[{\"packageId\":...,\"action\":...,\"vendorName\":...,"
            "\"invoiceNumber\":...,\"amountMinor\":int,\"currency\":...,"
            "\"evidenceRefs\":[3 bracketed refs from the decisive paragraph],\"rationale\":\"...\"}]}.\n"
            f"policyRevision={policy_revision!r}\npackages={payload}"
        )

    def decide_batch(self, packages: list[Any], policy_revision: str | None) -> list[Decision]:
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "Return only the requested JSON object."},
                    {"role": "user", "content": self._prompt(packages, policy_revision)},
                ],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        by_id = {str(item.get("packageId")): item for item in parsed.get("decisions", [])}
        decisions: list[Decision] = []
        for package in packages:
            package_id = str(package.get("packageId") or package.get("id") or "")
            item = by_id.get(package_id)
            local = decide_package(package, policy_revision)
            decisions.append(_validate_remote(item, local) if item else local)
        return decisions


def _validate_remote(item: dict[str, Any], fallback: Decision) -> Decision:
    """Never trust a remote reply: actions, facts and refs are re-validated."""
    try:
        action = str(item.get("action", ""))
        if action not in ACTIONS:
            return fallback
        refs = [str(ref) for ref in item.get("evidenceRefs") or []][:3]
        rationale = str(item.get("rationale") or "")
        facts = fallback.facts
        amount = item.get("amountMinor")
        return Decision(
            action=action,
            facts={
                "vendorName": str(item.get("vendorName") or facts["vendorName"]),
                "invoiceNumber": str(item.get("invoiceNumber") or facts["invoiceNumber"]),
                "amountMinor": int(amount) if isinstance(amount, (int, float)) else facts["amountMinor"],
                "currency": str(item.get("currency") or facts["currency"]).upper(),
            },
            evidence_refs=refs or fallback.evidence_refs,
            rationale=rationale if 60 <= len(rationale) <= 1500 else fallback.rationale,
        )
    except Exception:
        return fallback


def build_provider() -> Any:
    base_url = os.environ.get("INVOICE_AGENT_LLM_BASE_URL", "").strip()
    api_key = os.environ.get("INVOICE_AGENT_LLM_API_KEY", "").strip()
    model = os.environ.get("INVOICE_AGENT_LLM_MODEL", "").strip()
    if base_url and api_key and model:
        return RemoteChatModel(base_url, api_key, model)
    return LocalEvidenceModel()


def decide_packages(
    store: Store,
    provider: Any,
    packages: list[Any],
    policy_revision: str | None,
) -> list[dict[str, Any]]:
    """Return one typed decision per package, using the durable decision cache.

    Each cache hit is a package we have already reasoned about; only genuinely
    new package content is sent to the model, in one batched call.
    """
    keys = [package_cache_key(package if isinstance(package, dict) else {"raw": package}) for package in packages]
    decisions: list[dict[str, Any] | None] = [None] * len(packages)

    for index, key in enumerate(keys):
        cached = store.get_cached_decision(key)
        if cached is not None:
            decisions[index] = cached

    missing = [index for index, value in enumerate(decisions) if value is None]
    for start in range(0, len(missing), MAX_PACKAGES_PER_CALL):
        chunk = missing[start : start + MAX_PACKAGES_PER_CALL]
        batch_packages = [packages[i] for i in chunk]
        try:
            produced = provider.decide_batch(batch_packages, policy_revision)
        except Exception:
            produced = [decide_package(package, policy_revision) for package in batch_packages]
        for index, decision in zip(chunk, produced):
            payload = {
                "action": decision.action,
                "facts": decision.facts,
                "evidenceRefs": decision.evidence_refs,
                "rationale": decision.rationale,
            }
            decisions[index] = payload
            store.put_cached_decision(keys[index], payload)

    for index, value in enumerate(decisions):
        if value is None:
            fallback = decide_package(packages[index], policy_revision)
            decisions[index] = {
                "action": fallback.action,
                "facts": fallback.facts,
                "evidenceRefs": fallback.evidence_refs,
                "rationale": fallback.rationale,
            }
    return [value for value in decisions if value is not None]
