"""A2A 1.0 HTTP+JSON protocol objects: constants, Agent Card, envelopes."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

IN_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
IN_RESULTS = "application/vnd.ga5.invoice-action-results+json"
OUT_PROPOSALS = "application/vnd.ga5.invoice-action-proposals+json"
OUT_RECEIPTS = "application/vnd.ga5.invoice-action-receipts+json"

A2A_MEDIA_TYPE = "application/a2a+json"
A2A_VERSION = "1.0"

STATE_SUBMITTED = "TASK_STATE_SUBMITTED"
STATE_WORKING = "TASK_STATE_WORKING"
STATE_INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
STATE_COMPLETED = "TASK_STATE_COMPLETED"
STATE_CANCELED = "TASK_STATE_CANCELED"
TERMINAL_STATES = {STATE_COMPLETED, STATE_CANCELED}

ROLE_USER = "ROLE_USER"
ROLE_AGENT = "ROLE_AGENT"

MAX_RESPONSE_BYTES = 512 * 1024


def iso_timestamp(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms % 1000:03d}Z"


def agent_card(base_url: str) -> dict[str, Any]:
    """Public Agent Card for the submitted base URL."""
    canonical = base_url if base_url.endswith("/") else base_url + "/"
    interfaces = [
        {"url": canonical, "protocolBinding": "HTTP+JSON", "protocolVersion": A2A_VERSION},
        {"url": canonical.rstrip("/"), "protocolBinding": "HTTP+JSON", "protocolVersion": A2A_VERSION},
    ]
    return {
        "name": "EventEcho Invoice Action Agent",
        "description": (
            "Reads invoice claim batches, proposes exactly one typed business action per package "
            "(settle, request approval, hold, reject duplicate, open exception) with the three "
            "decisive bracketed evidence references, and executes only actions the reviewer accepts."
        ),
        "version": "1.0.0",
        "protocolVersion": A2A_VERSION,
        "url": canonical,
        "preferredTransport": "HTTP+JSON",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
            "extendedAgentCard": False,
        },
        "defaultInputModes": [IN_BATCH, IN_RESULTS, "application/json"],
        "defaultOutputModes": [OUT_PROPOSALS, OUT_RECEIPTS],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice action decisions",
                "description": (
                    "Chooses one of settle_invoice, request_approval, hold_invoice, reject_duplicate "
                    "or open_exception for every invoice package, cites the decisive evidence "
                    "references, waits for approval results, and then records the executions."
                ),
                "tags": [
                    "invoice",
                    "accounts-payable",
                    "approval-limits",
                    "duplicate-payment",
                    "evidence-citations",
                    "a2a",
                ],
                "examples": [
                    "Settle a reconciled invoice that is within autonomous authority.",
                    "Request approval for an invoice above the delegated limit.",
                ],
                "inputModes": [IN_BATCH, IN_RESULTS],
                "outputModes": [OUT_PROPOSALS, OUT_RECEIPTS],
            }
        ],
        "supportedInterfaces": interfaces,
    }


def normalize_message(message: dict[str, Any]) -> dict[str, Any]:
    """Keep the received message exactly as history requires, minus transport noise."""
    parts: list[dict[str, Any]] = []
    for part in message.get("parts") or []:
        if not isinstance(part, dict):
            continue
        clean: dict[str, Any] = {}
        if "mediaType" in part:
            clean["mediaType"] = part["mediaType"]
        if "data" in part:
            clean["data"] = part["data"]
        if "text" in part:
            clean["text"] = part["text"]
        if "metadata" in part:
            clean["metadata"] = part["metadata"]
        if clean:
            parts.append(clean)
    normalized: dict[str, Any] = {
        "messageId": message.get("messageId"),
        "role": message.get("role") or ROLE_USER,
        "parts": parts,
    }
    for key in ("taskId", "contextId"):
        if message.get(key):
            normalized[key] = message[key]
    return normalized


def task_json(task: dict[str, Any]) -> dict[str, Any]:
    """Serialise a stored task row into the A2A Task object (ProtoJSON strings)."""
    artifacts: list[dict[str, Any]] = []
    proposals = task.get("proposals")
    if proposals is not None:
        artifacts.append(
            {
                "artifactId": f"{task['id']}-proposals",
                "name": "invoice-action-proposals",
                "description": "One typed action proposal per invoice package.",
                "parts": [
                    {
                        "mediaType": OUT_PROPOSALS,
                        "data": {"batchId": task.get("batch_id"), "proposals": proposals},
                    }
                ],
            }
        )
    receipts = task.get("receipts")
    if receipts is not None:
        artifacts.append(
            {
                "artifactId": f"{task['id']}-receipts",
                "name": "invoice-action-receipts",
                "description": "Executions for accepted proposals only.",
                "parts": [{"mediaType": OUT_RECEIPTS, "data": receipts}],
            }
        )

    status: dict[str, Any] = {
        "state": task["state"],
        "timestamp": iso_timestamp(int(task.get("updated_ms") or task.get("created_ms") or 0)),
    }

    view: dict[str, Any] = {
        "id": task["id"],
        "contextId": task["context_id"],
        "status": status,
        "artifacts": artifacts,
        "history": task.get("history") or [],
    }
    metadata = task.get("metadata") or {}
    if metadata:
        view["metadata"] = metadata
    return view


def error_body(code: str, message: str) -> dict[str, Any]:
    return {"error": {"code": code, "message": message}}
