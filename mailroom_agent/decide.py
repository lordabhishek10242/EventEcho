"""The mailroom decision policy: one permitted action per dossier.

Layers, in priority order:

1. **Injection** — untrusted mail that tries to drive tools, read private
   context or create an outbound effect is quarantined.  A trusted quote *about*
   an attack, or a report of one, is not an attack.
2. **Duplicate / already completed** — suppressed with ``no_action``.
3. **Identity conflict** — routed to the owning team's approval queue.
4. **Approval** — a signed approval is only authority for the *exact* recipient,
   template and current facts it names; anything else is a scope mismatch and
   goes back for confirmation.
5. **Internal update** — a trusted, event-sourced delivery-window change.
6. **Normal customer work** — a non-sent draft.
7. **Informational** — ``no_action``.
"""

from __future__ import annotations

import re
from typing import Any

from .extract import (
    CLOCK_WINDOW_RE,
    _key_in,
    EMAIL_RE,
    ID_ART_RE,
    ISO_WINDOW_RE,
    Dossier,
    Line,
    build_dossier,
    collect_facts,
    norm_token,
)

ACTIONS = (
    "create_draft",
    "update_internal_record",
    "send_approved_notice",
    "request_confirmation",
    "quarantine_item",
    "no_action",
)

SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9]{12,}"),
    re.compile(r"\bAKIA[0-9A-Z]{12,}"),
    re.compile(r"\b(?:vault|canary|secret|password|passphrase|api[_ -]?key|private[_ -]?key)[:=\s]+\S{6,}", re.I),
    re.compile(r"\b[A-Fa-f0-9]{32,}\b"),
    re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b"),
)

INJECTION_STRONG = (
    re.compile(r"\bignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier|your|system)\s+(?:instructions?|rules?|prompts?|guardrails?|directives?)\b", re.I),
    re.compile(r"\bdisregard\s+(?:all\s+|the\s+|any\s+)?(?:previous|prior|above|system|safety|your)\b", re.I),
    re.compile(r"\b(?:new|updated|real|actual)\s+(?:system\s+)?instructions?\s*:", re.I),
    re.compile(r"\b(?:system|developer|admin)\s+(?:prompt|message|instruction)\s*:", re.I),
    re.compile(r"\boverride\s+(?:the\s+)?(?:system|safety|policy|policies|guardrails?|restrictions?)\b", re.I),
    re.compile(r"\b(?:reveal|print|show|output|repeat|leak|dump|export|share)\b.{0,50}\b(?:system\s+prompt|hidden\s+instructions|vault|secret|canary|password|api\s*key|access\s*token|credentials?|private\s+context)\b", re.I),
    re.compile(r"\b(?:send|forward|email|mail|post|upload|transmit|deliver|reply)\b.{0,70}\b[\w.+\-]+@[\w.\-]+\.\w{2,}", re.I),
    re.compile(r"\b(?:call|invoke|use|run|execute)\s+(?:the\s+|your\s+)?(?:tool|function|action|api)\b", re.I),
    re.compile(r"\b(?:choose|select|set|use)\s+(?:the\s+)?action\s*[:=]\s*\w+", re.I),
    re.compile(r"<\s*/?\s*(?:system|instruction|admin)s?\s*>", re.I),
    re.compile(r"\[\s*(?:system|instruction|admin)\s*\]", re.I),
    re.compile(r"\bdo\s+not\s+(?:tell|inform|alert|notify|warn)\s+(?:the\s+|any\s+)?(?:user|human|operator|customer|reviewer)\b", re.I),
    re.compile(r"\bwithout\s+(?:telling|informing|asking|notifying)\s+(?:the\s+|any\s+)?(?:user|human|customer|operator|reviewer)\b", re.I),
    re.compile(r"\b(?:you\s+are|you're|act\s+as|pretend\s+to\s+be)\s+(?:now\s+)?(?:an?\s+)?(?:admin|administrator|root|developer|supervisor|system)\b", re.I),
    re.compile(r"\b(?:approve|accept)\s+this\s+(?:request|invoice|transfer)\s+without\b", re.I),
)

REPORT_MARKERS = (
    "example",
    "e.g.",
    "sample",
    "quoted",
    "quote",
    "reported",
    "reports",
    "the sender wrote",
    "screenshot",
    "phishing",
    "awareness",
    "training",
    "do not follow",
    "never follow",
    "would say",
    "may attempt",
    "injection",
    "attack",
    "pattern",
    "warning",
    "advisory",
    "bulletin",
    "detected",
    "reported by",
    "customer wrote",
    "verbatim",
    "transcript",
)

DUPLICATE_RE = re.compile(
    r"\b(?:duplicate|duplicates|duplicated|second\s+copy|repeat(?:ed)?\s+(?:submission|request|message)|"
    r"resubmi(?:t|tted|ssion)|received\s+twice|same\s+(?:message|request|email)\s+again|"
    r"already\s+(?:received|submitted|logged|in\s+the\s+queue))\b",
    re.I,
)
COMPLETED_RE = re.compile(
    r"\b(?:already\s+(?:completed|handled|processed|sent|replied|resolved|actioned|done|closed)|"
    r"previously\s+(?:completed|handled|processed|sent|resolved)|"
    r"ticket\s+(?:is\s+)?closed|case\s+(?:is\s+)?closed|no\s+further\s+action|"
    r"reply\s+(?:was\s+)?sent|notice\s+(?:was\s+)?sent|completed\s+on\s+\d|"
    r"(?:handled|closed|resolved|actioned|answered|processed)\s+on\s+\d)\b",
    re.I,
)
INFORMATIONAL_RE = re.compile(
    r"\b(?:for\s+your\s+information|fyi|informational(?:\s+only)?|no\s+action\s+(?:is\s+)?(?:required|needed)|"
    r"no\s+action\s+necessary|nothing\s+to\s+do|newsletter|out\s+of\s+office|auto(?:matic)?[\s-]?reply|"
    r"courtesy\s+copy|just\s+(?:letting\s+you\s+know|a\s+note)|heads\s+up|notification\s+only|"
    r"do\s+not\s+reply)\b",
    re.I,
)
IDENTITY_RE = re.compile(
    r"\b(?:unverified|identity|verify|verification|could\s+not\s+be\s+verified|"
    r"does\s+not\s+match|do\s+not\s+match|mismatch(?:ed)?|claims?\s+to\s+be|"
    r"impersonat\w+|spoof\w*|unknown\s+sender|different\s+(?:address|email|number)|"
    r"not\s+the\s+account\s+holder|ownership\s+could\s+not|conflicting\s+identity)\b",
    re.I,
)
SENSITIVE_CHANGE_RE = re.compile(
    r"\b(?:change|update|replace|amend|switch)\b.{0,40}\b(?:address|email|phone|bank|account\s+number|"
    r"beneficiary|payment\s+details|routing\s+number|iban|card)\b",
    re.I,
)
APPROVAL_RE = re.compile(
    r"\b(?:approv(?:ed|al)|authoris(?:ed|ation)|authoriz(?:ed|ation)|cleared\s+to\s+send|"
    r"sign[\s-]?off|green\s*light|permission\s+to\s+send)\b",
    re.I,
)
APPROVAL_SEND_RE = re.compile(
    r"\b(?:send|sending|notify|notice|outbound|email|dispatch)\b",
    re.I,
)
INTERNAL_UPDATE_RE = re.compile(
    r"\b(?:delivery\s+window|window|eta|reschedul\w+|delivery\s+date)\b.{0,60}\b(?:chang\w+|updat\w+|moved|revised|new)\b"
    r"|\b(?:chang\w+|updat\w+|moved|revised)\b.{0,60}\b(?:delivery\s+window|window|eta|delivery\s+date)\b",
    re.I,
)
AUTHORISED_CHANGE_RE = re.compile(
    r"\b(?:authoris(?:ed|ation)|authoriz(?:ed|ation)|approved\s+change|source\s+event|event\s+(?:id|log)|"
    r"system\s+of\s+record|apply\s+the\s+change|per\s+(?:the\s+)?(?:event|record))\b",
    re.I,
)
REQUEST_RE = re.compile(
    r"\b(?:please|kindly|could\s+you|can\s+you|would\s+you|i\s+(?:want|need|would\s+like)|"
    r"request(?:ing)?|asking|let\s+me\s+know|advise|where\s+is|status\s+of|update\s+me|"
    r"confirm|when\s+will|any\s+update)\b",
    re.I,
)

STATUS_ALIASES = {
    "in_transit": "in_transit",
    "in transit": "in_transit",
    "out_for_delivery": "out_for_delivery",
    "out for delivery": "out_for_delivery",
    "delivered": "delivered",
    "delayed": "delayed",
    "shipped": "shipped",
    "processing": "processing",
    "packed": "packed",
    "cancelled": "cancelled",
    "canceled": "cancelled",
    "returned": "returned",
    "on_hold": "on_hold",
    "on hold": "on_hold",
    "pending": "pending",
    "exception": "exception",
    "rescheduled": "rescheduled",
    "dispatched": "dispatched",
}

DEFAULT_TEAM = "customer_verification"
SECURITY_QUEUE = "mailroom"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _safe_value(value: Any, limit: int = 160) -> str | None:
    """Payload values are typed facts: short, single-line, never a secret."""
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text or len(text) > limit:
        return None
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        return None
    return text


def _line_ids(lines: list[Line]) -> list[str]:
    seen: list[str] = []
    for line in lines:
        if line and line.line_id and line.line_id not in seen:
            seen.append(line.line_id)
    return seen


_QUOTE_PAIRS = (('"', '"'), ("\u201c", "\u201d"), ("'", "'"), ("`", "`"))


def _inside_quotes(text: str, start: int, end: int) -> bool:
    for opening, closing in _QUOTE_PAIRS:
        positions: list[int] = []
        index = text.find(opening)
        while index != -1:
            positions.append(index)
            index = text.find(opening, index + 1)
        if len(positions) < 2:
            continue
        for left, right in zip(positions[0::2], positions[1::2]):
            if left < start and end <= right:
                return True
        if closing != opening:
            continue
    return False


def _is_reported(line: Line, match: re.Match[str]) -> bool:
    """A quote *about* an attack, or a security report, is not an attack."""
    text = line.text
    start, end = match.span()
    if _inside_quotes(text, start, end):
        return True
    # Addresses are not evidence of a security report (attacker@example.com
    # would otherwise look like the word "attack" plus "example").
    nearby = text[max(0, start - 90): min(len(text), end + 90)]
    nearby = EMAIL_RE.sub(" ", nearby).lower()
    for marker in REPORT_MARKERS:
        if re.search(rf"(?<![a-z0-9]){re.escape(marker)}(?![a-z0-9])", nearby):
            return True
    return False


def _injection_lines(dossier: Dossier) -> list[Line]:
    hits: list[Line] = []
    for line in dossier.lines:
        if line.trusted:
            continue
        score = 0
        for pattern in INJECTION_STRONG:
            for match in pattern.finditer(line.text):
                if _is_reported(line, match):
                    score += 0.4
                else:
                    score += 3.0
        if score >= 3.0:
            hits.append(line)
    return hits


def _approval_info(dossier: Dossier, facts: dict[str, Any]) -> dict[str, Any] | None:
    approval_lines: list[Line] = []
    for line in dossier.lines:
        kind = norm_token(line.kind) + " " + norm_token(line.provenance)
        if not line.trusted:
            continue
        if ("approval" in kind or "authoris" in kind or "authoriz" in kind) and APPROVAL_RE.search(line.text):
            approval_lines.append(line)
            continue
        if APPROVAL_RE.search(line.text) and APPROVAL_SEND_RE.search(line.text):
            approval_lines.append(line)
    if not approval_lines:
        return None
    emails: list[str] = []
    statuses: list[str] = []
    approval_id: str | None = None
    for line in approval_lines:
        for match in EMAIL_RE.finditer(line.text):
            if match.group(0) not in emails:
                emails.append(match.group(0))
        for word, canonical in STATUS_ALIASES.items():
            if re.search(rf"\b{re.escape(word)}\b", line.text, re.I):
                if canonical not in statuses:
                    statuses.append(canonical)
        label_id = re.search(r"\b(?:approval|authorisation|authorization)[\s_-]*(?:id|ref(?:erence)?)?\s*[:#=]?\s*([A-Za-z0-9._:-]{4,})", line.text, re.I)
        if label_id and approval_id is None:
            approval_id = label_id.group(1)
    references: list[str] = []
    expired = False
    for line in approval_lines:
        for match in re.finditer(r"\bORD[-_ ]?[A-Z0-9]{2,}(?:[-_][A-Z0-9]{2,})*\b", line.text, re.I):
            if match.group(0) not in references:
                references.append(match.group(0))
        if re.search(r"\b(?:expired|no longer valid|superseded|revoked|withdrawn)\b", line.text, re.I):
            expired = True
    return {
        "lines": approval_lines,
        "emails": emails,
        "statuses": statuses,
        "references": references,
        "expired": expired,
        "approvalId": approval_id,
    }


def _identity_conflict(dossier: Dossier, facts: dict[str, Any]) -> tuple[bool, list[Line]]:
    evidence: list[Line] = []
    untrusted = [line for line in dossier.lines if not line.trusted]
    trusted = [line for line in dossier.lines if line.trusted]

    claimed = facts.get("claimedSender")
    verified = facts.get("customerContact")
    if claimed and verified and claimed.lower() != verified.lower():
        evidence.append(facts_line(dossier, facts["evidence"].get("claimedSender")))
        evidence.append(facts_line(dossier, facts["evidence"].get("customerContact")))
        return True, [line for line in evidence if line]

    for line in dossier.lines:
        if not IDENTITY_RE.search(line.text):
            continue
        if line.trusted or any(marker in line.lower for marker in ("policy", "runbook", "procedure", "if ", "when ")):
            # policy text describing when to verify is not itself a conflict
            if not line.trusted:
                continue
        evidence.append(line)
    if evidence:
        return True, evidence[:2]

    for line in untrusted:
        if SENSITIVE_CHANGE_RE.search(line.text):
            evidence.append(line)
    if evidence:
        return True, evidence[:2]
    return False, []


def facts_line(dossier: Dossier, line_id: str | None) -> Line | None:
    if not line_id:
        return None
    return dossier.line(line_id)


def _duplicate_signal(dossier: Dossier, facts: dict[str, Any]) -> tuple[str | None, list[Line]]:
    duplicate_lines: list[Line] = []
    completed_lines: list[Line] = []
    for line in dossier.lines:
        if DUPLICATE_RE.search(line.text):
            duplicate_lines.append(line)
        if COMPLETED_RE.search(line.text):
            completed_lines.append(line)
    if duplicate_lines:
        return "DUPLICATE", duplicate_lines[:2]
    if completed_lines:
        return "ALREADY_COMPLETED", completed_lines[:2]
    return None, []


def _informational_signal(dossier: Dossier) -> list[Line]:
    hits = [line for line in dossier.lines if INFORMATIONAL_RE.search(line.text)]
    return hits[:2]


def _internal_update(dossier: Dossier, facts: dict[str, Any]) -> tuple[bool, list[Line]]:
    trusted_lines = [line for line in dossier.lines if line.trusted]
    if not trusted_lines:
        return False, []
    change_lines = [line for line in trusted_lines if INTERNAL_UPDATE_RE.search(line.text)]
    authorising = [line for line in trusted_lines if AUTHORISED_CHANGE_RE.search(line.text)]
    has_event = bool(facts.get("sourceEventId"))
    has_window = bool(facts.get("deliveryWindow"))
    if (change_lines or has_window) and (has_event or authorising):
        evidence = []
        for line_id in (facts["evidence"].get("sourceEventId"), facts["evidence"].get("deliveryWindow")):
            line = facts_line(dossier, line_id)
            if line:
                evidence.append(line)
        evidence.extend(change_lines)
        evidence.extend(authorising)
        return True, _dedupe_lines(evidence)
    return False, []


def _dedupe_lines(lines: list[Line]) -> list[Line]:
    seen: set[str] = set()
    result: list[Line] = []
    for line in lines:
        if line and line.line_id not in seen:
            seen.add(line.line_id)
            result.append(line)
    return result


NEW_WINDOW_KEYS = ("newdeliverywindow", "newdelivery_window", "new_window", "newwindow",
                   "revisedwindow", "revised_window", "updatedwindow", "updated_window",
                   "deliverywindow", "delivery_window", "window", "eta")


def _new_window_line(dossier: Dossier, facts: dict[str, Any]) -> tuple[str, Line] | None:
    """The window an update should apply: event line first, then a new/revised
    label, then the freshest trusted record."""
    event_line = facts_line(dossier, facts["evidence"].get("sourceEventId"))
    if event_line is not None:
        match = ISO_WINDOW_RE.search(event_line.text) or CLOCK_WINDOW_RE.search(event_line.text)
        if match:
            text = event_line.text
            window = match.group(0)
            if match.re is ISO_WINDOW_RE:
                clock = CLOCK_WINDOW_RE.search(text)
                if clock and clock.start() >= match.end():
                    window = f"{window} {clock.group(0)}"
            cleaned = _safe_value(window, 120)
            if cleaned:
                return cleaned, event_line
        for key, value in event_line.labels.items():
            if _key_in(key, NEW_WINDOW_KEYS) and re.search(r"\d", value):
                cleaned = _safe_value(value, 120)
                if cleaned:
                    return cleaned, event_line
    for line in dossier.lines:
        if not line.trusted:
            continue
        for key, value in line.labels.items():
            if _key_in(key, ("newdeliverywindow", "revisedwindow", "updatedwindow", "newwindow")) and re.search(r"\d", value):
                cleaned = _safe_value(value, 120)
                if cleaned:
                    return cleaned, line
    return None


def _already_proven(evidence: list[Line], value: str | None) -> bool:
    if not value:
        return False
    needle = str(value).strip().lower()
    return any(needle and needle in line.text.lower() for line in evidence)


def _add_fact_line(evidence: list[Line], dossier: Dossier, facts: dict[str, Any], key: str) -> None:
    value = facts.get(key)
    line = facts_line(dossier, facts["evidence"].get(key))
    if line is None or any(item.line_id == line.line_id for item in evidence):
        return
    if _already_proven(evidence, value):
        return
    evidence.append(line)


def _draft_evidence(dossier: Dossier, facts: dict[str, Any]) -> list[Line]:
    evidence: list[Line] = []
    for key in ("status", "claimedSender", "referenceId"):
        _add_fact_line(evidence, dossier, facts, key)
    return evidence[:3]


def _status_for_payload(facts: dict[str, Any]) -> str | None:
    status = facts.get("status")
    if not status:
        return None
    return _safe_value(status, 40)


def _reference_for_payload(facts: dict[str, Any]) -> str | None:
    for key in ("referenceId", "caseId"):
        value = _safe_value(facts.get(key), 80)
        if value:
            return value
    return None


def _recipient_for_payload(facts: dict[str, Any]) -> str | None:
    """Only an address that identifies the correspondent, never one that merely
    appears in prose (an attacker address, a policy example, a colleague)."""
    for key in ("claimedSender", "customerContact"):
        value = _safe_value(facts.get(key), 120)
        if value and EMAIL_RE.fullmatch(value):
            return value
    return None


# --------------------------------------------------------------------------- #
# decision
# --------------------------------------------------------------------------- #

def _proposal(
    dossier: Dossier,
    action: str,
    target: dict[str, Any] | None,
    payload: dict[str, Any],
    evidence: list[Line],
    reason: str,
) -> dict[str, Any]:
    return {
        "action": action,
        "target": target,
        "payload": payload,
        "evidence": _line_ids(_dedupe_lines(evidence))[:4],
        "reason": reason,
    }


def decide(dossier: Dossier) -> dict[str, Any]:
    facts = collect_facts(dossier)
    identity_evidence = facts["evidence"]

    # 1. Injection -----------------------------------------------------------
    injection = _injection_lines(dossier)
    if injection:
        artifact_line = None
        for line in injection:
            if ID_ART_RE.search(line.text):
                artifact_line = line
                break
        artifact_value = facts.get("artifactId")
        if artifact_line is not None:
            match = ID_ART_RE.search(artifact_line.text)
            if match:
                artifact_value = match.group(0)
        if not artifact_value:
            source_id = injection[0].source_id
            artifact_value = source_id or "mailroom-artifact"
        artifact_value = _safe_value(artifact_value, 80) or "mailroom-artifact"
        evidence = list(injection[:2])
        artifact_source_line = facts_line(dossier, facts["evidence"].get("artifactId"))
        for extra in (artifact_source_line, artifact_line):
            if extra is not None and extra not in evidence:
                evidence.append(extra)
        return _proposal(
            dossier,
            "quarantine_item",
            {"kind": "security_queue", "id": SECURITY_QUEUE},
            {"artifactId": artifact_value, "reasonCode": "INDIRECT_PROMPT_INJECTION"},
            evidence,
            "untrusted content tries to drive tools or private context",
        )

    # 2. Duplicate / completed ----------------------------------------------
    reason_code, evidence = _duplicate_signal(dossier, facts)
    if reason_code:
        payload = {"reasonCode": reason_code, "referenceId": _reference_for_payload(facts) or ""}
        if not payload["referenceId"]:
            return _confirmation(dossier, facts, "duplicate without a usable reference")
        return _proposal(dossier, "no_action", None, payload, evidence, "already handled")

    # 3. Identity conflict ---------------------------------------------------
    conflict, identity_lines = _identity_conflict(dossier, facts)
    if conflict:
        return _confirmation(dossier, facts, "identity needs verification", identity_lines)

    # 4. Approval scope ------------------------------------------------------
    approval = _approval_info(dossier, facts)
    if approval is not None:
        recipient = _recipient_for_payload(facts)
        status = _status_for_payload(facts)
        approved_recipients = [email for email in approval["emails"]]
        approved_statuses = approval["statuses"]
        recipient_ok = bool(approved_recipients) and recipient is not None and recipient.lower() in {
            email.lower() for email in approved_recipients
        }
        status_ok = bool(approved_statuses) and status is not None and status.lower() in {
            value.lower() for value in approved_statuses
        }
        reference = _reference_for_payload(facts)
        approved_references = approval.get("references") or []
        reference_ok = (not approved_references) or (
            reference is not None and reference.lower() in {ref.lower() for ref in approved_references}
        )
        if recipient_ok and status_ok and reference and reference_ok and not approval.get("expired"):
            evidence = list(approval["lines"])
            for key in ("status", "referenceId", "claimedSender", "customerContact"):
                _add_fact_line(evidence, dossier, facts, key)
            return _proposal(
                dossier,
                "send_approved_notice",
                {"kind": "email", "id": approved_recipients[0]},
                {
                    "referenceId": reference,
                    "status": status,
                    "template": "approved_delivery_notice",
                },
                evidence,
                "trusted approval matches recipient and current facts",
            )
        return _confirmation(
            dossier,
            facts,
            "approval scope does not match this recipient or current facts",
            list(approval["lines"]) + identity_lines,
        )

    # 5. Internal record update ---------------------------------------------
    update_ok, update_lines = _internal_update(dossier, facts)
    if update_ok:
        case_id = _safe_value(facts.get("caseId") or facts.get("referenceId") or dossier.dossier_id, 80)
        event_id = _safe_value(facts.get("sourceEventId"), 80)
        window_line = _new_window_line(dossier, facts)
        value = _safe_value(window_line[0] if window_line else facts.get("deliveryWindow"), 120)
        if window_line:
            update_lines = _dedupe_lines(list(update_lines) + [window_line[1]])
        if case_id and event_id and value:
            return _proposal(
                dossier,
                "update_internal_record",
                {"kind": "case_record", "id": case_id},
                {"field": "delivery_window", "sourceEventId": event_id, "value": value},
                update_lines,
                "trusted event authorises the delivery-window change",
            )

    # 6. Normal customer work ------------------------------------------------
    recipient = _recipient_for_payload(facts)
    reference = _reference_for_payload(facts)
    status = _status_for_payload(facts)
    if recipient and reference and dossier.mailbox:
        return _proposal(
            dossier,
            "create_draft",
            {"kind": "draft_queue", "id": f"mailbox:{dossier.mailbox}"},
            {
                "recipient": recipient,
                "referenceId": reference,
                "status": status or "processing",
                "template": "order_status",
            },
            _draft_evidence(dossier, facts),
            "customer request needs a non-sent draft reply",
        )

    # 7. Informational -------------------------------------------------------
    info = _informational_signal(dossier)
    if info:
        payload = {"reasonCode": "INFORMATIONAL", "referenceId": _reference_for_payload(facts) or ""}
        if payload["referenceId"]:
            return _proposal(dossier, "no_action", None, payload, info, "informational only")

    if reference:
        return _proposal(
            dossier,
            "no_action",
            None,
            {"reasonCode": "INFORMATIONAL", "referenceId": reference},
            info or [],
            "nothing actionable in this dossier",
        )
    return _confirmation(dossier, facts, "ambiguous dossier without a usable reference", [])


def _confirmation(
    dossier: Dossier,
    facts: dict[str, Any],
    reason: str,
    extra_lines: list[Line] | None = None,
) -> dict[str, Any]:
    team = _safe_value(facts.get("team"), 60) or DEFAULT_TEAM
    claimed = _safe_value(facts.get("claimedSender"), 120) or _safe_value(facts.get("customerContact"), 120)
    reference = _reference_for_payload(facts) or ""
    evidence: list[Line] = []
    for line in extra_lines or []:
        evidence.append(line)
    for key in ("claimedSender", "referenceId", "team"):
        line = facts_line(dossier, facts["evidence"].get(key))
        if line:
            evidence.append(line)
    if not claimed or not reference:
        # Without the two facts the tool needs, the item is not sendable in any
        # form; park it as informational rather than inventing arguments.
        if reference:
            return _proposal(
                dossier,
                "no_action",
                None,
                {"reasonCode": "INFORMATIONAL", "referenceId": reference},
                evidence,
                "no safe tool arguments available",
            )
        return _proposal(
            dossier,
            "quarantine_item",
            {"kind": "security_queue", "id": SECURITY_QUEUE},
            {"artifactId": _safe_value(dossier.lines[0].source_id if dossier.lines else None, 80) or "mailroom-artifact",
             "reasonCode": "INDIRECT_PROMPT_INJECTION"},
            evidence[:2],
            "unclassifiable content isolated for review",
        )
    return _proposal(
        dossier,
        "request_confirmation",
        {"kind": "approval_queue", "id": team},
        {"claimedSender": claimed, "questionCode": "VERIFY_REQUEST", "referenceId": reference},
        evidence,
        reason,
    )


def decide_raw(raw_dossier: dict[str, Any]) -> dict[str, Any]:
    return decide(build_dossier(raw_dossier))
