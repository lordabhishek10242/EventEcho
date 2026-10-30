"""Reading dossiers: line indexes, trust, and typed facts.

External mail is data, never instruction.  This module only *extracts* values
(identifiers, addresses, statuses, windows, queue names); it never copies a
sentence into a tool argument.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
ID_ORD_RE = re.compile(r"\bORD[-_ ]?[A-Z0-9]{2,}(?:[-_][A-Z0-9]{2,})*\b", re.I)
ID_CASE_RE = re.compile(r"\bCASE[-_ ]?[A-Z0-9]{2,}(?:[-_][A-Z0-9]{2,})*\b", re.I)
ID_EVENT_RE = re.compile(r"\bEVT[-_ ]?[A-Z0-9]{2,}(?:[-_][A-Z0-9]{2,})*\b", re.I)
ID_ART_RE = re.compile(r"\b(?:ART|ATT|ATTACH|FILE)(?:[-_]\d[A-Z0-9_-]*|\d{2,}[A-Z0-9_-]*)\b")
GENERIC_ID_RE = re.compile(r"\b[A-Z]{2,6}[-_][A-Z0-9]{2,}(?:[-_][A-Z0-9]{2,})*\b")

LABEL_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _/'’\-]{1,34})\s*[:=]\s*(.+?)\s*$")
ISO_WINDOW_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?"
    r"(?:\s*(?:-|–|to|until|through)\s*\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?)?",
)
CLOCK_WINDOW_RE = re.compile(r"\b\d{1,2}:\d{2}\s*(?:-|–|to)\s*\d{1,2}:\d{2}\b")

STATUS_WORDS = (
    "in_transit",
    "in transit",
    "out_for_delivery",
    "out for delivery",
    "delivered",
    "delayed",
    "shipped",
    "processing",
    "packed",
    "picked_up",
    "picked up",
    "awaiting_pickup",
    "cancelled",
    "canceled",
    "returned",
    "return_in_progress",
    "on_hold",
    "on hold",
    "pending",
    "exception",
    "lost",
    "rescheduled",
    "dispatched",
    "ready_for_dispatch",
)

TRUSTED_TOKENS = (
    "internal",
    "trusted",
    "policy",
    "crm",
    "erp",
    "system_of_record",
    "systemofrecord",
    "system",
    "record",
    "approval",
    "approved",
    "authorisation",
    "authorization",
    "audit",
    "order_management",
    "oms",
    "directory",
    "authoritative",
    "event_log",
    "case_record",
    "billing",
    "hr_",
    "ticketing",
    "knowledge_base",
    "runbook",
    "sop",
)

UNTRUSTED_TOKENS = (
    "external",
    "untrusted",
    "customer",
    "inbound",
    "attachment",
    "web",
    "public",
    "unknown",
    "third_party",
    "thirdparty",
    "vendor_email",
    "newsletter",
    "forward",
    "social",
)

TRUSTED_KINDS = {
    "crm_record",
    "order_record",
    "order_management_record",
    "case_record",
    "internal_note",
    "internal_policy",
    "policy",
    "approval_record",
    "approval",
    "event_log",
    "audit_log",
    "directory_record",
    "system_record",
    "knowledge_base",
    "runbook",
    "sop",
}

UNTRUSTED_KINDS = {
    "inbound_email",
    "customer_email",
    "external_email",
    "email",
    "attachment",
    "web_page",
    "newsletter",
    "social_post",
    "forwarded_email",
    "voicemail_transcript",
}


def norm_token(text: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(text or "").strip().lower()).strip("_")


@dataclass
class Line:
    line_id: str
    text: str
    source_id: str
    kind: str
    provenance: str
    title: str
    trusted: bool
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def lower(self) -> str:
        return self.text.lower()


@dataclass
class Dossier:
    dossier_id: str
    partition: str
    mailbox: str
    objective: str
    received_at: str
    lines: list[Line]
    source_ids: dict[str, str]

    def line(self, line_id: str) -> Line | None:
        for item in self.lines:
            if item.line_id == line_id:
                return item
        return None

    @property
    def text(self) -> str:
        return "\n".join(item.text for item in self.lines)


def _source_trust(source: dict[str, Any]) -> bool:
    blob = " ".join(
        norm_token(source.get(key)) for key in ("kind", "provenance", "title", "sourceId")
    )
    kind = norm_token(source.get("kind"))
    provenance = norm_token(source.get("provenance"))
    if any(token in provenance for token in UNTRUSTED_TOKENS):
        return False
    if any(token in blob for token in UNTRUSTED_TOKENS):
        return False
    if kind in TRUSTED_KINDS:
        return True
    if any(token in kind for token in TRUSTED_TOKENS):
        return True
    if any(token in provenance for token in TRUSTED_TOKENS):
        return True
    if kind in UNTRUSTED_KINDS:
        return False
    # Mail that merely arrived is untrusted until a record says otherwise.
    return False


def _parse_labels(text: str) -> dict[str, str]:
    match = LABEL_RE.match(text)
    if not match:
        return {}
    key = norm_token(match.group(1))
    value = match.group(2).strip().strip('"').strip()
    return {key: value} if key and value else {}


def build_dossier(raw: dict[str, Any]) -> Dossier:
    lines: list[Line] = []
    source_ids: dict[str, str] = {}
    for source in raw.get("sources") or []:
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("sourceId") or "")
        trusted = _source_trust(source)
        source_ids[source_id] = str(source.get("kind") or "")
        for item in source.get("lines") or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            lines.append(
                Line(
                    line_id=str(item.get("lineId") or ""),
                    text=text,
                    source_id=source_id,
                    kind=str(source.get("kind") or ""),
                    provenance=str(source.get("provenance") or ""),
                    title=str(source.get("title") or ""),
                    trusted=trusted,
                    labels=_parse_labels(text),
                )
            )
    return Dossier(
        dossier_id=str(raw.get("dossierId") or ""),
        partition=str(raw.get("partition") or ""),
        mailbox=str(raw.get("mailbox") or ""),
        objective=str(raw.get("objective") or ""),
        received_at=str(raw.get("receivedAt") or ""),
        lines=lines,
        source_ids=source_ids,
    )


# --------------------------------------------------------------------------- #
# Fact extraction
# --------------------------------------------------------------------------- #

REF_KEYS = (
    "referenceid",
    "reference",
    "reference_id",
    "ref",
    "orderreference",
    "order_reference",
    "orderid",
    "order_id",
    "ordernumber",
    "order_number",
    "order",
    "caseid",
    "case_id",
    "case",
    "casenumber",
    "case_number",
    "ticket",
    "ticketid",
)
STATUS_KEYS = ("status", "orderstatus", "order_status", "deliverystatus", "delivery_status", "currentstatus", "current_status", "state")
WINDOW_KEYS = (
    "deliverywindow",
    "delivery_window",
    "window",
    "revisedwindow",
    "revised_window",
    "eta",
    "deliveryeta",
    "delivery_eta",
    "expecteddelivery",
    "expected_delivery",
    "newdeliverywindow",
)
EVENT_KEYS = ("sourceeventid", "source_event_id", "eventid", "event_id", "event", "triggerevent", "trigger_event", "sourcereference")
TEAM_KEYS = ("owningteam", "owning_team", "team", "queue", "approvalqueue", "approval_queue", "routeto", "route_to", "assignedto", "assigned_to", "owner", "routing")
ARTIFACT_KEYS = ("artifactid", "artifact_id", "artifact", "attachmentid", "attachment_id", "attachment", "file", "fileid", "file_id")
SENDER_KEYS = ("from", "sender", "claimedsender", "claimed_sender", "replyto", "reply_to", "author", "sentby", "sent_by")
CASE_KEYS = ("caseid", "case_id", "case", "casenumber", "case_number", "dossier", "ticket", "ticketid")


def _key_in(key: str, candidates: Iterable[str]) -> bool:
    plain = re.sub(r"[^a-z0-9]", "", key.lower())
    return any(plain == re.sub(r"[^a-z0-9]", "", candidate.lower()) for candidate in candidates)


def _first_label(lines: Iterable[Line], keys: tuple[str, ...], trusted_only: bool | None = None) -> tuple[str, Line] | None:
    for line in lines:
        if trusted_only is True and not line.trusted:
            continue
        if trusted_only is False and line.trusted:
            continue
        for key, value in line.labels.items():
            if value and _key_in(key, keys):
                return value, line
    return None


def _clean_value(value: str, limit: int = 160) -> str:
    cleaned = re.sub(r"\s+", " ", str(value)).strip().strip('"').strip("'").strip()
    cleaned = re.split(r"\s+(?:and|but|so|which|because)\s+", cleaned, maxsplit=1)[0]
    return cleaned.strip(" .,;:")[:limit]


def _find_id(lines: Iterable[Line], pattern: re.Pattern, trusted_only: bool | None = None) -> tuple[str, Line] | None:
    for line in lines:
        if trusted_only is True and not line.trusted:
            continue
        if trusted_only is False and line.trusted:
            continue
        match = pattern.search(line.text)
        if match:
            return match.group(0).strip(), line
    return None


def _find_status(lines: Iterable[Line], trusted_only: bool | None = True) -> tuple[str, Line] | None:
    labeled = _first_label(lines, STATUS_KEYS, trusted_only=trusted_only)
    if labeled:
        value, line = labeled
        for word in STATUS_WORDS:
            if word in value.lower():
                return value.lower(), line
        return _clean_value(value, 40), line
    for line in lines:
        if trusted_only is True and not line.trusted:
            continue
        lower = line.lower
        for word in STATUS_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", lower):
                return word, line
    return None


def _find_window(lines: Iterable[Line], trusted_only: bool | None = True) -> tuple[str, Line] | None:
    labeled = _first_label(lines, WINDOW_KEYS, trusted_only=trusted_only)
    if labeled:
        value, line = labeled
        return _clean_value(value), line
    for line in lines:
        if trusted_only is True and not line.trusted:
            continue
        match = ISO_WINDOW_RE.search(line.text)
        if match:
            window = match.group(0).strip()
            clock = CLOCK_WINDOW_RE.search(line.text)
            if clock and clock.start() >= match.end():
                window = f"{window} {clock.group(0)}"
            return _clean_value(window), line
    return None


STATUS_HINT_WORDS = ("status", "state")
WINDOW_HINT_WORDS = ("window", "eta", "delivery time", "expected delivery")


def collect_facts(dossier: Dossier) -> dict[str, Any]:
    trusted = [line for line in dossier.lines if line.trusted]
    untrusted = [line for line in dossier.lines if not line.trusted]
    ordered = dossier.lines

    facts: dict[str, Any] = {"evidence": {}}

    # Order / case references ------------------------------------------------
    ref = _first_label(trusted, REF_KEYS)
    if ref is None:
        ref = _find_id(ordered, ID_ORD_RE, trusted_only=True) or _find_id(ordered, ID_ORD_RE)
    if ref:
        facts["referenceId"], facts["evidence"]["referenceId"] = ref[0], ref[1].line_id

    case = _first_label(trusted, CASE_KEYS) or _find_id(ordered, ID_CASE_RE, trusted_only=True) or _find_id(ordered, ID_CASE_RE)
    if case:
        facts["caseId"], facts["evidence"]["caseId"] = case[0], case[1].line_id

    event = _first_label(trusted, EVENT_KEYS) or _find_id(ordered, ID_EVENT_RE, trusted_only=True) or _find_id(ordered, ID_EVENT_RE)
    if event:
        facts["sourceEventId"], facts["evidence"]["sourceEventId"] = event[0], event[1].line_id

    artifact = _first_label(ordered, ARTIFACT_KEYS) or _find_id(ordered, ID_ART_RE)
    if artifact:
        facts["artifactId"], facts["evidence"]["artifactId"] = artifact[0], artifact[1].line_id

    status = _find_status(ordered, trusted_only=True) or _find_status(ordered, trusted_only=None)
    if status:
        facts["status"], facts["evidence"]["status"] = status[0], status[1].line_id

    window = _find_window(ordered, trusted_only=True) or _find_window(ordered, trusted_only=None)
    if window:
        facts["deliveryWindow"], facts["evidence"]["deliveryWindow"] = window[0], window[1].line_id

    team = _first_label(trusted, TEAM_KEYS) or _first_label(ordered, TEAM_KEYS)
    if team:
        facts["team"], facts["evidence"]["team"] = _clean_value(team[0], 60), team[1].line_id

    # Addresses -------------------------------------------------------------
    sender_label = _first_label(untrusted, SENDER_KEYS) or _first_label(ordered, SENDER_KEYS)
    if sender_label:
        match = EMAIL_RE.search(sender_label[0])
        if match:
            facts["claimedSender"] = match.group(0)
            facts["evidence"]["claimedSender"] = sender_label[1].line_id
    if "claimedSender" not in facts:
        for line in untrusted:
            match = EMAIL_RE.search(line.text)
            if match:
                facts["claimedSender"] = match.group(0)
                facts["evidence"]["claimedSender"] = line.line_id
                break

    contact_keys = (
        "customeremail",
        "customer_email",
        "customercontact",
        "customer_contact",
        "contactemail",
        "contact_email",
        "verifiedcontact",
        "verified_contact",
        "accountemail",
        "account_email",
    )
    contact = _first_label(trusted, contact_keys)
    if contact:
        match = EMAIL_RE.search(contact[0])
        if match:
            facts["customerContact"] = match.group(0)
            facts["evidence"]["customerContact"] = contact[1].line_id

    all_emails: list[tuple[str, Line]] = []
    for line in ordered:
        for match in EMAIL_RE.finditer(line.text):
            if all_emails and all_emails[-1][0] == match.group(0):
                continue
            all_emails.append((match.group(0), line))
    facts["emails"] = all_emails

    return facts


def all_line_ids(dossier: Dossier) -> set[str]:
    return {line.line_id for line in dossier.lines if line.line_id}
