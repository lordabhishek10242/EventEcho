"""Evidence-scored invoice decision model.

For every package the model must choose exactly one of:

    settle_invoice | request_approval | hold_invoice | reject_duplicate |
    open_exception

Design notes
------------
* Documents mix decisive facts with cover sheets, archive examples, training
  decoys, negations and action words used for something else.  Paragraphs are
  therefore *scored*, and the single paragraph that decides an action wins.
* Negation is handled twice: an explicit negative lexicon (``no material
  conflict``, ``payment was for a different invoice``) and a same-sentence
  negator window that flips a positive cue.
* Structured hints (``alreadyPaid: true``) and numeric limit comparisons are
  used as a safety net when a package also ships machine-readable fields.
* The three decisive bracketed references are taken from the decisive
  paragraph, never from the cover sheet, archive examples or training decoys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .documents import (
    CURRENCY_TOKEN,
    Paragraph,
    extract_facts,
    extract_paragraphs,
    parse_minor,
    sentences,
)

ACTIONS = (
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
)
SPECIFIC_ACTIONS = ("reject_duplicate", "open_exception", "hold_invoice", "request_approval")

# --------------------------------------------------------------------------- #
# Cue lexicons
# --------------------------------------------------------------------------- #

POSITIVE_CUES: dict[str, list[tuple[str, float]]] = {
    "reject_duplicate": [
        (r"\balready\s+(?:been\s+)?(?:paid|settled|released|remitted|processed)\b", 5.0),
        (r"\bduplicate\s+(?:payment|presentment|submission|claim|invoice|request)\b", 5.0),
        (r"\b(?:second|repeat|double)[\s-]?(?:payment|presentment|submission|claim)\b", 5.0),
        (r"\bdouble[\s-]?paid\b", 5.5),
        (r"\bpaid\s+twice\b", 5.0),
        (r"\bsame\s+(?:commercial\s+)?invoice\b", 3.5),
        (r"\bsame\s+invoice\s+number\b", 4.0),
        (r"\bpreviously\s+(?:paid|settled|remitted)\b", 5.0),
        (r"\b(?:this|the)\s+invoice\s+(?:was|is|has\s+been)\s+(?:already\s+)?paid\b", 5.0),
        (r"\bwas\s+paid\b", 3.5),
        (r"\bhad\s+been\s+paid\b", 4.5),
        (r"\bpayment\s+(?:was|has\s+been|is)\s+(?:recorded|released|remitted|made|processed)\b", 3.5),
        (r"\bremittance\s+(?:was\s+)?(?:made|released|sent|processed)\b", 3.5),
        (r"\bpaid\s+(?:in\s+full\s+)?on\s+\d", 3.5),
        (r"\bsettled\s+(?:on|in)\b", 2.5),
        (r"\bduplicate\s+of\b", 4.5),
        (r"\bsubmitted\s+(?:twice|again)\b", 4.5),
        (r"\b(?:payment|remittance)\s+reference\b", 1.5),
        (r"\bwas\s+remitted\b", 3.5),
        (r"\bpayment\s+(?:has\s+|was\s+)?already\s+(?:been\s+)?(?:made|released|processed)\b", 5.0),
        (r"\bpreviously\s+settled\b", 5.0),
        (r"\bsecond\s+presentment\b", 5.0),
        (r"\bidentical\s+(?:invoice|claim|package)\b", 3.5),
        (r"\bre-?submitted\b", 4.0),
        (r"\bpaid\s+(?:under|by|via)\s+(?:remittance|utr|neft|rtgs|wire|ach|sepa|upi)\b", 4.0),
        (r"\bpresented\s+(?:again|twice|a\s+second\s+time)\b", 4.5),
        (r"\bdouble\s+(?:payment|disbursement)\b", 5.0),
    ],
    "open_exception": [
        (r"\bmaterial\s+(?:records?|documents?|data)\s+conflict\b", 5.5),
        (r"\b(?:records?|documents?|data)\s+conflict\b", 5.0),
        (r"\bconflicting\s+(?:records?|documents?|quantit|amount|prices?|values?)\b", 4.5),
        (r"\bconflict(?:s|ing)?\b", 3.0),
        (r"\bmismatch(?:es|ed)?\b", 4.0),
        (r"\bdiscrepanc(?:y|ies)\b", 4.0),
        (r"\b(?:does|do|did)\s+not\s+(?:match|agree|tie|reconcile)\b", 4.5),
        (r"\b(?:doesn'?t|don'?t|didn'?t)\s+(?:match|agree|tie)\b", 4.5),
        (r"\bdiffers?\s+from\s+(?:the\s+)?(?:purchase\s+order|po|grn|goods\s+receipt|receipt|invoice)\b", 5.0),
        (r"\b(?:quantity|price|rate|unit\s+price|amount)\s+(?:variance|difference)\b", 3.5),
        (r"\bexception\s+(?:workflow|queue|handling|process|case|review)\b", 5.0),
        (r"\bunresolved\s+(?:conflict|discrepancy|mismatch|variance|difference)\b", 5.0),
        (r"\b(?:cannot|can\s?not)\s+be\s+reconciled\b", 5.0),
        (r"\b(?:three[\s-]way\s+)?match\s+fails?\b", 5.0),
        (r"\bfails?\s+(?:the\s+)?(?:three[\s-]way\s+)?match\b", 4.5),
        (r"\bfails?\s+reconciliation\b", 4.5),
        (r"\bover[\s-]?receipt\b", 3.0),
        (r"\bexceeds?\s+(?:the\s+)?(?:purchase\s+order|po)\s+(?:quantity|value|amount)\b", 3.5),
        (r"\b(?:unmatched|unreconciled)\b", 3.5),
        (r"\btwo\s+records?\s+(?:disagree|conflict)\b", 4.5),
        (r"\brecords?\s+(?:do\s+not|don'?t)\s+(?:agree|match|tie)\b", 4.5),
        (r"\bcannot\s+be\s+matched\b", 4.0),
        (r"\bexception\s+(?:required|needed|raised)\b", 4.5),
        (r"\bescalat\w+\s+(?:to\s+)?(?:exception|dispute)\b", 4.0),
        (r"\bdisputed\s+(?:quantity|amount|receipt)\b", 3.5),
        (r"\bshort\s+(?:supply|receipt|delivery)\b", 3.0),
        (r"\bpartial\s+(?:receipt|delivery)\b", 3.0),
        (r"\bquanti(?:ty|ties)\s+(?:received|recorded)\s+(?:differs|does\s+not|is\s+short|falls\s+short)\b", 4.0),
        (r"\bprices?\s+(?:on|in)\s+the\s+(?:invoice|po|purchase\s+order)\s+(?:differ|disagree)\b", 4.0),
    ],
    "hold_invoice": [
        (r"\bpending\s+(?:bank\s+|vendor\s+|callback\s+)?(?:verification|confirmation|validation|callback)\b", 5.0),
        (r"\bverification\s+(?:is\s+)?(?:pending|outstanding|incomplete|not\s+complete|unfinished|open)\b", 5.0),
        (r"\bawait(?:s|ing)?\s+(?:the\s+|a\s+)?(?:bank\s+|vendor\s+|callback\s+)?(?:verification|confirmation|callback|response|clearance)\b", 4.5),
        (r"\buntil\s+(?:the\s+)?(?:(?:bank|vendor|callback|treasury|beneficiary)\s+)?verification\s+(?:completes|is\s+complete|finishes|clears)\b", 5.0),
        (r"\bhold\s+until\b", 4.0),
        (r"\buntil\s+(?:the\s+)?(?:bank|vendor|callback|treasury)?\s*(?:confirmation|clearance)\s+(?:arrives|is\s+received|completes)\b", 4.5),
        (r"\bon\s+hold\b", 4.0),
        (r"\bhold\s+(?:the\s+)?(?:payment|invoice|remittance)\b", 4.0),
        (r"\bpayment\s+(?:is\s+)?(?:paused|suspended|stopped|withheld|parked)\b", 4.5),
        (r"\bbank\s+(?:account\s+|detail\s*)?(?:change|update|change\s+request)\b", 4.0),
        (r"\bcallback\s+verification\b", 5.0),
        (r"\bnot\s+(?:yet\s+)?verified\b", 4.5),
        (r"\bverification\s+(?:callback|step|gate|call)\b", 4.0),
        (r"\bawaiting\s+(?:confirmation|clearance|bank\s+confirmation)\b", 3.5),
        (r"\bpayment\s+pauses?\b", 4.5),
        (r"\bwait\s+for\s+(?:the\s+)?(?:verification|confirmation)\b", 4.0),
        (r"\bheld?\s+(?:pending|until|awaiting)\b", 4.5),
        (r"\bpayment\s+(?:is\s+)?blocked\b", 4.0),
        (r"\b(?:bank|beneficiary|account)\s+(?:details?\s+)?(?:changed|change|updated|update)\b", 4.0),
        (r"\bverify\s+(?:the\s+)?(?:bank\s+account|beneficiary|vendor\s+account)\b", 4.0),
        (r"\b(?:know[-\s]your[-\s]vendor|kyc)\s+verification\b", 3.5),
        (r"\breleased?\s+only\s+(?:after|once)\b", 3.5),
        (r"\bonly\s+(?:after|once)\s+(?:the\s+)?(?:verification|confirmation|callback)\b", 4.5),
        (r"\bawaiting\s+(?:bank|vendor)\s+(?:confirmation|verification|response)\b", 4.5),
        (r"\bconfirmation\s+(?:is\s+)?(?:pending|outstanding)\b", 4.5),
    ],
    "request_approval": [
        (r"\bexceeds?\s+(?:the\s+)?(?:delegated|autonomous|approval|spend|authority|authoriz|authoris)\b", 5.0),
        (r"\boutside\s+(?:the\s+)?(?:delegated|autonomous)\s+authority\b", 5.0),
        (r"\boutside\s+(?:the\s+|our\s+|my\s+|its\s+|this\s+)?(?:agent'?s\s+|approver'?s\s+)?(?:delegated|autonomous|approval)\s+(?:authority|limit|remit|threshold)\b", 5.0),
        (r"\boutside\s+(?:the|our|my|this)\s+(?:agent'?s\s+)?(?:authority|limit)\b", 4.5),
        (r"\b(?:above|beyond|over)\s+(?:this|the|our|my)\s+(?:agent'?s\s+|approver'?s\s+)?(?:limit|authority|threshold|remit|ceiling)\b", 5.0),
        (r"\brests?\s+(?:above|outside|beyond)\b", 3.5),
        (r"\b(?:above|over|beyond)\s+(?:the\s+)?(?:autonomous|delegated|approval|spend)\s+(?:limit|threshold|authority|ceiling)\b", 5.0),
        (r"\brequires?\s+(?:cfo|chief\s+financial|board|treasury|committee|manager|director|human|second|manual|additional|executive|controller)\s+approval\b", 5.0),
        (r"\bapproval\s+(?:is\s+)?(?:required|needed|necessary)\b", 4.0),
        (r"\bnot\s+within\s+(?:the\s+)?(?:autonomous|delegated)\b", 5.0),
        (r"\b(?:un|non)[\s-]?delegated\b", 3.0),
        (r"\bauthority\s+limit\b", 3.0),
        (r"\bdelegated\s+authority\b", 2.5),
        (r"\bapproval\s+(?:workflow|gate|queue|matrix)\b", 3.0),
        (r"\bsend\s+(?:it\s+|this\s+)?(?:for|to)\s+approval\b", 4.0),
        (r"\bescalat\w+\s+(?:for|to)\s+(?:cfo|approval|review)\b", 3.5),
        (r"\bover\s+(?:the\s+)?(?:autonomous|delegated)\s+(?:limit|threshold)\b", 4.5),
        (r"\bexceeds?\s+(?:the\s+)?(?:limit|threshold|ceiling)\b", 4.5),
        (r"\b(?:above|over|beyond)\s+(?:the\s+)?(?:INR|USD|EUR|GBP|SGD|AED|[\u20b9$\u20ac\u00a3])\s*[0-9]", 4.5),
        (r"\bneeds?\s+(?:cfo|finance|manager|director|committee|controller)\s+(?:sign[-\s]?off|approval)\b", 4.5),
        (r"\bsign[-\s]?off\s+(?:is\s+)?(?:required|needed)\b", 4.0),
        (r"\bbeyond\s+(?:my|our|its|the\s+agent'?s)\s+(?:authority|limit|remit)\b", 4.5),
        (r"\bhuman\s+(?:approval|review|sign[-\s]?off)\b", 3.5),
        (r"\brequires?\s+(?:higher|another|second[-\s]level)\s+authority\b", 4.0),
        (r"\bsits?\s+(?:above|outside)\s+(?:the\s+)?(?:autonomous|delegated)\b", 4.5),
    ],
    "settle_invoice": [
        (r"\bwithin\s+(?:the\s+)?(?:autonomous|delegated)\s+(?:authority|limit|threshold|authoriz|authoris)\b", 5.0),
        (r"\bwithin\s+(?:the\s+)?(?:autonomous|delegated)\b", 4.0),
        (r"\bwithin\s+(?:our|my|its)\s+(?:autonomous\s+)?authority\b", 4.0),
        (r"\bfull(?:y)?\s+(?:three[\s-]way\s+)?match(?:ed)?\b", 4.0),
        (r"\bthree[\s-]way\s+match\s+(?:is\s+)?(?:clean|successful|complete|satisfied|agrees)\b", 4.5),
        (r"\breconcil(?:e|es|ed|iation|es)\b", 3.0),
        (r"\bclean\s+(?:match|reconciliation)\b", 4.0),
        (r"\ball\s+(?:three\s+)?(?:records|documents)\s+(?:agree|match|reconcile)\b", 5.0),
        (r"\bapproved\s+for\s+payment\b", 4.0),
        (r"\bclear(?:ed)?\s+for\s+payment\b", 4.0),
        (r"\bauto[\s-]?(?:settle|approve|pay|post|release)\b", 4.0),
        (r"\bno\s+(?:material\s+)?(?:conflict|discrepancy|mismatch|exception|variance)\b", 3.5),
        (r"\b(?:does\s+not|doesn'?t)\s+exceed\b", 3.5),
        (r"\bno\s+approval\s+(?:is\s+)?(?:required|needed)\b", 3.5),
        (r"\bmatches?\s+(?:the\s+)?(?:purchase\s+order|po|grn|goods\s+receipt|receipt|invoice)\b", 3.5),
        (r"\bpayment\s+(?:can|may|should)\s+(?:now\s+)?be\s+(?:released|processed|settled|paid)\b", 3.5),
        (r"\bvalid\s+and\s+(?:reconciled|matched|approved)\b", 4.0),
        (r"\bready\s+(?:for|to)\s+(?:payment|settle|settlement|release)\b", 3.5),
        (r"\bapproved\s+and\s+(?:matched|reconciled)\b", 4.0),
        (r"\bwithin\s+(?:the\s+)?(?:agent'?s\s+)?authority\b", 4.0),
        (r"\bno\s+reason\s+to\s+(?:hold|withhold)\b", 4.0),
        (r"\bwithin\s+(?:the\s+)?(?:delegated\s+)?(?:limit|ceiling)\b", 3.0),
        (r"\bverified\s+against\s+(?:the\s+)?(?:purchase\s+order|po|grn|goods\s+receipt)\b", 3.5),
        (r"\bthree[-\s]way\s+match\s+(?:was\s+)?performed\b", 3.0),
        (r"\bno\s+(?:open\s+)?(?:issue|blocker|dispute)\b", 2.0),
    ],
}

# Explicit negations.  They subtract from the action they are arguing against.
NEGATIVE_CUES: dict[str, list[tuple[str, float]]] = {
    "reject_duplicate": [
        (r"\bnot\s+(?:a\s+|the\s+)?duplicate\b", 6.0),
        (r"\bno\s+(?:duplicate|prior|previous)\s+(?:payment|remittance|invoice)?", 5.0),
        (r"\bdifferent\s+(?:commercial\s+)?invoice\b", 5.0),
        (r"\b(?:was|is|has)\s+not\s+(?:been\s+)?(?:paid|settled|remitted)\b", 5.0),
        (r"\bnever\s+(?:been\s+)?paid\b", 5.0),
        (r"\bunpaid\b", 4.5),
        (r"\bremains?\s+(?:unpaid|outstanding)\b", 4.0),
        (r"\bdistinct\s+invoice\b", 5.0),
        (r"\bpayment\s+(?:was\s+|is\s+)?for\s+(?:a\s+)?different\s+invoice\b", 6.5),
        (r"\bno\s+(?:payment|remittance)\s+(?:has\s+been|was|is)\s+(?:made|recorded|released)\b", 5.0),
        (r"\bnot\s+(?:been\s+)?previously\s+paid\b", 5.5),
        (r"\bnever\s+settled\b", 5.0),
    ],
    "open_exception": [
        (r"\bno\s+(?:material\s+)?(?:conflicts?|discrepanc(?:y|ies)|mismatch(?:es)?|exceptions?|variances?)\b", 5.5),
        (r"\b(?:records|documents|data)\s+(?:fully\s+)?(?:agree|match|reconcile)\b", 4.0),
        (r"\bfully\s+reconcil", 4.0),
        (r"\bwithin\s+(?:the\s+)?(?:agreed\s+|stated\s+)?tolerance\b", 4.0),
        (r"\bno\s+open\s+(?:conflict|exception|variance|discrepancy)\b", 4.5),
        (r"\b(?:conflict|mismatch|discrepancy)\s+(?:was|is|has\s+been)\s+(?:resolved|cleared|closed)\b", 4.0),
        (r"\bno\s+material\s+difference\b", 4.0),
    ],
    "hold_invoice": [
        (r"\bverification\s+(?:has\s+been\s+|was\s+|is\s+)?(?:completed|cleared|closed|satisfied|done)\b", 5.0),
        (r"\bverified\s+(?:successfully|in\s+full|today|completely)\b", 4.0),
        (r"\bno\s+(?:outstanding|pending)\s+(?:verification|confirmation|callback)\b", 5.0),
        (r"\bcallback\s+(?:was\s+)?(?:completed|cleared|done)\b", 4.0),
        (r"\bverification\s+(?:is\s+)?complete\b", 5.0),
        (r"\bbank\s+details?\s+(?:are\s+)?(?:confirmed|verified|unchanged|correct)\b", 4.5),
        (r"\bnot\s+pending\b", 4.0),
        (r"\bno\s+hold\b", 5.0),
    ],
    "request_approval": [
        (r"\bwithin\s+(?:the\s+)?(?:autonomous|delegated)\b", 4.0),
        (r"\b(?:does\s+not|doesn'?t)\s+(?:exceed|require)\b", 4.0),
        (r"\bno\s+approval\s+(?:is\s+)?(?:required|needed|necessary)\b", 4.0),
        (r"\bbelow\s+(?:the\s+)?(?:autonomous|delegated|approval)\b", 4.0),
        (r"\bunder\s+(?:the\s+)?(?:autonomous|delegated|approval)\s+(?:limit|threshold)\b", 4.0),
        (r"\bapproval\s+(?:is\s+)?not\s+required\b", 4.0),
        (r"\bpre[\s-]?approved\b", 2.0),
    ],
    "settle_invoice": [
        (r"\b(?:cannot|can\s?not)\s+be\s+(?:settled|paid|reconciled|released)\b", 4.0),
        (r"\b(?:not|does\s+not|doesn'?t|cannot)\s+reconcil", 4.0),
        (r"\bunmatched\b", 4.0),
        (r"\bnot\s+matched\b", 4.0),
        (r"\bmust\s+not\s+be\s+(?:paid|settled|released)\b", 4.5),
    ],
}

_NEGATOR = re.compile(
    r"\b(not|no|never|without|nor|none|zero|neither|isn'?t|aren'?t|wasn'?t|weren'?t|"
    r"doesn'?t|don'?t|didn'?t|cannot|can'?t|hasn'?t|haven'?t|hadn'?t)\b",
    re.I,
)

_COMPILED_POS = {
    action: [(re.compile(pat, re.I), weight) for pat, weight in cues]
    for action, cues in POSITIVE_CUES.items()
}
_COMPILED_NEG = {
    action: [(re.compile(pat, re.I), weight) for pat, weight in cues]
    for action, cues in NEGATIVE_CUES.items()
}

_AMOUNT_TOKEN = re.compile(rf"(?:{CURRENCY_TOKEN})\s*[0-9][0-9,]*(?:\.[0-9]{{1,2}})?", re.I)
_CURRENCY_RE = re.compile(r"\b(INR|USD|EUR|GBP|SGD|AED|JPY)\b|([₹$€£¥])", re.I)
_LIMIT_RE = re.compile(
    rf"(?:limit|threshold|ceiling|authority)\D{{0,40}}?({CURRENCY_TOKEN})\s*([0-9][0-9,]*(?:\.[0-9]{{1,2}})?)",
    re.I,
)


def _blank_scores() -> dict[str, float]:
    return {action: 0.0 for action in ACTIONS}


_CONTRAST = (" but ", " however", " yet ", " although", " nevertheless", " whereas", " nonetheless",
             " though ", " instead ")


def _sentence_bounds(text: str, pos: int) -> tuple[int, int]:
    start = max(
        text.rfind(".", 0, pos),
        text.rfind(";", 0, pos),
        text.rfind("!", 0, pos),
        text.rfind("?", 0, pos),
    )
    ends = [text.find(mark, pos) for mark in (".", ";", "!", "?")]
    ends = [value for value in ends if value != -1]
    end = min(ends) if ends else len(text)
    return start + 1, end


def _contrast_after(text: str, pos: int) -> bool:
    """A ``but``/``however`` after a negation reverses what it negated."""
    _start, end = _sentence_bounds(text, pos)
    tail = text[pos:end]
    # the contrast must still be before the next reference marker
    bracket = tail.find("[")
    if bracket != -1:
        tail = tail[:bracket]
    return any(marker.strip() in tail for marker in _CONTRAST)


def _negated_same_sentence(text: str, start: int) -> bool:
    """True when a negator sits shortly before the cue in the same sentence."""
    sentence_start, _end = _sentence_bounds(text, start)
    contrast = max((text.rfind(marker, sentence_start, start) for marker in _CONTRAST), default=-1)
    window_start = max(sentence_start, start - 60, contrast + 1 if contrast >= 0 else sentence_start)
    window = text[window_start:start]
    return bool(_NEGATOR.search(window))


def score_text(text: str) -> dict[str, float]:
    """Score one paragraph against the five actions."""
    scores = _blank_scores()
    lower = text.lower()
    for action, cues in _COMPILED_POS.items():
        for pattern, weight in cues:
            for match in pattern.finditer(lower):
                if _negated_same_sentence(lower, match.start()):
                    scores[action] -= weight * 0.55
                else:
                    scores[action] += weight
    for action, cues in _COMPILED_NEG.items():
        for pattern, weight in cues:
            for match in pattern.finditer(lower):
                # "not a duplicate, but this repeats the same invoice" — a
                # contrast clause reverses the negation it follows.
                factor = 0.45 if _contrast_after(lower, match.end()) else 1.0
                scores[action] -= weight * factor
    return scores


# --------------------------------------------------------------------------- #
# Structured hints
# --------------------------------------------------------------------------- #

_TRUE = {"true", "yes", "1", "y", "paid", "duplicate", "settled", "pending", "required", "open", "material", "conflict", "blocked"}
_FALSE = {"false", "no", "0", "n", "unpaid", "notpaid", "none", "cleared", "completed", "approved", "resolved", "closed", "notrequired", "verified"}


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _walk_scalars(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, dict):
                yield from _walk_scalars(child)
            elif isinstance(child, list):
                for item in child:
                    if not isinstance(item, (dict, list)):
                        yield str(key), item
                    else:
                        yield from _walk_scalars(item)
            else:
                yield str(key), child
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield from _walk_scalars(item)


KEY_GROUPS: dict[str, tuple[str, ...]] = {
    "duplicate": ("alreadypaid", "duplicate", "paids", "paidstatus", "paymentstatus", "duplicateof", "priorpayment", "previouslypaid", "settledstatus"),
    "hold": ("verification", "verificationstatus", "verificationpending", "bankverification", "callback", "callbackverification", "hold", "holdstatus", "paymenthold", "onhold"),
    "approval": ("approval", "approvalrequired", "approvalstatus", "requiresapproval", "withinauthority", "autonomousauthority", "authoritystatus", "withindelegatedauthority"),
    "exception": ("conflict", "recordsconflict", "conflictdetected", "mismatch", "discrepancy", "exception", "exceptionrequired", "matchstatus", "reconciliationstatus"),
    "settle": ("reconciled", "reconciliation", "matched", "matchstatus", "withinauthority", "settleable"),
}


def structured_signals(package: Any) -> dict[str, float]:
    """Packages may carry machine-readable flags; use them as a safety net."""
    signals = _blank_scores()
    for key, value in _walk_scalars(package):
        key_norm = _norm(key)
        if not key_norm or len(key_norm) > 40:
            continue
        value_norm = _norm(str(value)) if not isinstance(value, bool) else ("true" if value else "false")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            continue

        groups = [group for group, aliases in KEY_GROUPS.items() if key_norm in aliases]
        if not groups:
            continue
        if value_norm in _TRUE:
            for group in groups:
                if group == "duplicate":
                    signals["reject_duplicate"] += 6.0
                elif group == "hold":
                    signals["hold_invoice"] += 5.0
                elif group == "approval":
                    signals["request_approval"] += 5.0
                elif group == "exception":
                    signals["open_exception"] += 5.5
                elif group == "settle":
                    signals["settle_invoice"] += 3.0
        elif value_norm in _FALSE:
            for group in groups:
                if group == "duplicate":
                    signals["reject_duplicate"] -= 4.0
                    signals["settle_invoice"] += 1.0
                elif group == "hold":
                    signals["hold_invoice"] -= 4.0
                    signals["settle_invoice"] += 1.5
                elif group == "approval":
                    signals["request_approval"] -= 4.0
                    signals["settle_invoice"] += 1.5
                elif group == "exception":
                    signals["open_exception"] -= 4.0
                    signals["settle_invoice"] += 1.5
                elif group == "settle":
                    signals["settle_invoice"] += 2.0
    return signals


def numeric_limit_signal(facts: dict[str, Any], paragraphs: list[Paragraph]) -> dict[str, float]:
    """Compare the invoice amount to any stated authority limit in the documents."""
    signals = _blank_scores()
    amount = facts.get("amountMinor") or 0
    if amount <= 0:
        return signals
    for paragraph in paragraphs:
        if paragraph.is_decoy:
            continue
        for match in _LIMIT_RE.finditer(paragraph.text):
            limit_minor = parse_minor(f"{match.group(1)} {match.group(2)}")
            if not limit_minor or limit_minor <= 0:
                continue
            if amount > limit_minor * 1.0001:
                signals["request_approval"] += 4.5
                signals["settle_invoice"] -= 2.0
            else:
                signals["settle_invoice"] += 1.5
                signals["request_approval"] -= 1.0
    return signals


# --------------------------------------------------------------------------- #
# Decision
# --------------------------------------------------------------------------- #


@dataclass
class Decision:
    action: str
    facts: dict[str, Any]
    evidence_refs: list[str]
    rationale: str
    scores: dict[str, float] = field(default_factory=dict)
    decisive_ref: str | None = None
    warnings: list[str] = field(default_factory=list)


def _fact_bonus(text: str, facts: dict[str, Any]) -> float:
    bonus = 0.0
    invoice = str(facts.get("invoiceNumber") or "")
    if invoice and invoice != "UNKNOWN" and invoice.lower() in text.lower():
        bonus += 1.5
    if _AMOUNT_TOKEN.search(text):
        bonus += 1.0
    vendor = str(facts.get("vendorName") or "")
    if vendor and vendor != "Unknown vendor":
        head = vendor.split()[0].lower()
        if head and head in text.lower():
            bonus += 0.4
    return bonus


def _decisiveness(scores: dict[str, float], paragraph: Paragraph, facts: dict[str, Any]) -> float:
    specific = max(scores[action] for action in SPECIFIC_ACTIONS)
    value = specific + 0.9 * scores["settle_invoice"] + _fact_bonus(paragraph.text, facts)
    value += 0.25 * min(len(paragraph.refs), 3)
    if paragraph.is_cover:
        value -= 4.5
    if paragraph.is_decoy:
        value -= 6.0
    return value


def _choose_action(total: dict[str, float], has_assertive_score: bool) -> str:
    specific = max(SPECIFIC_ACTIONS, key=lambda a: total[a])
    specific_score = total[specific]
    settle_score = total["settle_invoice"]

    if specific_score >= 2.0 and specific_score >= settle_score * 0.7:
        return specific
    if settle_score >= 2.0:
        return "settle_invoice"
    if specific_score > 0.5:
        return specific
    if settle_score > 0.0:
        return "settle_invoice"
    # Nothing survives negation: an unresolved package needs an exception
    # workflow rather than an unauthorised payment.
    return "open_exception"


def _select_evidence(
    decisive: Paragraph,
    action: str,
    facts: dict[str, Any],
    paragraphs: list[Paragraph],
) -> list[str]:
    """Pick exactly three decisive references, all from the decisive paragraph
    when possible, in document order."""

    spans = [(m.start(), m.end(), m.group(1).strip()) for m in re.finditer(r"\[([^\[\]\n]{1,80})\]", decisive.text)]
    scored: list[tuple[float, int, str]] = []
    for pos, (start, end, ref) in enumerate(spans):
        prev_end = spans[pos - 1][1] if pos else 0
        next_start = spans[pos + 1][0] if pos + 1 < len(spans) else len(decisive.text)
        segment = decisive.text[prev_end:next_start]
        segment_scores = score_text(segment)
        value = segment_scores[action] + 0.35 * max(segment_scores[a] for a in SPECIFIC_ACTIONS)
        value += 0.5 * segment_scores["settle_invoice"]
        if action == "settle_invoice":
            value += 0.8 * segment_scores["settle_invoice"]
        value += 0.05 * max(0, 120 - len(segment)) / 120.0
        scored.append((value, pos, ref))

    chosen = [ref for _v, _p, ref in sorted(scored, key=lambda item: (-item[0], item[1]))[:3]]
    if len(chosen) < 3:
        fallback: list[tuple[float, int, str]] = []
        for paragraph in paragraphs:
            if paragraph is decisive or paragraph.is_cover or paragraph.is_decoy:
                continue
            scores = score_text(paragraph.text)
            value = scores[action] + 0.3 * max(scores[a] for a in SPECIFIC_ACTIONS)
            for ref_pos, ref in enumerate(paragraph.refs):
                if ref in chosen:
                    continue
                fallback.append((value, paragraph.index * 100 + ref_pos, ref))
        for _v, _p, ref in sorted(fallback, key=lambda item: (-item[0], item[1])):
            if len(chosen) >= 3:
                break
            chosen.append(ref)

    order = {ref: pos for pos, (_s, _e, ref) in enumerate(spans)}
    chosen.sort(key=lambda ref: (order.get(ref, 10_000), ref))
    return chosen[:3]


RATIONALE_REASON = {
    "settle_invoice": (
        "the three-way match reconciles the purchase order, goods receipt and invoice, no material "
        "record conflicts, and the amount is inside the delegated autonomous authority, so the invoice "
        "can be paid without escalation"
    ),
    "request_approval": (
        "the invoice is commercially valid and reconciles, but the amount sits outside the delegated "
        "autonomous authority, so it may only proceed once a human approver accepts it"
    ),
    "hold_invoice": (
        "the invoice is commercially valid, but payment must pause until the stated verification "
        "completes; paying now would bypass that control"
    ),
    "reject_duplicate": (
        "the identical commercial invoice has already been paid, so executing again would double-pay "
        "the same obligation"
    ),
    "open_exception": (
        "material records conflict and cannot be reconciled inside this workflow, so the package needs "
        "an exception workflow rather than payment"
    ),
}


def _quote(paragraph: Paragraph, limit: int = 260) -> str:
    text = re.sub(r"\[[^\[\]\n]{1,80}\]", "", paragraph.text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def _rationale(action: str, facts: dict[str, Any], refs: list[str], decisive: Paragraph) -> str:
    reason = RATIONALE_REASON[action]
    facts_text = (
        f"vendor {facts['vendorName']}, invoice {facts['invoiceNumber']}, "
        f"{facts['amountMinor']} {facts['currency']} minor units"
    )
    body = f"{action}: {reason}. Decisive facts — {facts_text}."
    quote = _quote(decisive)
    if quote:
        body += f" Decisive paragraph states: \"{quote}\"."
    if refs:
        body += " Supporting evidence: " + ", ".join(refs) + "."
    if len(body) < 60:
        body += " No further evidence contradicts this reading."
    if len(body) > 1500:
        body = body[:1460].rsplit(" ", 1)[0] + "."
    return body


def _aggregate(
    paragraph_scores: list[tuple[Paragraph, dict[str, float]]],
    decisive: Paragraph,
    structured: dict[str, float],
    numeric: dict[str, float],
) -> dict[str, float]:
    total = _blank_scores()
    for paragraph, scores in paragraph_scores:
        if paragraph is decisive:
            factor = 1.0
        elif paragraph.is_cover or paragraph.is_decoy:
            factor = 0.0
        else:
            factor = 0.3
        for action in ACTIONS:
            total[action] += scores[action] * factor
    for action in ACTIONS:
        total[action] += structured[action] + numeric[action]
    return total


def decide_package(package: Any, policy_revision: str | None = None) -> Decision:
    """Decide exactly one action for one package, with evidence and rationale."""
    paragraphs = extract_paragraphs(package)
    facts = extract_facts(package, paragraphs)

    if not paragraphs:
        return Decision(
            action="open_exception",
            facts=facts,
            evidence_refs=[],
            rationale=(
                "open_exception: the package contains no readable documents, so the records cannot be "
                "reconciled and the invoice cannot be paid without an exception review."
            ),
            warnings=["no_paragraphs"],
        )

    scored = [(paragraph, score_text(paragraph.text)) for paragraph in paragraphs]
    ranked = sorted(
        scored,
        key=lambda item: (-_decisiveness(item[1], item[0], facts), item[0].index),
    )
    decisive = ranked[0][0]
    if _decisiveness(ranked[0][1], decisive, facts) < 0.5:
        # Fall back to the most reference-rich non-cover paragraph.
        candidates = [p for p in paragraphs if not p.is_cover and not p.is_decoy] or paragraphs
        decisive = max(candidates, key=lambda p: (len(p.refs), len(p.text)))

    structured = structured_signals(package)
    numeric = numeric_limit_signal(facts, paragraphs)
    total = _aggregate(scored, decisive, structured, numeric)
    action = _choose_action(total, bool(structured) or bool(numeric))

    refs = _select_evidence(decisive, action, facts, paragraphs)
    rationale = _rationale(action, facts, refs, decisive)

    return Decision(
        action=action,
        facts=facts,
        evidence_refs=refs,
        rationale=rationale,
        scores={k: round(v, 3) for k, v in total.items()},
        decisive_ref=decisive.source,
    )
