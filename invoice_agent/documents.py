"""Document flattening, bracketed-reference extraction and fact extraction.

The grader delivers each invoice package as a bundle of documents whose text
is the only source of truth.  The exact key names are not guaranteed, so the
flattener walks any JSON shape and treats every long string (or any string
carrying ``[REF]`` markers) as a paragraph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

REF_RE = re.compile(r"\[([^\[\]\n]{1,80})\]")

COVER_RE = re.compile(
    r"(cover[\s\-_]?(sheet|page|note|slip)|routing\s+slip|batch\s+cover|"
    r"package\s+cover|transmittal|intake\s+sheet)",
    re.I,
)

DECOY_RE = re.compile(
    r"(archiv\w*|old\s+example|example\s+(only|from|of)|for\s+illustration|"
    r"illustrative\s+example|worked\s+example|training\s+(example|decoy|sample)|"
    r"decoy|template\s+only|sample\s+only|do\s+not\s+(use|act|rely)|"
    r"reference\s+only|prior[\s\-]year\s+example|historical\s+example|"
    r"not\s+part\s+of\s+this\s+(package|batch)|"
    r"historical\s+(?:record|cycle|package)|prior\s+cycle|"
    r"\billustrative\b|\bworked\s+through\b|"
    r"superseded|inactive|voided\s+example)",
    re.I,
)

# Words that turn a cue into its opposite inside one sentence.
NEGATOR_RE = re.compile(
    r"\b(not|no|never|without|nor|none|zero|neither|"
    r"isn'?t|aren'?t|wasn'?t|weren'?t|doesn'?t|don'?t|didn'?t|cannot|can'?t|"
    r"hasn'?t|haven'?t|hadn'?t)\b",
    re.I,
)

_PUNCT_SPLIT = re.compile(r"(?<=[.!?;])\s+")
_WS = re.compile(r"\s+")


@dataclass
class Paragraph:
    """One indivisible unit of evidence."""

    text: str
    refs: list[str] = field(default_factory=list)
    source: str = ""
    index: int = 0

    @property
    def is_cover(self) -> bool:
        return bool(COVER_RE.search(self.text))

    @property
    def is_decoy(self) -> bool:
        return bool(DECOY_RE.search(self.text))


REF_HINT_KEYS = (
    "ref",
    "reference",
    "refid",
    "referenceid",
    "docid",
    "doc_id",
    "documentid",
    "evidenceid",
    "citation",
    "evidence_ref",
    "evidenceref",
)
TEXT_KEYS = (
    "text",
    "content",
    "body",
    "paragraph",
    "statement",
    "narrative",
    "page",
    "document",
    "description",
    "summary",
)


def _scalar(value: Any, path: str, hint: str | None = None) -> Iterable[tuple[str, Any, str | None]]:
    """Walk any JSON shape, carrying a document reference hint with each string."""
    if isinstance(value, dict):
        own_hint = hint
        for key in REF_HINT_KEYS:
            for actual_key in value:
                if _norm_key(str(actual_key)) == _norm_key(key):
                    candidate = value[actual_key]
                    if isinstance(candidate, (str, int)) and 0 < len(str(candidate)) <= 80:
                        own_hint = str(candidate)
                    break
            if own_hint is not hint:
                break
        for key, child in value.items():
            key_norm = _norm_key(str(key))
            carries_text = key_norm in {_norm_key(k) for k in TEXT_KEYS}
            child_hint = own_hint if carries_text else hint
            yield from _scalar(child, f"{path}.{key}" if path else str(key), child_hint)
    elif isinstance(value, (list, tuple)):
        for pos, child in enumerate(value):
            yield from _scalar(child, f"{path}[{pos}]", hint)
    else:
        yield path, value, hint


def _iter_strings(value: Any, path: str = "") -> Iterable[tuple[str, str]]:
    for child_path, child, _hint in _scalar(value, path):
        if isinstance(child, str):
            yield child_path, child


def _iter_scalars(value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    for child_path, child, _hint in _scalar(value, path):
        yield child_path, child


def _split_blocks(text: str) -> list[str]:
    """Split a document body into paragraphs without breaking inline refs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks: list[str] = []
    for chunk in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in chunk.split("\n") if ln.strip()]
        if len(lines) > 1 and sum(1 for ln in lines if REF_RE.search(ln)) >= 2:
            blocks.extend(lines)
        elif lines:
            blocks.append(" ".join(lines))
    return blocks


def extract_paragraphs(package: Any) -> list[Paragraph]:
    """Flatten an arbitrarily-shaped package into ordered paragraphs."""
    paragraphs: list[Paragraph] = []
    seen: set[str] = set()
    for source, text, hint in _scalar(package, ""):
        if not isinstance(text, str):
            continue
        body = _WS.sub(" ", text).strip()
        if not body:
            continue
        has_ref = bool(REF_RE.search(body))
        if not has_ref and len(body) < 60:
            continue
        for block in _split_blocks(body):
            block = block.strip()
            if not block or block in seen:
                continue
            if not REF_RE.search(block) and len(block) < 60:
                continue
            seen.add(block)
            refs = [m.group(1).strip() for m in REF_RE.finditer(block)]
            if not refs and hint:
                # Some documents carry their reference in a sibling field
                # rather than inline brackets; use that identifier.
                refs = [hint]
            paragraphs.append(
                Paragraph(
                    text=block,
                    refs=refs,
                    source=source,
                    index=len(paragraphs),
                )
            )
    return paragraphs


def sentences(text: str) -> list[str]:
    return [s for s in _PUNCT_SPLIT.split(text) if s.strip()]


def ref_spans(paragraph: Paragraph) -> list[tuple[int, int, str]]:
    """Return ``(start, end, ref)`` for every bracketed reference."""
    return [(m.start(), m.end(), m.group(1).strip()) for m in REF_RE.finditer(paragraph.text)]


def segment_for_ref(paragraph: Paragraph, ref: str) -> str:
    """Text that surrounds a reference: the preceding clause plus the next one.

    Useful because decisive references are usually appended at the end of the
    sentence they support (``... exceeds the delegated authority [REF-1]``).
    """
    spans = ref_spans(paragraph)
    for pos, (start, end, name) in enumerate(spans):
        if name != ref:
            continue
        prev_end = spans[pos - 1][1] if pos else 0
        next_start = spans[pos + 1][0] if pos + 1 < len(spans) else len(paragraph.text)
        head = paragraph.text[prev_end:start]
        tail = paragraph.text[end:next_start]
        return f"{head} {ref} {tail}".strip()
    return ""


# --------------------------------------------------------------------------- #
# Fact extraction
# --------------------------------------------------------------------------- #

CURRENCY_ALIASES = {
    "₹": "INR",
    "rs": "INR",
    "rs.": "INR",
    "inr": "INR",
    "rupees": "INR",
    "$": "USD",
    "us$": "USD",
    "usd": "USD",
    "€": "EUR",
    "eur": "EUR",
    "£": "GBP",
    "gbp": "GBP",
    "sgd": "SGD",
    "aed": "AED",
    "jpy": "JPY",
    "¥": "JPY",
}

CURRENCY_TOKEN = r"(?:INR|USD|EUR|GBP|SGD|AED|JPY|Rs\.?|\u20b9|\$|\u20ac|\u00a3|\u00a5)"
NUMBER = r"[0-9][0-9,]*(?:\.[0-9]{1,2})?"

AMOUNT_WITH_TOKEN = re.compile(
    rf"(?<![A-Za-z])(?P<cur>{CURRENCY_TOKEN})\s*(?P<num>{NUMBER})", re.I
)
AMOUNT_REVERSED = re.compile(rf"(?P<num>{NUMBER})\s*(?P<cur>{CURRENCY_TOKEN})(?![A-Za-z])", re.I)
AMOUNT_CONTEXT = re.compile(
    rf"(?:amount|total|invoice\s+value|value|sum|balance|payable)\s*"
    rf"(?:due|of|is|:|=)?\s*(?P<cur>{CURRENCY_TOKEN})?\s*(?P<num>{NUMBER})",
    re.I,
)
INVOICE_NUMBER = re.compile(
    r"\b(?:invoice|bill|inv)\s*(?:number|no\.?|num|#|id|ref(?:erence)?)?\s*[:#\-–]?\s*"
    r"(?P<num>[A-Z0-9][A-Z0-9][A-Z0-9\-/_]{2,})",
    re.I,
)
INVOICE_TOKEN = re.compile(r"\b(INV[-/]?[A-Z0-9][A-Z0-9\-/_]{2,}|[A-Z]{2,5}-\d{3,}[A-Z0-9\-]*)\b")
VENDOR_LABEL = re.compile(
    r"\b(?:vendor|supplier|payee|billed\s+by|invoice\s+from|raised\s+by)\s*"
    r"(?:name)?\s*[:\-–=]\s*(?P<name>[A-Za-z0-9][^,.;\n\[\]]{2,70})",
    re.I,
)
COMPANY_TAIL = (
    r"Pvt\.?\s+Ltd\.?|Private\s+Limited|Ltd\.?|Limited|LLP|Inc\.?|Corporation|Corp\.?|"
    r"Industries|Technologies|Systems|Solutions|Enterprises|Services|Traders|Trading|"
    r"Supplies|Components|Logistics|Packaging|Distributors|Agencies|Group"
)
VENDOR_INLINE = re.compile(
    r"\b(?:[Vv]endor|[Ss]upplier|[Pp]ayee)\s+"
    rf"(?P<name>[A-Z][A-Za-z0-9&.'\-]*(?:\s+[A-Z][A-Za-z0-9&.'\-]*){{0,5}}"
    rf"\s+(?:{COMPANY_TAIL}))"
)
VENDOR_FROM = re.compile(
    rf"\b(?:from|issued\s+by|submitted\s+by|raised\s+by|billed\s+by)\s+"
    rf"(?P<name>[A-Z][A-Za-z0-9&.'\-]*(?:\s+[A-Z][A-Za-z0-9&.'\-]*){{0,5}}"
    rf"\s+(?:{COMPANY_TAIL}))"
)
COMPANY_SUFFIX = re.compile(
    rf"\b(?P<name>[A-Z][A-Za-z0-9&.'\-]*(?:\s+[A-Z][A-Za-z0-9&.'\-]*){{0,5}}"
    rf"\s+(?:{COMPANY_TAIL}))\b"
)

def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def parse_minor(raw: Any, currency_hint: str | None = None) -> int | None:
    """Convert an amount written in major units (``INR 1,234.50``) to minor units."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw * 100
    if isinstance(raw, float):
        return int(round(raw * 100))
    text = str(raw).strip()
    if not text:
        return None
    cleaned = text.replace(",", "").replace("_", "").replace(" ", "")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    number = match.group(0)
    if "." in number:
        return int(round(float(number) * 100))
    return int(number) * 100


def _detect_currency(text: str) -> str | None:
    match = re.search(r"\b(INR|USD|EUR|GBP|SGD|AED|JPY)\b", text, re.I)
    if match:
        return match.group(1).upper()
    for symbol, code in (("\u20b9", "INR"), ("$", "USD"), ("\u20ac", "EUR"), ("\u00a3", "GBP"), ("\u00a5", "JPY")):
        if symbol in text:
            return code
    for alias, code in CURRENCY_ALIASES.items():
        if alias.isalpha() and len(alias) > 2 and re.search(rf"\b{re.escape(alias)}\b", text, re.I):
            return code
    return None


MINOR_AMOUNT_KEYS = {
    "amountminor",
    "amountinminor",
    "amountminorunits",
    "totalminor",
    "minoramount",
    "amountminorvalue",
    "valueinminor",
}
MAJOR_AMOUNT_KEYS = {
    "amount",
    "amountmajor",
    "total",
    "invoiceamount",
    "amountdue",
    "grossamount",
    "invoicevalue",
    "value",
    "totalamount",
    "billedamount",
}
VENDOR_KEYS = {
    "vendorname",
    "vendor",
    "supplier",
    "suppliername",
    "payee",
    "payeename",
    "vendorlegalname",
    "companyname",
}
INVOICE_KEYS = {
    "invoicenumber",
    "invoiceno",
    "invoicenum",
    "invoiceid",
    "invoice",
    "billnumber",
    "billno",
    "invoiceref",
    "invoicereference",
    "documentnumber",
}
CURRENCY_KEYS = {"currency", "currencycode", "ccy", "isocurrency"}


def _as_minor(value: Any) -> int | None:
    """Interpret an explicitly-minor amount without scaling it again."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    text = str(value).strip().replace(",", "")
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return int(round(float(text) * 100))
    return parse_minor(value)


def _structured_facts(package: Any) -> dict[str, Any]:
    """Facts that a package ships as explicit fields (preferred over text)."""
    found: dict[str, Any] = {}
    for path, value in _iter_scalars(package):
        parts = [part for part in re.split(r"[.\[\]]+", path) if part]
        if not parts:
            continue
        key = _norm_key(parts[-1])
        if not key:
            continue

        if key in MINOR_AMOUNT_KEYS:
            if "amountMinor" in found:
                continue
            amount = _as_minor(value)
            if amount is not None:
                found["amountMinor"] = amount
        elif key in MAJOR_AMOUNT_KEYS:
            if "amountMinor" in found or "amountMajor" in found:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                found["amountMajor"] = parse_minor(value)
            elif isinstance(value, str) and re.search(r"\d", value) and len(value) <= 40:
                found["amountMajor"] = parse_minor(value)
        elif key in VENDOR_KEYS:
            if "vendorName" not in found and isinstance(value, str):
                name = value.strip()
                if 2 <= len(name) <= 90:
                    found["vendorName"] = name
        elif key in INVOICE_KEYS:
            if "invoiceNumber" not in found and isinstance(value, str):
                text = value.strip()
                if 1 < len(text) <= 40 and re.search(r"[A-Za-z0-9]", text):
                    found["invoiceNumber"] = text
        elif key in CURRENCY_KEYS:
            if "currency" not in found and isinstance(value, str):
                mapped = CURRENCY_ALIASES.get(value.strip().lower())
                if mapped:
                    found["currency"] = mapped
    return found


def extract_facts(package: Any, paragraphs: list[Paragraph]) -> dict[str, Any]:
    """Extract vendor, invoice number, amount (minor units) and currency."""
    structured = _structured_facts(package)

    ordered = sorted(
        paragraphs,
        key=lambda p: (
            p.is_decoy,
            not p.is_cover,
        ),
    )
    texts = [p.text for p in ordered]

    vendor = structured.get("vendorName")
    invoice = structured.get("invoiceNumber")
    amount_raw = structured.get("amountMinor")
    currency = structured.get("currency")

    if not vendor:
        vendor = _find_vendor(texts)
    if not invoice:
        invoice = _find_invoice_number(texts)
    if not currency:
        currency = _detect_currency(" ".join(texts[:4])) or "INR"

    amount_minor: int | None
    if amount_raw is not None:
        amount_minor = int(amount_raw)
    elif structured.get("amountMajor") is not None:
        amount_minor = int(structured["amountMajor"])
    else:
        text_amount = _find_amount(texts, invoice)
        amount_minor = parse_minor(text_amount) if text_amount is not None else 0
    if amount_minor is None:
        amount_minor = 0

    return {
        "vendorName": (vendor or "Unknown vendor").strip(),
        "invoiceNumber": (invoice or "UNKNOWN").strip(),
        "amountMinor": int(amount_minor),
        "currency": (currency or "INR").strip().upper(),
    }


def _clean_vendor(raw: str) -> str:
    name = raw.strip(" -\u2013\u2014:,.;()[]")
    name = re.split(
        r"\s+(?:billed|submitted|issued|raised|requested|requests|asks|asked|states|stated|"
        r"noted|dated|for|with|against|invoice|invoices|per|is|was|has|have)\s+",
        name,
        maxsplit=1,
    )[0]
    name = re.sub(r"\s+(?:billed|submitted|issued|raised|requests?|states?)$", "", name)
    return name.strip(" -\u2013\u2014:,.;()[]")


def _find_vendor(texts: list[str]) -> str | None:
    for text in texts:
        match = VENDOR_LABEL.search(text)
        if match:
            name = _clean_vendor(match.group("name"))
            if len(name) >= 2:
                return name
    for text in texts:
        match = VENDOR_INLINE.search(text)
        if match:
            name = _clean_vendor(match.group("name"))
            if len(name) >= 2:
                return name
    for text in texts:
        match = VENDOR_FROM.search(text)
        if match:
            name = _clean_vendor(match.group("name"))
            if len(name) >= 2:
                return name
    for text in texts:
        match = COMPANY_SUFFIX.search(text)
        if match:
            return _clean_vendor(match.group("name"))
    return None


def _find_invoice_number(texts: list[str]) -> str | None:
    labelled = re.compile(
        r"(?:invoice|bill)\s*(?:number|no\.?|num|#|id|ref(?:erence)?)?\s*[:#\-–]?\s*"
        r"(?P<num>(?:INV[-\s]?)?[A-Z0-9][A-Z0-9\-/_]{2,})",
        re.I,
    )
    for text in texts:
        for match in labelled.finditer(text):
            candidate = match.group("num").strip()
            if re.search(r"\d", candidate) and candidate.upper() not in {"NUMBER", "NO"}:
                return candidate.upper()
    for text in texts:
        match = INVOICE_TOKEN.search(text)
        if match:
            return match.group(1).upper()
    return None


def _find_amount(texts: list[str], invoice: str | None) -> str | None:
    for text in texts:
        match = AMOUNT_CONTEXT.search(text)
        if match:
            return f"{match.group('cur') or ''} {match.group('num')}".strip()
    for text in texts:
        match = AMOUNT_WITH_TOKEN.search(text)
        if match:
            return f"{match.group('cur')} {match.group('num')}"
    for text in texts:
        match = AMOUNT_REVERSED.search(text)
        if match:
            return f"{match.group('cur')} {match.group('num')}"
    return None
