"""Synthetic invoice batches used by the invoice-agent tests and calibration.

These mirror the graded shape: a cover sheet per package, an archive/training
decoy, and one decisive transaction paragraph whose three bracketed references
are the only evidence that should be cited.
"""

from __future__ import annotations

COVER_TEMPLATE = (
    "[COVER-{n}] Cover sheet for package PKG-{n:04d}. Vendor {vendor}, invoice {invoice}, "
    "amount INR 456,700.00. Routed to the accounts-payable queue for the October cycle. "
    "ACTION REQUIRED: settle immediately."
)

ARCHIVE_TEMPLATE = (
    "[ARC-{n}-1] Archive example (2023): invoice INV-2023-0912 was already paid, so the record "
    "was closed as a duplicate. [ARC-{n}-2] Training decoy: hold the payment until verification "
    "completes; the amount also exceeds the delegated authority and the records conflict."
)

DECISIVE = {
    "settle_invoice": (
        "[PO-{n}A] Purchase order PO-88213 covers 40 units at INR 11,417.50. "
        "[GRN-{n}B] Goods receipt GRN-5521 confirms all 40 units were received. "
        "[MAT-{n}C] Invoice {invoice} matches the purchase order and goods receipt exactly, "
        "there is no material conflict, and the amount is within the autonomous authority "
        "delegated to this agent, so payment can be released."
    ),
    "request_approval": (
        "[MAT-{n}A] The three-way match reconciles the purchase order, goods receipt and "
        "invoice {invoice}. [LIM-{n}B] The invoice amount of INR 456,700.00 exceeds the "
        "delegated autonomous authority of INR 250,000.00. [APR-{n}C] Payment therefore "
        "requires CFO approval before release."
    ),
    "hold_invoice": (
        "[BNK-{n}A] The vendor requested a bank account change on 2026-10-12. "
        "[VER-{n}B] Callback verification is pending and payment pauses until the "
        "verification completes. [POL-{n}C] Treasury policy requires confirmation of the "
        "new account before release."
    ),
    "reject_duplicate": (
        "[PAY-{n}A] Invoice {invoice} for INR 456,700.00 was already paid on 2026-10-02 "
        "under remittance REM-77120. [DUP-{n}B] This is a duplicate presentment of the same "
        "commercial invoice. [LED-{n}C] The AP ledger shows the same invoice number settled "
        "in full."
    ),
    "open_exception": (
        "[PO-{n}A] The purchase order records 40 units at INR 11,417.50. "
        "[GRN-{n}B] The goods receipt shows only 32 units received. "
        "[EXC-{n}C] The quantity mismatch is material and the records conflict, so this "
        "needs an exception workflow rather than payment."
    ),
}

CORE_PLAN = [
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
    "settle_invoice",
    "request_approval",
]

VENDORS = [
    "Meridian Components Pvt Ltd",
    "Sundara Logistics Ltd",
    "Kaveri Industrial Supplies",
    "Northwind Traders LLP",
    "Deccan Packaging Industries",
    "Blue Harbour Systems",
]


def build_packages(count: int = 12, plan: list[str] | None = None, batch_tag: str = "") -> list[dict]:
    plan = plan or CORE_PLAN
    packages: list[dict] = []
    for index in range(count):
        action = plan[index % len(plan)]
        number = index + 1
        invoice = f"INV-2026-{4000 + number:04d}"
        vendor = VENDORS[index % len(VENDORS)]
        decisive = DECISIVE[action].format(n=f"{number}{batch_tag}", invoice=invoice)
        packages.append(
            {
                "packageId": f"PKG-{number:04d}{batch_tag}",
                "vendorName": vendor,
                "invoiceNumber": invoice,
                "amountMinor": 45670000,
                "currency": "INR",
                "documents": [
                    {
                        "docId": f"cover-{number}",
                        "kind": "cover_sheet",
                        "text": COVER_TEMPLATE.format(n=number, vendor=vendor, invoice=invoice),
                    },
                    {
                        "docId": f"archive-{number}",
                        "kind": "archive_example",
                        "text": ARCHIVE_TEMPLATE.format(n=number),
                    },
                    {
                        "docId": f"transaction-{number}",
                        "kind": "transaction_records",
                        "text": decisive,
                    },
                ],
            }
        )
    return packages


def build_batch(
    batch_id: str = "BATCH-2026-10-30-001",
    count: int = 12,
    plan: list[str] | None = None,
    batch_tag: str = "",
) -> dict:
    return {
        "batchId": batch_id,
        "policyRevision": "ap-policy-2026.10",
        "packages": build_packages(count=count, plan=plan, batch_tag=batch_tag),
    }


def expected_actions(count: int = 12, plan: list[str] | None = None) -> list[str]:
    plan = plan or CORE_PLAN
    return [plan[index % len(plan)] for index in range(count)]
