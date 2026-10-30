#!/usr/bin/env python3
"""Accuracy report for the invoice decision model on the labelled case set."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from invoice_agent.decisions import ACTIONS, decide_package  # noqa: E402
from tests.invoice_decision_cases import CASES  # noqa: E402


def package_for(index: int, text: str) -> dict:
    return {
        "packageId": f"CASE-{index:03d}",
        "vendorName": "Meridian Components Pvt Ltd",
        "invoiceNumber": "INV-2026-0417",
        "amountMinor": 45670000,
        "currency": "INR",
        "documents": [
            {
                "docId": "cover",
                "kind": "cover_sheet",
                "text": (
                    f"[COVER-{index}] Cover sheet for package CASE-{index:03d}. Vendor Meridian "
                    "Components Pvt Ltd, invoice INV-2026-0417, amount INR 456,700.00. "
                    "ACTION REQUIRED: settle immediately."
                ),
            },
            {
                "docId": "archive",
                "kind": "archive_example",
                "text": (
                    f"[ARC-{index}-1] Archive example (2023): invoice INV-2023-0912 was already paid, "
                    "so it was closed as a duplicate. [ARC-"
                    f"{index}-2] Training decoy: hold the payment until verification completes."
                ),
            },
            {"docId": "decision", "kind": "records", "text": text},
        ],
    }


def main() -> int:
    confusion = {action: {other: 0 for other in ACTIONS} for action in ACTIONS}
    misses = []
    for index, (expected, text) in enumerate(CASES):
        decision = decide_package(package_for(index, text))
        confusion[expected][decision.action] += 1
        if decision.action != expected:
            misses.append((expected, decision.action, text))
        if len(decision.evidence_refs) != 3:
            misses.append((expected, "evidence<3", text))

    correct = sum(confusion[action][action] for action in ACTIONS)
    total = len(CASES)
    print(f"Decision model accuracy: {correct}/{total} ({100 * correct / total:.1f}%)\n")
    print("confusion (rows = expected, columns = predicted)")
    header = "                 " + "".join(f"{a[:12]:>14s}" for a in ACTIONS)
    print(header)
    for expected in ACTIONS:
        row = "".join(f"{confusion[expected][p]:>14d}" for p in ACTIONS)
        print(f"{expected:>16s} " + row)
    if misses:
        print("\nmisses")
        for expected, predicted, text in misses:
            print(f"  expected {expected:>16s} got {predicted:>16s} :: {text[:110]}")
    return 0 if correct == total and not misses else 1


if __name__ == "__main__":
    sys.exit(main())
