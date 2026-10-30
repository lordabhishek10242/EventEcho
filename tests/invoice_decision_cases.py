"""Labeled decision paragraphs for the invoice action model.

Each case is a single decisive paragraph taken from a synthetic invoice
package. They deliberately include negation, contrast clauses and action words
that belong to a different decision, so the model has to read the paragraph
rather than pattern-match a keyword.
"""

from __future__ import annotations

CASES: list[tuple[str, str]] = [
    # ---- settle_invoice ---------------------------------------------------
    ("settle_invoice",
     "[PO-1] The purchase order PO-88213 covers 40 units at INR 11,417.50. "
     "[GRN-2] Goods receipt GRN-5521 confirms 40 units received. "
     "[M-3] All three records agree, there is no material conflict, and the amount is within the "
     "autonomous authority, so the invoice can be paid."),
    ("settle_invoice",
     "[A-1] Three-way match performed against PO-88213 and GRN-5521. "
     "[B-2] The records fully reconcile with no discrepancy. "
     "[C-3] The value sits within the delegated autonomous limit of INR 250,000, so release payment now."),
    ("settle_invoice",
     "[X-9] Invoice INV-2026-0417 for INR 456,700.00 matches the purchase order and the goods receipt. "
     "[Y-8] No approval is required because the amount does not exceed the autonomous threshold. "
     "[Z-7] Approved and matched; auto-settle in this cycle."),
    ("settle_invoice",
     "[A-1] All three documents agree; the invoice is within the autonomous limit. [A-2] No discrepancy "
     "and no hold. [A-3] Cleared for payment."),
    ("settle_invoice",
     "[B-1] Validated: PO-1, GRN-2 and the invoice reconcile. [B-2] The amount is below the delegated "
     "threshold. [B-3] Release payment in this cycle."),
    ("settle_invoice",
     "[C-1] The package is clean: no discrepancy, no hold, no duplicate. [C-2] The invoice is fully "
     "matched. [C-3] The value is under the delegated ceiling, so auto-release is permitted."),
    ("settle_invoice",
     "[D-1] Invoice INV-2026-0417 is fully matched against PO and GRN with no exception. [D-2] "
     "INR 4,56,700.00 is within the agent's own authority. [D-3] Settle now."),
    ("settle_invoice",
     "[K2-41] An earlier payment of INR 100,000.00 was recorded, but it was for a different invoice "
     "(INV-2025-8800), not this one. [L2-42] The three-way match reconciles fully. [M2-43] The amount "
     "is within the autonomous authority, so the invoice can be settled."),
    ("settle_invoice",
     "[N2-51] Bank callback verification was completed on 2026-10-20. [O2-52] No material conflict "
     "exists between PO-88213, GRN-5521 and the invoice. [P2-53] The amount is within the delegated "
     "autonomous authority."),
    ("settle_invoice",
     "[Q2-61] There is no pending verification and no approval is required. [R2-62] The records "
     "reconcile and the amount is within the autonomous limit. [S2-63] Release payment."),
    ("settle_invoice",
     "[C1] Callback verification completed successfully yesterday and no hold remains. [C2] The records "
     "fully reconcile with the purchase order and goods receipt. [C3] The amount is within the "
     "autonomous authority."),
    ("settle_invoice",
     "[D1] The CFO pre-approved this vendor for the quarter. [D2] The three-way match is clean and the "
     "amount is within the delegated limit. [D3] Release payment."),
    ("settle_invoice",
     "[E1] No exception is needed: the GRN matches the invoice quantity exactly. [E2] There is no "
     "discrepancy and no pending verification. [E3] The invoice is within the autonomous authority, so "
     "settle it."),
    ("settle_invoice",
     "[J1] Vendor verification is complete and the bank details are confirmed. [J2] No material "
     "conflict exists between the purchase order, goods receipt and invoice. [J3] The value is within "
     "the delegated ceiling."),

    # ---- request_approval -------------------------------------------------
    ("request_approval",
     "[A-11] The three-way match reconciles the purchase order, goods receipt and invoice. "
     "[B-12] The invoice amount of INR 456,700.00 exceeds the delegated autonomous authority of "
     "INR 250,000.00. [C-13] Payment therefore requires CFO approval before release."),
    ("request_approval",
     "[D-21] The invoice is commercially valid and fully reconciled. [E-22] At INR 456,700.00 it sits "
     "above the INR 250,000 autonomous threshold. [F-23] Sign-off from the finance controller is "
     "required before payment."),
    ("request_approval",
     "[G-31] PO-88213 and GRN-5521 agree with the invoice. [H-32] The value is beyond our delegated "
     "authority for this agent. [I-33] Approval is required from the CFO; do not release payment "
     "autonomously."),
    ("request_approval",
     "[E-1] Commercially valid and reconciled, but at INR 812,300.00 it is outside the agent's "
     "delegated authority. [E-2] Route to the CFO for approval. [E-3] Do not pay autonomously."),
    ("request_approval",
     "[F-1] The amount is above the INR 500,000.00 limit. [F-2] Documents reconcile. [F-3] Approval by "
     "the finance committee is required before payment."),
    ("request_approval",
     "[G-1] The invoice is correct and fully reconciled. [G-2] Authority to pay rests above this "
     "agent's limit. [G-3] Escalate to the CFO for sign-off."),
    ("request_approval",
     "[H-1] Three-way match is clean. [H-2] The invoice total of INR 654,321.00 exceeds the autonomous "
     "spend threshold. [H-3] Human approval is required."),
    ("request_approval",
     "[H1] The invoice is valid and reconciles with PO-88213 and GRN-5521. [H2] At INR 456,700.00 it is "
     "above the INR 250,000.00 autonomous threshold. [H3] The finance director must approve before "
     "release."),

    # ---- hold_invoice -----------------------------------------------------
    ("hold_invoice",
     "[BNK-A] The vendor requested a bank account change on 2026-10-12. [VER-B] Callback verification "
     "is pending and payment pauses until the verification completes. [POL-C] Treasury policy requires "
     "confirmation of the new account before release."),
    ("hold_invoice",
     "[I-1] Payment is paused: the vendor's bank account changed and callback verification is "
     "outstanding. [I-2] Do not release funds until that verification completes. [I-3] Treasury must "
     "confirm the new account."),
    ("hold_invoice",
     "[J-1] Hold until the vendor verification completes. [J-2] The beneficiary account was updated "
     "yesterday. [J-3] The invoice itself reconciles with PO and GRN."),
    ("hold_invoice",
     "[K-1] Treasury has not confirmed the updated beneficiary account. [K-2] Payment remains on hold "
     "pending bank confirmation. [K-3] The records otherwise agree."),
    ("hold_invoice",
     "[B1] Verification of the updated bank details is not yet complete. [B2] Payment is held until the "
     "callback verification completes. [B3] The invoice reconciles with the purchase order."),
    ("hold_invoice",
     "[I1] The vendor's bank account changed on 2026-10-12 and treasury has not confirmed it. [I2] "
     "Payment pauses until the verification completes. [I3] The invoice is otherwise valid and "
     "reconciled."),

    # ---- reject_duplicate -------------------------------------------------
    ("reject_duplicate",
     "[PAY-A] Invoice INV-2026-0417 for INR 456,700.00 was already paid on 2026-10-02 under remittance "
     "REM-77120. [DUP-B] This is a duplicate presentment of the same commercial invoice. [LED-C] The AP "
     "ledger shows the same invoice number settled in full."),
    ("reject_duplicate",
     "[L-1] INV-2026-0417 was paid on 12 October via RTGS; this submission repeats the same commercial "
     "invoice. [L-2] The ledger shows a settled payment for this invoice number. [L-3] Reject as "
     "duplicate."),
    ("reject_duplicate",
     "[M-1] This is a second presentment; the earlier remittance settled it in full. [M-2] Payment "
     "UTR-8871 cleared on 2026-10-02. [M-3] Do not pay again."),
    ("reject_duplicate",
     "[N-1] The same commercial invoice was already paid in full under remittance REM-4411. [N-2] This "
     "package duplicates that claim. [N-3] Rejecting avoids a double payment."),
    ("reject_duplicate",
     "[A1] INV-2026-0417 was settled on 2026-10-02 under RTGS reference RTGS-9911. [A2] An earlier "
     "query about INV-2025-8800 was for a different invoice and is not a duplicate, but this payment "
     "repeats the same commercial invoice. [A3] Reject to avoid double payment."),
    ("reject_duplicate",
     "[F1] The AP ledger shows a payment for INV-2026-0417 released on 2026-10-02. [F2] Invoice "
     "INV-2026-9999 remains unpaid and is a different invoice, so it is not a duplicate. [F3] For "
     "INV-2026-0417 this is a duplicate of the same commercial invoice; do not pay again."),

    # ---- open_exception ---------------------------------------------------
    ("open_exception",
     "[PO-A] The purchase order records 40 units at INR 11,417.50. [GRN-B] The goods receipt shows only "
     "32 units received. [EXC-C] The quantity mismatch is material and the records conflict, so this "
     "needs an exception workflow rather than payment."),
    ("open_exception",
     "[O-1] The invoice bills 40 units but the goods receipt records 32. [O-2] The records conflict "
     "materially. [O-3] Raise an exception for investigation."),
    ("open_exception",
     "[P-1] PO value is INR 500,000.00 against an invoice of INR 612,000.00. [P-2] That is an "
     "unresolved price variance. [P-3] An exception workflow is required."),
    ("open_exception",
     "[Q-1] GRN-5521 and the invoice disagree on quantity received. [Q-2] The three-way match fails. "
     "[Q-3] Open an exception case."),
    ("open_exception",
     "[R-1] The purchase order and the invoice show different unit prices. [R-2] Records do not agree "
     "and the variance is unresolved. [R-3] Send to the exception queue."),
    ("open_exception",
     "[G1] The goods receipt records 32 units while the invoice bills 40 units. [G2] The purchase order "
     "agrees with the goods receipt on quantity. [G3] There is a material conflict between the records, "
     "so open an exception."),
]
