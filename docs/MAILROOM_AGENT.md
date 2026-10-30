# Mailroom Action Gate — `ga5-mailroom-action-gate/v2`

A public HTTPS endpoint that reads mailroom dossiers, proposes exactly one
permitted action per dossier with minimal line evidence, verifies the grader's
Ed25519-signed receipts, and only then records an outcome.

- **Endpoint (submitted URL):** `https://8001-i08mehv473n5yap8hpkdj.e2b.app/`
- **Health:** `GET https://8001-i08mehv473n5yap8hpkdj.e2b.app/healthz`
- **Code:** `mailroom_agent/` · **Tests:** `tests/test_mailroom_agent.py` ·
  **Live check:** `scripts/verify_mailroom_agent.py`

Standard library only: JSON, hashing, SQLite and a local evidence model. The
Ed25519 verifier is a self-contained RFC 8032 implementation (validated against
the published test vectors), so receipts are verified on-box with no network
call and no third-party crypto dependency.

---

## 1. Contract

```
POST <url>   { operation: "propose", evaluationId, receiptVerifier, dossiers:[...] }
        200  { profile, evaluationId, status:"awaiting_receipts", inputDigest, proposals:[...] }

POST <url>   { operation: "commit", evaluationId, inputDigest, receipts:[...] }
        200  { profile, evaluationId, status:"completed", inputDigest, outcomes:[...] }
```

* `inputDigest` = SHA-256 of the dossiers array in recursively key-sorted,
  compact JSON, UTF-8, lowercase hex.
* `proposalDigest` = SHA-256 of `{dossierId, callId, action, target, payload,
  evidence}` with evidence sorted; extra response fields are excluded.
* `callId` = `call-` + 32 hex chars derived from the dossier content
  fingerprint and dossier id, so the same stable dossier always yields the same
  proposal and call id across evaluations.
* Every response is non-redirecting `application/json`; successful bodies stay
  far below 512 KiB; bodies above 16 MiB are drained and rejected with 413.

### Error codes

| Status | When |
|---|---|
| 400 | malformed JSON, missing envelope fields, bad operation, missing sources/lines |
| 413 | request body over the limit |
| 409 | evaluation id reused with different content, unknown evaluation, digest/call/action/proposal mismatch, duplicate receipt, second commit |
| 422 | wrong profile, unusable verifier key, duplicate dossier/line ids, invalid receipt signature |

All validation happens before any model or tool work: a rejected request leaves
no evaluation behind.

---

## 2. Receipt verification

The `receiptVerifier` public key supplied with a propose call is stored with
that evaluation; nothing is hard-coded. A commit is rejected **before any
outcome is recorded** if one signature is invalid, missing, duplicated, or
moved to another receipt. The signed bytes are the canonical JSON of

```json
{"profile":"ga5-mailroom-action-gate/v2","evaluationId":"…","inputDigest":"…",
 "receipt":{ …every receipt field except receiptSignature… }}
```

`executed` is emitted only when the verified receipt says `accepted: true`;
otherwise `rejected`. Receipts are scoped to one evaluation, proposal digest
and call id.

---

## 3. Decision layer

Mail is data. External text never reaches a tool argument; payloads contain
only typed, short, extracted values (addresses, order references, statuses,
windows, ids), and values that look like secrets are dropped.

Priority:

1. **`quarantine_item`** — untrusted content that tries to drive tools, read
   private context or create an outbound effect. A *trusted* bulletin quoting an
   attack, or a customer forwarding one inside quotes ("is this a scam?"), is
   reported text, not an attack, and is handled normally.
2. **`no_action`** — `DUPLICATE` / `ALREADY_COMPLETED` when the mail is a repeat
   or the work is already done.
3. **`request_confirmation`** — identity conflicts (claimed sender vs verified
   contact) and approval-scope mismatches route to the owning approval queue.
4. **`send_approved_notice`** — only when a trusted approval names the exact
   recipient, the exact current status and the same order. An approval for a
   different recipient, for a stale status, for another order, or an expired
   approval is *not* authority and is routed back.
5. **`update_internal_record`** — a trusted, event-sourced delivery-window
   change (`sourceEventId` + the new window value).
6. **`create_draft`** — ordinary customer work produces a non-sent draft in the
   mailbox's draft queue.
7. **`no_action` `INFORMATIONAL`** — nothing to do.

Evidence is the smallest line set that establishes the authority and the exact
arguments: the status line and the sender line for a draft, the event and new
window for an update, the approval lines for a send, the malicious line and its
artifact line for a quarantine, and the duplicate/completed line for a
suppression. Duplicate line ids and unknown ids are impossible: they are
filtered against the dossier before the response is built.

---

## 4. Persistence

SQLite (WAL, `synchronous=FULL`, `BEGIN IMMEDIATE`):

| Table | Purpose |
|---|---|
| `evaluations` | propose hash, input digest, verifier key, exact response, commit state |
| `proposals` | the persisted proposal and digest for every dossier |
| `receipts` | every verified receipt, keyed by receipt id |
| `decision_cache` | dossier content fingerprint → decision |

* An exact propose or commit replay returns the stored response, byte for byte,
  with no model call and no repeated effect.
* Reusing an evaluation id with different content is `409`.
* Committing twice with different receipts is `409`.
* Stable dossiers are decided once: repeated Checks/Saves under new evaluation
  ids make no model call, while fresh audit dossiers are decided immediately.

---

## 5. Verification

| Check | Result |
|---|---|
| `python3 -m unittest tests.test_mailroom_agent` | **21 passed** |
| `python3 scripts/verify_mailroom_agent.py` | **27 / 27 live checks** |
| Labelled dossier corpus (`tests/mailroom_fixtures.py`) | **12 / 12** actions, schema-exact targets/payloads |
| Ed25519 | RFC 8032 vectors 1–3 verified; tampered signatures rejected |
| Propose latency, 12 dossiers, warm cache | ~20–30 ms |
| Replay latency | ~2 ms |

---

## 6. Run

```bash
python3 -m mailroom_agent.server          # 0.0.0.0:8001
MAILROOM_DB=… MAILROOM_PORT=… python3 -m mailroom_agent.server
```

Environment: `MAILROOM_PORT` / `PORT` (default 8001), `MAILROOM_DB`
(default `data/mailroom_agent.sqlite3`), `HOST`.
