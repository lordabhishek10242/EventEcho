# Invoice Action Agent — A2A 1.0 service

A public HTTPS service that reads an invoice claim batch, proposes **exactly one
typed business action per package** with the three decisive evidence
references, waits for the reviewer's results, and only then records executions.
It never pays on a proposal alone, and no retry can ever repeat a payment.

- **Agent Card (public):** `https://8000-i08mehv473n5yap8hpkdj.e2b.app/.well-known/agent-card.json`
- **Submitted A2A base URL:** `https://8000-i08mehv473n5yap8hpkdj.e2b.app/a2a/`
- **Skill id:** `invoice_action_agent`

> The base URL is served by the sandbox preview proxy. To move the agent to
> another host, run it there and submit that host's `/a2a/` base URL — the card
> derives `supportedInterfaces` from `INVOICE_AGENT_PUBLIC_BASE` (or from the
> request's `Host` / `X-Forwarded-*` headers).

---

## 1. Surface

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET` | `{origin}/.well-known/agent-card.json` | public | Agent Card |
| `POST` | `{base}/message:send` | Bearer | start a batch **or** deliver approval results |
| `GET` | `{base}/tasks/{id}` | Bearer | read one task |
| `GET` | `{base}/tasks` | Bearer | list **that principal's** tasks |
| `POST` | `{base}/tasks/{id}:cancel` | Bearer | cancel a non-terminal task |

Every non-card route requires `Authorization: Bearer <token>` (401 otherwise),
`A2A-Version: 1.0` (400 otherwise) and, for bodies, `Content-Type:
application/a2a+json` (415 otherwise). Successful responses are
`application/a2a+json` (the card is `application/json`), all JSON, and every
task stays far below the 512 KiB ceiling. ProtoJSON enum strings are used
throughout: `ROLE_USER`, `TASK_STATE_INPUT_REQUIRED`, `TASK_STATE_COMPLETED`,
`TASK_STATE_CANCELED`.

`POST /message:send` returns `{"task": Task}`; reads and cancellation return the
Task itself; listing returns `{"tasks": [Task, ...]}`. Errors are generic
(`{"error": {"code", "message"}}`) and never echo another principal's ids.

---

## 2. Decision layer

`invoice_agent/decisions.py` scores every paragraph of a package against five
lexicons (`settle_invoice`, `request_approval`, `hold_invoice`,
`reject_duplicate`, `open_exception`), then lets **the paragraph that decides
something win**. Facts and evidence come from that paragraph only.

What it guards against — exactly what the documents mix in:

| Noise | Handling |
|---|---|
| Cover sheet with action words | cover-sheet regex, −4.5 decisiveness, never a source of evidence |
| Archive examples / training decoys | decoy regex (`archive`, `example`, `training decoy`, `illustration`, …), −6.0 and excluded from evidence |
| Negation (`not a duplicate`, `different invoice`, `no material conflict`, `verification completed`) | explicit negative lexicon **plus** a same-sentence negator window that flips positive cues |
| Irrelevant action verbs | only decisive paragraphs are weighted fully; others contribute at 0.3 or 0 |
| Structured flags (`alreadyPaid: true`, `recordsConflict: true`) | structured hint scan as a safety net |
| Authority thresholds | numeric comparison of invoice amount against any stated limit, plus lexicon |

Evidence selection keeps the decisive paragraph's bracketed references, ranks
them by how close they sit to the deciding language, returns exactly three in
document order, and tops up from other non-cover, non-decoy paragraphs only if
the decisive paragraph has fewer than three.

The rationale (60–1500 chars) names the action, states the reasoning, quotes
the decisive paragraph and cites the evidence refs.

Two providers sit behind `invoice_agent/ai.py`:

- `local` (default) — the deterministic on-box evidence model above. No
  network, no key, no cost, and the same input always gives the same decision.
- `remote` — any OpenAI-compatible endpoint, enabled only with
  `INVOICE_AGENT_LLM_BASE_URL`, `INVOICE_AGENT_LLM_API_KEY` and
  `INVOICE_AGENT_LLM_MODEL`; replies are schema-validated and fall back to the
  local model on any error.

Either way, the protocol and storage layers are untouched.

---

## 3. State, idempotency, caching

`invoice_agent/store.py` keeps three SQLite tables (WAL, `synchronous=FULL`):

| Table | Purpose |
|---|---|
| `tasks` | one row per task: state, proposals, receipts, history, results |
| `idempotency` | `(principal, messageId) → message hash + task id` |
| `decision_cache` | canonical package content → typed decision |

* **Idempotency hash** = SHA-256 of recursively key-sorted compact JSON of the
  `message` object only. Reordered keys and a changed `returnImmediately`
  replay the same complete Task with no model call and no action.
* **Conflict** — the same `messageId` with different message content returns
  `409 IDEMPOTENCY_CONFLICT` and leaves the original untouched.
* **Concurrency** — task creation, cancellation and completion each run inside
  a `BEGIN IMMEDIATE` transaction, so concurrent equivalent requests return one
  Task, and a near-simultaneous cancel + receipt produces exactly one 200 and
  one 409.
* **Decision cache** — keyed by canonical package content with the opaque
  `packageId` stripped. Check/Save runs that re-issue the same packages under
  new batch/message/task namespaces hit the cache and make no model call;
  genuinely new audit packages are decided fresh.
* **Terminal immutability** — `updated_ms` is frozen at the transition, so a
  completed or canceled Task is byte-identical on every replay.
* **Isolation** — every task row is keyed by `(principal, id)`; an outsider's
  read, cancel or continuation is a generic 404, and their list is `{"tasks":[]}`.

---

## 4. Lifecycle

```
batch  ──►  TASK_STATE_INPUT_REQUIRED  (proposal artifact, history: [batch message])
                    │
      results ──────┼─────────────► TASK_STATE_COMPLETED  (proposal + receipts artifact,
                    │                                     history: [batch, results])
      cancel ───────┴─────────────► TASK_STATE_CANCELED   (proposal artifact only)
```

A result continuation is accepted only when principal, task, context, batch,
package id, action id and action all match the stored proposal; receipts carry
`ACCEPTED` executions only, and rejected proposals stay in history unexecuted.
Cancellation after completion (or a second result) is `409`.

---

## 5. Verification status

| Check | Result |
|---|---|
| Labelled decision cases (`scripts/eval_invoice_decisions.py`) | **40 / 40** (5 actions × 8 paraphrases with negation and decoys) |
| Unit + end-to-end tests (`tests.test_invoice_agent`) | **21 passed** |
| Live contract walkthrough (`scripts/verify_invoice_agent.py`) | **41 / 41 checks** |
| Batch latency, 12 packages, cold cache | ~45–90 ms (one batched model pass) |
| Idempotent replay latency | ~2 ms (no model call, no action) |
| Task size for 12 packages | ~25 KB (limit 512 KiB) |

The decision case set is deliberately adversarial: it contains "not a
duplicate" clauses, "but" reversals, cleared verifications, decoys that name a
different action, and archive paragraphs that shout "hold the payment".

## 6. Run and verify

```bash
python3 -m invoice_agent.server            # 0.0.0.0:8000, base path /a2a
python3 -m unittest tests.test_invoice_agent -v
python3 scripts/verify_invoice_agent.py    # live end-to-end walkthrough
```

Environment: `PORT`, `HOST`, `INVOICE_AGENT_BASE_PATH` (default `/a2a`),
`INVOICE_AGENT_DB` (default `data/invoice_agent.sqlite3`),
`INVOICE_AGENT_PUBLIC_BASE` (canonical origin used in the Agent Card).

No third-party dependencies — the service is standard library only.
