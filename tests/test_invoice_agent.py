"""End-to-end tests for the EventEcho Invoice Action Agent.

Run with:  python3 -m unittest tests.test_invoice_agent -v
"""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from invoice_agent import server as agent_server  # noqa: E402
from invoice_agent.ai import LocalEvidenceModel  # noqa: E402
from invoice_agent.decisions import decide_package  # noqa: E402
from invoice_agent.service import InvoiceService  # noqa: E402
from invoice_agent.store import Store  # noqa: E402
from tests.invoice_fixture import build_batch, expected_actions  # noqa: E402

IN_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
IN_RESULTS = "application/vnd.ga5.invoice-action-results+json"
OUT_PROPOSALS = "application/vnd.ga5.invoice-action-proposals+json"
OUT_RECEIPTS = "application/vnd.ga5.invoice-action-receipts+json"
A2A = "application/a2a+json"


class Client:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def call(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        token: str | None = "token-alpha",
        version: str | None = "1.0",
        content_type: str | None = A2A,
    ) -> tuple[int, dict, str]:
        url = self.base + path
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            request.add_header("Content-Type", content_type or A2A)
        if token is not None:
            request.add_header("Authorization", f"Bearer {token}")
        if version is not None:
            request.add_header("A2A-Version", version)
        request.add_header("X-Forwarded-Proto", "http")
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read()
                return response.status, json.loads(raw or b"{}"), response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                payload = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                payload = {"raw": raw.decode("utf-8", "replace")}
            return exc.code, payload, exc.headers.get("Content-Type", "")


def send_message(message: dict, accepted: list[str] | None = None, return_immediately: bool = False) -> dict:
    return {
        "message": message,
        "configuration": {
            "returnImmediately": return_immediately,
            "historyLength": 20,
            "acceptedOutputModes": accepted
            or ["application/vnd.ga5.invoice-action-proposals+json",
                "application/vnd.ga5.invoice-action-receipts+json"],
        },
    }


def batch_message(message_id: str, batch: dict) -> dict:
    return {
        "messageId": message_id,
        "role": "ROLE_USER",
        "parts": [{"mediaType": IN_BATCH, "data": batch}],
    }


def results_message(message_id: str, task_id: str, context_id: str, batch_id: str, results: list[dict]) -> dict:
    return {
        "messageId": message_id,
        "taskId": task_id,
        "contextId": context_id,
        "role": "ROLE_USER",
        "parts": [{"mediaType": IN_RESULTS, "data": {"batchId": batch_id, "results": results}}],
    }


class AgentServerCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        agent_server.Handler.log_message = lambda *args, **kwargs: None  # type: ignore[assignment]
        cls.tmpdir = tempfile.mkdtemp(prefix="invoice-agent-test-")
        db_path = os.path.join(cls.tmpdir, "state.sqlite3")
        ctx = agent_server.build_app(db_path=db_path, base_path="/a2a")
        cls.httpd = agent_server.AgentHTTPServer(("127.0.0.1", 0), ctx)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        cls.thread.start()
        cls.client = Client(f"http://127.0.0.1:{cls.port}/a2a")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()

    # -- helpers ------------------------------------------------------- #
    def start_task(self, token: str = "token-alpha", batch: dict | None = None, message_id: str | None = None):
        batch = batch or build_batch()
        message_id = message_id or f"msg-{uuid.uuid4()}"
        status, payload, ctype = self.client.call(
            "POST", "/message:send", send_message(batch_message(message_id, batch)), token=token
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(ctype, A2A)
        self.assertEqual(payload["task"]["status"]["state"], "TASK_STATE_INPUT_REQUIRED")
        return payload["task"], message_id, batch

    @staticmethod
    def proposals(task: dict) -> list[dict]:
        parts = [p for a in task["artifacts"] for p in a["parts"] if p.get("mediaType") == OUT_PROPOSALS]
        assert len(parts) == 1
        return parts[0]["data"]["proposals"]

    # -- protocol ------------------------------------------------------ #
    def test_agent_card_is_public_and_complete(self):
        # direct path, no auth, no version header
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/.well-known/agent-card.json",
            headers={"Host": f"127.0.0.1:{self.port}", "X-Forwarded-Proto": "https"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            self.assertEqual(response.status, 200)
            card = json.loads(response.read())
            self.assertTrue(response.headers.get("Content-Type", "").startswith("application/json"))
        self.assertTrue(card["name"] and card["description"] and card["version"])
        self.assertIsInstance(card["capabilities"], dict)
        skill = card["skills"][0]
        self.assertEqual(skill["id"], "invoice_action_agent")
        self.assertTrue(skill["name"] and skill["description"] and skill["tags"])
        self.assertIn(IN_BATCH, card["defaultInputModes"])
        self.assertIn(OUT_PROPOSALS, card["defaultOutputModes"])
        self.assertIn(OUT_RECEIPTS, card["defaultOutputModes"])
        expected_base = f"https://127.0.0.1:{self.port}/a2a/"
        interfaces = card["supportedInterfaces"]
        self.assertTrue(
            any(i["url"] == expected_base and i["protocolBinding"] == "HTTP+JSON" and i["protocolVersion"] == "1.0"
                for i in interfaces),
            interfaces,
        )

    def test_missing_auth_is_rejected(self):
        status, _, _ = self.client.call("GET", "/tasks", token=None)
        self.assertIn(status, (401, 403))

    def test_wrong_version_is_400(self):
        status, body, _ = self.client.call("GET", "/tasks", version="0.9")
        self.assertEqual(status, 400)
        status, _, _ = self.client.call("GET", "/tasks", version=None)
        self.assertEqual(status, 400)

    def test_wrong_media_type_is_rejected(self):
        batch = build_batch()
        status, _, _ = self.client.call(
            "POST",
            "/message:send",
            send_message(batch_message(f"msg-{uuid.uuid4()}", batch)),
            content_type="application/json",
        )
        self.assertEqual(status, 415)

    def test_batch_returns_input_required_with_typed_proposals(self):
        task, _msg_id, _batch = self.start_task()
        self.assertEqual(task["status"]["state"], "TASK_STATE_INPUT_REQUIRED")
        proposals = self.proposals(task)
        self.assertEqual(len(proposals), 12)
        self.assertEqual(len({p["packageId"] for p in proposals}), 12)
        self.assertEqual(len({p["actionId"] for p in proposals}), 12)
        for proposal in proposals:
            self.assertGreaterEqual(len(proposal["actionId"]), 12)
            self.assertIn(proposal["action"], {
                "settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception",
            })
            self.assertEqual(len(proposal["evidenceRefs"]), 3)
            self.assertTrue(60 <= len(proposal["rationale"]) <= 1500)
            refs = proposal["evidenceRefs"]
            self.assertTrue(any(ref in proposal["rationale"] for ref in refs))
            self.assertNotIn("COVER-", " ".join(refs))
            self.assertNotIn("ARC-", " ".join(refs))
        self.assertEqual([p["action"] for p in proposals], expected_actions())
        self.assertEqual(len(task["history"]), 1)
        self.assertEqual(task["history"][0]["role"], "ROLE_USER")

    def test_response_size_under_limit(self):
        task, _msg_id, _batch = self.start_task()
        raw = json.dumps({"task": task})
        self.assertLess(len(raw.encode()), 512 * 1024)

    # -- idempotency --------------------------------------------------- #
    def test_idempotent_replay_ignores_key_order_and_configuration(self):
        batch = build_batch()
        message_id = f"msg-{uuid.uuid4()}"
        first, _, _ = self.start_task(batch=batch, message_id=message_id)

        reordered = json.loads(json.dumps(batch_message(message_id, batch)))
        reordered = {k: reordered[k] for k in reversed(list(reordered))}
        reordered["parts"] = [
            {"data": {k: reordered["parts"][0]["data"][k] for k in reversed(list(reordered["parts"][0]["data"]))},
             "mediaType": IN_BATCH}
        ]
        status, payload, _ = self.client.call(
            "POST", "/message:send", send_message(reordered, return_immediately=True)
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["task"]["id"], first["id"])
        self.assertEqual(payload["task"], first)

    def test_message_id_reuse_with_changed_content_conflicts(self):
        batch = build_batch()
        message_id = f"msg-{uuid.uuid4()}"
        first, _, _ = self.start_task(batch=batch, message_id=message_id)
        changed = build_batch(batch_id="BATCH-OTHER")
        status, payload, _ = self.client.call(
            "POST", "/message:send", send_message(batch_message(message_id, changed))
        )
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"]["code"], "IDEMPOTENCY_CONFLICT")
        status, reread, _ = self.client.call("GET", f"/tasks/{first['id']}")
        self.assertEqual(status, 200)
        self.assertEqual(reread["task"], first)

    def test_concurrent_equivalent_requests_share_one_task(self):
        batch = build_batch()
        message_id = f"msg-{uuid.uuid4()}"
        results: list[tuple[int, dict]] = []
        lock = threading.Lock()

        def worker():
            status, payload, _ = self.client.call(
                "POST", "/message:send", send_message(batch_message(message_id, batch))
            )
            with lock:
                results.append((status, payload))

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        self.assertEqual(len(results), 6)
        ids = {payload["task"]["id"] for status, payload in results if status == 200}
        self.assertEqual(len(ids), 1, results)
        self.assertTrue(all(status == 200 for status, _ in results))

    def test_decision_cache_reused_for_renamed_namespaces(self):
        ctx = agent_server.build_app(db_path=os.path.join(self.tmpdir, "cache.sqlite3"), base_path="/a2a")
        service = InvoiceService(ctx.store, LocalEvidenceModel())
        batch_a = build_batch(batch_id="B-1")
        status_a = service.send_message("p1", send_message(batch_message("m-1", batch_a)))
        cached_after_first = ctx.store.stats()["cached_decisions"]
        self.assertEqual(cached_after_first, 12)

        # Same semantic packages, new package ids, batch and message.
        batch_b = build_batch(batch_id="B-2")
        for package in batch_b["packages"]:
            package["packageId"] = package["packageId"] + "-RENAMED"
        service.send_message("p1", send_message(batch_message("m-2", batch_b)))
        self.assertEqual(ctx.store.stats()["cached_decisions"], cached_after_first)
        self.assertEqual(status_a["task"]["status"]["state"], "TASK_STATE_INPUT_REQUIRED")

    # -- approval gate -------------------------------------------------- #
    def test_results_complete_task_with_accepted_receipts_only(self):
        task, _msg_id, batch = self.start_task()
        proposals = self.proposals(task)
        results = []
        for index, proposal in enumerate(proposals):
            outcome = "ACCEPTED" if index % 2 == 0 else "REJECTED"
            results.append(
                {
                    "packageId": proposal["packageId"],
                    "actionId": proposal["actionId"],
                    "action": proposal["action"],
                    "outcome": outcome,
                    "receiptNonce": f"nonce-{uuid.uuid4()}",
                }
            )
        status, payload, _ = self.client.call(
            "POST",
            "/message:send",
            send_message(
                results_message(f"msg-{uuid.uuid4()}", task["id"], task["contextId"], batch["batchId"], results)
            ),
        )
        self.assertEqual(status, 200, payload)
        completed = payload["task"]
        self.assertEqual(completed["status"]["state"], "TASK_STATE_COMPLETED")
        parts = [p for a in completed["artifacts"] for p in a["parts"]]
        self.assertEqual(len([p for p in parts if p["mediaType"] == OUT_PROPOSALS]), 1)
        receipt_parts = [p for p in parts if p["mediaType"] == OUT_RECEIPTS]
        self.assertEqual(len(receipt_parts), 1)
        executions = receipt_parts[0]["data"]["executions"]
        accepted = [r for r in results if r["outcome"] == "ACCEPTED"]
        self.assertEqual(len(executions), len(accepted))
        for execution in executions:
            proposal = next(p for p in proposals if p["packageId"] == execution["packageId"])
            self.assertEqual(execution["action"], proposal["action"])
            self.assertEqual(execution["actionId"], proposal["actionId"])
            self.assertEqual(execution["facts"], proposal["facts"])
            self.assertEqual(execution["evidenceRefs"], proposal["evidenceRefs"])
        self.assertEqual(len(completed["history"]), 2)

        # Terminal replay is byte-identical and does not create new work.
        status, replay, _ = self.client.call("GET", f"/tasks/{task['id']}")
        self.assertEqual(status, 200)
        self.assertEqual(replay["task"], completed)

    def test_mismatched_action_id_is_rejected(self):
        task, _msg_id, batch = self.start_task()
        proposal = self.proposals(task)[0]
        bad = {
            "packageId": proposal["packageId"],
            "actionId": "act-000000000000",
            "action": proposal["action"],
            "outcome": "ACCEPTED",
            "receiptNonce": "nonce-x",
        }
        status, payload, _ = self.client.call(
            "POST",
            "/message:send",
            send_message(
                results_message(f"msg-{uuid.uuid4()}", task["id"], task["contextId"], batch["batchId"], [bad])
            ),
        )
        self.assertEqual(status, 400)
        status, reread, _ = self.client.call("GET", f"/tasks/{task['id']}")
        self.assertEqual(reread["task"]["status"]["state"], "TASK_STATE_INPUT_REQUIRED")

    def test_wrong_action_for_proposal_is_rejected(self):
        task, _msg_id, batch = self.start_task()
        proposal = self.proposals(task)[0]
        swapped = next(a for a in ("settle_invoice", "request_approval") if a != proposal["action"])
        bad = {
            "packageId": proposal["packageId"],
            "actionId": proposal["actionId"],
            "action": swapped,
            "outcome": "ACCEPTED",
            "receiptNonce": "nonce-x",
        }
        status, _, _ = self.client.call(
            "POST",
            "/message:send",
            send_message(
                results_message(f"msg-{uuid.uuid4()}", task["id"], task["contextId"], batch["batchId"], [bad])
            ),
        )
        self.assertEqual(status, 400)

    def test_wrong_batch_or_context_is_rejected(self):
        task, _msg_id, batch = self.start_task()
        proposal = self.proposals(task)[0]
        good = {
            "packageId": proposal["packageId"],
            "actionId": proposal["actionId"],
            "action": proposal["action"],
            "outcome": "ACCEPTED",
            "receiptNonce": "nonce-x",
        }
        status, _, _ = self.client.call(
            "POST",
            "/message:send",
            send_message(results_message(f"msg-{uuid.uuid4()}", task["id"], task["contextId"], "WRONG-BATCH", [good])),
        )
        self.assertEqual(status, 400)
        status, _, _ = self.client.call(
            "POST",
            "/message:send",
            send_message(results_message(f"msg-{uuid.uuid4()}", task["id"], str(uuid.uuid4()), batch["batchId"], [good])),
        )
        self.assertEqual(status, 400)

    def test_second_result_continuation_is_terminal_conflict(self):
        task, _msg_id, batch = self.start_task()
        proposals = self.proposals(task)
        results = [
            {
                "packageId": p["packageId"],
                "actionId": p["actionId"],
                "action": p["action"],
                "outcome": "ACCEPTED",
                "receiptNonce": f"nonce-{uuid.uuid4()}",
            }
            for p in proposals
        ]
        status, _, _ = self.client.call(
            "POST",
            "/message:send",
            send_message(results_message(f"msg-{uuid.uuid4()}", task["id"], task["contextId"], batch["batchId"], results)),
        )
        self.assertEqual(status, 200)
        status, payload, _ = self.client.call(
            "POST",
            "/message:send",
            send_message(results_message(f"msg-{uuid.uuid4()}", task["id"], task["contextId"], batch["batchId"], results)),
        )
        self.assertEqual(status, 409)

    # -- lifecycle ------------------------------------------------------ #
    def test_cancel_then_result_conflicts(self):
        task, _msg_id, batch = self.start_task()
        status, payload, _ = self.client.call("POST", f"/tasks/{task['id']}:cancel")
        self.assertEqual(status, 200)
        self.assertEqual(payload["task"]["status"]["state"], "TASK_STATE_CANCELED")
        self.assertEqual(len(payload["task"]["artifacts"]), 1)
        status, _, _ = self.client.call("POST", f"/tasks/{task['id']}:cancel")
        self.assertEqual(status, 409)
        proposal = self.proposals(task)[0]
        status, _, _ = self.client.call(
            "POST",
            "/message:send",
            send_message(
                results_message(
                    f"msg-{uuid.uuid4()}",
                    task["id"],
                    task["contextId"],
                    batch["batchId"],
                    [
                        {
                            "packageId": proposal["packageId"],
                            "actionId": proposal["actionId"],
                            "action": proposal["action"],
                            "outcome": "ACCEPTED",
                            "receiptNonce": "nonce-x",
                        }
                    ],
                )
            ),
        )
        self.assertEqual(status, 409)

    def test_cancel_and_receipt_race_has_exactly_one_winner(self):
        for attempt in range(4):
            task, _msg_id, batch = self.start_task()
            proposal = self.proposals(task)[0]
            payload = send_message(
                results_message(
                    f"msg-{uuid.uuid4()}",
                    task["id"],
                    task["contextId"],
                    batch["batchId"],
                    [
                        {
                            "packageId": proposal["packageId"],
                            "actionId": proposal["actionId"],
                            "action": proposal["action"],
                            "outcome": "ACCEPTED",
                            "receiptNonce": f"nonce-{attempt}",
                        }
                    ],
                )
            )
            outcome: dict[str, int] = {}

            def do_cancel():
                outcome["cancel"] = self.client.call("POST", f"/tasks/{task['id']}:cancel")[0]

            def do_result():
                outcome["result"] = self.client.call("POST", "/message:send", payload)[0]

            threads = [threading.Thread(target=do_cancel), threading.Thread(target=do_result)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=20)
            self.assertEqual(sorted(outcome.values()), [200, 409], outcome)
            status, final, _ = self.client.call("GET", f"/tasks/{task['id']}")
            self.assertEqual(status, 200)
            state = final["task"]["status"]["state"]
            receipts = [p for a in final["task"]["artifacts"] for p in a["parts"] if p["mediaType"] == OUT_RECEIPTS]
            if state == "TASK_STATE_COMPLETED":
                self.assertEqual(len(receipts), 1)
            else:
                self.assertEqual(state, "TASK_STATE_CANCELED")
                self.assertEqual(receipts, [])

    # -- isolation ------------------------------------------------------ #
    def test_user_isolation(self):
        task, _msg_id, batch = self.start_task(token="token-alpha")
        status, payload, _ = self.client.call("GET", f"/tasks/{task['id']}", token="token-beta")
        self.assertEqual(status, 404)
        self.assertNotIn(task["id"], json.dumps(payload))
        status, payload, _ = self.client.call("POST", f"/tasks/{task['id']}:cancel", token="token-beta")
        self.assertEqual(status, 404)
        self.assertNotIn(task["id"], json.dumps(payload))
        status, payload, _ = self.client.call("GET", "/tasks", token="token-beta")
        self.assertEqual(status, 200)
        self.assertEqual(payload["tasks"], [])
        status, payload, _ = self.client.call("GET", "/tasks", token="token-alpha")
        self.assertEqual(status, 200)
        self.assertTrue(any(t["id"] == task["id"] for t in payload["tasks"]))
        status, reread, _ = self.client.call("GET", f"/tasks/{task['id']}", token="token-alpha")
        self.assertEqual(reread["task"]["status"]["state"], "TASK_STATE_INPUT_REQUIRED")

    def test_outsider_cannot_continue_task(self):
        task, _msg_id, batch = self.start_task(token="token-alpha")
        proposal = self.proposals(task)[0]
        result = {
            "packageId": proposal["packageId"],
            "actionId": proposal["actionId"],
            "action": proposal["action"],
            "outcome": "ACCEPTED",
            "receiptNonce": "nonce-x",
        }
        status, payload, _ = self.client.call(
            "POST",
            "/message:send",
            send_message(results_message(f"msg-{uuid.uuid4()}", task["id"], task["contextId"], batch["batchId"], [result])),
            token="token-beta",
        )
        self.assertEqual(status, 404)
        self.assertNotIn(task["id"], json.dumps(payload))


class DecisionModelCase(unittest.TestCase):
    def test_negated_duplicate_and_cleared_verification(self):
        package = build_batch(count=1, plan=["reject_duplicate"])["packages"][0]
        package["documents"][2]["text"] = (
            "[NEG-1] An earlier payment of INR 100,000.00 was recorded, but it was for a different "
            "invoice (INV-2025-8800), not this one. [NEG-2] The three-way match reconciles fully. "
            "[NEG-3] The amount is within the autonomous authority, so the invoice can be settled."
        )
        decision = decide_package(package)
        self.assertEqual(decision.action, "settle_invoice")
        self.assertEqual(decision.evidence_refs, ["NEG-1", "NEG-2", "NEG-3"])

    def test_archive_examples_never_supply_evidence(self):
        for package in build_batch()["packages"]:
            decision = decide_package(package)
            self.assertFalse(any(ref.startswith("ARC-") or ref.startswith("COVER-") for ref in decision.evidence_refs))
            self.assertGreaterEqual(len(decision.evidence_refs), 3)


if __name__ == "__main__":
    unittest.main()
