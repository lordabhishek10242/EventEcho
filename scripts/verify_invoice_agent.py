#!/usr/bin/env python3
"""Live end-to-end walkthrough of the Invoice Action Agent.

Starts an in-process server on a scratch database, then exercises the full
graded contract: card, auth, version, media types, proposals, idempotent
replay, conflict, approval gate, terminal replay, isolation and the
cancel-versus-receipt race.

    python3 scripts/verify_invoice_agent.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from invoice_agent import server as agent_server  # noqa: E402
from tests.invoice_fixture import build_batch, expected_actions  # noqa: E402

A2A = "application/a2a+json"
IN_BATCH = "application/vnd.ga5.invoice-claim-batch+json"
IN_RESULTS = "application/vnd.ga5.invoice-action-results+json"
OUT_RECEIPTS = "application/vnd.ga5.invoice-action-receipts+json"

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((PASS if condition else FAIL, name, detail))
    print(f"  [{PASS if condition else FAIL}] {name}{(' — ' + detail) if detail and not condition else ''}")


class Client:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def call(self, method, path, body=None, token="grader-alpha", version="1.0", ctype=A2A):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            request.add_header("Content-Type", ctype)
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        if version:
            request.add_header("A2A-Version", version)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read()), dict(response.headers)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            return exc.code, json.loads(raw or b"{}"), dict(exc.headers)


def batch_envelope(message_id: str, batch: dict) -> dict:
    return {
        "message": {
            "messageId": message_id,
            "role": "ROLE_USER",
            "parts": [{"mediaType": IN_BATCH, "data": batch}],
        },
        "configuration": {
            "returnImmediately": False,
            "historyLength": 20,
            "acceptedOutputModes": [
                "application/vnd.ga5.invoice-action-proposals+json",
                "application/vnd.ga5.invoice-action-receipts+json",
            ],
        },
    }


def results_envelope(message_id: str, task_id: str, context_id: str, batch_id: str, results_data: list[dict]) -> dict:
    return {
        "message": {
            "messageId": message_id,
            "taskId": task_id,
            "contextId": context_id,
            "role": "ROLE_USER",
            "parts": [{"mediaType": IN_RESULTS, "data": {"batchId": batch_id, "results": results_data}}],
        }
    }


def main() -> int:
    agent_server.Handler.log_message = lambda *a, **k: None  # type: ignore[assignment]
    tmp = tempfile.mkdtemp(prefix="invoice-agent-verify-")
    ctx = agent_server.build_app(db_path=os.path.join(tmp, "verify.sqlite3"), base_path="/a2a")
    httpd = agent_server.AgentHTTPServer(("127.0.0.1", 0), ctx)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.03}, daemon=True).start()
    origin = f"http://127.0.0.1:{port}"
    client = Client(origin + "/a2a")
    print(f"Invoice Action Agent verification against {origin}")

    print("\n1. Agent card")
    with urllib.request.urlopen(origin + "/.well-known/agent-card.json", timeout=10) as response:
        card = json.loads(response.read())
    check("card is public (no token)", True)
    check("name/description/version present", bool(card.get("name") and card.get("description") and card.get("version")))
    check("capabilities is an object", isinstance(card.get("capabilities"), dict))
    skill = (card.get("skills") or [{}])[0]
    check("invoice_action_agent skill", skill.get("id") == "invoice_action_agent" and bool(skill.get("tags")))
    check("input mode advertised", IN_BATCH in card.get("defaultInputModes", []))
    check("both output modes advertised", len([m for m in card.get("defaultOutputModes", []) if m.endswith("+json")]) == 2)
    check(
        "supportedInterfaces carries the base URL",
        any(i.get("url", "").rstrip("/") == origin + "/a2a" and i.get("protocolBinding") == "HTTP+JSON"
            and i.get("protocolVersion") == "1.0" for i in card.get("supportedInterfaces", [])),
    )

    print("\n2. Transport guards")
    check("missing token -> 401", client.call("GET", "/tasks", token=None)[0] == 401)
    check("wrong version -> 400", client.call("GET", "/tasks", version="2.0")[0] == 400)
    check(
        "wrong media type -> 415",
        client.call("POST", "/message:send", batch_envelope("m-x", build_batch()), ctype="application/json")[0] == 415,
    )

    print("\n3. Batch -> proposals")
    batch = build_batch(batch_id="BATCH-VERIFY-1")
    message_id = f"msg-{uuid.uuid4()}"
    started = time.time()
    status, payload, headers = client.call("POST", "/message:send", batch_envelope(message_id, batch))
    elapsed = (time.time() - started) * 1000
    task = payload.get("task", {})
    proposals = task.get("artifacts", [{}])[0].get("parts", [{}])[0].get("data", {}).get("proposals", [])
    check("200 with A2A media type", status == 200 and headers.get("Content-Type") == A2A)
    check("state INPUT_REQUIRED", task.get("status", {}).get("state") == "TASK_STATE_INPUT_REQUIRED")
    check("one proposal per package", len(proposals) == 12)
    check("unique package and action ids", len({p["packageId"] for p in proposals}) == 12 == len({p["actionId"] for p in proposals}))
    check("actions match the synthetic truth", [p["action"] for p in proposals] == expected_actions())
    check("three decisive refs each, no cover/archive", all(
        len(p["evidenceRefs"]) == 3 and not any(r.startswith(("COVER-", "ARC-")) for r in p["evidenceRefs"])
        for p in proposals
    ))
    check("rationale 60-1500 chars", all(60 <= len(p["rationale"]) <= 1500 for p in proposals))
    check("history keeps the initial message", len(task.get("history", [])) == 1 and task["history"][0]["role"] == "ROLE_USER")
    check("batch latency budget", elapsed < 5000, f"{elapsed:.0f} ms")
    print(f"        ({elapsed:.0f} ms for 12 packages, {len(json.dumps(payload))} bytes)")

    print("\n4. Idempotency and conflict")
    replay_request = batch_envelope(message_id, batch)
    replay_request["message"] = {k: replay_request["message"][k] for k in reversed(list(replay_request["message"]))}
    replay_request["configuration"]["returnImmediately"] = True
    started = time.time()
    replay_status, replay_payload, _ = client.call("POST", "/message:send", replay_request)
    replay_ms = (time.time() - started) * 1000
    check("replay returns the same task", replay_status == 200 and replay_payload["task"] == task)
    check("replay makes no model call", replay_ms < 150, f"{replay_ms:.0f} ms")
    conflict_status, conflict_payload, _ = client.call(
        "POST", "/message:send", batch_envelope(message_id, build_batch(batch_id="BATCH-OTHER"))
    )
    check("changed content -> 409 IDEMPOTENCY_CONFLICT", conflict_status == 409 and conflict_payload["error"]["code"] == "IDEMPOTENCY_CONFLICT")
    check("original untouched", client.call("GET", f"/tasks/{task['id']}")[1]["task"] == task)

    print("\n5. Approval gate")
    receipts = [
        {
            "packageId": p["packageId"],
            "actionId": p["actionId"],
            "action": p["action"],
            "outcome": "ACCEPTED" if index % 3 else "REJECTED",
            "receiptNonce": f"nonce-{uuid.uuid4()}",
        }
        for index, p in enumerate(proposals)
    ]
    bad_status, _, _ = client.call(
        "POST",
        "/message:send",
        results_envelope(f"msg-{uuid.uuid4()}", task["id"], task["contextId"], batch["batchId"],
                         [{**receipts[0], "actionId": "act-000000000000"}]),
    )
    check("mismatched action id -> 400", bad_status == 400)
    status, payload, _ = client.call(
        "POST",
        "/message:send",
        results_envelope(f"msg-{uuid.uuid4()}", task["id"], task["contextId"], batch["batchId"], receipts),
    )
    completed = payload.get("task", {})
    executions = [
        part["data"]["executions"]
        for artifact in completed.get("artifacts", [])
        for part in artifact["parts"]
        if part.get("mediaType") == OUT_RECEIPTS
    ]
    check("task COMPLETED", status == 200 and completed.get("status", {}).get("state") == "TASK_STATE_COMPLETED")
    check("exactly one receipts part", len(executions) == 1)
    accepted = [r for r in receipts if r["outcome"] == "ACCEPTED"]
    check("only accepted results executed", len(executions[0]) == len(accepted) if executions else False)
    check("history records the continuation", len(completed.get("history", [])) == 2)
    check("terminal replay is identical", client.call("GET", f"/tasks/{task['id']}")[1]["task"] == completed)
    second_status, _, _ = client.call(
        "POST",
        "/message:send",
        results_envelope(f"msg-{uuid.uuid4()}", task["id"], task["contextId"], batch["batchId"], receipts),
    )
    check("second continuation -> 409", second_status == 409)

    print("\n6. Isolation")
    check("outsider read -> 404", client.call("GET", f"/tasks/{task['id']}", token="grader-beta")[0] == 404)
    check("outsider cancel -> 404", client.call("POST", f"/tasks/{task['id']}:cancel", token="grader-beta")[0] == 404)
    check("outsider list is empty", client.call("GET", "/tasks", token="grader-beta")[1] == {"tasks": []})
    check("owner list contains the task", any(t["id"] == task["id"] for t in client.call("GET", "/tasks")[1]["tasks"]))
    outsider_body = json.dumps(client.call("GET", f"/tasks/{task['id']}", token="grader-beta")[1])
    check("outsider error does not echo the id", task["id"] not in outsider_body)

    print("\n7. Cancel versus receipt race")
    for attempt in range(3):
        task2 = client.call("POST", "/message:send", batch_envelope(f"msg-{uuid.uuid4()}", build_batch(batch_id=f"BATCH-RACE-{attempt}")))[1]["task"]
        race_results = [
            {"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
             "outcome": "ACCEPTED", "receiptNonce": f"nonce-{attempt}"}
            for p in task2["artifacts"][0]["parts"][0]["data"]["proposals"]
        ]
        envelope = results_envelope(f"msg-{uuid.uuid4()}", task2["id"], task2["contextId"], f"BATCH-RACE-{attempt}", race_results)
        outcome: dict[str, int] = {}

        def do_cancel():
            outcome["cancel"] = client.call("POST", f"/tasks/{task2['id']}:cancel")[0]

        def do_result():
            outcome["result"] = client.call("POST", "/message:send", envelope)[0]

        threads = [threading.Thread(target=do_cancel), threading.Thread(target=do_result)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=20)
        check(f"race {attempt + 1}: exactly one winner", sorted(outcome.values()) == [200, 409], str(outcome))
        final = client.call("GET", f"/tasks/{task2['id']}")[1]["task"]
        receipt_parts = [p for a in final["artifacts"] for p in a["parts"] if p.get("mediaType") == OUT_RECEIPTS]
        ok = (final["status"]["state"] == "TASK_STATE_COMPLETED" and len(receipt_parts) == 1) or (
            final["status"]["state"] == "TASK_STATE_CANCELED" and not receipt_parts
        )
        check(f"race {attempt + 1}: never both states", ok)

    httpd.shutdown()
    httpd.server_close()

    failures = [item for item in results if item[0] == FAIL]
    print(f"\n{len(results) - len(failures)}/{len(results)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
