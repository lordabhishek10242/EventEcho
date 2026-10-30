#!/usr/bin/env python3
"""Live transcript check for the mailroom action gate.

Starts the service in-process on a scratch database, then walks the graded
lifecycle: propose, exact replay, signed commit, commit replay, tampered
signature, cross-evaluation receipt, evaluation conflict and caching.

    python3 scripts/verify_mailroom_agent.py
"""

from __future__ import annotations

import base64
import copy
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

from mailroom_agent import ed25519, server as mailroom_server  # noqa: E402
from mailroom_agent.canonical import (  # noqa: E402
    PROFILE,
    digest_of,
    proposal_digest,
    receipt_signing_payload,
)
from mailroom_agent.decide import ACTIONS  # noqa: E402
from tests.mailroom_fixtures import EXPECTED, build_dossiers  # noqa: E402

results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((ok, name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{' — ' + detail if detail and not ok else ''}")


class Client:
    def __init__(self, url: str):
        self.url = url

    def post(self, payload: dict):
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode(), method="POST")
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
                return response.status, json.loads(raw), raw, response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            return exc.code, json.loads(raw or b"{}"), raw, exc.headers.get("Content-Type", "")


def main() -> int:
    mailroom_server.Handler.log_message = lambda *a, **k: None  # type: ignore[assignment]
    tmp = tempfile.mkdtemp(prefix="mailroom-verify-")
    ctx = mailroom_server.build_app(db_path=os.path.join(tmp, "verify.sqlite3"))
    httpd = mailroom_server.MailroomHTTPServer(("127.0.0.1", 0), ctx)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.03}, daemon=True).start()
    client = Client(f"http://127.0.0.1:{port}/")
    print(f"Mailroom action gate verification against http://127.0.0.1:{port}/")

    seed = os.urandom(32)
    _sig, public = ed25519.sign(seed, b"probe")
    jwk = {"kty": "OKP", "crv": "Ed25519", "x": base64.urlsafe_b64encode(public).decode().rstrip("=")}
    dossiers = build_dossiers()
    propose = {
        "profile": PROFILE,
        "operation": "propose",
        "evaluationId": f"eval-{uuid.uuid4()}",
        "receiptVerifier": {"algorithm": "Ed25519", "publicKeyJwk": jwk},
        "corpus": {"coreId": "core-1", "auditId": "audit-1", "stableCount": 58, "freshCount": 6},
        "allowedActions": list(ACTIONS),
        "dossiers": dossiers,
    }

    print("\n1. propose")
    started = time.time()
    status, body, raw, ctype = client.post(propose)
    elapsed = (time.time() - started) * 1000
    check("HTTP 200 + application/json", status == 200 and ctype == "application/json")
    check("profile/evaluationId echoed", body.get("profile") == PROFILE and body.get("evaluationId") == propose["evaluationId"])
    check("status awaiting_receipts", body.get("status") == "awaiting_receipts")
    check("inputDigest = sha256(canonical dossiers)", body.get("inputDigest") == digest_of(dossiers))
    check("one proposal per dossier", len(body.get("proposals", [])) == len(dossiers))
    check("callIds unique and well-formed", len({p["callId"] for p in body["proposals"]}) == len(dossiers)
          and all(len(p["callId"]) >= 12 for p in body["proposals"]))
    check("actions match the labelled truth", {p["dossierId"]: p["action"] for p in body["proposals"]} == EXPECTED)
    check("evidence is line-id only", all(
        isinstance(e, str) and e for p in body["proposals"] for e in p["evidence"]
    ))
    check("latency budget", elapsed < 5000, f"{elapsed:.0f} ms")
    print(f"        ({elapsed:.0f} ms for {len(dossiers)} dossiers, {len(raw)} bytes)")

    print("\n2. replay / conflict")
    replay_status, replay_body, replay_raw, _ = client.post(propose)
    check("exact replay is byte-identical", replay_status == 200 and replay_raw == raw)
    changed = copy.deepcopy(propose)
    changed["dossiers"][0]["objective"] = "changed objective"
    conflict_status, conflict_body, _, _ = client.post(changed)
    check("changed content -> 409", conflict_status == 409 and conflict_body["error"]["code"] == "EVALUATION_CONFLICT")

    print("\n3. commit")
    receipts = []
    for index, proposal in enumerate(body["proposals"]):
        receipt = {
            "dossierId": proposal["dossierId"],
            "callId": proposal["callId"],
            "action": proposal["action"],
            "accepted": index % 3 != 0,
            "proposalDigest": proposal_digest(proposal),
            "receiptId": f"rcpt-{uuid.uuid4()}",
        }
        message = receipt_signing_payload(PROFILE, body["evaluationId"], body["inputDigest"], receipt)
        signature, _public = ed25519.sign(seed, message)
        receipt["receiptSignature"] = base64.b64encode(signature).decode()
        receipts.append(receipt)
    commit = {
        "profile": PROFILE,
        "operation": "commit",
        "evaluationId": body["evaluationId"],
        "inputDigest": body["inputDigest"],
        "receipts": receipts,
    }
    commit_status, commit_body, _, _ = client.post(commit)
    check("commit -> completed", commit_status == 200 and commit_body.get("status") == "completed")
    check("one outcome per receipt", len(commit_body.get("outcomes", [])) == len(receipts))
    expected_statuses = ["rejected" if index % 3 == 0 else "executed" for index in range(len(receipts))]
    check("accepted receipts executed, rejected ones not",
          [o["status"] for o in commit_body["outcomes"]] == expected_statuses)
    check("outcome digests match proposals",
          [o["proposalDigest"] for o in commit_body["outcomes"]] == [r["proposalDigest"] for r in receipts])
    replay_status, replay_body, _, _ = client.post(commit)
    check("commit replay identical", replay_status == 200 and replay_body == commit_body)

    print("\n4. receipt safety")
    second_eval = copy.deepcopy(propose)
    second_eval["evaluationId"] = f"eval-{uuid.uuid4()}"
    _, second_body, _, _ = client.post(second_eval)
    foreign = {
        "profile": PROFILE,
        "operation": "commit",
        "evaluationId": second_body["evaluationId"],
        "inputDigest": second_body["inputDigest"],
        "receipts": [copy.deepcopy(receipts[0])],
    }
    foreign_status, foreign_body, _, _ = client.post(foreign)
    check("receipt signed for another evaluation is rejected", foreign_status == 422)

    tampered = copy.deepcopy(receipts[1])
    tampered["receiptId"] = "rcpt-tampered"
    bad = {
        "profile": PROFILE,
        "operation": "commit",
        "evaluationId": second_body["evaluationId"],
        "inputDigest": second_body["inputDigest"],
        "receipts": [tampered],
    }
    bad_status, bad_body, _, _ = client.post(bad)
    check("tampered receipt is rejected", bad_status == 422 and bad_body["error"]["code"] == "INVALID_SIGNATURE")
    _, still_open, _, _ = client.post(second_eval)
    check("rejected commit leaves the evaluation untouched", still_open["status"] == "awaiting_receipts")

    moved = {
        "profile": PROFILE,
        "operation": "commit",
        "evaluationId": second_body["evaluationId"],
        "inputDigest": second_body["inputDigest"],
        "receipts": [
            {**copy.deepcopy(receipts[0]), "callId": second_body["proposals"][1]["callId"]},
        ],
    }
    moved_status, _, _, _ = client.post(moved)
    check("moved receipt (wrong callId) is rejected", 400 <= moved_status < 500)

    print("\n5. schema")
    bad_profile = copy.deepcopy(propose)
    bad_profile["profile"] = "other/profile"
    check("wrong profile -> 422", client.post(bad_profile)[0] == 422)
    dup = copy.deepcopy(propose)
    dup["evaluationId"] = f"eval-{uuid.uuid4()}"
    dup["dossiers"].append(copy.deepcopy(dup["dossiers"][0]))
    check("duplicate dossierId -> 422", client.post(dup)[0] == 422)
    bad_op = copy.deepcopy(propose)
    bad_op["operation"] = "nope"
    check("unknown operation -> 400", client.post(bad_op)[0] == 400)
    unknown = dict(commit)
    unknown["evaluationId"] = f"eval-{uuid.uuid4()}"
    check("commit for unknown evaluation -> 409", client.post(unknown)[0] == 409)

    print("\n6. caching")
    cached_before = ctx.store.stats()["cachedDecisions"]
    third = copy.deepcopy(propose)
    third["evaluationId"] = f"eval-{uuid.uuid4()}"
    third_status, third_body, _, _ = client.post(third)
    check("repeat check reuses the dossier cache", ctx.store.stats()["cachedDecisions"] == cached_before)
    check("same dossiers -> same callIds",
          [p["callId"] for p in third_body["proposals"]] == [p["callId"] for p in body["proposals"]])
    check("new evaluation id still served", third_status == 200 and third_body["evaluationId"] == third["evaluationId"])

    httpd.shutdown()
    httpd.server_close()

    failures = [item for item in results if not item[0]]
    print(f"\n{len(results) - len(failures)}/{len(results)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
