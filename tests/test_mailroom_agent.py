"""Contract tests for the mailroom action gate.

Run with:  python3 -m unittest tests.test_mailroom_agent -v
"""

from __future__ import annotations

import base64
import copy
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mailroom_agent import ed25519, server as mailroom_server  # noqa: E402
from mailroom_agent.canonical import (  # noqa: E402
    PROFILE,
    canonical_json,
    digest_of,
    proposal_digest,
    receipt_signing_payload,
)
from mailroom_agent.decide import ACTIONS  # noqa: E402
from tests.mailroom_fixtures import build_dossiers  # noqa: E402

ALLOWED = list(ACTIONS)


def make_keypair() -> tuple[bytes, dict]:
    seed = os.urandom(32)
    signature, public = ed25519.sign(seed, b"probe")
    assert ed25519.verify(signature, b"probe", public)
    jwk = {
        "kty": "OKP",
        "crv": "Ed25519",
        "x": base64.urlsafe_b64encode(public).decode().rstrip("="),
    }
    return seed, jwk


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


class Client:
    def __init__(self, url: str):
        self.url = url

    def post(self, payload, raw: bytes | None = None):
        data = raw if raw is not None else json.dumps(payload).encode()
        request = urllib.request.Request(self.url, data=data, method="POST")
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
                return response.status, json.loads(body), body, response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            body = exc.read()
            try:
                parsed = json.loads(body or b"{}")
            except json.JSONDecodeError:
                parsed = {"raw": body.decode("utf-8", "replace")}
            return exc.code, parsed, body, exc.headers.get("Content-Type", "")


class MailroomCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mailroom_server.Handler.log_message = lambda *a, **k: None  # type: ignore[assignment]
        cls.tmpdir = tempfile.mkdtemp(prefix="mailroom-test-")
        ctx = mailroom_server.build_app(db_path=os.path.join(cls.tmpdir, "state.sqlite3"))
        cls.ctx = ctx
        cls.httpd = mailroom_server.MailroomHTTPServer(("127.0.0.1", 0), ctx)
        port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, kwargs={"poll_interval": 0.03}, daemon=True).start()
        cls.client = Client(f"http://127.0.0.1:{port}/")

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    # -- helpers ------------------------------------------------------- #
    def propose_payload(self, evaluation_id=None, dossiers=None, seed=None, jwk=None):
        if seed is None or jwk is None:
            seed, jwk = make_keypair()
        payload = {
            "profile": PROFILE,
            "operation": "propose",
            "evaluationId": evaluation_id or f"eval-{uuid.uuid4()}",
            "receiptVerifier": {"algorithm": "Ed25519", "publicKeyJwk": jwk},
            "corpus": {"coreId": "core-1", "auditId": f"audit-{uuid.uuid4()}", "stableCount": 58, "freshCount": 6},
            "allowedActions": ALLOWED,
            "dossiers": copy.deepcopy(dossiers if dossiers is not None else build_dossiers()),
        }
        return payload, seed

    def sign_receipt(self, seed, evaluation_id, input_digest, receipt):
        message = receipt_signing_payload(PROFILE, evaluation_id, input_digest, receipt)
        signature, _public = ed25519.sign(seed, message)
        return b64(signature)

    def commit_payload(self, propose_response, seed, accepted_map=None, mutate=None):
        evaluation_id = propose_response["evaluationId"]
        input_digest = propose_response["inputDigest"]
        receipts = []
        for proposal in propose_response["proposals"]:
            accepted = True
            if accepted_map is not None:
                accepted = accepted_map.get(proposal["dossierId"], True)
            receipt = {
                "dossierId": proposal["dossierId"],
                "callId": proposal["callId"],
                "action": proposal["action"],
                "accepted": accepted,
                "proposalDigest": proposal_digest(proposal),
                "receiptId": f"rcpt-{uuid.uuid4()}",
            }
            if mutate is not None:
                receipt = mutate(receipt, proposal)
            receipt["receiptSignature"] = self.sign_receipt(seed, evaluation_id, input_digest, receipt)
            receipts.append(receipt)
        return {
            "profile": PROFILE,
            "operation": "commit",
            "evaluationId": evaluation_id,
            "inputDigest": input_digest,
            "receipts": receipts,
        }

    # -- propose -------------------------------------------------------- #
    def test_propose_envelope(self):
        payload, _seed = self.propose_payload()
        status, body, _raw, ctype = self.client.post(payload)
        self.assertEqual(status, 200, body)
        self.assertEqual(ctype, "application/json")
        self.assertEqual(body["profile"], PROFILE)
        self.assertEqual(body["evaluationId"], payload["evaluationId"])
        self.assertEqual(body["status"], "awaiting_receipts")
        self.assertEqual(body["inputDigest"], digest_of(payload["dossiers"]))
        self.assertEqual(len(body["proposals"]), len(payload["dossiers"]))
        self.assertEqual(
            [p["dossierId"] for p in body["proposals"]], [d["dossierId"] for d in payload["dossiers"]]
        )
        call_ids = [p["callId"] for p in body["proposals"]]
        self.assertEqual(len(call_ids), len(set(call_ids)))
        line_ids = {
            line["lineId"]
            for dossier in payload["dossiers"]
            for source in dossier["sources"]
            for line in source["lines"]
        }
        for proposal in body["proposals"]:
            self.assertIn(proposal["action"], ALLOWED)
            self.assertRegex(proposal["callId"], r"^[A-Za-z0-9._:\-]{12,128}$")
            self.assertIn("target", proposal)
            self.assertIn("payload", proposal)
            self.assertTrue(proposal["evidence"])
            self.assertEqual(len(proposal["evidence"]), len(set(proposal["evidence"])))
            for line_id in proposal["evidence"]:
                self.assertIn(line_id, line_ids)
            self.assertNotIn("_proposalDigest", proposal)

    def test_target_and_payload_schemas(self):
        payload, _seed = self.propose_payload()
        _, body, _, _ = self.client.post(payload)
        for proposal in body["proposals"]:
            action = proposal["action"]
            if action == "no_action":
                self.assertIsNone(proposal["target"])
                self.assertEqual(set(proposal["payload"]), {"reasonCode", "referenceId"})
                self.assertIn(proposal["payload"]["reasonCode"], {"ALREADY_COMPLETED", "DUPLICATE", "INFORMATIONAL"})
            elif action == "create_draft":
                self.assertEqual(proposal["target"]["kind"], "draft_queue")
                self.assertTrue(proposal["target"]["id"].startswith("mailbox:"))
                self.assertEqual(set(proposal["payload"]), {"recipient", "referenceId", "status", "template"})
                self.assertEqual(proposal["payload"]["template"], "order_status")
            elif action == "update_internal_record":
                self.assertEqual(proposal["target"]["kind"], "case_record")
                self.assertEqual(set(proposal["payload"]), {"field", "sourceEventId", "value"})
                self.assertEqual(proposal["payload"]["field"], "delivery_window")
            elif action == "send_approved_notice":
                self.assertEqual(proposal["target"]["kind"], "email")
                self.assertEqual(set(proposal["payload"]), {"referenceId", "status", "template"})
                self.assertEqual(proposal["payload"]["template"], "approved_delivery_notice")
            elif action == "request_confirmation":
                self.assertEqual(proposal["target"]["kind"], "approval_queue")
                self.assertEqual(set(proposal["payload"]), {"claimedSender", "questionCode", "referenceId"})
                self.assertEqual(proposal["payload"]["questionCode"], "VERIFY_REQUEST")
            elif action == "quarantine_item":
                self.assertEqual(proposal["target"], {"kind": "security_queue", "id": "mailroom"})
                self.assertEqual(set(proposal["payload"]), {"artifactId", "reasonCode"})
                self.assertEqual(proposal["payload"]["reasonCode"], "INDIRECT_PROMPT_INJECTION")

    def test_expected_actions(self):
        from tests.mailroom_fixtures import EXPECTED

        payload, _seed = self.propose_payload()
        _, body, _, _ = self.client.post(payload)
        got = {p["dossierId"]: p["action"] for p in body["proposals"]}
        self.assertEqual(got, EXPECTED)

    def test_exact_replay_is_byte_identical(self):
        payload, _seed = self.propose_payload()
        status_a, body_a, raw_a, _ = self.client.post(payload)
        self.assertEqual(status_a, 200)
        status_b, body_b, raw_b, _ = self.client.post(payload)
        self.assertEqual(status_b, 200)
        self.assertEqual(body_a, body_b)
        self.assertEqual(raw_a, raw_b)

    def test_changed_content_same_evaluation_conflicts(self):
        payload, _seed = self.propose_payload()
        self.client.post(payload)
        changed = copy.deepcopy(payload)
        changed["dossiers"][0]["objective"] = "something else entirely"
        status, body, _, _ = self.client.post(changed)
        self.assertEqual(status, 409, body)
        self.assertEqual(body["error"]["code"], "EVALUATION_CONFLICT")

    def test_cached_decisions_across_evaluations(self):
        dossiers = build_dossiers()
        payload_a, _ = self.propose_payload(evaluation_id=f"eval-{uuid.uuid4()}", dossiers=dossiers)
        _, body_a, _, _ = self.client.post(payload_a)
        cached_after_a = self.ctx.store.stats()["cachedDecisions"]
        self.assertEqual(cached_after_a, len(dossiers))

        payload_b, _ = self.propose_payload(evaluation_id=f"eval-{uuid.uuid4()}", dossiers=dossiers)
        _, body_b, _, _ = self.client.post(payload_b)
        self.assertEqual(self.ctx.store.stats()["cachedDecisions"], cached_after_a)
        self.assertEqual(
            [p["callId"] for p in body_a["proposals"]], [p["callId"] for p in body_b["proposals"]]
        )
        self.assertEqual(
            [p["action"] for p in body_a["proposals"]], [p["action"] for p in body_b["proposals"]]
        )
        self.assertNotEqual(body_a["evaluationId"], body_b["evaluationId"])

    def test_fresh_audit_dossier_is_decided(self):
        core = build_dossiers()
        audit = copy.deepcopy(core[0])
        audit["dossierId"] = "AUD-0001"
        audit["partition"] = "fresh_audit"
        audit["sources"][0]["lines"][2]["text"] = "Please confirm the status of order ORD-2026-4417."
        payload, _ = self.propose_payload(dossiers=core + [audit])
        status, body, _, _ = self.client.post(payload)
        self.assertEqual(status, 200)
        self.assertEqual(len(body["proposals"]), len(core) + 1)
        self.assertEqual(body["proposals"][-1]["dossierId"], "AUD-0001")
        self.assertEqual(body["proposals"][-1]["action"], "create_draft")

    # -- commit --------------------------------------------------------- #
    def test_commit_records_outcomes(self):
        payload, seed = self.propose_payload()
        _, proposed, _, _ = self.client.post(payload)
        accepted_map = {
            proposal["dossierId"]: index % 2 == 0 for index, proposal in enumerate(proposed["proposals"])
        }
        commit = self.commit_payload(proposed, seed, accepted_map=accepted_map)
        status, body, _, _ = self.client.post(commit)
        self.assertEqual(status, 200, body)
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["evaluationId"], proposed["evaluationId"])
        self.assertEqual(body["inputDigest"], proposed["inputDigest"])
        self.assertEqual(len(body["outcomes"]), len(proposed["proposals"]))
        for outcome, proposal in zip(body["outcomes"], proposed["proposals"]):
            self.assertEqual(outcome["dossierId"], proposal["dossierId"])
            self.assertEqual(outcome["callId"], proposal["callId"])
            self.assertEqual(outcome["action"], proposal["action"])
            self.assertEqual(outcome["proposalDigest"], proposal_digest(proposal))
            self.assertEqual(
                outcome["status"], "executed" if accepted_map[proposal["dossierId"]] else "rejected"
            )
            self.assertTrue(outcome["receiptId"])

        replay_status, replay_body, _, _ = self.client.post(commit)
        self.assertEqual(replay_status, 200)
        self.assertEqual(replay_body, body)

    def test_commit_after_completed_with_new_receipts_conflicts(self):
        payload, seed = self.propose_payload()
        _, proposed, _, _ = self.client.post(payload)
        self.client.post(self.commit_payload(proposed, seed))
        second = self.commit_payload(proposed, seed)
        status, body, _, _ = self.client.post(second)
        self.assertEqual(status, 409, body)

    def test_tampered_signature_is_rejected(self):
        payload, seed = self.propose_payload()
        _, proposed, _, _ = self.client.post(payload)
        commit = self.commit_payload(proposed, seed)
        commit["receipts"][0]["receiptId"] = "rcpt-swapped-after-signing"
        status, body, _, _ = self.client.post(commit)
        self.assertEqual(status, 422, body)
        self.assertEqual(body["error"]["code"], "INVALID_SIGNATURE")
        # the evaluation is still open and the propose replay unchanged
        _, replay, _, _ = self.client.post(payload)
        self.assertEqual(replay, proposed)

    def test_signature_from_another_evaluation_is_rejected(self):
        payload_a, seed_a = self.propose_payload(evaluation_id=f"eval-a-{uuid.uuid4()}")
        _, proposed_a, _, _ = self.client.post(payload_a)
        payload_b, _seed_b = self.propose_payload(evaluation_id=f"eval-b-{uuid.uuid4()}")
        _, proposed_b, _, _ = self.client.post(payload_b)
        # valid signature, but minted for evaluation A's digest
        commit = self.commit_payload(proposed_b, seed_a)
        status, body, _, _ = self.client.post(commit)
        self.assertEqual(status, 422, body)

    def test_receipt_for_another_proposal_is_rejected(self):
        payload, seed = self.propose_payload()
        _, proposed, _, _ = self.client.post(payload)
        commit = self.commit_payload(proposed, seed)
        first, second = commit["receipts"][0], commit["receipts"][1]
        first["callId"], second["callId"] = second["callId"], first["callId"]
        status, body, _, _ = self.client.post(commit)
        self.assertEqual(status, 409, body)

    def test_wrong_action_or_digest_is_rejected(self):
        payload, seed = self.propose_payload()
        _, proposed, _, _ = self.client.post(payload)

        commit = self.commit_payload(proposed, seed)
        commit["receipts"][0]["action"] = "no_action"
        self.assertEqual(self.client.post(commit)[0], 409)

        commit = self.commit_payload(proposed, seed)
        commit["receipts"][0]["proposalDigest"] = "0" * 64
        self.assertEqual(self.client.post(commit)[0], 409)

        commit = self.commit_payload(proposed, seed)
        commit["inputDigest"] = "f" * 64
        self.assertEqual(self.client.post(commit)[0], 409)

    def test_unknown_evaluation_is_rejected(self):
        _payload, seed = self.propose_payload()
        payload, _ = self.propose_payload()
        _, proposed, _, _ = self.client.post(payload)
        commit = self.commit_payload(proposed, seed)
        commit["evaluationId"] = f"eval-unknown-{uuid.uuid4()}"
        status, _, _, _ = self.client.post(commit)
        self.assertEqual(status, 409)

    def test_duplicate_receipts_are_rejected(self):
        payload, seed = self.propose_payload()
        _, proposed, _, _ = self.client.post(payload)
        commit = self.commit_payload(proposed, seed)
        commit["receipts"].append(copy.deepcopy(commit["receipts"][0]))
        status, _, _, _ = self.client.post(commit)
        self.assertEqual(status, 409)

    # -- schema / robustness -------------------------------------------- #
    def test_malformed_requests_are_rejected_before_work(self):
        evaluation_id = f"eval-{uuid.uuid4()}"
        payload, _ = self.propose_payload(evaluation_id=evaluation_id)

        status, body, _, _ = self.client.post(payload, raw=b"{not json")
        self.assertEqual(status, 400)

        bad = copy.deepcopy(payload)
        bad["operation"] = "explode"
        self.assertEqual(self.client.post(bad)[0], 400)

        bad = copy.deepcopy(payload)
        bad["profile"] = "other/profile"
        self.assertEqual(self.client.post(bad)[0], 422)

        bad = copy.deepcopy(payload)
        bad["dossiers"].append(copy.deepcopy(bad["dossiers"][0]))
        self.assertEqual(self.client.post(bad)[0], 422)

        bad = copy.deepcopy(payload)
        bad["dossiers"][0]["sources"][0]["lines"].append(
            copy.deepcopy(bad["dossiers"][0]["sources"][0]["lines"][0])
        )
        self.assertEqual(self.client.post(bad)[0], 422)

        bad = copy.deepcopy(payload)
        bad["receiptVerifier"]["publicKeyJwk"]["x"] = "not-a-key"
        self.assertEqual(self.client.post(bad)[0], 422)

        bad = copy.deepcopy(payload)
        del bad["dossiers"][0]["sources"]
        self.assertEqual(self.client.post(bad)[0], 400)

        # none of the rejected requests created an evaluation
        self.assertIsNone(self.ctx.store.get_evaluation(evaluation_id))

    def test_oversized_body_is_rejected(self):
        payload, _ = self.propose_payload()
        payload["padding"] = "x" * (17 * 1024 * 1024)
        try:
            status, _, _, _ = self.client.post(payload)
        except urllib.error.URLError:
            # a reset connection is also a rejection; the body was never parsed
            return
        self.assertIn(status, (400, 413))

    def test_response_size_stays_small(self):
        core = build_dossiers()
        dossiers = []
        for index in range(70):
            clone = copy.deepcopy(core[index % len(core)])
            clone["dossierId"] = f"DOS-{index:04d}"
            dossiers.append(clone)
        payload, _ = self.propose_payload(dossiers=dossiers)
        status, body, raw, _ = self.client.post(payload)
        self.assertEqual(status, 200)
        self.assertLess(len(raw), 512 * 1024)
        self.assertEqual(len(body["proposals"]), 70)
        self.assertEqual(len({p["callId"] for p in body["proposals"]}), 70)


class Ed25519Case(unittest.TestCase):
    def test_rfc8032_vectors(self):
        vectors = [
            (
                "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a",
                "",
                "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b",
            ),
            (
                "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c",
                "72",
                "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00",
            ),
        ]
        for public, message, signature in vectors:
            self.assertTrue(
                ed25519.verify(bytes.fromhex(signature), bytes.fromhex(message), bytes.fromhex(public))
            )
            self.assertFalse(
                ed25519.verify(bytes.fromhex(signature), bytes.fromhex(message) + b"x", bytes.fromhex(public))
            )

    def test_canonical_json_is_key_sorted_and_compact(self):
        self.assertEqual(canonical_json({"b": 1, "a": [2, {"d": 4, "c": 3}]}), '{"a":[2,{"c":3,"d":4}],"b":1}')

    def test_proposal_digest_sorts_evidence_only(self):
        proposal = {
            "dossierId": "D1",
            "callId": "call-abc",
            "action": "no_action",
            "target": None,
            "payload": {"reasonCode": "DUPLICATE", "referenceId": "ORD-1"},
            "evidence": ["L2", "L1"],
            "ignored": "extra",
        }
        reordered = dict(proposal)
        reordered["evidence"] = ["L1", "L2"]
        self.assertEqual(proposal_digest(proposal), proposal_digest(reordered))


if __name__ == "__main__":
    unittest.main()
