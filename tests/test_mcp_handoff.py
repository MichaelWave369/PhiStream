"""MCP R0.2: test read-only evidence imports never turn into authority."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from phistream.mcp_gateway import MANIFEST_SCHEMA, MCPGateError, load_manifest
from phistream.mcp_handoff import (
    AGENTORA_SCHEMA, PHIMADA_SCHEMA, SCHEMA, read_receipt, review_handoff,
)

SHA = "a" * 40
OTHER_SHA = "b" * 40


class HandoffTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(json.dumps({
            "schema": MANIFEST_SCHEMA, "servers": {"local": {
                "command": sys.executable, "allowed_tools": {
                    "inspect": "read", "dangerous": "write",
                },
            }},
        }), encoding="utf-8")
        self.manifest = load_manifest(self.manifest_path)
        self.agentora = {
            "schema": AGENTORA_SCHEMA,
            "scope": "public_only", "source_scan_id": 4,
            "actual_jobs_dispatched": 0,
            "merge_authorized": False, "verified_complete": False,
            "missions": [{
                "mission_id": "test-001", "github_id": 1729,
                "repository": "MichaelWave369/PhiMadaPrime",
                "kind": "PUBLIC_STANDALONE_READINESS_AUDIT",
                "state": "QUEUED", "worker": None, "reviewer": None,
                "source_sha": SHA,
            }],
        }
        self.phimada = {
            "schema": PHIMADA_SCHEMA,
            "state": "LIVE_GITHUB_R1_RETURN_CORRELATED",
            "relation": "VERIFIED_TESTED_PR_HEAD",
            "repository": "MichaelWave369/Agentora",
            "repositoryId": 2727, "prNumber": 32,
            "selectedSha": SHA, "testedSha": SHA,
            "sourceSha": OTHER_SHA, "mergeSha": None,
            "agentIdentityAuthenticated": False,
            "aiModelExecutionProven": False,
            "browserExecutedWork": False,
            "productVerifiedComplete": False,
            "mergeAuthorized": False,
            "dispatchAuthorized": False,
        }

    def review_agentora(self, **kwargs):
        return review_handoff(
            self.agentora, kind="agentora-r9", manifest=self.manifest,
            server="local", tool="inspect", expected_sha=SHA,
            mission_id="test-001", **kwargs,
        )

    def test_agentora_review_bound_to_sha_tool_manifest_and_no_authority(self):
        result = self.review_agentora()
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["status"], "HELD_OPERATOR_REVIEW_ONLY")
        self.assertEqual(result["evidence"]["work_id"], "test-001")
        self.assertEqual(result["evidence"]["repository"],
                         "MichaelWave369/PhiMadaPrime")
        self.assertEqual(result["proposed_mcp"]["operator_declared_risk"], "read")
        self.assertEqual(len(result["bindings"]["source_receipt_sha256"]), 64)
        for key in ("execution_authorized", "server_launch_authorized",
                    "dispatch_authorized", "code_write_authorized",
                    "merge_authorized", "deployment_authorized"):
            self.assertIs(result[key], False)
        self.assertFalse(any(result["trust"].values()))

    def test_selected_sha_must_match_exact_mission_source(self):
        with self.assertRaisesRegex(MCPGateError, "SHA"):
            review_handoff(
                self.agentora, kind="agentora-r9", manifest=self.manifest,
                server="local", tool="inspect", expected_sha=OTHER_SHA,
                mission_id="test-001",
            )

    def test_agentora_held_assigned_or_qa_cannot_be_promoted(self):
        for state in ("ASSIGNED_IMPLEMENT", "QA_HELD", "ARCHIVED_NOT_APPROVED",
                      "UNKNOWN", "QA_QUEUED"):
            with self.subTest(state=state):
                self.agentora["missions"][0]["state"] = state
                with self.assertRaises(MCPGateError):
                    self.review_agentora()

    def test_agentora_no_duplicate_or_spoofed_jobs_or_approvals(self):
        cases = [
            ("missions", self.agentora["missions"] * 2),
            ("actual_jobs_dispatched", 1),
            ("merge_authorized", True),
            ("verified_complete", True),
            ("schema", "evil.source"),
            ("scope", "private"),
        ]
        for field, bad in cases:
            with self.subTest(field=field):
                original = self.agentora[field]
                self.agentora[field] = bad
                with self.assertRaises(MCPGateError):
                    self.review_agentora()
                self.agentora[field] = original

    def test_unknown_tool_cannot_be_requested_even_read_only(self):
        with self.assertRaisesRegex(MCPGateError, "allowlisted"):
            review_handoff(
                self.agentora, kind="agentora-r9", manifest=self.manifest,
                server="local", tool="unreviewed", expected_sha=SHA,
                mission_id="test-001",
            )

    def test_write_tool_proposal_still_never_approves_write(self):
        result = review_handoff(
            self.agentora, kind="agentora-r9", manifest=self.manifest,
            server="local", tool="dangerous", expected_sha=SHA,
            mission_id="test-001",
        )
        self.assertEqual(result["proposed_mcp"]["operator_declared_risk"], "write")
        self.assertFalse(result["code_write_authorized"])
        self.assertFalse(result["execution_authorized"])

    def test_phimada_r31_supports_tested_sha_without_promoting_merge(self):
        result = review_handoff(
            self.phimada, kind="phimada-r31", manifest=self.manifest,
            server="local", tool="inspect", expected_sha=SHA,
        )
        self.assertEqual(result["evidence"]["source"], "phimada-r31-chronicle")
        self.assertFalse(result["merge_authorized"])
        self.assertFalse(result["trust"]["imported_snapshot_authenticated"])
        self.assertEqual(result["evidence"]["work_id"], "agentora-pr-32")

    def test_phimada_verified_merge_relation_is_still_only_a_snapshot(self):
        self.phimada["relation"] = "VERIFIED_MERGE_COMMIT"
        self.phimada["mergeSha"] = OTHER_SHA
        self.phimada["selectedSha"] = OTHER_SHA
        result = review_handoff(
            self.phimada, kind="phimada-r31", manifest=self.manifest,
            server="local", tool="inspect", expected_sha=OTHER_SHA,
        )
        self.assertEqual(result["evidence"]["source_sha"], OTHER_SHA)
        self.assertFalse(result["merge_authorized"])
        self.assertFalse(result["deployment_authorized"])

    def test_phimada_spoofed_permissions_or_mismatched_sha_rejected(self):
        for field, bad in [
            ("dispatchAuthorized", True), ("mergeAuthorized", True),
            ("agentIdentityAuthenticated", True),
            ("aiModelExecutionProven", True),
            ("productVerifiedComplete", True),
            ("selectedSha", OTHER_SHA), ("repository", "Elsewhere/Fake"),
        ]:
            with self.subTest(field=field):
                old = self.phimada[field]
                self.phimada[field] = bad
                with self.assertRaises(MCPGateError):
                    review_handoff(
                        self.phimada, kind="phimada-r31",
                        manifest=self.manifest, server="local",
                        tool="inspect", expected_sha=SHA,
                    )
                self.phimada[field] = old

    def test_read_receipt_is_bounded_json_and_rejects_symlinks(self):
        path = self.root / "good.json"
        path.write_text(json.dumps(self.agentora), encoding="utf-8")
        self.assertEqual(read_receipt(path)["schema"], AGENTORA_SCHEMA)
        path.write_text("{bad json", encoding="utf-8")
        with self.assertRaises(MCPGateError):
            read_receipt(path)
        path.write_text("x" * (128 * 1024 + 1), encoding="utf-8")
        with self.assertRaises(MCPGateError):
            read_receipt(path)
        link = self.root / "link.json"
        try:
            link.symlink_to(path)
        except (OSError, NotImplementedError):
            return
        with self.assertRaises(MCPGateError):
            read_receipt(link)

    def test_duplicate_invalid_mission_id_and_foreign_type_fail_closed(self):
        with self.assertRaises(MCPGateError):
            review_handoff(
                self.agentora, kind="agentora-r9", manifest=self.manifest,
                server="local", tool="inspect", expected_sha=SHA,
                mission_id="bad id",
            )
        with self.assertRaises(MCPGateError):
            review_handoff(
                self.agentora, kind="surprise", manifest=self.manifest,
                server="local", tool="inspect", expected_sha=SHA,
                mission_id="test-001",
            )


if __name__ == "__main__":
    unittest.main()
