"""R0.7 field-intake contracts: synthetic images and fixture bytes only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from phistream.mcp_gateway import MCPGateError, _digest
from phistream.mcp_unreal_qualification import MARKER_SCHEMA
from phistream.mcp_unreal_actor_pilot import (
    RECEIPT, SCHEMA as PLAN_SCHEMA,
)
from phistream.mcp_unreal_field_qualification import (
    FIELDS, IMAGES, PACKET, SCHEMA, create_template, field_assessment,
    screenshot_bundle,
)
from phistream.mcp_unreal_recovery import (
    capture_baseline, restore_to_new_copy,
)

ENDPOINT = "http://127.0.0.1:8000/mcp"
TOOL = "EditorToolset.ActorToolset.SpawnActor"


def png_fixture() -> bytes:
    # Sufficient for bounded PNG header hashing, NOT a genuine captured image.
    return (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" +
            struct.pack(">II", 1280, 720) + b"\x08\x02\x00\x00\x00" +
            b"\x00\x00\x00\x00" + b"trailing-fixture")


class R07Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.source = self.base / "PhiMCPSandboxR07"
        self.source.mkdir()
        self.uproject = self.source / "PhiMCPSandboxR07.uproject"
        self.uproject.write_text('{"EngineAssociation":"5.9"}', encoding="utf-8")
        self.marker = self.source / "PHI_MCP_DISPOSABLE.json"
        self.marker.write_text(json.dumps({
            "schema": MARKER_SCHEMA, "project_filename": self.uproject.name,
            "project_sha256": self.sha(self.uproject), "disposable": True,
        }), encoding="utf-8")
        (self.source / "Content").mkdir()
        self.map = self.source / "Content" / "Scratch.umap"
        self.map.write_bytes(b"UE_FIXTURE_BEFORE")
        self.snapshot_parent = self.base / "baseline-store"
        self.snapshot_parent.mkdir()
        self.snapshot = self.snapshot_parent / "r06-r07field"
        capture_baseline(self.source, self.snapshot_parent, "r06-r07field")
        self.recovery_parent = self.base / "recovery"
        self.recovery_parent.mkdir()
        restore_to_new_copy(self.snapshot, self.recovery_parent)
        self.recovered = self.recovery_parent / self.source.name
        self.private = self.base / "private"
        self.private.mkdir()
        self.intents = self.private / "intents"
        self.intents.mkdir()
        self.screens = self.private / "images"
        self.screens.mkdir()
        self.operator = self.private / "operator.json"
        self.plan_path = self.private / "plan.json"
        self.receipt_path = self.private / "receipt.json"
        self.plan = {
            "schema": PLAN_SCHEMA,
            "project_sha256": self.sha(self.uproject),
            "marker_sha256": self.sha(self.marker),
            "expected_instance_id": "mock-369",
            "expected_process_id": 369,
            "tool_name": TOOL,
            "tool_schema_sha256": "a" * 64,
            "arguments": {
                "actor_class": "/Script/Engine.PointLight",
                "actor_label": "PhiMCPProbe_R07",
            },
        }
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        self.intent_hash = _digest({
            "endpoint": ENDPOINT, "plan": self.plan, "root": str(self.source.resolve()),
        })
        (self.intents / (self.intent_hash + ".attempt")).write_text(json.dumps({
            "schema": RECEIPT, "status": "ATTEMPT_RESERVED_UNKNOWN",
            "intent_sha256": self.intent_hash,
        }), encoding="utf-8")
        self.receipt = {
            "schema": RECEIPT,
            "status": "WRITE_SENT_MANUAL_EDITOR_REVIEW_REQUIRED",
            "intent_sha256": self.intent_hash,
            "tool_name": TOOL,
            "arguments_sha256": _digest(self.plan["arguments"]),
            "tool_reported_error": False,
            "tool_calls_sent": 1,
            "actor_exists_verified": False,
            "rollback_verified": False,
            "automated_retry_allowed": False,
            "production_authorized": False,
        }
        self.receipt_path.write_text(json.dumps(self.receipt), encoding="utf-8")
        self.record = {
            "schema": SCHEMA,
            "expected_actor_label": "PhiMCPProbe_R07",
            "observations": {field: True for field in FIELDS},
        }
        self.write_record()
        for name in IMAGES:
            (self.screens / name).write_bytes(png_fixture())
        self.map.write_bytes(b"UE_FIXTURE_AFTER_SAVED")

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_record(self):
        self.operator.write_text(json.dumps(self.record), encoding="utf-8")

    def assess(self, **kwargs):
        opts = {
            "snapshot_dir": self.snapshot,
            "project_root": self.source,
            "recovered_project_root": self.recovered,
            "plan_path": self.plan_path,
            "intent_dir": self.intents,
            "endpoint": ENDPOINT,
            "operator_record": self.operator,
            "screenshot_dir": self.screens,
            "receipt_path": self.receipt_path,
        }
        opts.update(kwargs)
        return field_assessment(**opts)

    def test_strong_local_consistency_still_not_editor_attestation(self):
        result = self.assess()
        self.assertEqual(result["schema"], PACKET)
        self.assertEqual(result["status"],
                         "OPERATOR_REPORTED_FIELD_CHECKS_CONSISTENT_UNVERIFIED")
        self.assertEqual(result["attempt_intent_sha256"], self.intent_hash)
        self.assertTrue(result["saved_umap_delta_observed"])
        self.assertTrue(result["operator_reported_all_checks_passed"])
        self.assertTrue(result["recovered_disk_subset_matches_baseline"])
        self.assertTrue(result["r05_receipt_correlated_as_self_report"])
        self.assertEqual(len(result["screenshot_evidence"]), 4)
        self.assertFalse(result["actor_existence_automatically_verified"])
        self.assertFalse(result["actual_editor_session_authenticated"])
        self.assertFalse(result["field_test_independently_qualified"])
        self.assertFalse(result["production_authorized"])
        self.assertFalse(result["merge_authorized"])
        self.assertFalse(result["automatic_retry_authorized"])
        self.assertNotIn(str(self.source), json.dumps(result))
        self.assertNotIn("UE_FIXTURE_AFTER", json.dumps(result))

    def test_incomplete_manual_observation_holds(self):
        self.record["observations"][FIELDS[0]] = False
        self.write_record()
        result = self.assess()
        self.assertEqual(result["status"], "HOLD_FIELD_CHECKS_INCOMPLETE_OR_CONFLICTING")
        self.assertFalse(result["operator_reported_all_checks_passed"])

    def test_screenshot_digest_is_not_image_content_validation(self):
        result = self.assess()
        for picture in result["screenshot_evidence"].values():
            self.assertFalse(picture["pixel_contents_verified"])
            self.assertFalse(picture["real_editor_origin_verified"])
            self.assertEqual(picture["reported_dimensions"], [1280, 720])
            self.assertEqual(picture["sha256"], hashlib.sha256(png_fixture()).hexdigest())

    def test_no_saved_map_delta_is_hold_even_if_operator_claims_yes(self):
        self.map.write_bytes(b"UE_FIXTURE_BEFORE")
        result = self.assess()
        self.assertFalse(result["saved_umap_delta_observed"])
        self.assertEqual(result["status"], "HOLD_FIELD_CHECKS_INCOMPLETE_OR_CONFLICTING")

    def test_missing_or_failed_r05_receipt_cannot_qualify(self):
        result = self.assess(receipt_path=None)
        self.assertEqual(result["status"], "HOLD_FIELD_CHECKS_INCOMPLETE_OR_CONFLICTING")
        self.receipt["tool_reported_error"] = True
        self.receipt_path.write_text(json.dumps(self.receipt), encoding="utf-8")
        result = self.assess()
        self.assertTrue(result["tool_reported_error_untrusted"])
        self.assertEqual(result["status"], "HOLD_FIELD_CHECKS_INCOMPLETE_OR_CONFLICTING")

    def test_recovered_copy_tamper_is_rejected(self):
        restored_map = self.recovered / "Content" / "Scratch.umap"
        restored_map.write_bytes(b"UNEXPECTED")
        with self.assertRaisesRegex(MCPGateError, "recovered copy"):
            self.assess()

    def test_wrong_or_missing_evidence_is_rejected(self):
        self.record["expected_actor_label"] = "PhiMCPProbe_wrong"
        self.write_record()
        with self.assertRaisesRegex(MCPGateError, "different planned actor"):
            self.assess()
        self.record["expected_actor_label"] = "PhiMCPProbe_R07"
        self.record["observations"].pop(FIELDS[0])
        self.write_record()
        with self.assertRaisesRegex(MCPGateError, "contract invalid"):
            self.assess()
        self.record["observations"][FIELDS[0]] = True
        self.write_record()
        (self.screens / IMAGES[0]).unlink()
        with self.assertRaisesRegex(MCPGateError, "screenshot"):
            self.assess()

    def test_fake_png_header_symlink_and_outside_parent(self):
        file = self.screens / IMAGES[0]
        file.write_bytes(b"not an image, just text"*3)
        with self.assertRaisesRegex(MCPGateError, "PNG"):
            self.assess()
        file.unlink()
        try:
            file.symlink_to(self.screens / IMAGES[1])
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        with self.assertRaisesRegex(MCPGateError, "symlink"):
            self.assess()
        with self.assertRaisesRegex(MCPGateError, "separate"):
            screenshot_bundle(self.source, separate_from=(
                self.source, self.recovered, self.snapshot))

    def test_production_or_original_as_recovery_refused(self):
        with self.assertRaises(MCPGateError):
            self.assess(recovered_project_root=self.source)

    def test_project_anchor_change_prevents_consistent_candidate(self):
        # R0.6 compares anchors without assuming the editor did nothing.
        self.uproject.write_text("changed", encoding="utf-8")
        result = self.assess()
        self.assertFalse(result["original_project_anchors_unchanged"])
        self.assertEqual(result["status"], "HOLD_FIELD_CHECKS_INCOMPLETE_OR_CONFLICTING")

    def test_forged_r05_receipt_permission_is_rejected(self):
        self.receipt["actor_exists_verified"] = True
        self.receipt_path.write_text(json.dumps(self.receipt), encoding="utf-8")
        with self.assertRaisesRegex(MCPGateError, "unsupported authority"):
            self.assess()

    def test_template_defaults_false_and_never_overwrites(self):
        dest = self.private / "form-template.json"
        output = create_template(dest, self.plan_path)
        self.assertEqual(output["status"], "EMPTY_OPERATOR_FORM_CREATED")
        data = json.loads(dest.read_text(encoding="utf-8"))
        self.assertFalse(any(data["observations"].values()))
        self.assertEqual(data["expected_actor_label"], "PhiMCPProbe_R07")
        with self.assertRaisesRegex(MCPGateError, "never overwrite"):
            create_template(dest, self.plan_path)


if __name__ == "__main__":
    unittest.main()
