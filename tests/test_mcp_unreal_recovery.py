"""PhiStream R0.6 offline evidence / scratch recovery regression matrix."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from phistream.mcp_gateway import MCPGateError, _digest
from phistream.mcp_unreal_qualification import MARKER_SCHEMA
from phistream.mcp_unreal_actor_pilot import (
    RECEIPT, SCHEMA as PLAN_SCHEMA,
)
from phistream.mcp_unreal_recovery import (
    REPORT, RESTORE, capture_baseline, compare_baseline,
    inventory, read_baseline, review_actor_attempt, restore_to_new_copy,
)

ENDPOINT = "http://127.0.0.1:8000/mcp"
TOOL = "EditorToolset.ActorToolset.SpawnActor"


class R06EvidenceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.project = self.root / "PhiMCPSandboxR06"
        self.project.mkdir()
        self.uproject = self.project / "PhiMCPSandboxR06.uproject"
        self.uproject.write_text('{"EngineAssociation":"5.9"}', encoding="utf-8")
        self.marker = self.project / "PHI_MCP_DISPOSABLE.json"
        self.marker.write_text(json.dumps({
            "schema": MARKER_SCHEMA,
            "project_filename": self.uproject.name,
            "project_sha256": self.sha(self.uproject),
            "disposable": True,
        }), encoding="utf-8")
        self.content = self.project / "Content"
        self.content.mkdir()
        self.asset = self.content / "Scratch.umap"
        self.asset.write_bytes(b"UE-FIXTURE-BEFORE-NOT-REAL-MAP")
        self.config = self.project / "Config"
        self.config.mkdir()
        (self.config / "DefaultEngine.ini").write_text(
            "[Engine]\nname=fixture\n", encoding="utf-8")
        self.backups = self.root / "private-baselines"
        self.backups.mkdir()
        self.recovery_parent = self.root / "recover-only"
        self.recovery_parent.mkdir()
        self.plan_path = self.root / "pilot-plan.json"
        self.intents = self.root / "private-intents"
        self.intents.mkdir()
        self.receipt = self.root / "actor-receipt.json"

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def capture(self, name="r06-fixture01"):
        return capture_baseline(self.project, self.backups, name)

    def snapshot(self):
        return self.backups / "r06-fixture01"

    def plan(self):
        info = {
            "schema": PLAN_SCHEMA,
            "project_sha256": self.sha(self.uproject),
            "marker_sha256": self.sha(self.marker),
            "expected_instance_id": "mock-369", "expected_process_id": 3333,
            "tool_name": TOOL, "tool_schema_sha256": "a" * 64,
            "arguments": {"actor_class": "/Script/Engine.PointLight",
                          "actor_label": "PhiMCPProbe_369"},
        }
        self.plan_path.write_text(json.dumps(info), encoding="utf-8")
        return info

    def make_intent(self, plan, *, receipt=True):
        intent = _digest({
            "endpoint": ENDPOINT, "plan": plan, "root": str(self.project.resolve())
        })
        (self.intents / (intent + ".attempt")).write_text(json.dumps({
            "schema": RECEIPT, "status": "ATTEMPT_RESERVED_UNKNOWN",
            "intent_sha256": intent,
        }), encoding="utf-8")
        if receipt:
            self.receipt.write_text(json.dumps({
                "schema": RECEIPT,
                "status": "WRITE_SENT_MANUAL_EDITOR_REVIEW_REQUIRED",
                "intent_sha256": intent, "tool_name": TOOL,
                "arguments_sha256": _digest(plan["arguments"]),
                "tool_reported_error": False,
                "tool_calls_sent": 1,
                "actor_exists_verified": False, "rollback_verified": False,
                "automated_retry_allowed": False, "production_authorized": False,
            }), encoding="utf-8")
        return intent

    def test_capture_verifies_copied_bytes_and_no_editor_connection(self):
        result = self.capture()
        self.assertEqual(result["status"], "BOUNDED_BASELINE_COPY_VERIFIED")
        self.assertFalse(result["all_unreal_files_backed_up"])
        self.assertFalse(result["live_editor_stopped_verified"])
        snap, manifest = read_baseline(self.snapshot())
        self.assertEqual(snap, self.snapshot())
        self.assertEqual(manifest["inventory_sha256"], result["inventory_sha256"])
        self.assertEqual(manifest["files"], inventory(self.project))
        self.assertEqual(compare_baseline(self.snapshot(), self.project)["counts"],
                         {"added": 0, "missing": 0, "modified": 0})

    def test_modified_added_and_deleted_assets_are_detected(self):
        self.capture()
        self.asset.write_bytes(b"UE-FIXTURE-AFTER")
        (self.config / "DefaultEngine.ini").unlink()
        (self.content / "New.umap").write_bytes(b"new test bytes")
        report = compare_baseline(self.snapshot(), self.project)
        self.assertEqual(report["schema"], REPORT)
        self.assertFalse(report["monitored_files_equal"])
        self.assertEqual(report["counts"],
                         {"added": 1, "missing": 1, "modified": 1})
        self.assertTrue(report["project_identity_anchors_unchanged"])
        self.assertFalse(report["actor_existence_verified"])
        self.assertFalse(report["saved_ue_map_semantics_verified"])
        self.assertFalse(report["in_memory_world_state_verified"])
        self.assertEqual(
            {item["path"]: item["kind"] for item in report["changes"]},
            {"Config/DefaultEngine.ini": "missing",
             "Content/New.umap": "added",
             "Content/Scratch.umap": "modified"})

    def test_modified_project_identity_anchor_is_detected(self):
        self.capture()
        self.uproject.write_text("changed project", encoding="utf-8")
        report = compare_baseline(self.snapshot(), self.project)
        self.assertFalse(report["project_identity_anchors_unchanged"])
        self.assertEqual(report["counts"]["modified"], 1)

    def test_snapshot_collision_and_inside_source_are_refused(self):
        self.capture()
        with self.assertRaisesRegex(MCPGateError, "already exists"):
            self.capture()
        self.assertEqual(len(list(self.backups.iterdir())), 1)
        with self.assertRaisesRegex(MCPGateError, "separate"):
            capture_baseline(self.project, self.project, "r06-inside")

    def test_manifest_tampering_fails_before_compare_or_restore(self):
        self.capture()
        docfile = self.snapshot() / "baseline.json"
        doc = json.loads(docfile.read_text(encoding="utf-8"))
        doc["files"][0]["sha256"] = "f" * 64
        docfile.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaisesRegex(MCPGateError, "digest mismatch"):
            compare_baseline(self.snapshot(), self.project)
        with self.assertRaises(MCPGateError):
            restore_to_new_copy(self.snapshot(), self.recovery_parent)
        self.assertFalse((self.recovery_parent / self.project.name).exists())

    def test_baseline_payload_mutation_is_rejected(self):
        self.capture()
        altered = self.snapshot() / "payload" / self.project.name / "Content" / "Scratch.umap"
        altered.write_text("tampered snapshot", encoding="utf-8")
        with self.assertRaisesRegex(MCPGateError, "payload bytes"):
            read_baseline(self.snapshot())

    def test_symlinks_and_oversize_in_scope_hold_without_snapshot(self):
        link = self.content / "linked-asset.umap"
        try:
            link.symlink_to(self.asset)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation unavailable")
        with self.assertRaisesRegex(MCPGateError, "symlink"):
            self.capture()
        link.unlink()
        huge = self.content / "large.umap"
        with huge.open("wb") as out:
            out.truncate(16 * 1024 * 1024 + 1)
        with self.assertRaisesRegex(MCPGateError, "scratch pilot size"):
            self.capture()

    def test_recovery_is_copy_only_and_verified_in_distinct_location(self):
        result = self.capture()
        self.asset.write_bytes(b"AFTER_ACTOR_FIXTURE")
        new = restore_to_new_copy(self.snapshot(), self.recovery_parent)
        self.assertEqual(new["schema"], RESTORE)
        self.assertEqual(new["status"],
                         "NEW_SCRATCH_COPY_MONITORED_FILES_VERIFIED")
        self.assertFalse(new["original_live_project_overwritten"])
        self.assertFalse(new["in_memory_editor_state_recovered"])
        self.assertFalse(new["actor_absence_verified"])
        restored = self.recovery_parent / self.project.name
        self.assertEqual(restored.joinpath("Content", "Scratch.umap").read_bytes(),
                         b"UE-FIXTURE-BEFORE-NOT-REAL-MAP")
        self.assertEqual(self.asset.read_bytes(), b"AFTER_ACTOR_FIXTURE")
        self.assertEqual(inventory(restored),
                         read_baseline(self.snapshot())[1]["files"])
        self.assertEqual(new["verified_inventory_sha256"],
                         result["inventory_sha256"])

    def test_recovery_refuses_existing_destination_and_nested_parent(self):
        self.capture()
        self.recovery_parent.joinpath(self.project.name).mkdir()
        with self.assertRaisesRegex(MCPGateError, "already exists"):
            restore_to_new_copy(self.snapshot(), self.recovery_parent)
        new_parent = self.project / "Content" / "RecoverHere"
        new_parent.mkdir()
        with self.assertRaisesRegex(MCPGateError, "separate"):
            restore_to_new_copy(self.snapshot(), new_parent)

    def test_actor_review_correlates_intent_but_never_proves_actor(self):
        self.capture()
        plan = self.plan()
        intent = self.make_intent(plan)
        self.asset.write_bytes(b"NEW_ASSET_BYTES")
        report = review_actor_attempt(
            self.snapshot(), self.project, self.plan_path,
            self.intents, ENDPOINT, self.receipt)
        self.assertEqual(report["status"],
                         "HOLD_MANUAL_ACTOR_INSPECTION_AND_RECOVERY_REVIEW")
        self.assertEqual(report["actor_intent_sha256"], intent)
        self.assertTrue(report["one_shot_reservation_correlated"])
        self.assertTrue(report["r05_receipt_correlated_as_self_report"])
        self.assertEqual(report["counts"]["modified"], 1)
        self.assertFalse(report["actor_exists_verified"])
        self.assertFalse(report["rollback_verified"])
        self.assertFalse(report["automatic_retry_authorized"])
        self.assertFalse(report["independent_editor_readback_performed"])

    def test_actor_review_without_receipt_stays_unknown_not_clean(self):
        self.capture()
        plan = self.plan()
        self.make_intent(plan, receipt=False)
        report = review_actor_attempt(
            self.snapshot(), self.project, self.plan_path, self.intents, ENDPOINT)
        self.assertFalse(report["r05_receipt_correlated_as_self_report"])
        self.assertFalse(report["actor_exists_verified"])
        self.assertFalse(report["automatic_retry_authorized"])
        self.assertTrue(report["monitored_files_equal"])

    def test_missing_reservation_and_forged_receipt_are_blocked(self):
        self.capture()
        self.plan()
        with self.assertRaisesRegex(MCPGateError, "receipt/intent"):
            review_actor_attempt(
                self.snapshot(), self.project, self.plan_path,
                self.intents, ENDPOINT)
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        self.make_intent(plan)
        data = json.loads(self.receipt.read_text(encoding="utf-8"))
        data["actor_exists_verified"] = True
        self.receipt.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(MCPGateError, "unsupported authority"):
            review_actor_attempt(
                self.snapshot(), self.project, self.plan_path,
                self.intents, ENDPOINT, self.receipt)

    def test_wrong_plan_source_and_production_named_project_refused(self):
        self.capture()
        plan = self.plan()
        self.make_intent(plan)
        plan["project_sha256"] = "f" * 64
        self.plan_path.write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaisesRegex(MCPGateError, "baseline"):
            review_actor_attempt(
                self.snapshot(), self.project, self.plan_path,
                self.intents, ENDPOINT)
        production = self.root / "RaceToOblivion"
        production.mkdir()
        (production / "RaceToOblivion.uproject").write_text("{}",
                                                           encoding="utf-8")
        with self.assertRaisesRegex(MCPGateError, "PhiMCPSandbox"):
            inventory(production)


if __name__ == "__main__":
    unittest.main()
