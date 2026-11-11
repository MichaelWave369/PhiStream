"""Unreal R0.5: one-shot actor-policy tests, no real editor or assets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from phistream.mcp_gateway import AuditLog, MCPGateError, _digest
from phistream.mcp_unreal_qualification import MARKER_SCHEMA
from phistream.mcp_unreal_actor_pilot import (
    CONFIRM, INSTANCE_META, RECEIPT, SCHEMA, actor_attempt,
    inspect_instance, read_plan, validate_plan,
)

TOOL = "EditorToolset.ActorToolset.SpawnActor"
TOOL_SCHEMA = {
    "type": "object", "properties": {
        "actor_class": {"type": "string"},
        "actor_label": {"type": "string"},
    }, "required": ["actor_class", "actor_label"],
}


class FakeClient:
    def __init__(self, endpoint, *, timeout, tracker):
        tracker["started"] += 1
        self.tracker = tracker

    def initialize(self):
        self.tracker["methods"].append("initialize")
        info = dict(self.tracker["instance"])
        return {
            "serverInfo": {"name": self.tracker.get("server_name", "unreal-mcp")},
            "_meta": {INSTANCE_META: info},
        }

    def list_tools(self):
        self.tracker["methods"].append("tools/list")
        return [{"name": TOOL,
                 "inputSchema": self.tracker.get("schema", TOOL_SCHEMA)},
                {"name": "call_tool", "inputSchema": {"type": "object"}}]

    def _post(self, method, params):
        self.tracker["methods"].append(method)
        self.tracker["calls"].append(params)
        if self.tracker.get("modify_file"):
            self.tracker["modify_file"].write_text("changed by fake editor")
        if self.tracker.get("tool_error"):
            return {"isError": True}
        if self.tracker.get("timeout"):
            raise TimeoutError("mock uncertainty after possible send")
        return {"content": [
            {"type": "text", "text": "PRIVATE_ACTOR_RESULT_NOT_VERIFIED"}],
            "isError": False}


class R05Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.project = self.root / "PhiMCPSandboxR05"
        self.project.mkdir()
        self.uproject = self.project / "PhiMCPSandboxR05.uproject"
        self.uproject.write_text('{"EngineAssociation":"5.9"}', encoding="utf-8")
        self.marker = self.project / "PHI_MCP_DISPOSABLE.json"
        project_digest = hashlib.sha256(self.uproject.read_bytes()).hexdigest()
        self.marker.write_text(json.dumps({
            "schema": MARKER_SCHEMA, "project_filename": self.uproject.name,
            "project_sha256": project_digest, "disposable": True,
        }), encoding="utf-8")
        self.audit = AuditLog(self.root / "audit.jsonl")
        self.intent = self.root / "private-intents"
        self.intent.mkdir()
        self.tracker = {
            "started": 0, "methods": [], "calls": [],
            "instance": {
                "instance_id": "epic-test-instance-369",
                "process_id": 3333,
                "engine_version": "5.9.0",
                "project_name": self.project.name,
                "project_folder": str(self.project),
            },
        }
        self.plan = {
            "schema": SCHEMA,
            "project_sha256": project_digest,
            "marker_sha256": hashlib.sha256(self.marker.read_bytes()).hexdigest(),
            "expected_instance_id": "epic-test-instance-369",
            "expected_process_id": 3333,
            "tool_name": TOOL,
            "tool_schema_sha256": _digest(TOOL_SCHEMA),
            "arguments": {
                "actor_class": "/Script/Engine.PointLight",
                "actor_label": "PhiMCPProbe_369",
            },
        }

    def client(self, endpoint, timeout):
        return FakeClient(endpoint, timeout=timeout, tracker=self.tracker)

    def run_attempt(self, **changes):
        options = {
            "project_root": self.project,
            "endpoint": "http://127.0.0.1:8000/mcp",
            "audit": self.audit,
            "intent_dir": self.intent,
            "plan": self.plan,
            "approve_connect": True,
            "approve_write": True,
            "confirmation": CONFIRM,
            "client_factory": self.client,
        }
        options.update(changes)
        return actor_attempt(**options)

    def test_read_only_identity_observation_without_tool_call(self):
        observed = inspect_instance(
            self.project, "http://127.0.0.1:8000/mcp",
            approve_connect=True, client_factory=self.client)
        self.assertEqual(observed["instance"]["process_id"], 3333)
        self.assertEqual(observed["status"], "REPORTED_INSTANCE_NOT_AUTHENTICATED")
        self.assertFalse(observed["tool_execution_authorized"])
        self.assertEqual(self.tracker["methods"], ["initialize"])

    def test_identity_denial_never_connects(self):
        with self.assertRaisesRegex(MCPGateError, "approve-connect"):
            inspect_instance(self.project, "http://127.0.0.1:8000/mcp",
                             client_factory=self.client)
        self.assertEqual(self.tracker["started"], 0)

    def test_one_approved_attempt_produces_review_not_success(self):
        receipt = self.run_attempt()
        self.assertEqual(receipt["schema"], RECEIPT)
        self.assertEqual(receipt["status"], "WRITE_SENT_MANUAL_EDITOR_REVIEW_REQUIRED")
        self.assertEqual(self.tracker["methods"],
                         ["initialize", "tools/list", "tools/call"])
        self.assertEqual(self.tracker["calls"], [{
            "name": TOOL, "arguments": self.plan["arguments"]
        }])
        self.assertFalse(receipt["actor_exists_verified"])
        self.assertFalse(receipt["instance_authenticated"])
        self.assertFalse(receipt["production_authorized"])
        self.assertFalse(receipt["automated_retry_allowed"])
        serialized = json.dumps(receipt) + self.audit.path.read_text(encoding="utf-8")
        self.assertNotIn("PRIVATE_ACTOR_RESULT_NOT_VERIFIED", serialized)
        self.assertNotIn("PhiMCPProbe_369", serialized)
        self.assertEqual(len(list(self.intent.glob("*.attempt"))), 1)

    def test_replaying_identical_plan_never_sends_second_write(self):
        self.run_attempt()
        with self.assertRaisesRegex(MCPGateError, "already attempted"):
            self.run_attempt()
        self.assertEqual(len(self.tracker["calls"]), 1)

    def test_no_approvals_no_reservation_or_connection(self):
        for changes in [
            {"approve_write": False}, {"approve_connect": False},
            {"confirmation": "approved"}, {"confirmation": ""},
        ]:
            with self.subTest(changes=changes), self.assertRaisesRegex(
                    MCPGateError, "approvals required"):
                self.run_attempt(**changes)
        self.assertEqual(self.tracker["started"], 0)
        self.assertFalse(list(self.intent.iterdir()))

    def test_wrong_instance_project_or_server_denied_before_tool_list(self):
        bad_cases = [
            ("instance_id", "different-instance"),
            ("process_id", 4444),
            ("project_name", "RaceToOblivion"),
            ("project_folder", str(self.root)),
            ("engine_version", "5.8"),
        ]
        for field, bad in bad_cases:
            with self.subTest(field=field):
                old = self.tracker["instance"][field]
                self.tracker["instance"][field] = bad
                with self.assertRaisesRegex(MCPGateError, "do NOT retry"):
                    self.run_attempt()
                self.tracker["instance"][field] = old
                self.assertEqual(self.tracker["calls"], [])
                self.assertEqual(self.tracker["methods"][-1], "initialize")
                # Distinct intent is mandatory after a held/unknown outcome.
                self.plan["arguments"]["actor_label"] += "_new"
                self.tracker["methods"].clear()
        self.assertEqual(len(list(self.intent.glob("*.attempt"))), 5)

    def test_wrong_tool_schema_blocks_before_write(self):
        self.tracker["schema"] = {"type": "object"}
        with self.assertRaisesRegex(MCPGateError, "do NOT retry"):
            self.run_attempt()
        self.assertEqual(self.tracker["methods"], ["initialize", "tools/list"])
        self.assertEqual(self.tracker["calls"], [])

    def test_unknown_outcome_leaves_permanent_attempt_reservation(self):
        self.tracker["timeout"] = True
        with self.assertRaisesRegex(MCPGateError, "do NOT retry"):
            self.run_attempt()
        self.assertEqual(len(self.tracker["calls"]), 1)
        with self.assertRaisesRegex(MCPGateError, "already attempted"):
            self.run_attempt()
        self.assertEqual(len(self.tracker["calls"]), 1)

    def test_local_file_changes_flagged_and_no_rollback_claim(self):
        monitored = self.project / "Content"
        monitored.mkdir()
        source = monitored / "Map.untitled"
        source.write_text("before", encoding="utf-8")
        self.tracker["modify_file"] = source
        result = self.run_attempt()
        self.assertTrue(result["project_files_changed_in_monitored_scope"])
        self.assertFalse(result["rollback_verified"])
        self.assertTrue(result["project_identity_anchors_unchanged"])

    def test_malicious_or_generic_tool_and_class_denied(self):
        for bad in (
            "call_tool", "EditorToolset.ActorToolset.DeleteAllActors",
            "EditorToolset.ActorToolset.ExecutePython",
            "SystemActorTool.SpawnActor",
        ):
            with self.subTest(tool=bad):
                self.plan["tool_name"] = bad
                with self.assertRaisesRegex(MCPGateError, "SpawnActor"):
                    validate_plan(self.plan)
        self.plan["tool_name"] = TOOL
        for bad in (
            "/Script/Engine.WorldSettings", "exec('rm -rf')",
            "/Game/RaceToOblivion/Actor", "PointLight;rebuild",
        ):
            with self.subTest(value=bad):
                self.plan["arguments"]["actor_class"] = bad
                with self.assertRaises(MCPGateError):
                    validate_plan(self.plan)
        self.assertEqual(self.tracker["started"], 0)

    def test_script_arguments_unbounded_transform_and_bad_actor_label_denied(self):
        for args in [
            {"actor_class": "PointLight", "actor_label": "YourActor"},
            {"actor_class": "PointLight", "actor_label": "PhiMCPProbe_a",
             "execute_python": "print('hello')"},
            {"actor_class": "PointLight", "actor_label": "PhiMCPProbe_a",
             "location": {"x": 20000, "y": 0, "z": 0}},
            {"actor_class": "PointLight", "actor_label": "PhiMCPProbe_a",
             "location": {"x": True, "y": 0, "z": 0}},
            {"actor_class": "PointLight", "actor_label": "PhiMCPProbe_a",
             "location": {"x": 0, "y": float("nan"), "z": 0}},
        ]:
            with self.subTest(args=args):
                self.plan["arguments"] = args
                with self.assertRaises((MCPGateError, ValueError)):
                    validate_plan(self.plan)

    def test_read_plan_bounded_and_rejects_nonmatching_marker(self):
        planfile = self.root / "plan.json"
        planfile.write_text(json.dumps(self.plan), encoding="utf-8")
        # prior test mutates self.plan only within that test's own setUp
        self.assertEqual(read_plan(planfile)["schema"], SCHEMA)
        self.marker.write_text(self.marker.read_text(encoding="utf-8") + " ",
                               encoding="utf-8")
        with self.assertRaisesRegex(MCPGateError, "no longer matches"):
            self.run_attempt()
        self.assertEqual(self.tracker["started"], 0)
        self.assertFalse(list(self.intent.iterdir()))

    def test_pending_editor_effect_does_not_become_verified(self):
        self.tracker["tool_error"] = True
        report = self.run_attempt()
        self.assertTrue(report["tool_reported_error"])
        self.assertFalse(report["actor_exists_verified"])
        self.assertEqual(report["status"], "WRITE_SENT_MANUAL_EDITOR_REVIEW_REQUIRED")


if __name__ == "__main__":
    unittest.main()
