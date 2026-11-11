"""Unreal R0.4: inspect a disposable project without granting editor mutation."""
from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

from phistream.mcp_gateway import AuditLog, MCPGateError, _digest
from phistream.mcp_unreal_qualification import (
    MARKER_SCHEMA, SCHEMA, inspect_metadata, project_probe, read_arguments,
)

SCHEMA_DOC = {
    "type": "object", "properties": {
        "names": {"type": "array", "items": {"type": "string"}}
    },
}


class FakeClient:
    def __init__(self, _endpoint, *, timeout, tracker):
        tracker["started"] += 1
        self.tracker = tracker

    def initialize(self):
        self.tracker["methods"].append("initialize")
        return {"serverInfo": {"name": "untrusted-mock"}}

    def list_tools(self):
        self.tracker["methods"].append("tools/list")
        return [{"name": "describe_tools", "inputSchema": SCHEMA_DOC},
                {"name": "call_tool", "inputSchema": {"type": "object"}}]

    def _post(self, method, params):
        self.tracker["methods"].append(method)
        self.tracker["calls"].append(params)
        return {"content": [{"type": "text", "text": "SENSITIVE_TOOL_RESPONSE"}],
                "isError": False}


class RealLocalHandler(BaseHTTPRequestHandler):
    """Local mock HTTP MCP server with no file-system or actor operations."""
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        return

    def do_POST(self):
        assert self.path == "/mcp"
        message = json.loads(self.rfile.read(
            int(self.headers["Content-Length"])).decode("utf-8"))
        self.server.methods.append(message["method"])
        if message["method"] == "notifications/initialized":
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if message["method"] == "initialize":
            result = {"protocolVersion": "2025-06-18",
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "0"}}
        elif message["method"] == "tools/list":
            result = {"tools": [
                {"name": "describe_tools", "inputSchema": SCHEMA_DOC},
                {"name": "call_tool", "inputSchema": {"type": "object"}},
            ]}
        elif message["method"] == "tools/call":
            if message["params"]["name"] != "describe_tools":
                raise AssertionError("unallowed metadata tool invoked")
            result = {"content": [
                {"type": "text", "text": "MOCK_ONLY_NO_UNREAL"}
            ], "isError": False}
        else:
            raise AssertionError("unexpected method " + message["method"])
        raw = json.dumps({"jsonrpc": "2.0", "id": message["id"],
                          "result": result}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class QualificationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.project = self.root / "PhiMCPSandboxTest"
        self.project.mkdir()
        self.uproject = self.project / "PhiMCPSandboxTest.uproject"
        self.uproject.write_text('{"EngineAssociation":"5.9"}',
                                 encoding="utf-8")
        self.marker = self.project / "PHI_MCP_DISPOSABLE.json"
        self.audit = AuditLog(self.root / "audit.jsonl")
        self.tracker = {"started": 0, "methods": [], "calls": []}
        self.args = {"names": ["EditorToolset.Example"]}
        self.expected = _digest(SCHEMA_DOC)

    def make_marker(self):
        digest = hashlib.sha256(self.uproject.read_bytes()).hexdigest()
        self.marker.write_text(json.dumps({
            "schema": MARKER_SCHEMA,
            "project_filename": self.uproject.name,
            "project_sha256": digest,
            "disposable": True,
        }), encoding="utf-8")

    def call(self, **changes):
        options = {
            "project_root": self.project,
            "endpoint": "http://127.0.0.1:8000/mcp",
            "audit": self.audit, "tool": "describe_tools",
            "arguments": self.args,
            "expected_schema_sha256": self.expected,
            "approve_connect": True, "approve_inspect": True,
            "client_factory": lambda ep, timeout: FakeClient(
                ep, timeout=timeout, tracker=self.tracker),
        }
        options.update(changes)
        return inspect_metadata(**options)

    def events(self):
        return [json.loads(line) for line in
                self.audit.path.read_text(encoding="utf-8").splitlines()]

    def test_offline_preflight_never_launches_or_creates_marker(self):
        result = project_probe(self.project)
        self.assertEqual(result["status"], "AWAITING_OPERATOR_DISPOSABLE_MARKER")
        self.assertFalse(result["network_access_performed"])
        self.assertFalse(self.marker.exists())
        self.assertEqual(self.tracker["started"], 0)

    def test_marker_and_metadata_response_do_not_authorize_anything(self):
        self.make_marker()
        result = self.call()
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["status"], "LOCAL_METADATA_RESPONSE_RECORDED_NOT_ATTESTED")
        self.assertTrue(result["local_project_identity_anchor_unchanged"])
        self.assertFalse(result["open_editor_project_verified"])
        self.assertFalse(result["editor_scene_changes_verified_absent"])
        self.assertFalse(result["general_tool_dispatch_authorized"])
        self.assertFalse(result["production_authorized"])
        self.assertEqual(result["tool_calls_sent"], 1)
        self.assertEqual(self.tracker["methods"],
                         ["initialize", "tools/list", "tools/call"])
        self.assertEqual(self.tracker["calls"][0]["name"], "describe_tools")
        serialized = json.dumps(result) + self.audit.path.read_text(encoding="utf-8")
        self.assertNotIn("SENSITIVE_TOOL_RESPONSE", serialized)
        self.assertNotIn("EditorToolset.Example", serialized)
        self.assertEqual([x["status"] for x in self.events()],
                         ["STARTED", "COMPLETED"])

    def test_denied_before_connect(self):
        self.make_marker()
        for flags in (
            {"approve_connect": False},
            {"approve_inspect": False},
        ):
            with self.subTest(flags=flags), self.assertRaisesRegex(
                MCPGateError, "both --approve"):
                self.call(**flags)
        self.assertEqual(self.tracker["started"], 0)
        self.assertEqual([x["status"] for x in self.events()],
                         ["DENIED_APPROVAL", "DENIED_APPROVAL"])

    def test_missing_marker_rejected_before_network(self):
        with self.assertRaisesRegex(MCPGateError, "marker"):
            self.call()
        self.assertEqual(self.tracker["started"], 0)

    def test_forbid_generic_tool_dispatcher_even_with_approvals(self):
        self.make_marker()
        for tool in ("call_tool", "spawn_actor", "execute_python",
                     "PlayerController.ConsoleCommand", "tools/call"):
            with self.subTest(tool=tool), self.assertRaisesRegex(
                MCPGateError, "three fixed"):
                self.call(tool=tool)
        self.assertEqual(self.tracker["started"], 0)

    def test_schema_mismatch_blocks_before_tools_call(self):
        self.make_marker()
        with self.assertRaisesRegex(MCPGateError, "inspection failed"):
            self.call(expected_schema_sha256="f" * 64)
        self.assertEqual(self.tracker["calls"], [])
        self.assertEqual(self.tracker["methods"], ["initialize", "tools/list"])
        self.assertEqual([x["status"] for x in self.events()],
                         ["STARTED", "FAILED"])

    def test_unexpected_project_modification_fails_closed(self):
        self.make_marker()

        class ModifiedClient(FakeClient):
            def _post(inner, method, params):
                result = super()._post(method, params)
                self.uproject.write_text("modified", encoding="utf-8")
                return result

        with self.assertRaisesRegex(MCPGateError, "must be checked manually"):
            self.call(client_factory=lambda ep, timeout: ModifiedClient(
                ep, timeout=timeout, tracker=self.tracker))
        self.assertEqual([x["status"] for x in self.events()],
                         ["STARTED", "FAILED"])

    def test_marker_forgery_invalid_and_symlink_rejected(self):
        self.make_marker()
        data = json.loads(self.marker.read_text(encoding="utf-8"))
        data["disposable"] = False
        self.marker.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(MCPGateError, "marker"):
            project_probe(self.project, require_marker=True)
        self.marker.unlink()
        try:
            self.marker.symlink_to(self.uproject)
        except (OSError, NotImplementedError):
            return
        with self.assertRaises(MCPGateError):
            project_probe(self.project, require_marker=True)

    def test_production_like_project_name_is_denied(self):
        old = self.project
        production = self.root / "RaceToOblivion"
        old.rename(production)
        with self.assertRaisesRegex(MCPGateError, "PhiMCPSandbox"):
            project_probe(production)

    def test_argument_file_invalid_and_non_object(self):
        path = self.root / "args.json"
        path.write_text('["not an object"]', encoding="utf-8")
        with self.assertRaisesRegex(MCPGateError, "object"):
            read_arguments(path)
        path.write_text('{"names":["ok"]}', encoding="utf-8")
        self.assertEqual(read_arguments(path), {"names": ["ok"]})
        path.write_text("x" * (16 * 1024 + 1), encoding="utf-8")
        with self.assertRaisesRegex(MCPGateError, "bounded"):
            read_arguments(path)

    def test_real_loopback_http_fixture_call_method(self):
        self.make_marker()
        server = ThreadingHTTPServer(("127.0.0.1", 0), RealLocalHandler)
        server.daemon_threads = True
        server.methods = []
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = self.call(
                endpoint=f"http://127.0.0.1:{server.server_port}/mcp",
                client_factory=None if False else
                __import__("phistream.mcp_unreal_discovery",
                           fromlist=["LoopbackUnrealClient"]).LoopbackUnrealClient)
            self.assertEqual(result["tool_calls_sent"], 1)
            self.assertEqual(server.methods, [
                "initialize", "notifications/initialized",
                "tools/list", "tools/call",
            ])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
