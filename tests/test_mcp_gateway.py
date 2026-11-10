"""MCP gateway R0.1: no GPU, remote accounts, or real engine writes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

from phistream.mcp_gateway import (
    AuditLog, AUDIT_SCHEMA, MANIFEST_SCHEMA, MCPGateError, MCPGateway,
    StdioMCPClient, load_manifest,
)


SERVER = r'''
import json
import sys

def reply(ident, result):
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": ident, "result": result}) + "\n")
    sys.stdout.flush()

for line in sys.stdin:
    message = json.loads(line)
    if "id" not in message:
        continue
    method = message["method"]
    if method == "initialize":
        reply(message["id"], {"protocolVersion": "2025-06-18",
                              "capabilities": {"tools": {}},
                              "serverInfo": {"name": "fixture", "version": "1"}})
    elif method == "tools/list":
        if message["params"].get("cursor") == "next":
            reply(message["id"], {"tools": [{"name": "write_fixture",
                                            "description": "mock write"}]})
        else:
            reply(message["id"], {"tools": [{"name": "ping", "description": "read data"}],
                                  "nextCursor": "next"})
    elif method == "tools/call":
        reply(message["id"], {"content": [{"type": "text",
                                          "text": message["params"]["arguments"].get("secret", "pong")}],
                              "isError": False})
'''


class FakeClient:
    """No subprocess: count calls and assert no tool execution on denial."""

    def __init__(self, config, tracker):
        tracker["launched"] += 1
        self.tracker = tracker

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def list_tools(self):
        return [{"name": "ping"}, {"name": "write_fixture"}]

    def call_tool(self, name, arguments):
        self.tracker["called"].append((name, arguments))
        return {"content": [{"type": "text", "text": "PRIVATE_RESULT"}],
                "isError": False}


class MCPGatewayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.script = self.root / "server.py"
        self.script.write_text(textwrap.dedent(SERVER), encoding="utf-8")
        self.manifest_path = self.root / "manifest.json"
        self.audit_path = self.root / "audit.jsonl"
        self.config = {
            "schema": MANIFEST_SCHEMA,
            "servers": {"test": {
                "command": sys.executable,
                "args": [str(self.script)],
                "allowed_tools": {"ping": "read", "write_fixture": "write"},
                "timeout_seconds": 3,
            }},
        }
        self.save()
        self.tracker = {"launched": 0, "called": []}

    def save(self):
        self.manifest_path.write_text(json.dumps(self.config), encoding="utf-8")

    def gateway(self, *, fake=True):
        if fake:
            return MCPGateway(load_manifest(self.manifest_path),
                              AuditLog(self.audit_path),
                              lambda config: FakeClient(config, self.tracker))
        return MCPGateway(load_manifest(self.manifest_path), AuditLog(self.audit_path))

    def events(self):
        return [json.loads(line) for line in
                self.audit_path.read_text(encoding="utf-8").splitlines()]

    def test_manifest_requires_absolute_executable_and_explicit_tools(self):
        self.config["servers"]["test"]["command"] = "python"
        self.save()
        with self.assertRaisesRegex(MCPGateError, "absolute"):
            load_manifest(self.manifest_path)
        self.config["servers"]["test"]["command"] = sys.executable
        self.config["servers"]["test"]["allowed_tools"] = {"*": "read"}
        self.save()
        with self.assertRaisesRegex(MCPGateError, "name"):
            load_manifest(self.manifest_path)
        self.config["servers"]["test"]["allowed_tools"] = {"ping": "admin"}
        self.save()
        with self.assertRaisesRegex(MCPGateError, "risk"):
            load_manifest(self.manifest_path)

    def test_discovery_requires_explicit_launch(self):
        with self.assertRaisesRegex(MCPGateError, "approve-launch"):
            self.gateway().catalog("test")
        self.assertEqual(self.tracker["launched"], 0)
        self.assertEqual(self.events()[0]["status"], "DENIED_NO_LAUNCH_APPROVAL")

    def test_call_requires_two_operator_approvals(self):
        for choices in ({}, {"approve_launch": True},
                        {"approve_execute": True}):
            with self.assertRaisesRegex(MCPGateError, "approval missing"):
                self.gateway().call("test", "ping", {"x": 1}, **choices)
        self.assertEqual(self.tracker["launched"], 0)
        self.assertEqual(self.tracker["called"], [])

    def test_unknown_tool_denied_before_subprocess(self):
        with self.assertRaisesRegex(MCPGateError, "not allowlisted"):
            self.gateway().call("test", "unreviewed", {},
                                approve_launch=True, approve_execute=True)
        self.assertEqual(self.tracker["launched"], 0)

    def test_write_tools_need_separate_write_approval(self):
        with self.assertRaisesRegex(MCPGateError, "approval missing"):
            self.gateway().call("test", "write_fixture", {},
                                approve_launch=True, approve_execute=True)
        self.assertEqual(self.tracker["launched"], 0)
        result = self.gateway().call("test", "write_fixture", {"sample": 1},
                                     approve_launch=True, approve_execute=True,
                                     approve_write=True)
        self.assertFalse(result["isError"])
        self.assertEqual(self.tracker["called"], [("write_fixture", {"sample": 1})])

    def test_audit_records_only_hashes_not_private_arguments_or_results(self):
        argument = {"secret": "PRIVATE_ARGUMENT"}
        self.gateway().call("test", "ping", argument,
                            approve_launch=True, approve_execute=True)
        audit_text = self.audit_path.read_text(encoding="utf-8")
        self.assertNotIn("PRIVATE_ARGUMENT", audit_text)
        self.assertNotIn("PRIVATE_RESULT", audit_text)
        self.assertEqual([row["status"] for row in self.events()],
                         ["STARTED", "COMPLETED"])
        self.assertTrue(all(row["schema"] == AUDIT_SCHEMA for row in self.events()))
        self.assertEqual(len(self.events()[1]["result_sha256"]), 64)
        if os.name == "posix":
            self.assertEqual(self.audit_path.stat().st_mode & 0o077, 0)

    def test_discovery_is_not_execution(self):
        result = self.gateway().catalog("test", approve_launch=True)
        self.assertFalse(result["tool_execution_authorized"])
        self.assertEqual([item["name"] for item in result["tools"]],
                         ["ping", "write_fixture"])
        self.assertEqual(self.tracker["called"], [])

    def test_fail_closed_if_audit_directory_does_not_exist(self):
        gateway = MCPGateway(load_manifest(self.manifest_path),
                             AuditLog(self.root / "missing" / "audit.jsonl"),
                             lambda config: FakeClient(config, self.tracker))
        with self.assertRaisesRegex(MCPGateError, "audit path"):
            gateway.call("test", "ping", {}, approve_launch=True,
                         approve_execute=True)
        self.assertEqual(self.tracker["launched"], 0)

    def test_real_stdio_handshake_paginated_discovery_and_echo(self):
        gateway = self.gateway(fake=False)
        catalog = gateway.catalog("test", approve_launch=True)
        self.assertEqual([row["name"] for row in catalog["tools"]],
                         ["ping", "write_fixture"])
        self.assertTrue(all(item["allowlisted"] for item in catalog["tools"]))
        response = gateway.call("test", "ping", {"secret": "SENT_ONLY_TO_FIXTURE"},
                                approve_launch=True, approve_execute=True)
        self.assertEqual(response["content"][0]["text"], "SENT_ONLY_TO_FIXTURE")
        self.assertNotIn("SENT_ONLY_TO_FIXTURE",
                         self.audit_path.read_text(encoding="utf-8"))

    def test_no_real_write_without_approve_write(self):
        gateway = self.gateway(fake=False)
        with self.assertRaises(MCPGateError):
            gateway.call("test", "write_fixture", {},
                         approve_launch=True, approve_execute=True)
        self.assertEqual([row["status"] for row in self.events()],
                         ["DENIED_BY_POLICY"])

    def test_missing_advertised_tool_blocks_call(self):
        class MissingTool(FakeClient):
            def list_tools(self):
                return [{"name": "elsewhere"}]
        gateway = MCPGateway(load_manifest(self.manifest_path),
                             AuditLog(self.audit_path),
                             lambda config: MissingTool(config, self.tracker))
        with self.assertRaisesRegex(MCPGateError, "tool call failed"):
            gateway.call("test", "ping", {}, approve_launch=True,
                         approve_execute=True)
        self.assertEqual(self.tracker["called"], [])
        self.assertEqual([row["status"] for row in self.events()],
                         ["STARTED", "FAILED"])


if __name__ == "__main__":
    unittest.main()
