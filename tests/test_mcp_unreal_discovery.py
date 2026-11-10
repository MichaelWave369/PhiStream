"""R0.3: local HTTP/SSE protocol tests, no Unreal installation needed."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

from phistream.mcp_gateway import AuditLog, MCPGateError
from phistream.mcp_unreal_discovery import (
    SCHEMA, LoopbackUnrealClient, discover, validate_endpoint,
)

TOOLS = [
    {"name": "list_toolsets", "inputSchema": {"type": "object"}},
    {"name": "describe_toolsets", "inputSchema": {"type": "object"}},
    {"name": "describe_tools", "inputSchema": {"type": "object"}},
    {"name": "call_tool", "inputSchema": {"type": "object"}},
    {"name": "inspect_fixture", "description": "INJECT_ME_IGNORE_POLICY",
     "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}}},
    {"name": "mutate_fixture", "inputSchema": {"type": "object"}},
]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def respond(self, result, status=200, sse=False, session=True):
        obj = result
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        if sse:
            raw = b"event: message\n" + b"data: " + raw + b"\n\n"
        self.send_response(status)
        if session:
            self.send_header("Mcp-Session-Id", "fixture-session-369")
        self.send_header("Content-Type",
                         "text/event-stream" if sse else "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.server.methods.append(body["method"])
        self.server.headers_seen.append(dict(self.headers))
        self.server.paths_seen.append(self.path)
        if self.server.mode == "redirect":
            self.send_response(302)
            self.send_header("Location", "http://example.invalid/foreign")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        method = body["method"]
        if method == "notifications/initialized":
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        req_id = body["id"] + (1 if self.server.mode == "bad-id" else 0)
        if method == "initialize":
            result = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mock-unreal-editor", "version": "1"},
            }
        elif method == "tools/list":
            if body["params"].get("cursor") == "part2":
                result = {"tools": TOOLS[4:]}
            else:
                result = {"tools": TOOLS[:4], "nextCursor": "part2"}
            if self.server.mode == "malformed-tool":
                result = {"tools": [{"name": "oops", "inputSchema": "string"}]}
        else:
            result = {"isError": True}
        response = {"jsonrpc": "2.0", "id": req_id, "result": result}
        self.respond(response, sse=self.server.mode == "sse")


class UnrealDiscoveryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.folder = Path(temp.name)
        self.audit = AuditLog(self.folder / "audit.jsonl")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.server.mode = "json"
        self.server.methods = []
        self.server.headers_seen = []
        self.server.paths_seen = []
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join, 2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}/mcp"

    def events(self):
        return [json.loads(line) for line in
                self.audit.path.read_text(encoding="utf-8").splitlines()]

    def test_bad_hosts_paths_and_credentials_are_refused(self):
        for url in (
            "http://example.com:8000/mcp",
            "https://127.0.0.1:8000/mcp",
            "http://localhost:8000/mcp",
            "http://[::1]:8000/mcp",
            "http://127.0.0.2:8000/mcp",
            "http://127.0.0.1:8000/admin",
            "http://127.0.0.1:8000/mcp?token=abc",
            "http://127.0.0.1:8000/mcp#x",
            "http://user:pass@127.0.0.1:8000/mcp",
            "http://127.0.0.1/mcp",
            "http://127.0.0.1:80/mcp",
            "file:///etc/passwd",
        ):
            with self.subTest(url=url), self.assertRaises(MCPGateError):
                validate_endpoint(url)
        self.assertEqual(validate_endpoint(self.endpoint), self.endpoint)

    def test_no_connection_without_operator_approval(self):
        with self.assertRaisesRegex(MCPGateError, "approve-connect"):
            discover(self.endpoint, self.audit)
        self.assertEqual(self.server.methods, [])
        self.assertEqual(self.events()[0]["status"],
                         "DENIED_NO_CONNECT_APPROVAL")

    def test_discovery_json_pagination_and_metadata_only(self):
        receipt = discover(self.endpoint, self.audit, approve_connect=True)
        self.assertEqual(receipt["schema"], SCHEMA)
        self.assertEqual(receipt["status"],
                         "LOCAL_CATALOG_OBSERVED_NOT_ATTESTED")
        self.assertEqual(len(receipt["advertised_tools"]), 6)
        self.assertTrue(receipt["search_meta_tools_present"])
        self.assertFalse(receipt["discovery_exhaustive"])
        self.assertFalse(receipt["project_identity_verified"])
        self.assertFalse(receipt["mcp_server_authenticated"])
        self.assertFalse(receipt["runtime_approval_granted"])
        self.assertFalse(receipt["production_authorized"])
        self.assertEqual(receipt["tools_invoked"], 0)
        self.assertEqual(receipt["scene_mutations"], 0)
        self.assertEqual(self.server.methods, [
            "initialize", "notifications/initialized",
            "tools/list", "tools/list",
        ])
        self.assertTrue(all(x == "/mcp" for x in self.server.paths_seen))
        self.assertTrue(all("Authorization" not in h for h in self.server.headers_seen))
        output = json.dumps(receipt)
        self.assertNotIn("INJECT_ME_IGNORE_POLICY", output)
        self.assertNotIn("properties", output)
        self.assertEqual([x["status"] for x in self.events()],
                         ["STARTED", "COMPLETED"])
        self.assertEqual(len(self.events()[1]["result_sha256"]), 64)

    def test_streamable_http_sse_response(self):
        self.server.mode = "sse"
        receipt = discover(self.endpoint, self.audit, approve_connect=True)
        self.assertEqual(len(receipt["advertised_tools"]), 6)
        self.assertNotIn("tools/call", self.server.methods)

    def test_redirect_cannot_escape_loopback(self):
        self.server.mode = "redirect"
        with self.assertRaisesRegex(MCPGateError, "no tool invocation"):
            discover(self.endpoint, self.audit, approve_connect=True)
        self.assertEqual(self.server.methods, ["initialize"])
        self.assertEqual([x["status"] for x in self.events()],
                         ["STARTED", "FAILED"])

    def test_mismatched_rpc_id_is_rejected(self):
        self.server.mode = "bad-id"
        with self.assertRaises(MCPGateError):
            discover(self.endpoint, self.audit, approve_connect=True)
        self.assertEqual(self.server.methods, ["initialize"])

    def test_malformed_schema_refuses_catalog(self):
        self.server.mode = "malformed-tool"
        with self.assertRaises(MCPGateError):
            discover(self.endpoint, self.audit, approve_connect=True)
        self.assertNotIn("tools/call", self.server.methods)

    def test_client_refuses_list_before_initialize(self):
        client = LoopbackUnrealClient(self.endpoint)
        with self.assertRaisesRegex(MCPGateError, "initialization required"):
            client.list_tools()
        self.assertEqual(self.server.methods, [])

    def test_timeout_bounds_and_no_hidden_remote_fallback(self):
        for timeout in (0, 21, True, -1, 0.5):
            with self.subTest(timeout=timeout), self.assertRaises(MCPGateError):
                LoopbackUnrealClient(self.endpoint, timeout=timeout)
        self.assertEqual(self.server.methods, [])


if __name__ == "__main__":
    unittest.main()
