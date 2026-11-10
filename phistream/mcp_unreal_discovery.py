"""Unreal MCP R0.3: loopback-only, discovery-only Streamable HTTP pilot.

Epic Unreal Engine 5.9's experimental MCP server uses HTTP/SSE, not stdio.
This module NEVER sends tools/call. It cannot edit Unreal scenes, load assets,
run tests, issue fleet jobs, or authorize R0.1/R0.2 tool execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import (
    HTTPRedirectHandler, ProxyHandler, Request, build_opener,
)

from .mcp_gateway import (
    AuditLog, MCPGateError, MCP_VERSION, MAX_JSON_BYTES, MAX_TOOLS, _digest,
)

SCHEMA = "phistream.mcp-unreal.loopback-discovery.v0.3"
SESSION_RE = re.compile(r"[A-Za-z0-9._~-]{1,256}\Z")
TOOL_RE = re.compile(r"[A-Za-z0-9_.:/-]{1,128}\Z")
MAX_PAGES = 20
MAX_MESSAGES = 64
MAX_LINE_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = MAX_JSON_BYTES


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Never follow a localhost response to another origin or service."""
        return None


def validate_endpoint(endpoint: str) -> str:
    """Allow exactly a local numeric IPv4 host on the documented /mcp path."""
    if type(endpoint) is not str or len(endpoint) > 120:
        raise MCPGateError("MCP endpoint must be a short string")
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as exc:
        raise MCPGateError("invalid MCP loopback URL") from exc
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path != "/mcp"
            or port is None or not 1024 <= port <= 65535
            or parsed.netloc != f"127.0.0.1:{port}"):
        raise MCPGateError(
            "only http://127.0.0.1:<port>/mcp is allowed; no redirects")
    return endpoint


def _jsonrpc_message(payload: bytes, request_id: int) -> dict:
    if len(payload) > MAX_RESPONSE_BYTES:
        raise MCPGateError("HTTP MCP response exceeded size limit")
    try:
        response = json.loads(payload.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise MCPGateError("invalid HTTP MCP JSON") from exc
    if (type(response) is not dict or response.get("jsonrpc") != "2.0"
            or type(response.get("id")) is not int
            or response["id"] != request_id or "error" in response
            or type(response.get("result")) is not dict):
        raise MCPGateError("HTTP MCP invalid or mismatched JSON-RPC response")
    return response["result"]


def _read_sse(response, ident: int) -> dict:
    """Read one response event without waiting for a long-lived SSE stream EOF."""
    total = 0
    lines: list[bytes] = []
    events_seen = 0
    while total <= MAX_RESPONSE_BYTES and events_seen < MAX_MESSAGES:
        line = response.readline(MAX_LINE_BYTES + 1)
        if not line:
            break
        total += len(line)
        if len(line) > MAX_LINE_BYTES or total > MAX_RESPONSE_BYTES:
            raise MCPGateError("SSE event exceeded size limit")
        if line.strip() == b"":
            if not lines:
                continue
            events_seen += 1
            data = b"\n".join(lines)
            lines = []
            try:
                obj = json.loads(data.decode("utf-8"))
            except (UnicodeError, ValueError) as exc:
                raise MCPGateError("invalid SSE MCP JSON event") from exc
            if type(obj) is dict and "id" not in obj:
                continue  # MCP notification, not a reply
            return _jsonrpc_message(data, ident)
        if line.startswith(b"data:"):
            lines.append(line[5:].strip())
        elif line.startswith(b":") or line.startswith(b"event:") or line.startswith(b"id:"):
            continue
        else:
            raise MCPGateError("unsupported SSE line")
    raise MCPGateError("no bounded matching JSON-RPC SSE event")


class LoopbackUnrealClient:
    """One serialized MCP operation per request, no remote host or proxy."""

    def __init__(self, endpoint: str, *, timeout: int = 8):
        self.endpoint = validate_endpoint(endpoint)
        if type(timeout) is not int or not 1 <= timeout <= 20:
            raise MCPGateError("HTTP MCP timeout must be 1..20 seconds")
        self.timeout = timeout
        self.session_id: str | None = None
        self.initialized = False
        self.next_id = 0
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())

    def _post(self, method: str, params: dict, *, notification: bool = False) -> dict | None:
        if notification:
            envelope = {"jsonrpc": "2.0", "method": method, "params": params}
        else:
            self.next_id += 1
            envelope = {"jsonrpc": "2.0", "id": self.next_id,
                        "method": method, "params": params}
        body = json.dumps(envelope, separators=(",", ":"), allow_nan=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id is not None:
            headers["Mcp-Session-Id"] = self.session_id
        if self.initialized:
            headers["MCP-Protocol-Version"] = MCP_VERSION
        request = Request(self.endpoint, data=body, headers=headers, method="POST")
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                new_session = response.headers.get("Mcp-Session-Id")
                if new_session is not None:
                    if (not SESSION_RE.fullmatch(new_session) or
                            (self.session_id is not None and
                             self.session_id != new_session)):
                        raise MCPGateError("unexpected or unsafe MCP session identifier")
                    self.session_id = new_session
                if notification:
                    if response.status not in (200, 202, 204):
                        raise MCPGateError("MCP notification not accepted")
                    return None
                if response.status != 200:
                    raise MCPGateError("MCP HTTP status was not 200")
                media_type = response.headers.get_content_type()
                if media_type == "application/json":
                    return _jsonrpc_message(
                        response.read(MAX_RESPONSE_BYTES + 1), self.next_id)
                if media_type == "text/event-stream":
                    return _read_sse(response, self.next_id)
                raise MCPGateError("MCP response has an unsupported media type")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise MCPGateError("local Unreal MCP endpoint unavailable or rejected request") from exc

    def initialize(self) -> dict:
        result = self._post("initialize", {
            "protocolVersion": MCP_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "PhiStream-Unreal-Discovery",
                           "version": "0.3.0"},
        })
        if (type(result) is not dict or
                result.get("protocolVersion") != MCP_VERSION or
                type(result.get("serverInfo")) is not dict or
                type(result["serverInfo"].get("name")) is not str):
            raise MCPGateError("MCP initialization failed or unsupported version")
        self.initialized = True
        self._post("notifications/initialized", {}, notification=True)
        return result

    def list_tools(self) -> list[dict]:
        if not self.initialized:
            raise MCPGateError("MCP initialization required")
        tools: list[dict] = []
        names: set[str] = set()
        cursors: set[str] = set()
        cursor: str | None = None
        for _ in range(MAX_PAGES):
            data = self._post("tools/list", {"cursor": cursor} if cursor else {})
            if type(data) is not dict or type(data.get("tools")) is not list:
                raise MCPGateError("invalid MCP tool catalog")
            page = data["tools"]
            if len(page) + len(tools) > MAX_TOOLS:
                raise MCPGateError("too many advertised tools")
            for descriptor in page:
                if (type(descriptor) is not dict or
                        type(descriptor.get("name")) is not str or
                        TOOL_RE.fullmatch(descriptor["name"]) is None or
                        type(descriptor.get("inputSchema")) is not dict or
                        descriptor["name"] in names):
                    raise MCPGateError("invalid or duplicate MCP tool descriptor")
                names.add(descriptor["name"])
                tools.append(descriptor)
            cursor = data.get("nextCursor")
            if cursor is None:
                return tools
            if (type(cursor) is not str or not 1 <= len(cursor) <= 256
                    or cursor in cursors):
                raise MCPGateError("invalid MCP pagination cursor")
            cursors.add(cursor)
        raise MCPGateError("MCP pagination limit reached")


def discover(endpoint: str, audit: AuditLog, *, approve_connect: bool = False,
             timeout: int = 8, client_factory=LoopbackUnrealClient) -> dict:
    """Only inspect advertised tool names and input-schema hashes."""
    endpoint = validate_endpoint(endpoint)
    if approve_connect is not True:
        audit.append("unreal-discovery", "DENIED_NO_CONNECT_APPROVAL", "loopback")
        raise MCPGateError("explicit --approve-connect required")
    audit.append("unreal-discovery", "STARTED", "loopback")
    try:
        client = client_factory(endpoint, timeout=timeout)
        handshake = client.initialize()
        tools = client.list_tools()
        names = [tool["name"] for tool in tools]
        result = {
            "schema": SCHEMA,
            "status": "LOCAL_CATALOG_OBSERVED_NOT_ATTESTED",
            "transport": "STREAMABLE_HTTP_LOOPBACK_ONLY",
            "endpoint": endpoint,
            "server_name_untrusted": handshake["serverInfo"]["name"][:128],
            "advertised_tools": [
                {"name": tool["name"],
                 "input_schema_sha256": _digest(tool["inputSchema"])}
                for tool in tools
            ],
            "search_meta_tools_present": all(name in names for name in (
                "list_toolsets", "describe_toolsets", "describe_tools", "call_tool")),
            "discovery_exhaustive": False,
            "unreal_editor_identity_verified": False,
            "project_identity_verified": False,
            "mcp_server_authenticated": False,
            "tools_invoked": 0,
            "scene_mutations": 0,
            "runtime_approval_granted": False,
            "production_authorized": False,
        }
        audit.append("unreal-discovery", "COMPLETED", "loopback",
                     result_sha256=_digest(result))
        return result
    except Exception as exc:
        audit.append("unreal-discovery", "FAILED", "loopback")
        raise MCPGateError(
            "local Unreal MCP discovery failed; no tool invocation attempted"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unreal 5.9 local HTTP MCP read-only tool discovery")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/mcp")
    parser.add_argument("--audit", required=True,
                        help="Private existing-parent audit JSONL path")
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--approve-connect", action="store_true")
    args = parser.parse_args()
    try:
        result = discover(args.endpoint, AuditLog(args.audit),
                          approve_connect=args.approve_connect,
                          timeout=args.timeout)
        print(json.dumps(result, sort_keys=True, indent=2))
    except MCPGateError as exc:
        print(json.dumps({"status": "HELD_NO_EDITOR_MUTATION",
                          "reason": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
