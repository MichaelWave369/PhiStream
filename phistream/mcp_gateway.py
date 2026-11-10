"""PhiStream MCP R0.1: local stdio tools behind explicit policy and audit.

This does not change CLAVIS or GA108/Vessie routing authority. In particular,
neither a model proposal nor MCP tool metadata constitutes operator approval.
Only preconfigured local executables are supported in this first release.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import stat
import subprocess
import sys
import threading
import time

MANIFEST_SCHEMA = "phistream.mcp-gateway.manifest.v0.1"
AUDIT_SCHEMA = "phistream.mcp-gateway.audit.v0.1"
MCP_VERSION = "2025-06-18"
MAX_JSON_BYTES = 1024 * 1024
MAX_ARGUMENT_BYTES = 64 * 1024
MAX_TOOLS = 512
_NAME = re.compile(r"[a-zA-Z0-9_.:/-]{1,128}\Z")
_ENV = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class MCPGateError(ValueError):
    """Rejected policy, protocol, transport, or audit operation."""


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_manifest(path: str | Path) -> dict:
    """Validate an operator-written manifest. No model or server may widen it."""
    file = Path(path)
    if not file.is_file() or file.stat().st_size > MAX_JSON_BYTES:
        raise MCPGateError("manifest missing or too large")
    try:
        obj = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise MCPGateError("invalid manifest JSON") from exc
    if type(obj) is not dict or set(obj) != {"schema", "servers"}:
        raise MCPGateError("manifest fields invalid")
    if obj["schema"] != MANIFEST_SCHEMA or type(obj["servers"]) is not dict:
        raise MCPGateError("manifest schema mismatch")
    if not 1 <= len(obj["servers"]) <= 32:
        raise MCPGateError("manifest must have 1..32 servers")
    for server_id, config in obj["servers"].items():
        if type(server_id) is not str or not _NAME.fullmatch(server_id):
            raise MCPGateError("invalid server ID")
        if type(config) is not dict or not set(config) <= {
            "command", "args", "allowed_tools", "pass_env", "timeout_seconds"
        }:
            raise MCPGateError("invalid server configuration")
        command = config.get("command")
        if type(command) is not str or not Path(command).is_absolute():
            raise MCPGateError("server command must be an absolute executable path")
        args = config.get("args", [])
        if type(args) is not list or len(args) > 32 or any(
            type(v) is not str or len(v) > 4096 for v in args
        ):
            raise MCPGateError("server args must be bounded strings")
        allowed = config.get("allowed_tools")
        if type(allowed) is not dict or not 1 <= len(allowed) <= MAX_TOOLS:
            raise MCPGateError("explicit allowed_tools mapping required")
        for name, risk in allowed.items():
            if type(name) is not str or not _NAME.fullmatch(name):
                raise MCPGateError("invalid allowlisted tool name")
            if risk not in ("read", "write"):
                raise MCPGateError("tool risk must be read or write")
        env = config.get("pass_env", [])
        if type(env) is not list or len(env) > 20 or any(
            type(name) is not str or not _ENV.fullmatch(name) for name in env
        ) or len(set(env)) != len(env):
            raise MCPGateError("invalid pass_env names")
        timeout = config.get("timeout_seconds", 15)
        if type(timeout) is not int or not 1 <= timeout <= 60:
            raise MCPGateError("timeout_seconds must be 1..60")
    return obj


class AuditLog:
    """Append-only private JSONL metadata. Never stores tool arguments or output."""

    def __init__(self, filename: str | Path):
        self.path = Path(filename)

    def append(self, event: str, status: str, server: str, *,
               tool: str | None = None, arguments_sha256: str | None = None,
               result_sha256: str | None = None) -> None:
        if self.path.is_symlink() or not self.path.parent.is_dir():
            raise MCPGateError("audit path must have an existing directory and no symlink")
        record = {
            "schema": AUDIT_SCHEMA,
            "utc": datetime.now(timezone.utc).isoformat(),
            "event": event, "status": status, "server": server, "tool": tool,
            "arguments_sha256": arguments_sha256,
            "result_sha256": result_sha256,
        }
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self.path, flags, 0o600)
            try:
                meta = os.fstat(fd)
                if not stat.S_ISREG(meta.st_mode):
                    raise MCPGateError("audit destination is not a regular file")
                if os.name == "posix" and (stat.S_IMODE(meta.st_mode) & 0o077):
                    raise MCPGateError("audit log has unsafe permissions")
                raw = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
                while raw:
                    raw = raw[os.write(fd, raw):]
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError as exc:
            raise MCPGateError("audit write failed") from exc


class StdioMCPClient:
    """Synchronous JSON-RPC 2.0 over newline-delimited MCP stdio, bounded I/O."""

    def __init__(self, config: dict):
        self.config = config
        self.process: subprocess.Popen | None = None
        self.messages: queue.Queue = queue.Queue(maxsize=64)
        self.next_id = 0
        self.timeout = config.get("timeout_seconds", 15)

    def _reader(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        while True:
            raw = self.process.stdout.readline(MAX_JSON_BYTES + 1)
            if not raw:
                self.messages.put(MCPGateError("MCP server closed stdout"))
                return
            if len(raw) > MAX_JSON_BYTES:
                self.messages.put(MCPGateError("MCP response too large"))
                return
            try:
                message = json.loads(raw.decode("utf-8"))
            except (UnicodeError, ValueError):
                self.messages.put(MCPGateError("invalid MCP JSON-RPC response"))
                return
            self.messages.put(message)

    def _send(self, message: dict) -> None:
        if self.process is None or self.process.stdin is None:
            raise MCPGateError("MCP server is not running")
        data = (json.dumps(message, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        if len(data) > MAX_JSON_BYTES:
            raise MCPGateError("MCP request too large")
        try:
            self.process.stdin.write(data)
            self.process.stdin.flush()
        except OSError as exc:
            raise MCPGateError("MCP server stdin closed") from exc

    def request(self, method: str, params: dict) -> dict:
        self.next_id += 1
        ident = self.next_id
        self._send({"jsonrpc": "2.0", "id": ident, "method": method, "params": params})
        end = time.monotonic() + self.timeout
        skipped = 0
        while time.monotonic() < end:
            try:
                message = self.messages.get(timeout=max(0.001, end - time.monotonic()))
            except queue.Empty as exc:
                raise MCPGateError("MCP response timeout") from exc
            if isinstance(message, Exception):
                raise message
            if type(message) is not dict or message.get("jsonrpc") != "2.0":
                raise MCPGateError("malformed JSON-RPC message")
            if message.get("id") != ident:
                if "id" in message:
                    raise MCPGateError("unexpected JSON-RPC response ID")
                skipped += 1
                if skipped > 64:
                    raise MCPGateError("too many MCP notifications")
                continue
            if "error" in message or type(message.get("result")) is not dict:
                raise MCPGateError("MCP request rejected or malformed result")
            return message["result"]
        raise MCPGateError("MCP response timeout")

    def __enter__(self) -> "StdioMCPClient":
        base = ("PATH", "HOME", "USERPROFILE", "SystemRoot", "TEMP", "TMP", "TMPDIR")
        env = {key: os.environ[key] for key in base if key in os.environ}
        for key in self.config.get("pass_env", []):
            if key not in os.environ:
                raise MCPGateError("explicitly permitted environment variable is missing")
            env[key] = os.environ[key]
        try:
            self.process = subprocess.Popen(
                [self.config["command"], *self.config.get("args", [])],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, env=env, shell=False,
            )
            threading.Thread(target=self._reader, daemon=True).start()
            initialized = self.request("initialize", {
                "protocolVersion": MCP_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "PhiStream-MCP-Gateway", "version": "0.1.0"},
            })
            if type(initialized.get("protocolVersion")) is not str:
                raise MCPGateError("MCP handshake missing protocolVersion")
            self._send({"jsonrpc": "2.0", "method": "notifications/initialized",
                        "params": {}})
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def list_tools(self) -> list[dict]:
        tools: list[dict] = []
        cursor = None
        cursors = set()
        names = set()
        for _ in range(25):
            response = self.request("tools/list", {"cursor": cursor} if cursor else {})
            page = response.get("tools")
            if type(page) is not list or len(tools) + len(page) > MAX_TOOLS:
                raise MCPGateError("invalid or oversized MCP tool listing")
            for item in page:
                if type(item) is not dict or type(item.get("name")) is not str:
                    raise MCPGateError("malformed MCP tool descriptor")
                name = item["name"]
                if name in names:
                    raise MCPGateError("duplicate MCP tool name")
                names.add(name)
                tools.append(item)
            cursor = response.get("nextCursor")
            if cursor is None:
                return tools
            if type(cursor) is not str or not cursor or cursor in cursors:
                raise MCPGateError("invalid MCP pagination cursor")
            cursors.add(cursor)
        raise MCPGateError("MCP tool pagination limit reached")

    def call_tool(self, tool: str, arguments: dict) -> dict:
        return self.request("tools/call", {"name": tool, "arguments": arguments})

    def __exit__(self, _type, _value, _traceback) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=1)
            for handle in (self.process.stdin, self.process.stdout):
                if handle is not None:
                    handle.close()


class MCPGateway:
    """Policy and audit are enforced before launching any MCP child process."""

    def __init__(self, manifest: dict, audit: AuditLog, client_factory=StdioMCPClient):
        self.servers = manifest["servers"]
        self.audit = audit
        self.client_factory = client_factory

    def _server(self, server: str) -> dict:
        if server not in self.servers:
            raise MCPGateError("server not in operator manifest")
        return self.servers[server]

    def catalog(self, server: str, *, approve_launch: bool = False) -> dict:
        config = self._server(server)
        if approve_launch is not True:
            self.audit.append("catalog", "DENIED_NO_LAUNCH_APPROVAL", server)
            raise MCPGateError("explicit --approve-launch is required")
        self.audit.append("catalog", "STARTED", server)
        try:
            with self.client_factory(config) as client:
                listed = client.list_tools()
            result = {
                "schema": "phistream.mcp-gateway.catalog.v0.1",
                "server": server,
                "tools": [
                    {"name": item["name"],
                     "allowlisted": item["name"] in config["allowed_tools"],
                     "risk": config["allowed_tools"].get(item["name"]),
                     "description": str(item.get("description", ""))[:256]}
                    for item in listed
                ],
                "tool_execution_authorized": False,
            }
            self.audit.append("catalog", "COMPLETED", server,
                              result_sha256=_digest(result))
            return result
        except Exception as exc:
            self.audit.append("catalog", "FAILED", server)
            raise MCPGateError("MCP catalog failed; see server-local diagnostics") from exc

    def call(self, server: str, tool: str, arguments: dict, *,
             approve_launch: bool = False, approve_execute: bool = False,
             approve_write: bool = False) -> dict:
        config = self._server(server)
        if type(arguments) is not dict:
            raise MCPGateError("arguments must be a JSON object")
        try:
            raw = json.dumps(arguments, allow_nan=False).encode("utf-8")
            digest = _digest(arguments)
        except (TypeError, ValueError) as exc:
            raise MCPGateError("arguments must be valid JSON") from exc
        if len(raw) > MAX_ARGUMENT_BYTES:
            raise MCPGateError("tool arguments too large")
        risk = config["allowed_tools"].get(tool)
        if risk is None or approve_launch is not True or approve_execute is not True or (
            risk == "write" and approve_write is not True
        ):
            self.audit.append("call", "DENIED_BY_POLICY", server, tool=tool,
                              arguments_sha256=digest)
            raise MCPGateError("tool not allowlisted or explicit approval missing")
        self.audit.append("call", "STARTED", server, tool=tool,
                          arguments_sha256=digest)
        try:
            with self.client_factory(config) as client:
                advertised = {item["name"] for item in client.list_tools()}
                if tool not in advertised:
                    raise MCPGateError("allowlisted tool not advertised by MCP server")
                result = client.call_tool(tool, arguments)
            if type(result) is not dict:
                raise MCPGateError("malformed MCP tool result")
            status = "TOOL_REPORTED_ERROR" if result.get("isError") is True else "COMPLETED"
            self.audit.append("call", status, server, tool=tool,
                              arguments_sha256=digest, result_sha256=_digest(result))
            return result
        except Exception as exc:
            self.audit.append("call", "FAILED", server, tool=tool,
                              arguments_sha256=digest)
            raise MCPGateError("MCP tool call failed; review private audit") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Local MCP gateway, deny by default")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    sub = parser.add_subparsers(dest="action", required=True)
    catalog = sub.add_parser("catalog", help="Discover tools; server launch requires approval")
    catalog.add_argument("--server", required=True)
    catalog.add_argument("--approve-launch", action="store_true")
    call = sub.add_parser("call", help="Execute one allowlisted tool")
    call.add_argument("--server", required=True)
    call.add_argument("--tool", required=True)
    call.add_argument("--arguments-file", type=Path, help="JSON object in local file")
    call.add_argument("--approve-launch", action="store_true")
    call.add_argument("--approve-execute", action="store_true")
    call.add_argument("--approve-write", action="store_true")
    call.add_argument("--show-result", action="store_true",
                      help="Print untrusted raw MCP result; otherwise only SHA256")
    args = parser.parse_args()
    try:
        gateway = MCPGateway(load_manifest(args.manifest), AuditLog(args.audit))
        if args.action == "catalog":
            result = gateway.catalog(args.server, approve_launch=args.approve_launch)
        else:
            parameters = {}
            if args.arguments_file:
                if (not args.arguments_file.is_file() or
                        args.arguments_file.stat().st_size > MAX_ARGUMENT_BYTES):
                    raise MCPGateError("arguments file missing or too large")
                parameters = json.loads(args.arguments_file.read_text(encoding="utf-8"))
            response = gateway.call(
                args.server, args.tool, parameters,
                approve_launch=args.approve_launch,
                approve_execute=args.approve_execute,
                approve_write=args.approve_write,
            )
            result = (response if args.show_result else {
                "schema": "phistream.mcp-gateway.call-receipt.v0.1",
                "server": args.server, "tool": args.tool,
                "tool_reported_error": response.get("isError") is True,
                "result_sha256": _digest(response), "raw_result_exported": False,
            })
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    except (MCPGateError, ValueError, OSError) as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "reason": str(exc)}),
              file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
