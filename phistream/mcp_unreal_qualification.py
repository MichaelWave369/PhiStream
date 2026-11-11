"""PhiStream MCP R0.4: explicitly approved Unreal metadata-tool pilot.

A local project marker and MCP schema fingerprints are *review aids*, not
authentication of the editor or process listening on localhost. No arbitrary
MCP tool is callable here. The generic Unreal call_tool dispatcher is denied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from .mcp_gateway import AuditLog, MCPGateError, _digest
from .mcp_unreal_discovery import LoopbackUnrealClient, validate_endpoint

SCHEMA = "phistream.mcp-unreal.metadata-pilot.v0.4"
MARKER_SCHEMA = "phistream.mcp-unreal.disposable-project.v0.4"
META_TOOLS = frozenset(("list_toolsets", "describe_toolsets", "describe_tools"))
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
MAX_ARG_BYTES = 16 * 1024
MAX_UPROJECT_BYTES = 2 * 1024 * 1024


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise MCPGateError(reason)


def _file_hash(path: Path, limit: int) -> str:
    _require(path.is_file() and not path.is_symlink(),
             "expected regular local file, not symlink")
    _require(path.stat().st_size <= limit,
             "local file exceeds bounded verification limit")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_probe(project_root: str | Path, *, require_marker: bool = False) -> dict:
    """Read a single local .uproject + optional disposable marker; no editing."""
    raw = Path(project_root).expanduser()
    _require(raw.is_absolute() and raw.is_dir() and not raw.is_symlink(),
             "project_root must be an existing absolute directory, not symlink")
    root = raw.resolve(strict=True)
    projects = sorted(root.glob("*.uproject"))
    _require(len(projects) == 1 and projects[0].is_file()
             and not projects[0].is_symlink(),
             "exactly one regular .uproject is required in the sandbox root")
    project = projects[0]
    _require(root.name == project.stem and
             root.name.startswith("PhiMCPSandbox"),
             "project directory and .uproject must share PhiMCPSandbox* name")
    digest = _file_hash(project, MAX_UPROJECT_BYTES)
    marker = root / "PHI_MCP_DISPOSABLE.json"
    present = marker.exists() or marker.is_symlink()
    marker_sha = None
    if present:
        marker_sha = _file_hash(marker, 4096)
        try:
            obj = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise MCPGateError("invalid disposable project marker") from exc
        _require(type(obj) is dict and set(obj) == {
            "schema", "project_filename", "project_sha256", "disposable"
        } and obj["schema"] == MARKER_SCHEMA and obj["disposable"] is True
                 and obj["project_filename"] == project.name
                 and obj["project_sha256"] == digest,
                 "project marker must match local .uproject and assert disposable")
    if require_marker:
        _require(present, "disposable project marker must exist before inspection")
    return {
        "schema": "phistream.mcp-unreal.local-project-probe.v0.4",
        "status": "LOCAL_MARKER_MATCHED_NOT_EDITOR_AUTHENTICATED" if present else
                  "AWAITING_OPERATOR_DISPOSABLE_MARKER",
        "project_filename": project.name,
        "project_sha256": digest,
        "marker_sha256": marker_sha,
        "marker_present": present,
        "editor_identity_verified": False,
        "open_editor_project_verified": False,
        "network_access_performed": False,
        "scene_mutations": 0,
    }


def read_arguments(filename: str | Path) -> dict:
    path = Path(filename)
    _require(path.is_file() and not path.is_symlink() and
             path.stat().st_size <= MAX_ARG_BYTES,
             "tool arguments must be a bounded regular JSON file")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError, OSError) as exc:
        raise MCPGateError("invalid JSON arguments file") from exc
    _require(type(obj) is dict, "MCP metadata arguments must be an object")
    _require(len(json.dumps(obj, allow_nan=False).encode("utf-8")) <= MAX_ARG_BYTES,
             "metadata arguments exceed size limit")
    return obj


def inspect_metadata(
    project_root: str | Path, endpoint: str, audit: AuditLog,
    *, tool: str, arguments: dict, expected_schema_sha256: str,
    approve_connect: bool = False, approve_inspect: bool = False,
    timeout: int = 8, client_factory=LoopbackUnrealClient,
) -> dict:
    """Send one exact pinned metadata tool request, never a generic dispatcher.

    Metadata calls MAY have unexpected effects on a compromised server, so
    neither this function nor its audit claims absence of editor mutations.
    """
    validate_endpoint(endpoint)
    _require(type(tool) is str and tool in META_TOOLS,
             "only three fixed metadata inspection tools are permitted")
    _require(type(expected_schema_sha256) is str and
             SHA256.fullmatch(expected_schema_sha256) is not None,
             "exact expected SHA256 of advertised inputSchema is required")
    _require(type(arguments) is dict, "metadata arguments must be a JSON object")
    try:
        args_raw = json.dumps(arguments, allow_nan=False, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MCPGateError("metadata arguments must be JSON") from exc
    _require(len(args_raw) <= MAX_ARG_BYTES,
             "metadata tool arguments exceed size limit")

    # Missing approval is rejected before reading any project or starting HTTP.
    if approve_connect is not True or approve_inspect is not True:
        audit.append("unreal-schema-inspection", "DENIED_APPROVAL", "loopback",
                     tool=tool, arguments_sha256=_digest(arguments))
        raise MCPGateError("both --approve-connect and --approve-inspect required")

    before = project_probe(project_root, require_marker=True)
    audit.append("unreal-schema-inspection", "STARTED", "loopback",
                 tool=tool, arguments_sha256=_digest(arguments))
    try:
        client = client_factory(endpoint, timeout=timeout)
        handshake = client.initialize()
        catalog = client.list_tools()
        entries = [entry for entry in catalog
                   if type(entry) is dict and entry.get("name") == tool]
        _require(len(entries) == 1 and type(entries[0].get("inputSchema")) is dict,
                 "selected metadata tool must be advertised exactly once")
        observed_schema = _digest(entries[0]["inputSchema"])
        _require(observed_schema == expected_schema_sha256,
                 "live MCP metadata tool schema changed from operator pin")

        # The only location in R0.4 that calls tools/call.
        result = client._post("tools/call", {"name": tool, "arguments": arguments})
        _require(type(result) is dict and result.get("isError") is not True,
                 "MCP metadata tool reported failure")
        after = project_probe(project_root, require_marker=True)
        _require(after["project_sha256"] == before["project_sha256"] and
                 after["marker_sha256"] == before["marker_sha256"],
                 "local sandbox identity anchor changed during inspection")

        receipt = {
            "schema": SCHEMA,
            "status": "LOCAL_METADATA_RESPONSE_RECORDED_NOT_ATTESTED",
            "endpoint": endpoint,
            "selected_tool": tool,
            "tool_schema_sha256": observed_schema,
            "arguments_sha256": _digest(arguments),
            "response_sha256": _digest(result),
            "local_project_sha256": before["project_sha256"],
            "local_marker_sha256": before["marker_sha256"],
            "local_project_identity_anchor_unchanged": True,
            "local_process_authenticated": False,
            "open_editor_project_verified": False,
            "editor_scene_changes_verified_absent": False,
            "tool_result_trusted": False,
            "tool_calls_sent": 1,
            "actor_creation_authorized": False,
            "general_tool_dispatch_authorized": False,
            "code_write_authorized": False,
            "merge_authorized": False,
            "production_authorized": False,
        }
        audit.append("unreal-schema-inspection", "COMPLETED", "loopback",
                     tool=tool, arguments_sha256=_digest(arguments),
                     result_sha256=_digest(receipt))
        return receipt
    except Exception as exc:
        audit.append("unreal-schema-inspection", "FAILED", "loopback",
                     tool=tool, arguments_sha256=_digest(arguments))
        raise MCPGateError(
            "Unreal metadata inspection failed; editor state must be checked manually"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhiStream R0.4 Unreal disposable-project metadata pilot")
    sub = parser.add_subparsers(dest="action", required=True)
    preflight = sub.add_parser("preflight", help="Offline project/marker check only")
    preflight.add_argument("--project-root", type=Path, required=True)
    run = sub.add_parser("inspect", help="Operator-approved metadata call only")
    run.add_argument("--project-root", type=Path, required=True)
    run.add_argument("--endpoint", default="http://127.0.0.1:8000/mcp")
    run.add_argument("--audit", type=Path, required=True)
    run.add_argument("--tool", choices=sorted(META_TOOLS), required=True)
    run.add_argument("--arguments-file", type=Path, required=True)
    run.add_argument("--expected-schema-sha256", required=True)
    run.add_argument("--approve-connect", action="store_true")
    run.add_argument("--approve-inspect", action="store_true")
    run.add_argument("--timeout", type=int, default=8)
    opts = parser.parse_args()
    try:
        if opts.action == "preflight":
            report = project_probe(opts.project_root)
        else:
            report = inspect_metadata(
                opts.project_root, opts.endpoint, AuditLog(opts.audit),
                tool=opts.tool, arguments=read_arguments(opts.arguments_file),
                expected_schema_sha256=opts.expected_schema_sha256,
                approve_connect=opts.approve_connect,
                approve_inspect=opts.approve_inspect, timeout=opts.timeout)
        print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    except (MCPGateError, OSError) as exc:
        print(json.dumps({"status": "HELD_NO_GENERAL_EDITOR_EXECUTION",
                          "reason": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
