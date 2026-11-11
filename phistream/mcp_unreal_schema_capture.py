"""Unreal MCP R0.8: observe real tool schemas, save privately, never invoke tools.

This is a workstation-side discovery helper. It does not authenticate the
local HTTP service, grant a tool permission, or run an Unreal actor command.
Returned schema text is UNTRUSTED even when it resembles documentation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

from .mcp_gateway import AuditLog, MCPGateError, _digest
from .mcp_unreal_actor_pilot import _local_identity
from .mcp_unreal_discovery import LoopbackUnrealClient, validate_endpoint
from .mcp_unreal_qualification import project_probe

SCHEMA = "phistream.mcp-unreal.schema-capture.v0.8"
MAX_EXPORT_BYTES = 256 * 1024
MAX_CANDIDATES = 16
TOOL_NAME = re.compile(r"[a-zA-Z0-9_.:/-]{1,128}\Z")


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise MCPGateError(message)


def _private_output(path: str | Path, project_root: Path) -> Path:
    raw = Path(path).expanduser()
    _need(raw.is_absolute() and raw.name.endswith(".json") and
          raw.name != ".json" and not raw.exists() and not raw.is_symlink(),
          "output must be a new absolute .json file (no overwrite or symlink)")
    _need(raw.parent.is_dir() and not raw.parent.is_symlink(),
          "output parent directory must already exist")
    resolved_parent = raw.parent.resolve(strict=True)
    # Reject the live project and all ancestors/descendants as output parents.
    _need(resolved_parent != project_root and
          resolved_parent not in project_root.parents and
          project_root not in resolved_parent.parents,
          "private schema output must be outside project tree and ancestors")
    return resolved_parent / raw.name


def _recorded_tools(catalog: list[dict], exact_tool: str | None) -> tuple[list[dict], int]:
    _need(type(catalog) is list, "invalid MCP catalog")
    all_names: set[str] = set()
    for tool in catalog:
        _need(type(tool) is dict and
              type(tool.get("name")) is str and
              TOOL_NAME.fullmatch(tool["name"]) is not None and
              type(tool.get("inputSchema")) is dict and
              tool["name"] not in all_names,
              "invalid, duplicate or missing MCP tool descriptor")
        all_names.add(tool["name"])
    if exact_tool is not None:
        _need(type(exact_tool) is str and
              TOOL_NAME.fullmatch(exact_tool) is not None,
              "exact tool name invalid")
        selected = [x for x in catalog if x["name"] == exact_tool]
        _need(len(selected) == 1, "exact tool not advertised in this catalog")
    else:
        candidates = sorted(
            (tool for tool in catalog
             if "actor" in tool["name"].lower() or
                "spawn" in tool["name"].lower()),
            key=lambda tool: tool["name"],
        )
        selected = candidates[:MAX_CANDIDATES]
    entries = []
    for tool in selected:
        schema = tool["inputSchema"]
        # Nested schemas and free-text descriptions are data, never instructions.
        entries.append({
            "name": tool["name"], "input_schema_sha256": _digest(schema),
            "inputSchema_untrusted": schema,
        })
    return entries, sum(
        "actor" in tool["name"].lower() or "spawn" in tool["name"].lower()
        for tool in catalog
    )


def _write_new(path: Path, data: dict) -> str:
    raw = (json.dumps(data, sort_keys=True, indent=2,
                      ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    _need(len(raw) <= MAX_EXPORT_BYTES, "schema report exceeds private export size limit")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise MCPGateError("private schema file already exists; no overwrite") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise MCPGateError("failed to write private schema report") from exc
    return _digest(data)


def capture_schema(
    project_root: str | Path, endpoint: str, private_output: str | Path,
    audit: AuditLog, *, approve_connect: bool = False,
    approve_export: bool = False, exact_tool: str | None = None,
    timeout: int = 8, client_factory=LoopbackUnrealClient,
) -> dict:
    """Initialize + tools/list; no tools/call, filesystem writes only to private output."""
    endpoint = validate_endpoint(endpoint)
    # Explicit denial happens before opening a socket, reading files or creating output.
    if approve_connect is not True or approve_export is not True:
        audit.append("unreal-schema-capture", "DENIED_APPROVAL", "loopback")
        raise MCPGateError("explicit --approve-connect AND --approve-export required")
    probe = project_probe(project_root, require_marker=True)
    root = Path(project_root).resolve(strict=True)
    output = _private_output(private_output, root)
    if exact_tool is not None:
        _need(type(exact_tool) is str and
              TOOL_NAME.fullmatch(exact_tool) is not None,
              "invalid exact tool name")
    audit.append("unreal-schema-capture", "STARTED", "loopback")
    try:
        client = client_factory(endpoint, timeout=timeout)
        handshake = client.initialize()
        instance = _local_identity(handshake, root,
                                   Path(probe["project_filename"]).stem)
        catalog = client.list_tools()
        observed, total_candidate = _recorded_tools(catalog, exact_tool)
        names_and_hashes = [
            {"name": entry["name"],
             "input_schema_sha256": _digest(entry["inputSchema"])}
            for entry in sorted(catalog, key=lambda x: x["name"])
        ]
        names = set(item["name"] for item in catalog)
        tool_search = all(x in names for x in (
            "list_toolsets", "describe_toolsets", "describe_tools", "call_tool"
        )) and len(catalog) == 4
        result = {
            "schema": SCHEMA,
            "status": ("TOOL_SEARCH_MODE_ONLY_META_TOOLS"
                       if tool_search else "DIRECT_TOOL_CATALOG_OBSERVED"),
            "endpoint": endpoint,
            "reported_server": handshake["serverInfo"]["name"],
            "reported_engine_version": instance["engine_version"],
            "reported_instance_id": instance["instance_id"],
            "reported_process_id": instance["process_id"],
            "reported_project_name": instance["project_name"],
            "project_sha256": probe["project_sha256"],
            "marker_sha256": probe["marker_sha256"],
            "tools_total": len(catalog),
            "actor_or_spawn_candidates_total": total_candidate,
            "candidate_schemas_exported": len(observed),
            "candidate_subset_truncated": (
                exact_tool is None and total_candidate > MAX_CANDIDATES
            ),
            "tool_search_mode_apparent": tool_search,
            "advertised_tool_hashes": names_and_hashes,
            "selected_input_schemas_untrusted": observed,
            "remote_server_authenticated": False,
            "editor_process_authenticated": False,
            "tool_metadata_attested": False,
            "actual_tool_behavior_verified": False,
            "tools_called": 0,
            "write_authorized": False,
            "production_authorized": False,
            "merge_authorized": False,
        }
        digest = _write_new(output, result)
        audit.append("unreal-schema-capture", "COMPLETED", "loopback",
                     result_sha256=digest)
        return {
            "schema": SCHEMA,
            "status": result["status"],
            "private_output_created": True,
            "private_output_path": str(output),
            "report_sha256": digest,
            "advertised_tool_count": len(catalog),
            "actor_or_spawn_candidates_total": total_candidate,
            "candidate_schemas_exported": len(observed),
            "tool_search_mode_apparent": tool_search,
            "tools_called": 0,
            "write_authorized": False,
            "real_unreal_editor_verified": False,
            "next_action": (
                "Turn off Enable Tool Search in a disposable editor and reconnect"
                if tool_search else
                "Review private tool schema and stop if no compatible direct actor tool exists"
            ),
        }
    except Exception as exc:
        audit.append("unreal-schema-capture", "FAILED", "loopback")
        raise MCPGateError(
            "local schema capture failed; no editor tool invocation requested"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="R0.8 real Unreal MCP tool catalog to a private JSON file (discovery only)")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/mcp")
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--exact-tool",
                        help="Export schema for one exact advertised tool name")
    parser.add_argument("--approve-connect", action="store_true")
    parser.add_argument("--approve-export", action="store_true")
    parser.add_argument("--timeout", type=int, default=8)
    args = parser.parse_args()
    try:
        result = capture_schema(
            args.project_root, args.endpoint, args.private_output,
            AuditLog(args.audit), approve_connect=args.approve_connect,
            approve_export=args.approve_export,
            exact_tool=args.exact_tool, timeout=args.timeout)
        print(json.dumps(result, sort_keys=True, indent=2))
    except (MCPGateError, OSError, ValueError) as exc:
        print(json.dumps({"status": "HELD_NO_EDITOR_TOOL_INVOCATION",
                          "reason": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
