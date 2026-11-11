"""R0.5: one-shot, operator-approved PointLight experiment in a disposable UE project.

Experimental: this code has not been field-tested against a real Unreal Editor.
No generic MCP call_tool dispatcher, no arbitrary commands, no auto-retries,
and no claim that a successful tool response proves an actor was created.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

from .mcp_gateway import AuditLog, MCPGateError, _digest
from .mcp_unreal_discovery import LoopbackUnrealClient, validate_endpoint
from .mcp_unreal_qualification import (
    SHA256, project_probe, _require, _file_hash,
)

SCHEMA = "phistream.mcp-unreal.actor-pilot.v0.5"
RECEIPT = "phistream.mcp-unreal.actor-attempt.v0.5"
INSTANCE_META = "com.epicgames.unrealengine/instance"
CONFIRM = "ONE_DISPOSABLE_POINTLIGHT_ONE_ATTEMPT_NO_AUTO_RETRY"
TOOL_RE = re.compile(
    r"EditorToolset\.[A-Za-z0-9_]{1,64}\.SpawnActor(?:FromClass)?\Z"
)
INSTANCE_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}\Z")
LABEL_RE = re.compile(r"PhiMCPProbe_[A-Za-z0-9_]{1,40}\Z")
MAX_PLAN = 16 * 1024
MAX_FILES = 512
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MONITORED = ("Content", "Config", "Plugins")


def _valid_project_path(path: str, root: Path) -> bool:
    if type(path) is not str or not path:
        return False
    # Only use as a consistency check, NOT as process authentication.
    return os.path.normcase(os.path.normpath(path.rstrip("\\/"))) == (
        os.path.normcase(os.path.normpath(str(root).rstrip("\\/")))
    )


def _local_identity(handshake: dict, root: Path, project_name: str) -> dict:
    _require(type(handshake) is dict and
             type(handshake.get("serverInfo")) is dict and
             handshake["serverInfo"].get("name") == "unreal-mcp",
             "unexpected MCP server name")
    metadata = handshake.get("_meta")
    _require(type(metadata) is dict and
             type(metadata.get(INSTANCE_META)) is dict,
             "Unreal instance metadata absent")
    info = metadata[INSTANCE_META]
    instance = info.get("instance_id")
    pid = info.get("process_id")
    engine = info.get("engine_version")
    _require(type(instance) is str and INSTANCE_RE.fullmatch(instance),
             "invalid Unreal instance ID")
    _require(type(pid) is int and 1 <= pid <= 2147483647,
             "invalid reported editor process ID")
    _require(type(engine) is str and engine.startswith("5.9") and len(engine) <= 128,
             "only experimental Unreal 5.9 accepted")
    _require(info.get("project_name") == project_name and
             _valid_project_path(info.get("project_folder"), root),
             "reported editor project does not match disposable root")
    return {
        "instance_id": instance,
        "process_id": pid,
        "engine_version": engine,
        "project_name": project_name,
        "metadata_origin_authenticated": False,
    }


def _snapshot(root: Path) -> str:
    """Bound fingerprint of project/config/asset bytes. Not a whole disk audit."""
    files = [root / (root.name + ".uproject"), root / "PHI_MCP_DISPOSABLE.json"]
    for folder in MONITORED:
        top = root / folder
        if not top.exists():
            continue
        _require(top.is_dir() and not top.is_symlink(),
                 "monitored project directory is not regular")
        for candidate in top.rglob("*"):
            _require(not candidate.is_symlink(),
                     "project contains symlink under monitored paths")
            if candidate.is_file():
                files.append(candidate)
            elif not candidate.is_dir():
                raise MCPGateError("unsupported project filesystem entry")
    _require(len(files) <= MAX_FILES, "too many files for disposable project pilot")
    total = 0
    records = []
    for path in sorted(files, key=lambda p: str(p)):
        _require(path.is_file() and not path.is_symlink(),
                 "monitored project file missing or symlinked")
        size = path.stat().st_size
        total += size
        _require(size <= MAX_FILE_BYTES and total <= MAX_TOTAL_BYTES,
                 "sandbox file inventory exceeds pilot limit")
        records.append((path.relative_to(root).as_posix(),
                        hashlib.sha256(path.read_bytes()).hexdigest()))
    return _digest(records)


def _safe_value(key: str, value) -> bool:
    if key in ("actor_class", "actor_class_path", "class", "class_path",
               "class_name"):
        return type(value) is str and value in (
            "PointLight", "/Script/Engine.PointLight"
        )
    if key in ("name", "label", "actor_label", "actor_name"):
        return type(value) is str and LABEL_RE.fullmatch(value) is not None
    if key in ("location", "rotation"):
        return (type(value) is dict and set(value) == {"x", "y", "z"}
                and all(type(v) in (int, float) and math.isfinite(v)
                        and abs(v) <= (360 if key == "rotation" else 1000)
                        for v in value.values()))
    return False


def validate_plan(plan: dict) -> None:
    _require(type(plan) is dict and set(plan) == {
        "schema", "project_sha256", "marker_sha256",
        "expected_instance_id", "expected_process_id",
        "tool_name", "tool_schema_sha256", "arguments"
    } and plan["schema"] == SCHEMA, "invalid R0.5 plan schema")
    _require(all(type(plan.get(k)) is str and SHA256.fullmatch(plan[k])
                 for k in ("project_sha256", "marker_sha256",
                           "tool_schema_sha256")), "missing pinned SHA256")
    _require(type(plan["expected_instance_id"]) is str and
             INSTANCE_RE.fullmatch(plan["expected_instance_id"]) and
             type(plan["expected_process_id"]) is int and
             0 < plan["expected_process_id"] <= 2147483647,
             "missing pinned editor process/instance")
    _require(type(plan["tool_name"]) is str and
             TOOL_RE.fullmatch(plan["tool_name"]) is not None,
             "only specific EditorToolset SpawnActor tools are eligible")
    args = plan["arguments"]
    _require(type(args) is dict and len(args) in (2, 3, 4),
             "actor arguments must be a tiny fixed shape")
    classes = [k for k in args if k in (
        "actor_class", "actor_class_path", "class", "class_path", "class_name")]
    names = [k for k in args if k in ("name", "label", "actor_label", "actor_name")]
    _require(len(classes) == 1 and len(names) == 1 and
             all(_safe_value(k, v) for k, v in args.items()),
             "only a PointLight with PhiMCPProbe_ name and bounded transform allowed")
    _require(len(json.dumps(args, allow_nan=False).encode("utf-8")) <= MAX_PLAN,
             "plan arguments too large")


def read_plan(filename: str | Path) -> dict:
    path = Path(filename)
    _require(path.is_file() and not path.is_symlink() and
             0 < path.stat().st_size <= MAX_PLAN, "bounded plan file required")
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise MCPGateError("invalid JSON plan") from exc
    validate_plan(plan)
    return plan


def inspect_instance(project_root: Path, endpoint: str, *,
                     approve_connect: bool = False,
                     timeout: int = 8, client_factory=LoopbackUnrealClient) -> dict:
    validate_endpoint(endpoint)
    _require(approve_connect is True, "explicit --approve-connect required")
    probe = project_probe(project_root, require_marker=True)
    root = Path(project_root).resolve(strict=True)
    client = client_factory(endpoint, timeout=timeout)
    observed = _local_identity(client.initialize(), root,
                               Path(probe["project_filename"]).stem)
    return {
        "schema": "phistream.mcp-unreal.instance-observation.v0.5",
        "status": "REPORTED_INSTANCE_NOT_AUTHENTICATED",
        "instance": observed,
        "project_sha256": probe["project_sha256"],
        "marker_sha256": probe["marker_sha256"],
        "tool_execution_authorized": False,
    }


def actor_attempt(project_root: Path, endpoint: str, audit: AuditLog,
                  intent_dir: Path, plan: dict, *,
                  approve_connect: bool = False, approve_write: bool = False,
                  confirmation: str = "", timeout: int = 8,
                  client_factory=LoopbackUnrealClient) -> dict:
    """At most one network write per pinned plan. Never retries on uncertainty."""
    validate_endpoint(endpoint)
    validate_plan(plan)
    if (approve_connect is not True or approve_write is not True
            or confirmation != CONFIRM):
        audit.append("unreal-actor-pilot", "DENIED_APPROVAL", "loopback",
                     tool=plan["tool_name"], arguments_sha256=_digest(plan))
        raise MCPGateError("operator connection, write and exact intent approvals required")
    probe = project_probe(project_root, require_marker=True)
    _require(probe["project_sha256"] == plan["project_sha256"] and
             probe["marker_sha256"] == plan["marker_sha256"],
             "project or marker no longer matches approved plan")
    root = Path(project_root).resolve(strict=True)
    _require(intent_dir.is_dir() and not intent_dir.is_symlink()
             and intent_dir.resolve() != root
             and root not in intent_dir.resolve().parents,
             "private intent directory must exist outside project")
    original_snapshot = _snapshot(root)
    intent_hash = _digest({
        "endpoint": endpoint, "plan": plan, "root": str(root),
    })
    reservation = intent_dir / (intent_hash + ".attempt")
    # Intentionally consumed before any network action. Crash/timeout remains HOLD.
    try:
        fd = os.open(reservation,
                     os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                     getattr(os, "O_NOFOLLOW", 0), 0o600)
    except FileExistsError as exc:
        raise MCPGateError("this actor intent was already attempted; never auto-retry") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"schema": RECEIPT, "status": "ATTEMPT_RESERVED_UNKNOWN",
                       "intent_sha256": intent_hash}, handle)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise MCPGateError("failed to durably reserve actor attempt")
    audit.append("unreal-actor-pilot", "STARTED", "loopback",
                 tool=plan["tool_name"], arguments_sha256=intent_hash)
    try:
        client = client_factory(endpoint, timeout=timeout)
        observed = _local_identity(client.initialize(), root,
                                   Path(probe["project_filename"]).stem)
        _require(observed["instance_id"] == plan["expected_instance_id"] and
                 observed["process_id"] == plan["expected_process_id"],
                 "connected instance/PID differ from operator pin")
        # Tool Search must be OFF in the disposable editor. Do not use
        # Epic's generic call_tool search dispatcher.
        listed = client.list_tools()
        matches = [item for item in listed if item.get("name") == plan["tool_name"]]
        _require(len(matches) == 1 and type(matches[0].get("inputSchema")) is dict,
                 "actor tool unavailable in direct tools/list; turn tool search off")
        _require(_digest(matches[0]["inputSchema"]) ==
                 plan["tool_schema_sha256"],
                 "live actor tool schema differs from operator pin")
        result = client._post("tools/call", {
            "name": plan["tool_name"], "arguments": plan["arguments"]
        })
        # 'isError' can be true and the editor may STILL have changed.
        result_hash = _digest(result)
        after = _snapshot(root)
        new_probe = project_probe(root, require_marker=True)
        anchor_unchanged = (
            new_probe["project_sha256"] == probe["project_sha256"] and
            new_probe["marker_sha256"] == probe["marker_sha256"]
        )
        receipt = {
            "schema": RECEIPT,
            "status": "WRITE_SENT_MANUAL_EDITOR_REVIEW_REQUIRED",
            "intent_sha256": intent_hash,
            "tool_name": plan["tool_name"],
            "arguments_sha256": _digest(plan["arguments"]),
            "response_sha256": result_hash,
            "tool_reported_error": result.get("isError") is True,
            "project_files_changed_in_monitored_scope": after != original_snapshot,
            "project_identity_anchors_unchanged": anchor_unchanged,
            "instance_metadata_matched_operator_pin": True,
            "instance_authenticated": False,
            "actor_exists_verified": False,
            "rollback_verified": False,
            "tool_calls_sent": 1,
            "automated_retry_allowed": False,
            "general_editor_dispatch_allowed": False,
            "production_authorized": False,
            "merge_authorized": False,
        }
        audit.append("unreal-actor-pilot", "REVIEW_REQUIRED", "loopback",
                     tool=plan["tool_name"], arguments_sha256=intent_hash,
                     result_sha256=_digest(receipt))
        return receipt
    except Exception as exc:
        audit.append("unreal-actor-pilot", "UNKNOWN_NEEDS_MANUAL_REVIEW", "loopback",
                     tool=plan["tool_name"], arguments_sha256=intent_hash)
        raise MCPGateError(
            "actor attempt blocked or outcome unknown; do NOT retry without manual inspection"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One disposable Unreal PointLight, one attempt only")
    sub = parser.add_subparsers(dest="action", required=True)
    identity = sub.add_parser("identity", help="Read-only reported editor identity")
    identity.add_argument("--project-root", type=Path, required=True)
    identity.add_argument("--endpoint", default="http://127.0.0.1:8000/mcp")
    identity.add_argument("--approve-connect", action="store_true")
    identity.add_argument("--timeout", type=int, default=8)
    run = sub.add_parser("attempt", help="One explicitly approved disposable write")
    run.add_argument("--project-root", type=Path, required=True)
    run.add_argument("--endpoint", default="http://127.0.0.1:8000/mcp")
    run.add_argument("--audit", type=Path, required=True)
    run.add_argument("--intent-dir", type=Path, required=True)
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--approve-connect", action="store_true")
    run.add_argument("--approve-write", action="store_true")
    run.add_argument("--confirm", default="")
    run.add_argument("--timeout", type=int, default=8)
    opts = parser.parse_args()
    try:
        if opts.action == "identity":
            response = inspect_instance(
                opts.project_root, opts.endpoint,
                approve_connect=opts.approve_connect, timeout=opts.timeout)
        else:
            response = actor_attempt(
                opts.project_root, opts.endpoint, AuditLog(opts.audit),
                opts.intent_dir, read_plan(opts.plan),
                approve_connect=opts.approve_connect,
                approve_write=opts.approve_write,
                confirmation=opts.confirm, timeout=opts.timeout)
        print(json.dumps(response, indent=2, sort_keys=True, allow_nan=False))
    except (MCPGateError, OSError, ValueError) as exc:
        print(json.dumps({"status": "HELD_UNVERIFIED",
                          "reason": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
