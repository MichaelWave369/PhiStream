"""PhiStream MCP R0.6: offline, bounded evidence + scratch-only restore.

Does not connect to Unreal, parse .umap, prove actor existence, or overwrite
a running editor project. A snapshot is a local, operator-controlled copy,
not an authenticated backup or a complete UE recovery image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys

from .mcp_gateway import MCPGateError, _digest
from .mcp_unreal_qualification import project_probe
from .mcp_unreal_actor_pilot import (
    MAX_FILES, MAX_FILE_BYTES, MAX_TOTAL_BYTES, MONITORED, RECEIPT,
    read_plan,
)
from .mcp_unreal_discovery import validate_endpoint

SCHEMA = "phistream.mcp-unreal.recovery-baseline.v0.6"
REPORT = "phistream.mcp-unreal.evidence-review.v0.6"
RESTORE = "phistream.mcp-unreal.recovery-copy.v0.6"
ACTOR_RESULT = "phistream.mcp-unreal.actor-attempt.v0.5"
PROJECT_RE = re.compile(r"PhiMCPSandbox[A-Za-z0-9_-]{1,64}\Z")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
PATH_PART = re.compile(r"[A-Za-z0-9_ .()\-]{1,128}\Z")
MANIFEST_MAX_BYTES = 256 * 1024
ATTEMPT_MAX_BYTES = 16 * 1024
MAX_CHANGES_DISPLAY = 80


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise MCPGateError(reason)


def _dir(value: str | Path) -> Path:
    raw = Path(value).expanduser()
    _need(raw.is_absolute() and raw.is_dir() and not raw.is_symlink(),
          "required absolute, existing, non-symlink directory")
    return raw.resolve(strict=True)


def _separate(parent: Path, *roots: Path) -> None:
    _need(all(parent != root and root not in parent.parents and
              parent not in root.parents for root in roots),
          "backup/destination directory must be separate from project and snapshot")


def _path_ok(relative: str, project: str) -> bool:
    if type(relative) is not str or "\\" in relative or ":" in relative:
        return False
    normalized = PurePosixPath(relative)
    parts = normalized.parts
    if (relative != normalized.as_posix() or not parts or
            any(p in (".", "..") or PATH_PART.fullmatch(p) is None
                for p in parts)):
        return False
    if len(parts) == 1:
        return parts[0] in (project + ".uproject", "PHI_MCP_DISPOSABLE.json")
    return parts[0] in MONITORED and len(parts) >= 2


def _sha_file(path: Path) -> tuple[int, str]:
    _need(path.is_file() and not path.is_symlink(),
          "symlink/special file in monitored project scope")
    before = path.stat()
    _need(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_FILE_BYTES,
          "file exceeds scratch pilot size limit or is not regular")
    hash_ = hashlib.sha256()
    size = 0
    # O_NOFOLLOW protects final component on POSIX. This is not a sandbox.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        with os.fdopen(os.open(path, flags), "rb") as stream:
            while True:
                piece = stream.read(1024 * 1024)
                if not piece:
                    break
                size += len(piece)
                _need(size <= MAX_FILE_BYTES, "file changed beyond pilot limit")
                hash_.update(piece)
    except OSError as exc:
        raise MCPGateError("unable to hash monitored file") from exc
    after = path.stat()
    _need(before.st_size == size == after.st_size and
          before.st_mtime_ns == after.st_mtime_ns and
          before.st_ino == after.st_ino,
          "file changed while being hashed; no reliable snapshot")
    return size, hash_.hexdigest()


def inventory(project_root: str | Path) -> list[dict]:
    root = _dir(project_root)
    project = root.name
    _need(PROJECT_RE.fullmatch(project) is not None,
          "only a named PhiMCPSandbox* project can be inventoried")
    paths = [root / (project + ".uproject"),
             root / "PHI_MCP_DISPOSABLE.json"]
    for group in MONITORED:
        folder = root / group
        if not folder.exists() and not folder.is_symlink():
            continue
        _need(folder.is_dir() and not folder.is_symlink(),
              "monitored project folder is not an ordinary directory")
        for here, subdirs, files in os.walk(folder, followlinks=False):
            base = Path(here)
            for name in subdirs:
                _need(not (base / name).is_symlink(),
                      "symlink in monitored asset tree")
            for name in files:
                candidate = base / name
                _need(not candidate.is_symlink(),
                      "symlink in monitored asset tree")
                paths.append(candidate)
            _need(len(paths) <= MAX_FILES,
                  "too many project files for bounded recovery")
    records = []
    total = 0
    for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        if not path.exists() and not path.is_symlink():
            continue  # A deleted anchor is reported in after-state comparison.
        rel = path.relative_to(root).as_posix()
        _need(_path_ok(rel, project), "unsafe or unsupported project path")
        size, digest = _sha_file(path)
        total += size
        _need(len(records) < MAX_FILES and total <= MAX_TOTAL_BYTES,
              "project exceeds scratch recovery limits")
        records.append({"path": rel, "bytes": size, "sha256": digest})
    return records


def _validate_records(records: object, project: str) -> list[dict]:
    _need(type(records) is list and 2 <= len(records) <= MAX_FILES,
          "invalid baseline file count")
    paths = []
    total = 0
    for r in records:
        _need(type(r) is dict and set(r) == {"path", "bytes", "sha256"} and
              _path_ok(r["path"], project) and
              type(r["bytes"]) is int and 0 <= r["bytes"] <= MAX_FILE_BYTES and
              type(r["sha256"]) is str and SHA_RE.fullmatch(r["sha256"]),
              "invalid baseline manifest entry")
        paths.append(r["path"])
        total += r["bytes"]
    _need(len(set(paths)) == len(paths) and paths == sorted(paths) and
          project + ".uproject" in paths and
          "PHI_MCP_DISPOSABLE.json" in paths and
          total <= MAX_TOTAL_BYTES, "invalid duplicate/order/anchor/size in baseline")
    return records


def read_baseline(snapshot_dir: str | Path) -> tuple[Path, dict]:
    snapshot = _dir(snapshot_dir)
    manifest_file = snapshot / "baseline.json"
    _need(manifest_file.is_file() and not manifest_file.is_symlink() and
          0 < manifest_file.stat().st_size <= MANIFEST_MAX_BYTES,
          "complete baseline manifest missing or too large")
    try:
        doc = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise MCPGateError("invalid baseline JSON") from exc
    _need(type(doc) is dict and set(doc) == {
        "schema", "project", "capture_source_root", "files",
        "inventory_sha256", "snapshot_complete",
    } and doc["schema"] == SCHEMA and doc["snapshot_complete"] is True and
          type(doc["project"]) is str and
          PROJECT_RE.fullmatch(doc["project"]) is not None and
          type(doc["capture_source_root"]) is str and
          Path(doc["capture_source_root"]).is_absolute(),
          "invalid baseline header")
    _validate_records(doc["files"], doc["project"])
    _need(_digest(doc["files"]) == doc["inventory_sha256"],
          "baseline manifest digest mismatch")
    payload = snapshot / "payload" / doc["project"]
    _need(payload.is_dir() and not payload.is_symlink(),
          "baseline payload project absent or symlinked")
    _need(inventory(payload) == doc["files"],
          "baseline payload bytes or files differ from manifest")
    project_probe(payload, require_marker=True)
    return snapshot, doc


def capture_baseline(project_root: str | Path,
                     snapshot_parent: str | Path, name: str) -> dict:
    """Copy only bounded .uproject, marker, Content, Config and Plugins."""
    root = _dir(project_root)
    parent = _dir(snapshot_parent)
    _separate(parent, root)
    _need(PROJECT_RE.fullmatch(root.name) is not None,
          "source must be a disposable PhiMCPSandbox project")
    _need(type(name) is str and
          re.fullmatch(r"r06-[A-Za-z0-9_-]{4,64}", name) is not None,
          "snapshot name must be r06-* and unique")
    project_probe(root, require_marker=True)
    before = inventory(root)
    _validate_records(before, root.name)
    target = parent / name
    try:
        target.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise MCPGateError("snapshot already exists; never overwrite") from exc
    copied = target / "payload" / root.name
    copied.mkdir(parents=True, exist_ok=False)
    # A partially created target is deliberately left for human reconciliation.
    for item in before:
        source = root / Path(item["path"])
        dest = copied / Path(item["path"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        _need(not source.is_symlink() and source.is_file(),
              "source changed during snapshot copy")
        with source.open("rb") as r, dest.open("xb") as w:
            shutil.copyfileobj(r, w, 1024 * 1024)
        _need(_sha_file(dest) == (item["bytes"], item["sha256"]),
              "snapshot copy not byte-for-byte equal")
    _need(inventory(root) == before and inventory(copied) == before,
          "project changed during baseline capture")
    record = {
        "schema": SCHEMA,
        "project": root.name,
        "capture_source_root": str(root),
        "files": before,
        "inventory_sha256": _digest(before),
        "snapshot_complete": True,
    }
    raw = json.dumps(record, sort_keys=True, indent=2, allow_nan=False)
    _need(len(raw.encode("utf-8")) <= MANIFEST_MAX_BYTES,
          "complete snapshot metadata too large")
    with (target / "baseline.json").open("x", encoding="utf-8") as out:
        out.write(raw)
    read_baseline(target)  # Independently read and rehash the completed copy.
    return {
        "schema": SCHEMA, "status": "BOUNDED_BASELINE_COPY_VERIFIED",
        "project": root.name, "file_count": len(before),
        "inventory_sha256": record["inventory_sha256"],
        "snapshot_directory": str(target),
        "all_unreal_files_backed_up": False,
        "live_editor_stopped_verified": False,
    }


def compare_baseline(snapshot_dir: str | Path, project_root: str | Path) -> dict:
    _, doc = read_baseline(snapshot_dir)
    root = _dir(project_root)
    _need(root.name == doc["project"], "wrong project name for comparison")
    current = inventory(root)
    old = {r["path"]: r for r in doc["files"]}
    new = {r["path"]: r for r in current}
    added = sorted(set(new) - set(old))
    missing = sorted(set(old) - set(new))
    changed = sorted(k for k in set(old) & set(new) if old[k] != new[k])
    display = [{"path": p, "kind": "added" if p in added else
                "missing" if p in missing else "modified"}
               for p in sorted(added + missing + changed)[:MAX_CHANGES_DISPLAY]]
    old_project = doc["project"] + ".uproject"
    anchors_equal = all(old.get(k) == new.get(k) for k in (
        old_project, "PHI_MCP_DISPOSABLE.json"))
    return {
        "schema": REPORT,
        "status": "FILESYSTEM_SCOPE_COMPARED_ACTOR_NOT_VERIFIED",
        "baseline_inventory_sha256": doc["inventory_sha256"],
        "current_inventory_sha256": _digest(current),
        "monitored_files_equal": not (added or missing or changed),
        "project_identity_anchors_unchanged": anchors_equal,
        "counts": {"added": len(added), "missing": len(missing),
                   "modified": len(changed)},
        "changes": display,
        "changes_truncated": len(added) + len(missing) + len(changed) > MAX_CHANGES_DISPLAY,
        "actor_existence_verified": False,
        "in_memory_world_state_verified": False,
        "saved_ue_map_semantics_verified": False,
        "entire_project_scanned": False,
        "production_authorized": False,
    }


def _read_json_file(path: Path, max_bytes: int) -> dict:
    _need(path.is_file() and not path.is_symlink() and
          0 < path.stat().st_size <= max_bytes,
          "required receipt/intent JSON absent, symlinked, or too large")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise MCPGateError("invalid receipt/intent JSON") from exc
    _need(type(doc) is dict, "receipt/intent must be JSON object")
    return doc


def review_actor_attempt(snapshot_dir: str | Path, project_root: str | Path,
                         plan_path: str | Path, intent_dir: str | Path,
                         endpoint: str, receipt_path: str | Path | None = None) -> dict:
    """Correlate R0.5 immutable intent with independent R0.6 file scan.

    Does not trust the R0.5 return content as proof of an Unreal actor.
    """
    endpoint = validate_endpoint(endpoint)
    _, doc = read_baseline(snapshot_dir)
    root = _dir(project_root)
    _need(root.name == doc["project"] and
          str(root) == doc["capture_source_root"],
          "only the originally captured scratch root can be reviewed")
    plan = read_plan(plan_path)
    baseline = {r["path"]: r["sha256"] for r in doc["files"]}
    _need(plan["project_sha256"] == baseline[doc["project"] + ".uproject"] and
          plan["marker_sha256"] == baseline["PHI_MCP_DISPOSABLE.json"],
          "actor plan not bound to baseline project identity")
    intents = _dir(intent_dir)
    _separate(intents, root)
    intent_hash = _digest({"endpoint": endpoint, "plan": plan, "root": str(root)})
    reservation = _read_json_file(intents / (intent_hash + ".attempt"),
                                  ATTEMPT_MAX_BYTES)
    _need(reservation == {
        "schema": RECEIPT, "status": "ATTEMPT_RESERVED_UNKNOWN",
        "intent_sha256": intent_hash,
    }, "invalid or mismatched one-shot actor reservation")
    receipt_matched = False
    tool_reported_error = None
    if receipt_path is not None:
        receipt = _read_json_file(Path(receipt_path), ATTEMPT_MAX_BYTES)
        _need(receipt.get("schema") == ACTOR_RESULT and
              receipt.get("status") == "WRITE_SENT_MANUAL_EDITOR_REVIEW_REQUIRED"
              and receipt.get("intent_sha256") == intent_hash
              and receipt.get("tool_name") == plan["tool_name"]
              and receipt.get("arguments_sha256") == _digest(plan["arguments"])
              and type(receipt.get("tool_reported_error")) is bool
              and receipt.get("tool_calls_sent") == 1
              and receipt.get("actor_exists_verified") is False
              and receipt.get("rollback_verified") is False
              and receipt.get("automated_retry_allowed") is False
              and receipt.get("production_authorized") is False,
              "R0.5 receipt is inconsistent or claims unsupported authority")
        receipt_matched = True
        tool_reported_error = receipt["tool_reported_error"]
    comparison = compare_baseline(snapshot_dir, root)
    return {
        **comparison,
        "status": "HOLD_MANUAL_ACTOR_INSPECTION_AND_RECOVERY_REVIEW",
        "actor_intent_sha256": intent_hash,
        "one_shot_reservation_correlated": True,
        "r05_receipt_correlated_as_self_report": receipt_matched,
        "tool_reported_error_untrusted": tool_reported_error,
        "actor_exists_verified": False,
        "rollback_verified": False,
        "automatic_retry_authorized": False,
        "independent_editor_readback_performed": False,
        "next_action": "Inspect the disposable Unreal World Outliner and map manually; restore to a separate scratch copy",
    }


def restore_to_new_copy(snapshot_dir: str | Path,
                        destination_parent: str | Path) -> dict:
    """Reproduce snapshot's bounded files in new path, never overwrite source."""
    snap, doc = read_baseline(snapshot_dir)
    parent = _dir(destination_parent)
    original = Path(doc["capture_source_root"]).resolve(strict=False)
    _separate(parent, snap, original)
    dest = parent / doc["project"]
    try:
        dest.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise MCPGateError("recovery target already exists; never overwrite") from exc
    payload = snap / "payload" / doc["project"]
    for record in doc["files"]:
        src = payload / Path(record["path"])
        dst = dest / Path(record["path"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as reader, dst.open("xb") as writer:
            shutil.copyfileobj(reader, writer, 1024 * 1024)
        _need(_sha_file(dst) == (record["bytes"], record["sha256"]),
              "recovery target differs from baseline")
    _need(inventory(dest) == doc["files"],
          "recovery project does not match baseline files")
    project_probe(dest, require_marker=True)
    return {
        "schema": RESTORE,
        "status": "NEW_SCRATCH_COPY_MONITORED_FILES_VERIFIED",
        "restored_copy": str(dest),
        "file_count": len(doc["files"]),
        "verified_inventory_sha256": doc["inventory_sha256"],
        "original_live_project_overwritten": False,
        "original_live_project_deleted": False,
        "all_ue_files_recovered": False,
        "in_memory_editor_state_recovered": False,
        "editor_boot_verified": False,
        "actor_absence_verified": False,
        "production_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unreal R0.6 offline scratch evidence and new-copy recovery")
    sub = parser.add_subparsers(dest="action", required=True)
    snap = sub.add_parser("capture", help="BEFORE edit: preserve bounded source files")
    snap.add_argument("--project-root", type=Path, required=True)
    snap.add_argument("--snapshot-parent", type=Path, required=True)
    snap.add_argument("--name", required=True)
    compare = sub.add_parser("compare", help="AFTER edit: read-only file delta")
    compare.add_argument("--snapshot", type=Path, required=True)
    compare.add_argument("--project-root", type=Path, required=True)
    review = sub.add_parser("review", help="Correlate R0.5 attempt with disk delta")
    review.add_argument("--snapshot", type=Path, required=True)
    review.add_argument("--project-root", type=Path, required=True)
    review.add_argument("--plan", type=Path, required=True)
    review.add_argument("--intent-dir", type=Path, required=True)
    review.add_argument("--endpoint", default="http://127.0.0.1:8000/mcp")
    review.add_argument("--receipt", type=Path)
    restore = sub.add_parser("restore-copy", help="Restore into a NEW separate folder")
    restore.add_argument("--snapshot", type=Path, required=True)
    restore.add_argument("--destination-parent", type=Path, required=True)
    opts = parser.parse_args()
    try:
        if opts.action == "capture":
            out = capture_baseline(opts.project_root, opts.snapshot_parent, opts.name)
        elif opts.action == "compare":
            out = compare_baseline(opts.snapshot, opts.project_root)
        elif opts.action == "review":
            out = review_actor_attempt(opts.snapshot, opts.project_root, opts.plan,
                                       opts.intent_dir, opts.endpoint, opts.receipt)
        else:
            out = restore_to_new_copy(opts.snapshot, opts.destination_parent)
        print(json.dumps(out, sort_keys=True, indent=2, allow_nan=False))
    except (MCPGateError, OSError, ValueError) as exc:
        print(json.dumps({"status": "HELD_NO_LIVE_PROJECT_OVERWRITE",
                          "reason": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
