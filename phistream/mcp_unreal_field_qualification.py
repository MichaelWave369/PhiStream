"""PhiStream MCP R0.7: offline operator field-qualification packet for Unreal.

This module cannot see a real Unreal Editor, validate screenshot contents,
authenticate a human reviewer, verify an actor, or certify editor rollback.
It binds operator-supplied evidence to R0.5/R0.6 local file receipts and
keeps automated verification distinct from human-reported observations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys

from .mcp_gateway import MCPGateError, _digest
from .mcp_unreal_actor_pilot import read_plan
from .mcp_unreal_discovery import validate_endpoint
from .mcp_unreal_qualification import project_probe
from .mcp_unreal_recovery import (
    _dir, _separate, inventory, read_baseline, review_actor_attempt,
)

SCHEMA = "phistream.mcp-unreal.field-observation.v0.7"
PACKET = "phistream.mcp-unreal.field-qualification.v0.7"
FIELDS = (
    "original_editor_project_visually_matched",
    "test_actor_visible_in_original_outliner",
    "test_actor_class_is_pointlight",
    "test_actor_label_matched_plan",
    "original_level_saved_after_attempt",
    "restored_copy_opened_in_unreal_editor",
    "restored_map_loaded_without_error",
    "test_actor_absent_in_restored_outliner",
)
IMAGES = (
    "original-outliner.png",
    "original-actor-details.png",
    "original-level-saved.png",
    "recovered-outliner.png",
)
PNG = b"\x89PNG\r\n\x1a\n"
MAX_SCREENSHOT = 8 * 1024 * 1024
MAX_RECORD = 4096
LABEL = re.compile(r"PhiMCPProbe_[A-Za-z0-9_]{1,40}\Z")


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise MCPGateError(reason)


def _read_record(path: str | Path) -> dict:
    record = Path(path)
    _need(record.is_file() and not record.is_symlink() and
          0 < record.stat().st_size <= MAX_RECORD,
          "operator observation must be a bounded, non-symlink JSON file")
    try:
        value = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise MCPGateError("invalid operator observation JSON") from exc
    _need(type(value) is dict and set(value) == {
        "schema", "expected_actor_label", "observations"
    } and value["schema"] == SCHEMA and
          type(value["expected_actor_label"]) is str and
          LABEL.fullmatch(value["expected_actor_label"]) is not None and
          type(value["observations"]) is dict and
          set(value["observations"]) == set(FIELDS) and
          all(type(value["observations"][field]) is bool for field in FIELDS),
          "operator observation contract invalid or missing checks")
    return value


def create_template(path: str | Path, plan_path: str | Path) -> dict:
    """Create review form once; every assertion defaults to false."""
    plan = read_plan(plan_path)
    args = plan["arguments"]
    labels = [v for key, v in args.items()
              if key in ("name", "label", "actor_label", "actor_name")]
    _need(len(labels) == 1, "R0.5 plan must specify exactly one label")
    record = {
        "schema": SCHEMA,
        "expected_actor_label": labels[0],
        "observations": {field: False for field in FIELDS},
    }
    dest = Path(path)
    _need(dest.is_absolute() and dest.parent.is_dir()
          and not dest.parent.is_symlink(), "private form parent must already exist")
    try:
        with dest.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as exc:
        raise MCPGateError("operator form exists; will never overwrite") from exc
    return {"schema": SCHEMA, "status": "EMPTY_OPERATOR_FORM_CREATED",
            "all_checks_default_false": True,
            "operator_attestation_received": False}


def _image_digest(path: Path) -> dict:
    _need(path.is_file() and not path.is_symlink() and
          33 <= path.stat().st_size <= MAX_SCREENSHOT,
          "screenshot missing, symlinked or too large")
    # Parse only the fixed PNG signature/IHDR dimensions, not image content.
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
            _need(len(header) == 24 and header[:8] == PNG and
                  header[12:16] == b"IHDR", "screenshot is not PNG with IHDR")
            width, height = struct.unpack(">II", header[16:24])
            _need(320 <= width <= 8192 and 200 <= height <= 8192,
                  "screenshot dimensions outside bounded review window")
            h = hashlib.sha256()
            h.update(header)
            total = len(header)
            while True:
                part = handle.read(1024 * 1024)
                if not part:
                    break
                total += len(part)
                _need(total <= MAX_SCREENSHOT, "screenshot exceeded review limit")
                h.update(part)
        _need(path.stat().st_size == total,
              "screenshot changed while being hashed")
    except OSError as exc:
        raise MCPGateError("screenshot cannot be read") from exc
    return {"sha256": h.hexdigest(), "bytes": total,
            "reported_dimensions": [width, height],
            "pixel_contents_verified": False,
            "real_editor_origin_verified": False}


def screenshot_bundle(directory: str | Path, *, separate_from: tuple[Path, ...]) -> dict:
    folder = _dir(directory)
    _separate(folder, *separate_from)
    return {name: _image_digest(folder / name) for name in IMAGES}


def field_assessment(
    snapshot_dir: str | Path, project_root: str | Path,
    recovered_project_root: str | Path, plan_path: str | Path,
    intent_dir: str | Path, endpoint: str,
    operator_record: str | Path, screenshot_dir: str | Path,
    *, receipt_path: str | Path | None = None,
) -> dict:
    """Bind local evidence and operator testimony. Never grant execution."""
    endpoint = validate_endpoint(endpoint)
    snap, baseline = read_baseline(snapshot_dir)
    root = _dir(project_root)
    recovered = _dir(recovered_project_root)
    _need(root.name == baseline["project"] == recovered.name,
          "original, restored and baseline project names must match")
    _need(str(root) == baseline["capture_source_root"],
          "original root must match the captured baseline root")
    _separate(recovered, root, snap)
    _need(recovered != root, "recovery must be a separate scratch copy")
    project_probe(recovered, require_marker=True)
    _need(inventory(recovered) == baseline["files"],
          "recovered copy does not match baseline bytes")
    report = review_actor_attempt(snap, root, plan_path, intent_dir,
                                  endpoint, receipt_path)
    plan = read_plan(plan_path)
    record = _read_record(operator_record)
    labels = [v for k, v in plan["arguments"].items()
              if k in ("name", "label", "actor_label", "actor_name")]
    _need(len(labels) == 1 and record["expected_actor_label"] == labels[0],
          "human observation references a different planned actor")
    pictures = screenshot_bundle(
        screenshot_dir, separate_from=(root, recovered, snap))
    all_checked = all(record["observations"].values())
    # Without a saved .umap delta, an in-memory actor can evaporate on exit.
    saved_map_delta = any(
        item["path"].endswith(".umap") and
        item["kind"] in ("added", "modified")
        for item in report["changes"]
    )
    _need(not report["changes_truncated"],
          "too many changed files to qualify scratch scene evidence")
    no_tool_error = (
        report["r05_receipt_correlated_as_self_report"] and
        report["tool_reported_error_untrusted"] is False
    )
    anchors_ok = report["project_identity_anchors_unchanged"]
    candidate = (all_checked and saved_map_delta and
                 no_tool_error and anchors_ok)
    return {
        "schema": PACKET,
        "status": ("OPERATOR_REPORTED_FIELD_CHECKS_CONSISTENT_UNVERIFIED"
                   if candidate else "HOLD_FIELD_CHECKS_INCOMPLETE_OR_CONFLICTING"),
        "baseline_inventory_sha256": baseline["inventory_sha256"],
        "attempt_intent_sha256": report["actor_intent_sha256"],
        "original_current_inventory_sha256": report["current_inventory_sha256"],
        "recovered_inventory_sha256": _digest(inventory(recovered)),
        "recovered_disk_subset_matches_baseline": True,
        "original_project_anchors_unchanged": anchors_ok,
        "saved_umap_delta_observed": saved_map_delta,
        "r05_receipt_correlated_as_self_report":
            report["r05_receipt_correlated_as_self_report"],
        "tool_reported_error_untrusted": report["tool_reported_error_untrusted"],
        "operator_observation_record_sha256": _digest(record),
        "operator_assertions": record["observations"],
        "operator_reported_all_checks_passed": all_checked,
        "operator_reported_actor_label": record["expected_actor_label"],
        "screenshot_evidence": pictures,
        "screenshot_contents_independently_interpreted": False,
        "human_identity_authenticated": False,
        "actual_editor_session_authenticated": False,
        "actor_existence_automatically_verified": False,
        "restored_level_boot_automatically_verified": False,
        "field_test_independently_qualified": False,
        "automatic_retry_authorized": False,
        "editor_write_authorized": False,
        "production_authorized": False,
        "merge_authorized": False,
        "next_action": (
            "Independently inspect captured screenshots and a live disposable "
            "editor session; do not promote this packet into write authority."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unreal R0.7 offline human field-test evidence gate")
    sub = parser.add_subparsers(dest="command", required=True)
    form = sub.add_parser("template")
    form.add_argument("--plan", required=True, type=Path)
    form.add_argument("--output", required=True, type=Path)
    check = sub.add_parser("assess")
    check.add_argument("--snapshot", required=True, type=Path)
    check.add_argument("--project-root", required=True, type=Path)
    check.add_argument("--recovered-root", required=True, type=Path)
    check.add_argument("--plan", required=True, type=Path)
    check.add_argument("--intent-dir", required=True, type=Path)
    check.add_argument("--endpoint", default="http://127.0.0.1:8000/mcp")
    check.add_argument("--operator-record", required=True, type=Path)
    check.add_argument("--screenshots", required=True, type=Path)
    check.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "template":
            result = create_template(args.output, args.plan)
        else:
            result = field_assessment(
                args.snapshot, args.project_root, args.recovered_root,
                args.plan, args.intent_dir, args.endpoint,
                args.operator_record, args.screenshots,
                receipt_path=args.receipt)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    except (MCPGateError, OSError, ValueError) as exc:
        print(json.dumps({"status": "HOLD_NO_EDITOR_EFFECT",
                          "reason": str(exc)}), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
