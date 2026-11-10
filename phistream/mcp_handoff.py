"""MCP R0.2: offline, non-authorizing handoff from existing fleet evidence.

Imports Agentora R9 public mission status or PhiMadaPrime R31 correlation
records as *untrusted snapshots*. It NEVER launches an MCP client, reads a
private lease, verifies live GitHub history, or authorizes tool execution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from .mcp_gateway import MCPGateError, _digest, load_manifest

SCHEMA = "phistream.mcp-handoff.review.v0.2"
AGENTORA_SCHEMA = "agentora.fleet.mission-control.v0.9"
PHIMADA_SCHEMA = "phimada.agentora-r1-chronicle-correlation.v0.31"
_SHA = re.compile(r"[0-9a-f]{40}\Z")
_MISSION_ID = re.compile(r"[A-Za-z0-9_.-]{1,80}\Z")
MAX_RECEIPT_BYTES = 128 * 1024
# Only these fixed, previously-existing status contracts are accepted.
AGENTORA_KINDS = frozenset({
    "PUBLIC_STANDALONE_READINESS_AUDIT",
    "PHIMADA_FIXED_TEST",
    "PHIMADA_MODEL_PROPOSAL_REVIEW",
})
KNOWN_STATES = frozenset({
    "QUEUED", "CLAIMING_IMPLEMENT", "ASSIGNED_IMPLEMENT", "HANDOFF_IN_PROGRESS",
    "QA_QUEUED", "CLAIMING_QA", "ASSIGNED_QA", "QA_HELD", "HELD_RECONCILE",
    "ARCHIVED_NOT_APPROVED",
})


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise MCPGateError(reason)


def _sha(value: object) -> bool:
    return type(value) is str and _SHA.fullmatch(value) is not None


def _false_flags(record: dict, names: tuple[str, ...]) -> None:
    _need(all(record.get(name) is False for name in names),
          "source record claims unauthorized or unknown capabilities")


def read_receipt(filename: str | Path) -> dict:
    """Bound local input. Do not trust file name, provenance or copied approvals."""
    path = Path(filename)
    _need(path.is_file() and not path.is_symlink() and
          0 < path.stat().st_size <= MAX_RECEIPT_BYTES,
          "source receipt missing, linked, or oversized")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise MCPGateError("source receipt is not valid UTF-8 JSON") from exc
    _need(type(obj) is dict, "source receipt must be a JSON object")
    return obj


def _agentora(source: dict, mission_id: str) -> dict:
    _need(source.get("schema") == AGENTORA_SCHEMA and
          source.get("scope") == "public_only" and
          type(source.get("source_scan_id")) is int and
          source["source_scan_id"] >= 1 and
          type(source.get("missions")) is list and
          len(source["missions"]) <= 200 and
          source.get("actual_jobs_dispatched") == 0,
          "not an Agentora R9 public status snapshot")
    _false_flags(source, ("merge_authorized", "verified_complete"))
    _need(type(mission_id) is str and _MISSION_ID.fullmatch(mission_id),
          "invalid mission_id")
    missions = [x for x in source["missions"] if
                type(x) is dict and x.get("mission_id") == mission_id]
    _need(len(missions) == 1, "mission absent or duplicated in status snapshot")
    item = missions[0]
    _need(type(item.get("github_id")) is int and item["github_id"] > 0 and
          type(item.get("repository")) is str and
          re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
                       item["repository"]) is not None and
          item.get("kind") in AGENTORA_KINDS and
          item.get("state") in KNOWN_STATES and
          _sha(item.get("source_sha")),
          "Agentora mission identity or source SHA invalid")
    _need(item["state"] == "QUEUED",
          "Agentora mission is not in the review-only QUEUED state")
    return {
        "source": "agentora-r9-public-status",
        "repository": item["repository"],
        "repository_id": item["github_id"],
        "source_sha": item["source_sha"],
        "work_id": item["mission_id"],
        "evidence_kind": item["kind"],
        "source_state": item["state"],
    }


def _phimada(source: dict) -> dict:
    _need(source.get("schema") == PHIMADA_SCHEMA and
          source.get("state") == "LIVE_GITHUB_R1_RETURN_CORRELATED" and
          source.get("repository") == "MichaelWave369/Agentora" and
          type(source.get("repositoryId")) is int and
          source["repositoryId"] > 0 and
          type(source.get("prNumber")) is int and
          source["prNumber"] > 0 and
          source.get("relation") in (
              "VERIFIED_TESTED_PR_HEAD", "VERIFIED_MERGE_COMMIT",
          ) and
          all(_sha(source.get(name)) for name in (
              "selectedSha", "sourceSha", "testedSha"
          )) and
          (source.get("mergeSha") is None or _sha(source["mergeSha"])),
          "not a valid PhiMadaPrime R31 snapshot")
    _false_flags(source, (
        "agentIdentityAuthenticated", "aiModelExecutionProven",
        "browserExecutedWork", "productVerifiedComplete",
        "mergeAuthorized", "dispatchAuthorized",
    ))
    relation = source["relation"]
    _need((relation == "VERIFIED_TESTED_PR_HEAD" and
           source["selectedSha"] == source["testedSha"]) or
          (relation == "VERIFIED_MERGE_COMMIT" and
           source["mergeSha"] is not None and
           source["selectedSha"] == source["mergeSha"]),
          "PhiMadaPrime relation/SHA mismatch")
    return {
        "source": "phimada-r31-chronicle",
        "repository": source["repository"],
        "repository_id": source["repositoryId"],
        "source_sha": source["selectedSha"],
        "work_id": f"agentora-pr-{source['prNumber']}",
        "evidence_kind": relation,
        "source_state": source["state"],
    }


def review_handoff(source: dict, *, kind: str, manifest: dict,
                   server: str, tool: str, expected_sha: str,
                   mission_id: str | None = None) -> dict:
    """Return a proposal; NO execution API or approval conversion exists."""
    _need(type(source) is dict and type(manifest) is dict,
          "source and manifest must be objects")
    _need(_sha(expected_sha), "expected source SHA must be 40 lowercase hex")
    _need(type(server) is str and server in manifest.get("servers", {}),
          "MCP server is absent from operator manifest")
    config = manifest["servers"][server]
    _need(type(config) is dict and type(config.get("allowed_tools")) is dict,
          "invalid manifest server policy")
    _need(type(tool) is str and tool in config["allowed_tools"],
          "requested MCP tool is not operator allowlisted")
    risk = config["allowed_tools"][tool]
    _need(risk in ("read", "write"), "invalid tool risk classification")
    if kind == "agentora-r9":
        evidence = _agentora(source, mission_id)
    elif kind == "phimada-r31":
        _need(mission_id is None, "PhiMadaPrime R31 has no R9 mission ID")
        evidence = _phimada(source)
    else:
        raise MCPGateError("unknown handoff source type")
    _need(evidence["source_sha"] == expected_sha,
          "imported source SHA does not match operator-selected SHA")
    return {
        "schema": SCHEMA,
        "status": "HELD_OPERATOR_REVIEW_ONLY",
        "evidence": evidence,
        "proposed_mcp": {"server": server, "tool": tool,
                         "operator_declared_risk": risk},
        "bindings": {
            "source_receipt_sha256": _digest(source),
            "operator_manifest_sha256": _digest(manifest),
            "expected_source_sha": expected_sha,
        },
        "trust": {
            "imported_snapshot_authenticated": False,
            "current_git_head_revalidated": False,
            "mcp_server_attested": False,
            "tool_effects_verified": False,
            "runtime_approval_granted": False,
        },
        "execution_authorized": False,
        "server_launch_authorized": False,
        "dispatch_authorized": False,
        "code_write_authorized": False,
        "merge_authorized": False,
        "deployment_authorized": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="PhiStream R0.2 offline Agentora/PhiMadaPrime review handoff")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-kind", choices=["agentora-r9", "phimada-r31"],
                   required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--server", required=True)
    p.add_argument("--tool", required=True)
    p.add_argument("--expected-sha", required=True)
    p.add_argument("--mission-id", help="Required for an Agentora R9 status")
    args = p.parse_args()
    try:
        result = review_handoff(
            read_receipt(args.source), kind=args.source_kind,
            manifest=load_manifest(args.manifest),
            server=args.server, tool=args.tool,
            expected_sha=args.expected_sha, mission_id=args.mission_id,
        )
    except (MCPGateError, ValueError, OSError) as exc:
        p.error(str(exc))
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
