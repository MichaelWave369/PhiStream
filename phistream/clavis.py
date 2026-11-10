"""CLAVIS Φ369 R0.1: deterministic, offline cognitive *planning*, not model routing.

This module neither chooses a production model nor invokes inference, tools,
network services, training, or adapter mixing. GA108/Vessie retain routing
authority; the output is an untrusted proposal awaiting external approval.
"""
from __future__ import annotations

import argparse
import json
import math

SCHEMA = "phistream.clavis.route-plan.v0.1"
MEMORY_SCHEMA = "phistream.clavis.volatile-memory.v0.1"

CHAMBERS = (
    "perception", "memory", "reason", "imagination",
    "synthesis", "verification", "reflection",
)
MODES = {
    "discovery": ("perception", "memory", "reason", "imagination",
                  "synthesis", "verification", "reflection"),
    "engineering": ("perception", "memory", "reason", "synthesis",
                    "verification", "reflection"),
    "creative": ("perception", "memory", "imagination", "synthesis",
                 "verification", "reflection"),
    "production": ("perception", "memory", "reason", "synthesis",
                   "verification", "reflection"),
}
# These are review-only methodological lenses, matching Nine Mind R0.6.
# They are not historical-person simulations, model weights, or deployed agents.
SPECIALISTS = {
    "tesla": frozenset({"electrical", "energy", "engineering", "systems"}),
    "da-vinci": frozenset({"mechanics", "geometry", "prototype", "engineering", "visual"}),
    "feynman": frozenset({"physics", "evidence", "reasoning", "engineering"}),
    "escher": frozenset({"geometry", "spatial", "recursion", "visual"}),
    "dali": frozenset({"symbolism", "visual", "creative"}),
    "picasso": frozenset({"perspective", "visual", "creative"}),
    "bach": frozenset({"music", "harmony", "composition", "structure"}),
    "beethoven": frozenset({"music", "expression", "composition", "structure"}),
    "vivaldi": frozenset({"music", "rhythm", "composition", "structure"}),
}
CAPABILITIES = frozenset().union(*SPECIALISTS.values())
PREFERENCES = {
    "discovery": ("feynman", "da-vinci", "dali"),
    "engineering": ("tesla", "da-vinci", "feynman"),
    "creative": ("escher", "picasso", "dali", "bach"),
    "production": ("feynman", "tesla", "da-vinci"),
}
REGISTER_KINDS = ("observation", "hypothesis", "evidence", "decision")
REGISTER_LIMIT = 4


class ClavisError(ValueError):
    """Fail-closed validation for CLAVIS planning and volatile registers."""


def _labels(value, *, valid: frozenset[str], name: str, maximum: int) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= maximum:
        raise ClavisError(f"{name} must contain 1..{maximum} labels")
    if any(type(item) is not str or item not in valid for item in value):
        raise ClavisError(f"{name} contains an unknown label")
    if len(set(value)) != len(value):
        raise ClavisError(f"{name} contains duplicates")
    return tuple(value)


def plan(
    mode: str,
    needs: list[str] | tuple[str, ...],
    *,
    max_specialists: int = 3,
    token_budget: int = 2048,
    allowed_profiles: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Construct a categorical route proposal with explicit coverage and hold gates.

    Inputs are operator-declared capability labels, never raw private prompts.
    Capabilities are static declarations, NOT measured quality or benchmark data.
    No output of this function is a runtime authorization token.
    """
    if mode not in MODES:
        raise ClavisError("unknown CLAVIS mode")
    requested = _labels(needs, valid=CAPABILITIES, name="needs", maximum=12)
    if type(max_specialists) is not int or not 1 <= max_specialists <= 4:
        raise ClavisError("max_specialists must be an integer 1..4")
    if type(token_budget) is not int or not 128 <= token_budget <= 32768:
        raise ClavisError("token_budget must be an integer 128..32768")
    allowed = (
        tuple(SPECIALISTS)
        if allowed_profiles is None else
        _labels(allowed_profiles, valid=frozenset(SPECIALISTS),
                name="allowed_profiles", maximum=9)
    )

    remaining = set(requested)
    pool = list(allowed)
    selected = []
    priority = PREFERENCES[mode]
    while remaining and pool and len(selected) < max_specialists:
        ranked = sorted(
            pool,
            key=lambda profile: (
                -len(SPECIALISTS[profile] & remaining),
                -len(SPECIALISTS[profile] & set(requested)),
                priority.index(profile) if profile in priority else len(priority),
                list(SPECIALISTS).index(profile),
            ),
        )
        candidate = ranked[0]
        new_coverage = SPECIALISTS[candidate] & remaining
        if not new_coverage:
            break
        selected.append({
            "profile_id": candidate,
            "matched_capabilities": sorted(SPECIALISTS[candidate] & set(requested)),
            "newly_covered_capabilities": sorted(new_coverage),
            "selection_basis": "DECLARED_CAPABILITY_OVERLAP_ONLY",
        })
        remaining -= new_coverage
        pool.remove(candidate)

    status = "HELD_UNCOVERED_CAPABILITIES" if remaining else "PLANNING_ONLY_NO_EXECUTION"
    return {
        "schema": SCHEMA,
        "status": status,
        "mode": mode,
        "requested_capabilities": sorted(requested),
        "chambers": list(MODES[mode]),
        "specialists": selected,
        "uncovered_capabilities": sorted(remaining),
        "max_specialists": max_specialists,
        "proposed_token_budget": token_budget,
        "memory_registers": {
            kind: [f"{kind[0]}{index}" for index in range(1, REGISTER_LIMIT + 1)]
            for kind in REGISTER_KINDS
        },
        "gate": {
            "verification_required": True,
            "provenance_review_required": True,
            "operator_approval_required": True,
            "external_router_authority": "GA108/Vessie unchanged",
            "inference_authorized": False,
            "tool_execution_authorized": False,
            "adapter_activation_authorized": False,
            "deployment_authorized": False,
        },
        "evidence": {
            "model_quality_measured": False,
            "latency_measured": False,
            "risk_calibrated": False,
            "weights_loaded": False,
            "plan_has_executed": False,
        },
    }


class WorkingMemory:
    """Sixteen bounded, volatile registers; no disk, telemetry or hidden export.

    Confidence is supplied metadata, NOT a calibrated posterior. A decision
    must reference an actual evidence register. Snapshot omits all raw content
    and source details, even if the caller put private text into memory.
    """

    def __init__(self) -> None:
        self._slots: dict[str, list[dict]] = {kind: [] for kind in REGISTER_KINDS}

    def record(
        self, kind: str, content: str, *,
        source: str, confidence: float,
        evidence_refs: tuple[str, ...] = (),
    ) -> str:
        if kind not in REGISTER_KINDS:
            raise ClavisError("unknown register kind")
        if type(content) is not str or not 1 <= len(content) <= 4096:
            raise ClavisError("register content must contain 1..4096 characters")
        if type(source) is not str or not 1 <= len(source) <= 256:
            raise ClavisError("register provenance source is required")
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ClavisError("confidence is declared metadata in the range 0..1")
        if not isinstance(evidence_refs, tuple) or len(evidence_refs) > REGISTER_LIMIT or (
            any(type(ref) is not str for ref in evidence_refs)
        ) or len(set(evidence_refs)) != len(evidence_refs):
            raise ClavisError("invalid evidence references")
        if kind != "decision" and evidence_refs:
            raise ClavisError("only decision registers can refer to evidence")
        available = {entry["id"] for entry in self._slots["evidence"]}
        if kind == "decision" and (not evidence_refs or not set(evidence_refs) <= available):
            raise ClavisError("decisions require existing evidence registers")
        if len(self._slots[kind]) >= REGISTER_LIMIT:
            raise ClavisError("register bank full; deliberate reset required")
        identifier = f"{kind[0]}{len(self._slots[kind]) + 1}"
        self._slots[kind].append({
            "id": identifier, "content": content, "source": source,
            "confidence": float(confidence), "evidence_refs": evidence_refs,
        })
        return identifier

    def snapshot(self) -> dict:
        return {
            "schema": MEMORY_SCHEMA,
            "persistence": "NONE_VOLATILE_IN_PROCESS",
            "raw_content_exported": False,
            "source_details_exported": False,
            "registers": {
                kind: [
                    {"id": entry["id"], "confidence_unverified": entry["confidence"],
                     "evidence_refs": list(entry["evidence_refs"])}
                    for entry in self._slots[kind]
                ]
                for kind in REGISTER_KINDS
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLAVIS Φ369 offline categorical cognitive route proposal (never runs models)"
    )
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    parser.add_argument("--need", choices=sorted(CAPABILITIES), action="append", required=True)
    parser.add_argument("--allow-profile", choices=tuple(SPECIALISTS), action="append")
    parser.add_argument("--max-specialists", type=int, default=3)
    parser.add_argument("--token-budget", type=int, default=2048)
    args = parser.parse_args()
    try:
        result = plan(args.mode, args.need, max_specialists=args.max_specialists,
                      token_budget=args.token_budget, allowed_profiles=args.allow_profile)
    except ClavisError as exc:
        parser.exit(2, "CLAVIS HELD: " + str(exc) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
