"""Nine Mind R0.5: offline two/three-arm blind output review, never grading.

Compares operator-supplied base, method-prompted, and (later) LoRA-adapter
holdout output receipts under exactly one *existing* Model Forge dataset.
No model/API execution, training, automatic scoring, or Vessie route changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets

from .forge_review_pack import _load_receipt, _check_receipt
from .model_forge import ForgeError, load_examples
from .model_atlas import CATALOG_PATH, read_json, validate_catalog
from .nine_mind_seed import REPO_ROOT

PACK_SCHEMA = "phistream.nine-mind.blind-pack.v0.5"
KEY_SCHEMA = "phistream.nine-mind.blind-key.v0.5"
REVIEW_STATUS = "HUMAN_REVIEW_REQUIRED"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISIONS = re.compile(r"[0-9a-f]{40}\Z")
ARMS = ("base", "prompted", "adapter")
MANUAL_RUBRIC = (
    "Correctness and explicit uncertainty",
    "Responsiveness to the task and constraints",
    "Reproducible or useful reasoning/design method",
    "Originality without copying or unsupported claims",
    "Errors, omissions, and regressions",
)


def _digest(value: dict) -> str:
    """Canonical JSON digest of self-declared receipt; NOT source attestation."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()


def _validate_prompted(receipt, heldout, digest, profile):
    _check_receipt(receipt, "prompted", heldout, digest)
    if receipt.get("profile_id") != profile:
        raise ForgeError("prompted receipt specialist differs from requested profile")
    if receipt.get("recipe_sha256") is None or not isinstance(receipt["recipe_sha256"], str) or not _SHA256.fullmatch(receipt["recipe_sha256"]):
        raise ForgeError("prompted receipt requires a sha256 of the method-prompt recipe")
    if receipt.get("source_class") != "OPERATOR_ASSERTED_NOT_ATTESTED":
        raise ForgeError("prompted receipt cannot claim independent authenticity")
    if receipt.get("method_prompt_applied") is not True:
        raise ForgeError("prompted receipt must declare actual use of the recipe")
    if receipt.get("adapter_sha256") is not None:
        raise ForgeError("prompted baseline must not declare adapter weights")


def _check_adapter_provenance(receipt, mode):
    digest = receipt.get("adapter_sha256")
    if mode == "adapter":
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ForgeError("adapter receipt requires a declared 64-character SHA-256")
    elif digest is not None:
        raise ForgeError("untrained base/prompted receipt must not claim adapter weights")


def build_pack(dataset_path: Path, profile: str, base_path: Path, prompted_path: Path,
               adapter_path: Path | None = None, *,
               shuffle=None, session_factory=None):
    """Build blind material in memory; no writes and no model execution.

    The operator-supplied run receipts are *not* independently authenticated;
    outputs may be fabricated. Actual model quality is unknown.
    """
    catalog = read_json(CATALOG_PATH)
    validate_catalog(catalog)
    if profile not in {p["id"] for p in catalog["profiles"]}:
        raise ForgeError("unknown Nine Mind specialist")
    examples, dataset_digest = load_examples(dataset_path)
    heldout = {x["id"]: x for x in examples if x["split"] == "eval"}
    if not heldout:
        raise ForgeError("dataset has no held-out questions")
    base = _load_receipt(base_path)
    prompted = _load_receipt(prompted_path)
    arm_inputs = {"base": base, "prompted": prompted}
    if adapter_path is not None:
        arm_inputs["adapter"] = _load_receipt(adapter_path)
    for mode, receipt in arm_inputs.items():
        _check_receipt(receipt, mode, heldout, dataset_digest)
        _check_adapter_provenance(receipt, mode)
    _validate_prompted(prompted, heldout, dataset_digest, profile)
    for receipt in arm_inputs.values():
        revision = receipt.get("model_revision")
        if not isinstance(revision, str) or not _REVISIONS.fullmatch(revision):
            raise ForgeError("all arm receipts require an exact pinned 40-character model revision")
    models = {(r["model"], r["model_revision"]) for r in arm_inputs.values()}
    if len(models) != 1:
        raise ForgeError("all arms must declare exactly the same base model and revision")
    model, revision = next(iter(models))
    if not isinstance(revision, str) or not _REVISIONS.fullmatch(revision):
        raise ForgeError("all arm receipts require an exact pinned 40-character model revision")
    for r in arm_inputs.values():
        if r.get("review_status") != REVIEW_STATUS:
            raise ForgeError("run receipt must remain unapproved")
        if r.get("network_upload") is not False:
            raise ForgeError("run receipt must not claim network upload")
    if shuffle is None:
        shuffle = secrets.SystemRandom().shuffle
    if session_factory is None:
        session_factory = lambda: secrets.token_hex(16)
    session = session_factory()
    if not isinstance(session, str) or not re.fullmatch(r"[0-9a-f]{32}", session):
        raise ForgeError("invalid blinded review session token")
    outputs = {
        mode: {row["id"]: row["output"] for row in receipt["outputs"]}
        for mode, receipt in arm_inputs.items()
    }
    rows, mappings = [], []
    for key, example in heldout.items():
        mapping = list(arm_inputs)
        shuffle(mapping)
        if len(mapping) != len(arm_inputs) or set(mapping) != set(arm_inputs):
            raise ForgeError("invalid randomization of review arm labels")
        labels = {chr(65+i): mode for i, mode in enumerate(mapping)}
        answer_values = {label: outputs[mode][key] for label, mode in labels.items()}
        rows.append({
            "id": key, "topic": example["topic"],
            "prompt": example["messages"][0]["content"],
            "answers": answer_values,
            "reviewer_scores": {label: None for label in labels},
            "rubric": list(MANUAL_RUBRIC),
        })
        mappings.append({"id": key, "labels": labels})
    all_reviewed = all(row["review_status"] == "human-approved" for row in examples)
    pack = {
        "schema": PACK_SCHEMA, "session": session,
        "status": "HUMAN_REVIEW_REQUIRED",
        "dataset_sha256": dataset_digest,
        "arm_count": len(arm_inputs),
        "row_count": len(rows),
        "rows": rows,
        "dataset_review_status": "ALL_ROWS_HUMAN_APPROVED" if all_reviewed else "UNREVIEWED_MECHANICS_ONLY",
        "public_seed_status": "POSSIBLE_PUBLIC_EXPOSURE_NOT_INDEPENDENT_BENCHMARK",
        "quality_improvement_proven": False,
        "winner": None,
        "model_execution_by_this_tool": False,
        "production_authorized": False,
    }
    key = {
        "schema": KEY_SCHEMA, "session": session,
        "dataset_sha256": dataset_digest,
        "model_id": model, "model_revision": revision,
        "profile_id_asserted": profile,
        "source_receipt_content_sha256": {mode: _digest(receipt) for mode, receipt in arm_inputs.items()},
        "adapter_sha256_declared": arm_inputs["adapter"].get("adapter_sha256") if "adapter" in arm_inputs else None,
        "prompt_recipe_sha256_declared": prompted["recipe_sha256"],
        "mapping": mappings,
        "private_answer_key": True,
        "model_outputs_unattested": True,
        "quality_improvement_proven": False,
        "deployment_authorized": False,
    }
    return pack, key


def _new_private_file(path: Path, value: dict) -> None:
    """Exclusive-create private JSON; caller checks both destinations first."""
    raw = (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)


def write_pair(pack_path: Path, key_path: Path, pack: dict, key: dict):
    """Write both private files without overwriting existing data.

    Parent directories must already exist. Never write under Git checkout.
    """
    destinations = []
    for item in (pack_path, key_path):
        path = Path(item).expanduser()
        if not path.parent.exists() or not path.parent.is_dir():
            raise ForgeError("output directory must already exist")
        parent = path.parent.resolve(strict=True)
        final = parent / path.name
        if final == REPO_ROOT or REPO_ROOT in final.parents:
            raise ForgeError("private review files must not be committed inside repository")
        if path.is_symlink() or final.exists():
            raise ForgeError("refusing to overwrite existing private review file")
        destinations.append(final)
    first, second = destinations
    if first == second:
        raise ForgeError("review pack and answer key must have separate destinations")
    _new_private_file(first, pack)
    try:
        _new_private_file(second, key)
    except BaseException:
        first.unlink(missing_ok=True)
        raise
    return {"status": "HUMAN_REVIEW_REQUIRED", "session": pack["session"],
            "rows": len(pack["rows"]), "arms": pack["arm_count"],
            "review_written": True, "answer_key_separate": True,
            "automatic_scoring": False, "execution": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True,
                   help="human-reviewed Forge dataset or explicit public mechanics fixture")
    p.add_argument("--profile", required=True, help="one known Nine Mind profile ID")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--prompted", type=Path, required=True)
    p.add_argument("--adapter", type=Path, help="optional third arm when a trained adapter exists")
    p.add_argument("--pack", type=Path, required=True)
    p.add_argument("--key", type=Path, required=True)
    args = p.parse_args()
    try:
        pack, key = build_pack(args.dataset, args.profile, args.base, args.prompted, args.adapter)
        receipt = write_pair(args.pack, args.key, pack, key)
        print(json.dumps(receipt, indent=2, sort_keys=True))
    except (OSError, ForgeError, ValueError) as exc:
        p.exit(2, "Nine Mind review HELD: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
