"""Offline, shareable (no prompt/text/path) Phi Model Forge field evidence.

Operator-supplied local receipts are self-reported, not independent GPU witnesses.
Never loads a model, runs training, sends network requests or publishes artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any

from .forge_review_pack import _load_receipt, _check_receipt
from .model_forge import ForgeError, load_examples

SCHEMA = "phi.model-forge.field-evidence.v1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
MAX_ADAPTER_SIZE = 512 * 1024 * 1024
MAX_HEADER_SIZE = 1024 * 1024
DEFAULT_ROOT = Path.home() / "phi-model-forge"
DEFAULT_DATASET = DEFAULT_ROOT / "reviewed-r03.jsonl"


def _bounded_local(path: Path, limit: int) -> Path:
    path = Path(path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ForgeError("Expected a regular local file, not a symlink")
    size = path.stat().st_size
    if not 0 < size <= limit:
        raise ForgeError("Source file exceeds local field-report size policy")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safetensors_header(path: Path) -> int:
    """Verify a bounded header and tensor offset envelopes without loading weights."""
    file = _bounded_local(path, MAX_ADAPTER_SIZE)
    size = file.stat().st_size
    with file.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ForgeError("Adapter safetensors header is truncated")
        head_size = struct.unpack("<Q", prefix)[0]
        if not 0 < head_size <= MAX_HEADER_SIZE or head_size + 8 >= size:
            raise ForgeError("Invalid adapter safetensors header size")
        try:
            head = json.loads(stream.read(head_size).decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ForgeError("Unreadable safetensors header") from exc
    if not isinstance(head, dict):
        raise ForgeError("Safetensors header must be an object")
    spans: list[tuple[int, int]] = []
    count = 0
    available = size - 8 - head_size
    for name, tensor in head.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not name or not isinstance(tensor, dict):
            raise ForgeError("Malformed adapter tensor descriptor")
        offsets = tensor.get("data_offsets")
        shape = tensor.get("shape")
        if (not isinstance(offsets, list) or len(offsets) != 2
            or any(type(n) is not int for n in offsets)
            or offsets[0] < 0 or offsets[1] <= offsets[0] or offsets[1] > available
            or not isinstance(tensor.get("dtype"), str) or not tensor["dtype"]
            or not isinstance(shape, list) or len(shape) > 8
            or any(type(n) is not int or n < 0 for n in shape)):
            raise ForgeError("Malformed adapter tensor metadata/offset")
        spans.append((offsets[0], offsets[1]))
        count += 1
        if count > 4096:
            raise ForgeError("Adapter tensor count exceeds local intake limit")
    if not count:
        raise ForgeError("Adapter file has no tensor descriptors")
    for previous, following in zip(sorted(spans), sorted(spans)[1:]):
        if previous[1] > following[0]:
            raise ForgeError("Adapter tensors overlap in data offsets")
    return count


def verify_local_adapter(
    adapter: Path, dataset_sha256: str, train_count: int, eval_count: int,
    *, base_model: str | None = None, model_revision: str | None = None,
) -> dict[str, Any]:
    """Return no private paths or generated content; verify self-reported lineage."""
    directory = Path(adapter).expanduser()
    if directory.is_symlink() or not directory.is_dir():
        raise ForgeError("Adapter directory does not exist or is a symlink")
    receipt = _load_receipt(directory / "phi-training-receipt.json")
    config = _load_receipt(directory / "adapter_config.json")
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ForgeError("Local adapter is not a declared PEFT causal LoRA")
    if receipt.get("schema") != "phi.model-forge.training-receipt.v1" or receipt.get("status") != "LOCALLY_TRAINED_UNVERIFIED_QUALITY":
        raise ForgeError("Training receipt does not describe a completed candidate")
    model = receipt.get("base_model")
    if (not isinstance(model, str) or not model or len(model) > 256
        or config.get("base_model_name_or_path") != model):
        raise ForgeError("Adapter base model and training receipt disagree")
    if base_model is not None and model != base_model:
        raise ForgeError("Evaluation base model differs from trained adapter")
    revision = receipt.get("model_revision")
    if revision is not None and (not isinstance(revision, str) or len(revision) > 160):
        raise ForgeError("Invalid declared model revision")
    if model_revision is not None and revision != model_revision:
        raise ForgeError("Evaluation model revision differs from trained adapter")
    if (receipt.get("dataset_sha256") != dataset_sha256
        or not isinstance(dataset_sha256, str) or not SHA256.fullmatch(dataset_sha256)):
        raise ForgeError("Training receipt dataset SHA256 does not match selected reviewed data")
    if (type(receipt.get("train_example_count")) is not int or receipt["train_example_count"] != train_count
        or type(receipt.get("held_out_examples")) is not int or receipt["held_out_examples"] != eval_count):
        raise ForgeError("Training receipt example counts do not match selected reviewed data")
    steps = receipt.get("steps_requested")
    global_step = receipt.get("global_step")
    if type(steps) is not int or not 1 <= steps <= 300 or type(global_step) is not int or global_step != steps:
        raise ForgeError("Training receipt does not report the full requested optimizer steps")
    max_length = receipt.get("max_length")
    tokens = receipt.get("token_budget_audit")
    if (type(max_length) is not int or not 128 <= max_length <= 1024
        or not isinstance(tokens, dict)
        or tokens.get("schema") != "phi.model-forge.token-budget.v1"
        or tokens.get("status") != "TOKEN_BUDGET_PASSED"
        or tokens.get("max_length_configured") != max_length
        or tokens.get("train_count") != train_count
        or tokens.get("held_out_count") != eval_count
        or type(tokens.get("max_total_tokens")) is not int
        or tokens["max_total_tokens"] > max_length):
        raise ForgeError("Training token-budget receipt is missing or contradictory")
    if any(receipt.get(key) is not False for key in
           ("quality_improvement_proven", "production_authorized", "network_upload")):
        raise ForgeError("Training receipt has incompatible authority/quality declarations")
    weights = directory / "adapter_model.safetensors"
    observed_count = _safetensors_header(weights)
    observed_sha = _sha256_file(weights)
    if receipt.get("adapter_sha256") != observed_sha:
        raise ForgeError("Adapter weights changed since recorded training SHA256")
    return {
        "base_model": model,
        "model_revision_declared": revision,
        "adapter_sha256": observed_sha,
        "adapter_tensor_count": observed_count,
        "optimizer_steps_reported": global_step,
        "token_audit_passed": True,
    }


def field_report(dataset: Path = DEFAULT_DATASET, root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    """Only return safe flags/counts/fingerprints. No prompts, paths, logs or completions."""
    root = Path(root).expanduser()
    dataset = Path(dataset).expanduser()
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "WAITING_FOR_CURATED_DATA",
        "next_action": "review_examples",
        "dataset_reviewed": False,
        "adapter_verified_against_local_receipt": False,
        "base_eval_present": False,
        "adapter_eval_present": False,
        "blind_review_ready": False,
        "operator_hardware_witnessed": False,
        "model_quality_proven": False,
        "production_authorized": False,
        "contains_prompts_or_generated_text": False,
        "contains_local_paths": False,
        "network_upload": False,
    }
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ForgeError("Field root must be a local directory, not a symlink")
    if not dataset.exists():
        return report
    examples, digest = load_examples(dataset)
    if any(row.get("review_status") != "human-approved" for row in examples):
        report.update(status="DATASET_REVIEW_REQUIRED", next_action="review_examples")
        return report
    train_count = sum(x["split"] == "train" for x in examples)
    eval_count = sum(x["split"] == "eval" for x in examples)
    report.update(
        dataset_reviewed=True, dataset_sha256=digest,
        train_examples=train_count, eval_examples=eval_count,
        status="WAITING_FOR_LOCAL_TRAINING", next_action="train_on_operator_gpu",
    )
    adapter = root / "adapter-r03"
    receipt_path = adapter / "phi-training-receipt.json"
    weights_path = adapter / "adapter_model.safetensors"
    if not receipt_path.exists() and not weights_path.exists():
        return report
    if not receipt_path.exists() or not weights_path.exists():
        report.update(status="INCOMPLETE_ADAPTER_ARTIFACT", next_action="inspect_training_failure")
        return report
    verified = verify_local_adapter(adapter, digest, train_count, eval_count)
    report.update(
        status="WAITING_FOR_HELD_OUT_EVAL", next_action="run_base_and_adapter_eval",
        adapter_verified_against_local_receipt=True,
        base_model_declared=verified["base_model"],
        revision_declared=verified["model_revision_declared"],
        adapter_sha256=verified["adapter_sha256"],
        adapter_tensor_count=verified["adapter_tensor_count"],
        optimizer_steps_reported=verified["optimizer_steps_reported"],
    )
    base_path, adapted_path = root / "base-holdout-r03.json", root / "adapter-holdout-r03.json"
    report["base_eval_present"] = base_path.exists()
    report["adapter_eval_present"] = adapted_path.exists()
    if not (report["base_eval_present"] and report["adapter_eval_present"]):
        return report
    held_out = {x["id"]: x for x in examples if x["split"] == "eval"}
    base = _load_receipt(base_path)
    adapted = _load_receipt(adapted_path)
    _check_receipt(base, "base", held_out, digest)
    _check_receipt(adapted, "adapter", held_out, digest)
    if (base.get("model") != verified["base_model"]
        or adapted.get("model") != verified["base_model"]
        or base.get("model_revision") != verified["model_revision_declared"]
        or adapted.get("model_revision") != verified["model_revision_declared"]
        or adapted.get("adapter_sha256") != verified["adapter_sha256"]
        or base.get("adapter_sha256") not in (None,)):
        raise ForgeError("Evaluation receipts do not bind to the same trained adapter/base")
    report.update(
        status="HUMAN_BLIND_REVIEW_REQUIRED", next_action="score_blind_review_before_promotion",
        blind_review_ready=True, held_out_outputs_count=eval_count,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit a minimal private-text-free PhiOS model field report")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    try:
        print(json.dumps(field_report(args.dataset, args.root), indent=2, sort_keys=True, allow_nan=False))
    except (ForgeError, OSError, ValueError) as exc:
        # Never quote exception details from JSON/model outputs in a shareable report.
        parser.exit(2, f"Forge field evidence held: {exc}\n")


if __name__ == "__main__":
    main()
