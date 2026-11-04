"""PhiOS Model Forge R0.4: fail-closed tokenizer budget audit.

Tokenizer access is opt-in and uses existing Hugging Face cache by default.
No CUDA, weights, training, upload, or production inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .model_forge import DEFAULT_DATA, ForgeError, load_examples

SCHEMA = "phi.model-forge.token-budget.v1"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def audit_examples(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    system: str,
    max_length: int,
) -> dict[str, Any]:
    if not 128 <= max_length <= 1024:
        raise ForgeError("Model Forge sequence budget must be 128 to 1024 tokens")
    if not isinstance(system, str) or not system.strip():
        raise ForgeError("Missing operator system prompt")
    train = [item for item in examples if item["split"] == "train"]
    if not train:
        raise ForgeError("No training rows present")
    exceeded: list[str] = []
    mismatched_template: list[str] = []
    empty_target: list[str] = []
    totals: list[int] = []
    prompt_sizes: list[int] = []
    completion_sizes: list[int] = []
    for example in train:
        prompt = [{"role": "system", "content": system}, example["messages"][0]]
        completed = prompt + [example["messages"][1]]
        try:
            prompt_ids = tokenizer.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True
            )
            full_ids = tokenizer.apply_chat_template(
                completed, tokenize=True, add_generation_prompt=False
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise ForgeError("Tokenizer chat-template formatting failed") from exc
        if (not isinstance(prompt_ids, list) or not isinstance(full_ids, list)
                or not all(type(n) is int and n >= 0 for n in prompt_ids + full_ids)):
            raise ForgeError("Tokenizer must return nonnegative integer token IDs")
        if full_ids[:len(prompt_ids)] != prompt_ids:
            mismatched_template.append(example["id"])
            target_size = 0
        else:
            target_size = len(full_ids) - len(prompt_ids)
            if target_size < 2:
                empty_target.append(example["id"])
        if len(full_ids) > max_length:
            exceeded.append(example["id"])
        totals.append(len(full_ids))
        prompt_sizes.append(len(prompt_ids))
        completion_sizes.append(target_size)
    passed = not exceeded and not mismatched_template and not empty_target
    return {
        "schema": SCHEMA,
        "status": "TOKEN_BUDGET_PASSED" if passed else "BLOCKED_BY_TOKEN_BUDGET",
        "max_length_configured": max_length,
        "train_count": len(train),
        "held_out_count": sum(x["split"] == "eval" for x in examples),
        "max_total_tokens": max(totals),
        "max_prompt_tokens": max(prompt_sizes),
        "min_completion_tokens": min(completion_sizes),
        "max_completion_tokens": max(completion_sizes),
        "overflow_example_ids": exceeded,
        "template_mismatch_ids": mismatched_template,
        "missing_completion_ids": empty_target,
        "truncation_allowed": False,
        "model_weights_loaded": False,
        "training_started": False,
        "quality_proven": False,
        "production_authorized": False,
    }


def require_valid(report: dict[str, Any]) -> None:
    if report.get("schema") != SCHEMA or report.get("status") != "TOKEN_BUDGET_PASSED":
        raise ForgeError(
            "Token audit blocked training: inspect overflow/template/target example IDs "
            "and increase --max-length within 1024 or refine the dataset"
        )


def inspect(
    dataset: Path,
    *,
    model: str = DEFAULT_MODEL,
    revision: str | None = None,
    max_length: int = 256,
    allow_download: bool = False,
) -> dict[str, Any]:
    examples, digest = load_examples(dataset)
    if any(x["review_status"] != "human-approved" for x in examples):
        raise ForgeError("Token audit requires an individually reviewed local dataset")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ForgeError("Install the optional transformers package in an isolated environment") from exc
    # No model weights loaded and no download unless operator approved it.
    tokenizer = AutoTokenizer.from_pretrained(
        model, revision=revision, trust_remote_code=False,
        local_files_only=not allow_download,
    )
    # Use the exact shared system prompt used by the training script.
    from scripts.phi_model_train import SYSTEM
    report = audit_examples(examples, tokenizer, SYSTEM, max_length)
    report.update({
        "dataset_sha256": digest,
        "base_model_requested": model,
        "revision_requested": revision,
        "tokenizer_commit_reported": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "download_permission": allow_download,
    })
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit supervised chat-token budgets; no training")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--allow-download", action="store_true",
                        help="Allow downloading tokenizer files only; no model weights")
    args = parser.parse_args()
    try:
        result = inspect(args.dataset, model=args.model, revision=args.revision,
                         max_length=args.max_length, allow_download=args.allow_download)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        if result["status"] != "TOKEN_BUDGET_PASSED":
            parser.exit(2, "Token audit BLOCKED: no GPU training was started\n")
    except (ForgeError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"Token audit failed: {exc}\n")


if __name__ == "__main__":
    main()
