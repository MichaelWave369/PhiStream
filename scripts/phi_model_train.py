"""EXPLICIT local QLoRA experiment, separate from PhiStream's inference engine.

No model weights are included. By default only prints a dataset review receipt.
Requires --execute AND a manually approved dataset. Never uploads any files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from phistream.model_forge import DEFAULT_DATA, ForgeError, load_examples, review
from phistream.forge_token_audit import audit_examples, require_valid

SYSTEM = (
    "You are a PhiOS research operator advisor. Distinguish observations, hypotheses "
    "and verified results. Never claim to have run tools without evidence. "
    "Capability is not execution authority; seek explicit approval for side effects. "
    "Use retrieval for live repository facts and keep local privacy boundaries."
)


def train(args: argparse.Namespace) -> dict:
    examples, digest = load_examples(args.dataset)
    if any(x["review_status"] != "human-approved" for x in examples):
        raise ForgeError("Training blocked: each example requires human-approved review_status")
    if args.model != "Qwen/Qwen3-4B-Instruct-2507" and not args.allow_model_override:
        raise ForgeError("Non-default model requires --allow-model-override")
    if args.steps < 1 or args.steps > 300 or args.max_length < 128 or args.max_length > 1024:
        raise ForgeError("Training bounds: 1-300 steps; sequence length 128-1024 tokens")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ForgeError("Output directory already exists; do not overwrite a prior run")
    if output == Path.cwd().resolve() or Path.cwd().resolve() in output.parents:
        raise ForgeError("Use an output directory outside the Git checkout to avoid accidental commits")
    # Heavy libraries are imported only inside the explicitly requested training path.
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise ForgeError("Install optional local model-forge training dependencies first") from exc
    if not torch.cuda.is_available():
        raise ForgeError("No CUDA GPU available; refusing silent CPU fallback")
    free, total = torch.cuda.mem_get_info()
    if free < 6 * 1024 ** 3:
        raise ForgeError("Less than 6 GiB CUDA memory free; stop other GPU workloads and retry")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=False, local_files_only=not args.allow_download, revision=args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Check actual token counts BEFORE loading 4B model weights or using GPU memory.
    token_audit = audit_examples(examples, tokenizer, SYSTEM, args.max_length)
    require_valid(token_audit)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, device_map="auto",
        dtype=dtype, trust_remote_code=False,
        local_files_only=not args.allow_download, revision=args.revision,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    training_rows = [{
        "prompt": [{"role": "system", "content": SYSTEM}, x["messages"][0]],
        "completion": [x["messages"][1]],
    } for x in examples if x["split"] == "train"]
    dataset = Dataset.from_list(training_rows)
    output.mkdir(parents=True, exist_ok=False)
    config = SFTConfig(
        output_dir=str(output),
        max_steps=args.steps,
        max_length=args.max_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        packing=False,
        completion_only_loss=True,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        optim="paged_adamw_8bit",
        seed=args.seed,
        data_seed=args.seed,
    )
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model, args=config, train_dataset=dataset,
        processing_class=tokenizer, peft_config=lora,
    )
    result = trainer.train()
    trainer.model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    weights = output / "adapter_model.safetensors"
    if not weights.is_file():
        raise ForgeError("No adapter_model.safetensors was saved; training output requires inspection")
    hasher = hashlib.sha256()
    with weights.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    weights_sha256 = hasher.hexdigest()
    receipt = {
        "schema": "phi.model-forge.training-receipt.v1",
        "status": "LOCALLY_TRAINED_UNVERIFIED_QUALITY",
        "base_model": args.model,
        "model_revision": args.revision,
        "adapter_sha256": weights_sha256,
        "token_budget_audit": token_audit,
        "base_commit_reported": getattr(model.config, "_commit_hash", None),
        "tokenizer_commit_reported": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "adapter_path": str(output),
        "dataset_sha256": digest,
        "train_example_count": len(training_rows),
        "held_out_examples": sum(x["split"] == "eval" for x in examples),
        "steps_requested": args.steps,
        "global_step": int(result.global_step),
        "max_length": args.max_length,
        "seed": args.seed,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_free_bytes_before_load": free,
        "cuda_total_bytes": total,
        "bf16": dtype == torch.bfloat16,
        "license_assessment": "operator_review_required",
        "holdout_evaluated": False,
        "quality_improvement_proven": False,
        "production_authorized": False,
        "network_upload": False,
    }
    (output / "phi-training-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    p = argparse.ArgumentParser(description="Local QLoRA model training (requires explicit --execute)")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--output", type=Path, default=Path.home() / "phi-model-forge" / "adapter-r01")
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=369)
    p.add_argument("--revision", default=None, help="Optional pinned model revision; prefer an immutable Hugging Face commit SHA")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--allow-download", action="store_true",
                   help="Allow explicit Hugging Face base model download; otherwise use cache")
    p.add_argument("--allow-model-override", action="store_true")
    args = p.parse_args()
    try:
        report = train(args) if args.execute else review(args.dataset)
        print(json.dumps(report, indent=2, sort_keys=True))
        if not args.execute:
            print("DRY RUN ONLY: no training, no model loaded, no weights downloaded.")
    except (ForgeError, OSError, RuntimeError, ValueError) as exc:
        p.exit(2, f"Model Forge stopped: {exc}\n")


if __name__ == "__main__":
    main()
