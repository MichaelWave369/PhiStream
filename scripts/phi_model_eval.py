"""Opt-in BASE or ADAPTER generation against held-out public fixture prompts.

No automatic reward score, benchmark superiority claim, network upload or promotion.
Only one model is loaded on the GPU per invocation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from phistream.model_forge import DEFAULT_DATA, ForgeError, load_examples
from phistream.forge_protocol import SYSTEM


def evaluate(args: argparse.Namespace) -> dict:
    examples, digest = load_examples(args.dataset)
    held_out = [x for x in examples if x["split"] == "eval"]
    if not held_out:
        raise ForgeError("No held-out evaluation prompts")
    if args.mode == "adapter" and not args.adapter:
        raise ForgeError("Adapter mode requires --adapter local path")
    if args.output.exists():
        raise ForgeError("Refusing to overwrite an existing evaluation receipt")
    if args.max_new_tokens < 1 or args.max_new_tokens > 256:
        raise ForgeError("max_new_tokens must be 1..256")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise ForgeError("Install optional Hugging Face inference dependencies first") from exc
    if not torch.cuda.is_available():
        raise ForgeError("CUDA unavailable; no silent CPU downgrade")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=not args.allow_download, trust_remote_code=False, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, device_map="auto",
        dtype=dtype, trust_remote_code=False,
        local_files_only=not args.allow_download, revision=args.revision)
    if args.mode == "adapter":
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, args.adapter, is_trainable=False, local_files_only=True)
    model.eval()
    outputs = []
    for x in held_out:
        messages = [{"role": "system", "content": SYSTEM}, x["messages"][0]]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: val.to(model.device) for key, val in encoded.items()}
        with torch.inference_mode():
            tokens = model.generate(**encoded, do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id)
        completion = tokenizer.decode(
            tokens[0][encoded["input_ids"].shape[-1]:],
            skip_special_tokens=True)
        outputs.append({"id": x["id"], "topic": x["topic"],
                        "prompt": x["messages"][0]["content"],
                        "output": completion})
    receipt = {
        "schema": "phi.model-forge.holdout-output.v1",
        "model": args.model, "model_revision": args.revision, "mode": args.mode,
        "dataset_sha256": digest, "outputs": outputs,
        "review_status": "HUMAN_REVIEW_REQUIRED",
        "quality_proven": False, "deployment_authorized": False,
        "network_upload": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    p = argparse.ArgumentParser(description="Generate held-out outputs for manual review")
    p.add_argument("--mode", choices=["base", "adapter"], required=True)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--adapter", type=Path)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--revision", default=None, help="Optional pinned model revision")
    p.add_argument("--allow-download", action="store_true")
    args = p.parse_args()
    try:
        receipt = evaluate(args)
        print(json.dumps({"schema": receipt["schema"],
            "outputs": len(receipt["outputs"]), "review_status": receipt["review_status"],
            "output_path": str(args.output)}, indent=2))
    except (ForgeError, RuntimeError, ValueError, OSError) as exc:
        p.exit(2, f"Model Forge holdout stopped: {exc}\n")


if __name__ == "__main__":
    main()
