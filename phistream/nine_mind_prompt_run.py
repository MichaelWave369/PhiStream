"""Φ Nine Mind R0.6: explicit, cache-only local specialist-prompt evaluation.

No automatic model downloads, training, adapter modifications, remote inference,
router changes, or claims of model-quality improvement. Offline preflight is
the default. GPU inference is an operator-approved, separate physical test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

from .forge_baseline import PIN, inspect_pin
from .forge_protocol import SYSTEM
from .model_atlas import CATALOG_PATH, read_json
from .model_forge import ForgeError, load_examples
from .nine_mind_seed import REPO_ROOT, TOPIC

MAX_NEW_TOKENS = 256
MAX_CONTEXT_TOKENS = 1024
MAX_OUTPUT_CHARS = 10000
RUN_SCHEMA = "phi.model-forge.holdout-output.v1"
PLAN_SCHEMA = "phistream.nine-mind.prompt-plan.v0.6"

# These are *original methodological instructions*, not historical
# impersonations, copyrighted text, or claims about any person's thoughts.
METHODS = {
    "tesla": "Analyze electrical or energy-system ideas with explicit circuit assumptions, units, power and loss budgets. Separate feasible engineering from speculation; state how to measure each prediction.",
    "da-vinci": "Analyze the physical mechanism and geometry; propose a simple prototype, identify constraints and failure points, then describe measurable changes for the next iteration.",
    "feynman": "Reduce the question to testable predictions. Explain the physics from definitions, distinguish systematic from random error, and suggest a controlled falsification test.",
    "escher": "Develop original geometric or spatial designs using explicit transformations, tiling or recursion rules. Distinguish an intentional 2D illusion from realizable 3D construction.",
    "dali": "Invent original surreal symbolism with a coherent focal point and clear composition. Treat possible meanings as artistic interpretations, not clinical diagnoses or factual revelations.",
    "picasso": "Explore a subject through multiple simultaneous viewpoints and controlled abstraction. Specify which recognizability cues survive and which visual constraints are deliberately changed.",
    "bach": "Use original motivic structure and independent musical lines. Describe interval transformations, imitation and harmonic checks without quoting protected compositions.",
    "beethoven": "Develop an original short musical motif using recognizable transformations, contrasts and a tension-release arc. Distinguish a compositional plan from an audio recording.",
    "vivaldi": "Organize original musical ideas using recurring refrains, rhythmic momentum and contrasting movement textures. Explain continuity and transitions without copying known scores.",
}
_SHA = re.compile(r"[0-9a-f]{40}\Z")


def recipe(profile: str) -> str:
    """Exactly the full system text supplied to the tokenizer."""
    if profile not in METHODS:
        raise ForgeError("unknown Nine Mind research specialist")
    return SYSTEM + "\n\nNine Mind methodological lens (" + profile + "): " + METHODS[profile] + (
        "\nTreat the named profile as a design method, not as a simulated historical person. "
        "Use evidence, identify unknowns, do not claim to have used tools, and answer the actual user prompt."
    )


def _baseline():
    pin = read_json(PIN)
    catalog = read_json(CATALOG_PATH)
    inspect_pin(pin, catalog)
    if not _SHA.fullmatch(pin["revision"]):
        raise ForgeError("baseline needs an exact pinned commit revision")
    return pin


def prepare(dataset: Path, profile: str, max_new_tokens: int = 128):
    """CPU-only, dependency-free preflight. No CUDA or model loading."""
    if profile not in METHODS:
        raise ForgeError("unknown Nine Mind research specialist")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise ForgeError("max_new_tokens must be an integer from 1 to 256")
    pin = _baseline()
    examples, digest = load_examples(dataset)
    if not examples or not any(x["split"] == "eval" for x in examples):
        raise ForgeError("dataset has no held-out questions")
    # Topic check is a necessary, but not sufficient, profile-to-dataset binding.
    if any(x["topic"] != TOPIC[profile] for x in examples):
        raise ForgeError("dataset topic is incompatible with selected specialist")
    approved = all(x["review_status"] == "human-approved" for x in examples)
    text = recipe(profile)
    return {
        "schema": PLAN_SCHEMA,
        "status": "LOCAL_INFERENCE_ELIGIBLE_FOR_OPERATOR_REVIEW" if approved else "DATASET_HUMAN_REVIEW_REQUIRED",
        "profile_id": profile, "dataset_sha256": digest,
        "model": pin["model_id"], "model_revision": pin["revision"],
        "recipe_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "dataset_all_rows_approved": approved,
        "held_out_count": sum(x["split"] == "eval" for x in examples),
        "max_new_tokens": max_new_tokens, "max_context_tokens": MAX_CONTEXT_TOKENS,
        "cached_weights_required": True, "network_access_authorized": False,
        "model_loaded": False, "inference_started": False,
        "quality_proven": False, "production_authorized": False,
    }


def _private_destination(output: Path) -> Path:
    output = Path(output).expanduser()
    if output.name in ("", ".", "..") or output.suffix.lower() != ".json":
        raise ForgeError("choose a new .json receipt filename")
    if output.is_symlink() or output.exists():
        raise ForgeError("refusing to overwrite existing or symlinked receipt")
    try:
        parent = output.parent.resolve(strict=True)
    except OSError as exc:
        raise ForgeError("private receipt parent directory must already exist") from exc
    if not parent.is_dir():
        raise ForgeError("private receipt parent directory must already exist")
    target = parent / output.name
    if target == REPO_ROOT or REPO_ROOT in target.parents:
        raise ForgeError("private receipts must be outside the Git checkout")
    return target


def generate_cached(model_id: str, revision: str, held_out: list, system: str, max_new_tokens: int):
    """The only real GPU/inference path; invoked solely through explicit run()."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise ForgeError("optional local CUDA/Transformers dependencies are not installed") from exc
    if not torch.cuda.is_available():
        raise ForgeError("CUDA unavailable; no remote or CPU fallback")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    # Intentionally never download anything. Never load remote model code.
    tok = AutoTokenizer.from_pretrained(
        model_id, revision=revision, local_files_only=True, trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, local_files_only=True, trust_remote_code=False,
        quantization_config=quant, dtype=dtype, device_map={"": "cuda:0"},
    )
    model.eval()
    results = []
    for x in held_out:
        messages = [{"role": "system", "content": system}, x["messages"][0]]
        formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tok(formatted, return_tensors="pt")
        length = int(encoded["input_ids"].shape[-1])
        if length < 1 or length > MAX_CONTEXT_TOKENS:
            raise ForgeError("prompt token count outside 1..1024; no truncation")
        encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
        with torch.inference_mode():
            tokens = model.generate(
                **encoded, do_sample=False, max_new_tokens=max_new_tokens,
                pad_token_id=tok.eos_token_id,
            )
        response = tok.decode(tokens[0][length:], skip_special_tokens=True)
        results.append(response)
    return results


def run(dataset: Path, profile: str, output: Path, *, approve_local_inference: bool = False,
        max_new_tokens: int = 128, generator=None):
    """Operator-approved local execution; injectable generator for *synthetic tests*.

    Receipt content is self-reported even for real execution. External evidence
    still requires separate physical checks; outputs are not uploaded by us.
    """
    plan = prepare(dataset, profile, max_new_tokens=max_new_tokens)
    if not approve_local_inference:
        raise ForgeError("real evaluation requires explicit --approve-local-inference")
    if not plan["dataset_all_rows_approved"]:
        raise ForgeError("every dataset row requires individual human approval")
    target = _private_destination(output)
    examples, digest = load_examples(dataset)
    if digest != plan["dataset_sha256"]:
        raise ForgeError("dataset content changed since preflight")
    held_out = [x for x in examples if x["split"] == "eval"]
    callable_generator = generator if generator is not None else generate_cached
    generated = callable_generator(plan["model"], plan["model_revision"], held_out,
                                   recipe(profile), max_new_tokens)
    if not isinstance(generated, list) or len(generated) != len(held_out):
        raise ForgeError("local generator returned wrong held-out output count")
    output_rows = []
    for example, answer in zip(held_out, generated):
        if not isinstance(answer, str) or not answer.strip() or len(answer) > MAX_OUTPUT_CHARS:
            raise ForgeError("local output missing or outside bounded answer size")
        output_rows.append({
            "id": example["id"], "topic": example["topic"],
            "prompt": example["messages"][0]["content"], "output": answer,
        })
    receipt = {
        "schema": RUN_SCHEMA, "mode": "prompted",
        "model": plan["model"], "model_revision": plan["model_revision"],
        "dataset_sha256": digest, "outputs": output_rows,
        "adapter_sha256": None,
        "profile_id": profile, "recipe_sha256": plan["recipe_sha256"],
        "source_class": "OPERATOR_ASSERTED_NOT_ATTESTED",
        "method_prompt_applied": True,
        "review_status": "HUMAN_REVIEW_REQUIRED",
        "quality_proven": False, "deployment_authorized": False,
        "network_upload": False,
    }
    data = (json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    if len(data) > 2 * 1024 * 1024:
        raise ForgeError("local evaluation receipt exceeds 2 MiB")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(target, flags, 0o600)
    except FileExistsError as exc:
        raise ForgeError("refusing to overwrite existing private receipt") from exc
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    return {
        "status": "LOCAL_GENERATION_RECORDED_HUMAN_REVIEW_REQUIRED",
        "mode": "prompted", "profile_id": profile,
        "dataset_sha256": digest, "model_revision": plan["model_revision"],
        "recipe_sha256": plan["recipe_sha256"],
        "outputs_written": len(output_rows),
        "source_attested": False, "quality_proven": False,
        "production_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(METHODS), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--execute", action="store_true",
                        help="explicit local CUDA evaluation; no automatic model download")
    parser.add_argument("--approve-local-inference", action="store_true",
                        help="second explicit consent gate for local inference")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.execute:
            if args.output is None:
                raise ForgeError("--execute requires --output outside the repository")
            result = run(args.dataset, args.profile, args.output,
                         approve_local_inference=args.approve_local_inference,
                         max_new_tokens=args.max_new_tokens)
        else:
            if args.output or args.approve_local_inference:
                raise ForgeError("--output and --approve-local-inference require --execute")
            result = prepare(args.dataset, args.profile, max_new_tokens=args.max_new_tokens)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (OSError, ForgeError, ValueError, RuntimeError) as exc:
        parser.exit(2, "Nine Mind prompted evaluation HELD: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
