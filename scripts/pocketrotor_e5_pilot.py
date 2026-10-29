"""PocketRotor E5: bounded Granite native CPU smoke + opt-in pretrained pilot.

Native-smoke executes real Hugging Face Granite classes with tiny RANDOM
weights. Pretrained mode explicitly downloads a pinned PUBLIC IBM model,
then executes 32/16/16 teacher-forced public-text token steps on CPU.
The latter runs ONLY on manual command with --authorize-download.
Nothing writes actual model weights or prompt text to the JSON receipts.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.pocketrotor_e2 import analyze, _digest
from scripts.pocketrotor_e2_route_bridge import convert_capture, validate_capture
from scripts.pocketrotor_e5_granite import GraniteNativeRouterCapture

SCHEMA = "PHISTREAM_POCKETROTOR_E5_PILOT_RECEIPT_V1"
VERSION = "5.21.0"
MODEL_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"
MODEL_REVISION = "23d6629dbac8c0a0652f28991f62bb75157c02e3"
MAX_WEIGHT_BYTES = 3 * 1024**3
SPLIT_STEPS = {"train": 32, "same_family": 16, "shifted": 16}
PUBLIC_TEXT = {
    "train": (
        "Engineers study how information travels through a network of small computers. "
        "They collect measurements from each connection, check the values twice, and "
        "record which paths carry the most traffic. A balanced system can sometimes "
        "avoid unnecessary transfers by remembering which pieces of data were "
        "used recently. A rigorous experiment reports its misses as well as its "
        "successes, and compares every proposed improvement to a simple baseline. "
        "Different workloads create different patterns, so an honest evaluation "
        "keeps separate sequences for training and later testing."
    ),
    "same_family": (
        "A technician is troubleshooting a distributed sensor system in a workshop. "
        "Each sensor sends small messages at regular intervals. The technician "
        "measures latency, checks available memory, and compares several "
        "caching strategies. Some predictions help, but wrong predictions "
        "also waste bandwidth. Every experiment includes an ordinary "
        "replacement policy as a control, with observations written "
        "down in a reproducible log."
    ),
    "shifted": (
        "Along a coastal garden a naturalist observes seabirds returning "
        "to their nests. A tide pool is full of tiny creatures, and "
        "the wind moves through tall grass. The naturalist writes notes "
        "about seasons and animal behavior, then studies photographs "
        "of plants. Far away a chef prepares a vegetable soup and "
        "bakes fresh bread for a group of visiting travelers."
    ),
}


def _source(kind: str, identity: str, checkpoint_sha: str) -> dict:
    return {
        "kind": kind,
        "runtime": f"transformers-{VERSION}-granite-native",
        "model_id": identity,
        "checkpoint_sha256": checkpoint_sha,
        "route_semantics": "POST_NATIVE_ROUTER_SELECTED_EXPERT_IDS",
        "token_text_recorded": False,
    }


def _native_smoke() -> tuple[dict, dict]:
    import torch
    import transformers
    from torch import nn
    from transformers.models.granitemoe.modeling_granitemoe import (
        GraniteMoeMoE, GraniteMoeTopKRouter, GraniteMoeExperts,
    )
    if transformers.__version__ != VERSION:
        raise RuntimeError("Transformers version differs from E5 pinned version")
    torch.set_num_threads(1)
    torch.manual_seed(369)
    conf = SimpleNamespace(
        num_local_experts=32, num_experts_per_tok=8,
        hidden_size=16, intermediate_size=8, hidden_act="silu",
        _experts_implementation="eager",
    )

    class TinyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.block_sparse_moe = GraniteMoeMoE(conf)

        def forward(self, x):
            return self.block_sparse_moe(x)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([TinyLayer(), TinyLayer()])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = TinyModel().cpu().eval()
    for p in model.parameters():
        nn.init.uniform_(p, a=-.025, b=.025)
    if (type(model.layers[0].block_sparse_moe.router) is not GraniteMoeTopKRouter
            or type(model.layers[0].block_sparse_moe.experts) is not GraniteMoeExperts):
        raise RuntimeError("Real Granite router/expert class contract changed")
    with torch.no_grad():
        x = torch.randn((1, 1, 16))
        original = model(x).detach().clone()
        with GraniteNativeRouterCapture(
            model, source=_source("synthetic", "random-tiny-native-Granite", "0" * 64),
            expected_layers=2,
        ) as session:
            observed = []
            for split, count in (("train", 8), ("same_family", 4), ("shifted", 4)):
                for step in range(count):
                    observed.append(session.run_step(
                        split, step, lambda: model(x)).detach().clone())
    for output in observed:
        torch.testing.assert_close(original, output, atol=0, rtol=0)
    capture = session.export()
    if len(capture["events"]) != 32:
        raise RuntimeError("Native Granite CPU hook event count mismatch")
    return {
        "schema": SCHEMA, "status": "REAL_GRANITE_CLASSES_SYNTHETIC_CPU_PARITY",
        "native_library_executed": True,
        "pretrained_checkpoint_executed": False,
        "synthetic_weights": True,
        "native_router_experts_parity": True,
        "bitwise_output_parity": True,
        "expert_count": 32, "top_k": 8, "routed_layers": 2,
        "recorded_route_events": len(capture["events"]),
        "original_model_routing_unchanged": True,
        "actual_model_inference_speedup_proven": False,
        "cuda_used": False,
    }, capture


def _sha_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError("Refuse absent/symlinked pretrained weight file")
    h = sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(2 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def _pretrained(model_dir: Path) -> tuple[dict, dict]:
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download

    if transformers.__version__ != VERSION:
        raise RuntimeError("Pinned Transformers runtime required")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU-only pretrained pilot; CUDA must not initialize")
    torch.set_num_threads(2)
    # Caller deliberately authorizes public weights via explicit CLI flag,
    # and the manual workflow is opt-in only.
    snapshot = Path(snapshot_download(
        repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=str(model_dir),
        allow_patterns=[
            "config.json", "generation_config.json",
            "model.safetensors", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "added_tokens.json",
            "merges.txt", "vocab.json",
        ],
        max_workers=2,
    ))
    target = snapshot / "model.safetensors"
    if (not target.is_file() or target.is_symlink()
            or not 0 < target.stat().st_size <= MAX_WEIGHT_BYTES):
        raise RuntimeError("Expected bounded public single-shard safetensors checkpoint")
    weight_bytes = target.stat().st_size
    checkpoint_hash = _sha_file(target)
    conf = AutoConfig.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False)
    if (conf.model_type != "granitemoe"
            or conf.num_hidden_layers != 24
            or conf.num_local_experts != 32
            or conf.num_experts_per_tok != 8):
        raise RuntimeError("IBM Granite checkpoint architecture not recognized")
    conf._experts_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), config=conf, local_files_only=True,
        trust_remote_code=False, use_safetensors=True,
        dtype="auto",
    ).cpu().eval()
    if type(model).__name__ != "GraniteMoeForCausalLM":
        raise RuntimeError("Expected native GraniteMoeForCausalLM")
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise RuntimeError("Unexpected non-CPU pretrained parameters")

    expert_footprints = []
    for layer in model.model.layers:
        experts = layer.block_sparse_moe.experts
        tensors = (experts.gate_up_proj, experts.down_proj)
        if any(t.shape[0] != 32 for t in tensors):
            raise RuntimeError("Expected one uniform tensor group per expert")
        b = sum(t[0].numel() * t.element_size() for t in tensors)
        expert_footprints.append(b)
    if (not all(b == expert_footprints[0] for b in expert_footprints)
            or not 0 < expert_footprints[0] <= 16 * 1024**2):
        raise RuntimeError("Non-uniform or unsupported measured expert size")

    sequences = {}
    for label, text in PUBLIC_TEXT.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not isinstance(ids, list) or len(ids) < SPLIT_STEPS[label]:
            raise RuntimeError("Frozen public test sentence too short for split")
        sequences[label] = ids[:SPLIT_STEPS[label]]

    # Uninstrumented vs instrumented *identical first-token* full-model
    # output comparison, no need to fabricate an independent reference.
    start_token = torch.tensor([[sequences["train"][0]]], dtype=torch.long)
    with torch.inference_mode():
        baseline = model(input_ids=start_token, use_cache=True).logits.detach().clone()

    src = _source(
        "operator_asserted_native", f"{MODEL_ID}@{MODEL_REVISION}",
        checkpoint_hash)
    with torch.inference_mode():
        with GraniteNativeRouterCapture(
            model, source=src, expected_layers=24, expected_experts=32,
            expected_top_k=8,
        ) as session:
            output_parity_checked = False
            for split in ("train", "same_family", "shifted"):
                cache = None
                for step, token_id in enumerate(sequences[split]):
                    token = torch.tensor([[token_id]], dtype=torch.long)
                    result = session.run_step(
                        split, step,
                        lambda token=token, cache=cache: model(
                            input_ids=token, past_key_values=cache, use_cache=True))
                    if split == "train" and step == 0:
                        torch.testing.assert_close(
                            baseline, result.logits, atol=0, rtol=0)
                        output_parity_checked = True
                    cache = result.past_key_values
                    if cache is None:
                        raise RuntimeError("Native model failed to preserve stateful token cache")
    capture = session.export()
    verified = validate_capture(capture)
    if (verified["counts"] != SPLIT_STEPS
            or verified["layers"] != list(range(24))
            or not output_parity_checked):
        raise RuntimeError("Pretrained native token-step capture is incomplete")

    # Every byte budget is based on actual parameter shape/dtype, NOT an
    # invented 4096-byte expert proxy.
    expert_bytes = expert_footprints[0]
    per_layer = {}
    for layer in (0, 6, 12, 18, 23):
        trace, manifest = convert_capture(
            capture, layer=layer, expert_bytes=expert_bytes,
            capacity_bytes=expert_bytes * 12, max_prefetch=1,
        )
        result = analyze(trace)
        if (manifest["source_kind"] != "operator_asserted_native"
                or manifest["e2_trace_canonical_sha256"] != _digest(trace)
                or not result["simulated_expert_routing_only"]):
            raise RuntimeError("E2 route bridge provenance/budget contract failed")
        per_layer[str(layer)] = {
            "trace_sha256": manifest["e2_trace_canonical_sha256"],
            "holdouts": {
                name: {
                    "lru": hold["results"]["lru"],
                    "flat_markov": hold["results"]["flat-markov"],
                    "nested_bubble": hold["results"]["nested-bubble"],
                    "nested_total_bytes_vs_lru": hold["nested_total_bytes_vs_lru"],
                    "nested_demand_bytes_vs_flat_markov":
                        hold["nested_demand_bytes_vs_flat_markov"],
                } for name, hold in result["holdouts"].items()
            },
        }
    return {
        "schema": SCHEMA,
        "status": "OPERATOR_REPORTED_REAL_PRETRAINED_GRANITE_NATIVE_ROUTING",
        "model_id": MODEL_ID, "pinned_revision": MODEL_REVISION,
        "transformers_version": VERSION,
        "checkpoint_safetensors_sha256": checkpoint_hash,
        "checkpoint_weight_bytes_hashed": weight_bytes,
        "expert_weight_bytes_per_layer_measured_from_params": expert_bytes,
        "native_router_consumer_parity": True,
        "bitwise_first_token_output_parity": True,
        "training_and_holdout_steps": SPLIT_STEPS,
        "native_layer_count": 24, "native_routed_experts": 32, "native_top_k": 8,
        "native_events": len(capture["events"]),
        "source_capture_canonical_sha256": _digest(capture),
        "frozen_public_teacher_forced_text_only": True,
        "private_user_inputs_recorded": False,
        "model_outputs_or_token_ids_exported": False,
        "full_pretrained_model_executed_on_cpu": True,
        "real_gpu_weight_transfers_measured": False,
        "real_inference_speedup_proven": False,
        "runtime_source_independently_attested": False,
        "production_promoted": False,
        "per_layer_e2_replay_only": per_layer,
        "limitations": [
            "Teacher-forced, short, public text. Not a held-out population benchmark.",
            "Actual router selections from a pretrained Granite model, not Qwen3.5.",
            "Checkpoint content hashed locally; independent model authenticity not attested.",
            "Cache replay is a simulation: prefetch is not implemented in pretrained runtime.",
            "Real throughput, transfers, energy and output improvements have not been established.",
        ],
    }, capture


def _write_new(path: Path, obj: dict) -> None:
    if not path.parent.is_dir() or path.is_symlink():
        raise ValueError("Output folder missing or symlink refused")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E5 Granite CPU: real HF native smoke, optional pinned public pretrained pilot"
    )
    parser.add_argument("mode", choices=("native-smoke", "pretrained"))
    parser.add_argument("--authorize-download", action="store_true",
                        help="Required in pretrained mode; permits ~2.67GB public HF model")
    parser.add_argument("--model-dir", help="Existing working directory for public weights")
    parser.add_argument("--report-output", required=True, help="NEW JSON report path")
    parser.add_argument("--capture-output", required=True, help="NEW native routing capture path")
    args = parser.parse_args()
    report = Path(args.report_output)
    capture = Path(args.capture_output)
    if report.resolve() == capture.resolve() or report.exists() or capture.exists():
        raise ValueError("Refuse source/receipt collision or overwrite")
    if (not report.parent.is_dir() or not capture.parent.is_dir()
            or report.is_symlink() or capture.is_symlink()):
        raise ValueError("Output folder must already exist and be ordinary")
    if args.mode == "native-smoke":
        if args.authorize_download or args.model_dir:
            parser.error("native-smoke never downloads or accepts model directories")
        result, events = _native_smoke()
    else:
        if not args.authorize_download or not args.model_dir:
            parser.error("pretrained requires --authorize-download and --model-dir")
        location = Path(args.model_dir)
        if not location.is_dir() or location.is_symlink():
            raise ValueError("Model directory must already exist without symlinks")
        result, events = _pretrained(location)
    _write_new(report, result)
    _write_new(capture, events)


if __name__ == "__main__":
    main()
