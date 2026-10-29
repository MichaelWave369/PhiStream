"""PocketRotor E4: bounded CPU integration against real HF Qwen3.5 MoE classes.

This runner never calls from_pretrained(), AutoModel, a tokenizer, a model
hub, a GGUF backend or CUDA. All tensor values are synthetic; the *code*
executing the MoE block is the installed Hugging Face Transformers library.

Opt-in isolated CI job installs a pinned Transformers release. On any native
API drift or incomplete route capture: fail, never silently skip.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import inspect
import json
from pathlib import Path
from types import SimpleNamespace


SCHEMA = "PHISTREAM_POCKETROTOR_E4_NATIVE_CPU_SMOKE_V1"
TRANSFORMERS_VERSION = "5.21.0"


def native_cpu_smoke() -> dict:
    import torch
    import transformers
    from torch import nn
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeSparseMoeBlock,
        Qwen3_5MoeTopKRouter,
        Qwen3_5MoeExperts,
    )
    from scripts.pocketrotor_e3_preflight import inspect_source
    from scripts.pocketrotor_e3_qwen35 import (
        Qwen35NativeRouterCapture, adapter_report,
    )
    from scripts.pocketrotor_e2_route_bridge import convert_capture
    from scripts.pocketrotor_e2 import analyze

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError("Pinned transformers release changed")
    if torch.cuda.is_initialized():
        raise RuntimeError("This research job must never initialize CUDA")
    torch.set_num_threads(1)
    torch.manual_seed(369)

    module_path = Path(inspect.getfile(Qwen3_5MoeTopKRouter))
    if module_path.name != "modeling_qwen3_5_moe.py":
        raise RuntimeError("Qwen source module path drift")
    raw = module_path.read_bytes()
    if not 0 < len(raw) <= 2 * 1024 * 1024:
        raise RuntimeError("Oversized HF source")
    static = inspect_source(raw)
    if static["source_structure_status"] != "STATIC_AST_CONTRACT_MATCH_ONLY":
        raise RuntimeError("Native source contract unrecognized")

    # Real HF Qwen MoE block, but *tiny*, randomly initialized test weights.
    # 256 routed experts and top-k 8 are genuine architectural dimensions.
    # Hidden=16, expert width=8, two layers keep this test comfortably CPU only.
    conf = SimpleNamespace(
        num_experts=256, num_experts_per_tok=8,
        hidden_size=16, moe_intermediate_size=8,
        shared_expert_intermediate_size=8, hidden_act="silu",
        _experts_implementation="eager",
    )

    class OneLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = Qwen3_5MoeSparseMoeBlock(conf)

        def forward(self, x):
            return self.mlp(x)

    class TinyNativeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([OneLayer(), OneLayer()])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = TinyNativeModel().cpu().eval()
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise RuntimeError("Unexpected non-CPU parameter")
    for p in model.parameters():
        nn.init.uniform_(p, a=-0.025, b=0.025)

    gate = model.layers[0].mlp.gate
    experts = model.layers[0].mlp.experts
    if (type(gate) is not Qwen3_5MoeTopKRouter
            or type(experts) is not Qwen3_5MoeExperts
            or gate.num_experts != 256 or gate.top_k != 8):
        raise RuntimeError("Native Qwen router/expert types have changed")

    events = [("train", 8), ("same_family", 4), ("shifted", 4)]
    inputs = []
    for group_name, count in events:
        for index in range(count):
            # Intentionally synthetic activations, not tokenizer inputs.
            shift = 3.0 if group_name == "shifted" else 0.0
            x = torch.randn((1, 1, 16), dtype=torch.float32) * 0.03 + shift
            inputs.append((group_name, index, x))
    with torch.no_grad():
        baseline = [model(x).detach().clone() for _, _, x in inputs]

    source = {
        "kind": "synthetic",
        "runtime": f"transformers-{transformers.__version__}-real-native-block-cpu",
        "model_id": "synthetic-random-untrained-two-layer-256-expert",
        "checkpoint_sha256": "0" * 64,
        "route_semantics": "POST_NATIVE_ROUTER_SELECTED_EXPERT_IDS",
        "token_text_recorded": False,
    }
    with torch.no_grad():
        with Qwen35NativeRouterCapture(
            model, source=source,
            expected_experts=256, expected_top_k=8, expected_layers=2,
        ) as session:
            observed = [
                session.run_step(group, step, lambda x=x: model(x)).detach().clone()
                for group, step, x in inputs
            ]
    for original, hooked in zip(baseline, observed):
        torch.testing.assert_close(original, hooked, atol=0.0, rtol=0.0)
    capture = session.export()
    report = adapter_report(capture)
    if (capture["expert_count"] != 256
            or len(capture["events"]) != len(inputs) * 2
            or report["source_kind"] != "synthetic"
            or report["steps_per_split"] != dict(events)):
        raise AssertionError("Native capture incomplete or mislabeled")

    # Verify the existing bridge accepts actual native HF-selected indices.
    # Values remain synthetic because the weights/activations are synthetic.
    per_layer = {}
    for layer in (0, 1):
        trace, manifest = convert_capture(
            capture, layer=layer, expert_bytes=1024,
            capacity_bytes=8 * 1024, max_prefetch=1,
        )
        result = analyze(trace)
        if (manifest["source_authentication"] != "OPERATOR_ASSERTED_NOT_VERIFIED"
                or not manifest["synthetic_source"]
                or result["model_execution"]
                or not result["simulated_expert_routing_only"]
                or set(result["holdouts"]) != {"same_family", "shifted"}):
            raise AssertionError("E2/E2-R replay authority exceeded")
        per_layer[str(layer)] = {
            "native_recorded_steps": len(trace["train"])
            + sum(len(steps) for steps in trace["holdouts"].values()),
            "all_holds_preserved": list(result["holdouts"]),
            "simulation_status": result["status"],
            "capture_source_remains_synthetic": True,
        }
    return {
        "schema": SCHEMA,
        "status": "REAL_LIBRARY_NATIVE_BLOCK_CPU_HOOK_PARITY_SYNTHETIC_WEIGHTS",
        "transformers_version": transformers.__version__,
        "transformers_native_source_sha256": sha256(raw).hexdigest(),
        "transformers_native_source_path": "modeling_qwen3_5_moe.py",
        "native_library_classes_executed": True,
        "real_pretrained_model_executed": False,
        "real_checkpoint_weights_loaded": False,
        "synthetic_weights_and_inputs": True,
        "native_expert_count": 256,
        "native_top_k": 8,
        "tiny_test_layers": 2,
        "native_token_steps": len(inputs),
        "observed_native_gate_to_experts_events": len(capture["events"]),
        "complete_output_parity_with_unhooked_native_block": True,
        "capture_source_authenticated_as_pretrained_checkpoint": False,
        "per_layer_e2_bridge": per_layer,
        "cuda_initialized": False,
        "checkpoint_downloaded": False,
        "measured_gpu_performance": False,
        "model_output_quality_tested": False,
        "production_promoted": False,
        "limitations": [
            "Uses actual installed HF MoE router and expert classes, but tiny untrained weights.",
            "Does not execute a pretrained 35B checkpoint or its full text decoder.",
            "Does not verify an externally controlled real-model runtime or prompt sequence.",
            "Native CPU observation perturbs timing; no hardware speedup claims.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E4 real HF Qwen native-block CPU smoke with tiny synthetic weights"
    )
    parser.add_argument("--output", help="Optional new JSON receipt, otherwise stdout")
    args = parser.parse_args()
    result = native_cpu_smoke()
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        path = Path(args.output)
        if not path.parent.is_dir() or path.is_symlink():
            raise ValueError("Receipt output folder missing/symlinked")
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
