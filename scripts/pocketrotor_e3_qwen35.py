"""Qwen3.5 MoE: opt-in observation of native routed expert IDs.

Attaches to an ALREADY LOADED, eval-mode Hugging Face model. No model
loading, downloads, tokenizer, inference scheduling or expert substitution.
Observes selected IDs at the router output and independently checks that
the same IDs enter the native experts module. Batch=1, sequence=1 only.
"""
from __future__ import annotations

import re
from typing import Callable

from scripts.pocketrotor_e2_route_bridge import RouteEventRecorder, validate_capture

SCHEMA = "PHISTREAM_POCKETROTOR_E3_QWEN35_NATIVE_HOOK_V1"
PREFIX = "transformers.models.qwen3_5_moe."
ROUTER_CLASS = "Qwen3_5MoeTopKRouter"
EXPERTS_CLASS = "Qwen3_5MoeExperts"
PATTERN = re.compile(r"(?:^|[.])layers[.]([0-9]+)[.]mlp[.](gate|experts)$")


def _native_class(module, name: str) -> bool:
    cls = type(module)
    return cls.__name__ == name and cls.__module__.startswith(PREFIX)


class Qwen35NativeRouterCapture:
    """Always fail closed if native router/expert architecture drifts."""

    def __init__(self, model, *, source: dict,
                 expected_experts: int = 256, expected_top_k: int = 8,
                 expected_layers: int = 40):
        if type(expected_experts) is not int or not 4 <= expected_experts <= 256:
            raise ValueError("Expected routed experts must be 4..256")
        if type(expected_top_k) is not int or not 1 <= expected_top_k <= 8:
            raise ValueError("Expected top-k must be 1..8")
        if type(expected_layers) is not int or not 1 <= expected_layers <= 128:
            raise ValueError("Expected layer count must be 1..128")
        if getattr(model, "training", None) is not False:
            raise ValueError("Already-loaded model must be in eval mode")
        if not callable(getattr(model, "named_modules", None)):
            raise ValueError("Expected torch.nn.Module model")
        if source.get("kind") not in ("synthetic", "operator_asserted_native"):
            raise ValueError("Source must be explicitly labeled")
        self.recorder = RouteEventRecorder(source, expected_experts)
        self.expected_top_k = expected_top_k
        modules: dict[int, dict] = {}
        for path, module in model.named_modules():
            match = PATTERN.search(path)
            if not match:
                continue
            layer, kind = int(match.group(1)), match.group(2)
            pair = modules.setdefault(layer, {})
            if kind in pair:
                raise ValueError("Duplicate routed module")
            if kind == "gate":
                if not _native_class(module, ROUTER_CLASS):
                    raise ValueError("Not the recognized native Qwen3.5 gate class")
                if module.num_experts != expected_experts or module.top_k != expected_top_k:
                    raise ValueError("Native gate architecture mismatch")
            elif not _native_class(module, EXPERTS_CLASS):
                raise ValueError("Not the recognized native Qwen3.5 experts class")
            pair[kind] = module
        if (len(modules) != expected_layers or
                sorted(modules) != list(range(expected_layers)) or
                any(set(pair) != {"gate", "experts"} for pair in modules.values())):
            raise ValueError("Native Qwen3.5 gate/expert layer coverage mismatch")
        self.layers = sorted(modules)
        self.handles = []
        self.active: tuple[str, int] | None = None
        self.pending: dict[int, list[int]] = {}
        self.seen: set[int] = set()
        self.closed = False
        self.failed = False
        self.completed = 0
        try:
            for layer in self.layers:
                self.handles.append(
                    modules[layer]["gate"].register_forward_hook(self._gate_hook(layer))
                )
                self.handles.append(
                    modules[layer]["experts"].register_forward_pre_hook(
                        self._experts_hook(layer)
                    )
                )
        except BaseException:
            self.close()
            raise

    def _check(self):
        if self.closed or self.failed or self.active is None:
            self.failed = True
            raise RuntimeError("Native hook executed outside approved step")

    def _ids(self, selected, layer):
        import torch
        if (not isinstance(selected, torch.Tensor)
                or selected.ndim != 2
                or tuple(selected.shape) != (1, self.expected_top_k)
                or selected.dtype not in (torch.int32, torch.int64)):
            self.failed = True
            raise ValueError("Expected one native token and exact top-k integer tensor")
        ids = selected.detach().cpu().reshape(-1).tolist()
        if len(set(ids)) != len(ids) or any(
                type(v) is not int or not 0 <= v < self.recorder.expert_count
                for v in ids):
            self.failed = True
            raise ValueError(f"Invalid native router expert IDs in layer {layer}")
        return ids

    def _gate_hook(self, layer):
        def hook(_module, _inputs, output):
            self._check()
            if layer in self.pending or layer in self.seen:
                self.failed = True
                raise ValueError("Duplicate router output")
            if not isinstance(output, (tuple, list)) or len(output) != 3:
                self.failed = True
                raise ValueError("Unknown native router tuple layout")
            self.pending[layer] = self._ids(output[2], layer)
            return None
        return hook

    def _experts_hook(self, layer):
        def hook(_module, args):
            self._check()
            if layer not in self.pending or layer in self.seen:
                self.failed = True
                raise ValueError("Experts invoked without unique matching router")
            if not isinstance(args, tuple) or len(args) < 3:
                self.failed = True
                raise ValueError("Experts input layout changed")
            native_ids = self._ids(args[1], layer)
            if native_ids != self.pending.pop(layer):
                self.failed = True
                raise ValueError("Experts did not consume router-selected IDs")
            split, step = self.active
            self.recorder.record(split=split, layer=layer, step=step,
                                 experts=native_ids)
            self.seen.add(layer)
            return None
        return hook

    def run_step(self, split: str, step: int, forward: Callable[[], object]):
        """Execute caller's original single-token forward without modification."""
        if self.closed or self.failed or self.active is not None:
            raise RuntimeError("Capture is failed, closed or already running")
        if (not isinstance(split, str) or type(step) is not int
                or not 0 <= step < 4096 or not callable(forward)):
            raise ValueError("Invalid split, step or forward callback")
        self.active = (split, step)
        self.pending.clear()
        self.seen.clear()
        try:
            result = forward()
            if self.pending or self.seen != set(self.layers):
                raise ValueError("Every layer must execute gate and experts once")
            self.completed += 1
            return result
        except BaseException:
            self.failed = True
            raise
        finally:
            self.active = None

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.active is not None:
            self.failed = True
        self.closed = True

    def export(self) -> dict:
        if not self.closed or self.failed or not self.completed:
            raise RuntimeError("Cannot export active, incomplete or failed capture")
        capture = self.recorder.export()
        validate_capture(capture)
        return capture

    def __enter__(self):
        if self.closed:
            raise RuntimeError("Closed recorder cannot be reused")
        return self

    def __exit__(self, exc_type, _exc, _tb):
        if exc_type is not None:
            self.failed = True
        self.close()
        return False


def adapter_report(capture: dict) -> dict:
    """Inspect exported routes, never claim model provenance authentication."""
    checked = validate_capture(capture)
    return {
        "schema": SCHEMA,
        "status": "NATIVE_GATE_TO_EXPERTS_ID_PARITY_OPERATOR_ASSERTED",
        "source_kind": checked["source"]["kind"],
        "layers": checked["layers"],
        "splits": checked["splits"],
        "steps_per_split": checked["counts"],
        "checkpoint_hash_independently_verified": False,
        "runtime_provenance_attested": False,
        "model_executed_by_report": False,
        "production_promoted": False,
    }
