"""PocketRotor E5: read-only native IBM Granite MoE router instrumentation.

The caller owns the original, already-loaded Hugging Face Granite model.
Hooks only observe native router indices and confirm exact expert inputs.
No model loading, downloading, routing modifications or prompt capture here.
"""
from __future__ import annotations

import re
from typing import Callable

from scripts.pocketrotor_e2_route_bridge import RouteEventRecorder, validate_capture

SCHEMA = "PHISTREAM_POCKETROTOR_E5_GRANITE_HOOK_V1"
MODULE_PREFIX = "transformers.models.granitemoe."
MODULE_PATTERN = re.compile(r"(?:^|[.])layers[.]([0-9]+)[.]block_sparse_moe[.](router|experts)$")


def _native(module, class_name):
    cls = type(module)
    return cls.__name__ == class_name and cls.__module__.startswith(MODULE_PREFIX)


class GraniteNativeRouterCapture:
    """A hook pair per actual Granite MoE layer, fail-closed and single-token."""

    def __init__(self, model, *, source: dict, expected_layers: int = 24,
                 expected_experts: int = 32, expected_top_k: int = 8):
        if type(expected_layers) is not int or not 1 <= expected_layers <= 128:
            raise ValueError("Invalid native layer count")
        if type(expected_experts) is not int or not 4 <= expected_experts <= 256:
            raise ValueError("Invalid native expert count")
        if type(expected_top_k) is not int or not 1 <= expected_top_k <= 8:
            raise ValueError("Invalid native routed top-k")
        if getattr(model, "training", None) is not False:
            raise ValueError("Model must already be in eval mode")
        if not callable(getattr(model, "named_modules", None)):
            raise ValueError("Missing native torch module registry")
        self.recorder = RouteEventRecorder(source, expected_experts)
        self.expected_top_k = expected_top_k
        found: dict[int, dict] = {}
        for path, module in model.named_modules():
            match = MODULE_PATTERN.search(path)
            if not match:
                continue
            layer, kind = int(match.group(1)), match.group(2)
            pair = found.setdefault(layer, {})
            if kind in pair:
                raise ValueError("Duplicate native expert/router module")
            if kind == "router":
                if not _native(module, "GraniteMoeTopKRouter"):
                    raise ValueError("Unknown native Granite router class")
                if module.num_experts != expected_experts or module.top_k != expected_top_k:
                    raise ValueError("Native Granite router expert count/top-k mismatch")
            elif not _native(module, "GraniteMoeExperts"):
                raise ValueError("Unknown native Granite experts class")
            pair[kind] = module
        if (len(found) != expected_layers or sorted(found) != list(range(expected_layers))
                or any(set(pair) != {"router", "experts"} for pair in found.values())):
            raise ValueError("Native Granite layer/router/experts coverage mismatch")
        self.layers = sorted(found)
        self.handles = []
        self.active: tuple[str, int] | None = None
        self.pending: dict[int, list[int]] = {}
        self.seen: set[int] = set()
        self.failed = False
        self.closed = False
        self.completed_steps = 0
        try:
            for layer in self.layers:
                self.handles.append(found[layer]["router"].register_forward_hook(
                    self._router_hook(layer)))
                self.handles.append(found[layer]["experts"].register_forward_pre_hook(
                    self._experts_hook(layer)))
        except BaseException:
            self.close()
            raise

    def _guard(self):
        if self.failed or self.closed or self.active is None:
            self.failed = True
            raise RuntimeError("Native Granite hook ran outside approved step")

    def _indices(self, tensor):
        import torch
        if (not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != (1, self.expected_top_k)
                or tensor.dtype not in (torch.int32, torch.int64)):
            self.failed = True
            raise ValueError("Expected one native token and integer top-k expert IDs")
        ids = tensor.detach().cpu().reshape(-1).tolist()
        if len(set(ids)) != len(ids) or any(
                type(x) is not int or x < 0 or x >= self.recorder.expert_count for x in ids):
            self.failed = True
            raise ValueError("Invalid native Granite expert selection")
        return ids

    def _router_hook(self, layer):
        def hook(_module, _args, outputs):
            self._guard()
            if layer in self.pending or layer in self.seen:
                self.failed = True
                raise ValueError("Duplicate Granite router execution")
            if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
                self.failed = True
                raise ValueError("Native Granite router output contract changed")
            self.pending[layer] = self._indices(outputs[2])
            return None
        return hook

    def _experts_hook(self, layer):
        def hook(_module, args):
            self._guard()
            if layer not in self.pending or layer in self.seen:
                self.failed = True
                raise ValueError("Native Granite experts called without matching router")
            if not isinstance(args, tuple) or len(args) < 3:
                self.failed = True
                raise ValueError("Granite experts input contract changed")
            consumed = self._indices(args[1])
            if consumed != self.pending.pop(layer):
                self.failed = True
                raise ValueError("Experts did not consume native router-selected IDs")
            split, step = self.active
            self.recorder.record(
                split=split, step=step, layer=layer, experts=consumed)
            self.seen.add(layer)
            return None
        return hook

    def run_step(self, split: str, step: int, forward: Callable[[], object]):
        if self.closed or self.failed or self.active is not None:
            raise RuntimeError("Granite recorder closed/failed/already executing")
        if (not isinstance(split, str) or type(step) is not int
                or not 0 <= step < 4096 or not callable(forward)):
            raise ValueError("Invalid split, step or forward callback")
        self.active = (split, step)
        self.pending.clear()
        self.seen.clear()
        try:
            output = forward()
            if self.pending or self.seen != set(self.layers):
                raise ValueError("Every layer must execute router and experts once")
            self.completed_steps += 1
            return output
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

    def export(self):
        if not self.closed or self.failed or not self.completed_steps:
            raise RuntimeError("Refuse incomplete or failed native capture")
        result = self.recorder.export()
        validate_capture(result)
        return result

    def __enter__(self):
        if self.closed:
            raise RuntimeError("Native recorder already closed")
        return self

    def __exit__(self, exc_type, _exc, _tb):
        if exc_type is not None:
            self.failed = True
        self.close()
        return False
