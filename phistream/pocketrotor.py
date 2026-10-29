"""ΦPocketRotor E0: opt-in bounded tensor residency for canonical Llama.

This is a *cache*, not an MoE router, neural compression or GPU prefetcher.
The canonical StreamingLlama execution and weights remain unchanged.
"""
from __future__ import annotations

from collections import OrderedDict
import math
from typing import Callable

import torch

from .llama import LAYER_NAMES, StreamingLlama
from .safeweights import LocalCheckpoint

SCHEMA = "PHISTREAM_POCKETROTOR_E0_CACHE_V1"
POLICIES = ("flat-lru", "nested-pin")


def _quad(index: int, depth: int) -> str:
    """Canonical NW/NE/SW/SE digit address (0,1,2,3)."""
    if type(index) is not int or type(depth) is not int or depth < 1 or index < 0 or index >= 4 ** depth:
        raise ValueError("Pocket address exceeds finite quadtree")
    digits = []
    for shift in reversed(range(depth)):
        digits.append(str((index // (4 ** shift)) % 4))
    return "".join(digits)


def pocket_inventory(checkpoint: LocalCheckpoint) -> list[dict]:
    """Finite, stable logical addresses; tensor identities never depend on cache state."""
    layers = checkpoint.config["num_hidden_layers"]
    if type(layers) is not int or layers < 1:
        raise ValueError("Invalid layer count")
    depth = 1
    while 4 ** depth < layers:
        depth += 1
    result = []
    for layer in range(layers):
        for slot, suffix in enumerate(LAYER_NAMES):
            name = f"model.layers.{layer}.{suffix}"
            entry = checkpoint.entries[name]
            size = math.prod(entry.shape) * 4  # Decoder always stages FP32.
            result.append({
                "name": name, "layer": layer,
                "address": _quad(layer, depth) + "/" + _quad(slot, 2),
                "bytes_fp32": size,
            })
    return result


class PocketRotorCache:
    """Exact-budget FP32 cache. No speculative loads and no hidden fallback.

    flat-lru is a conventional per-tensor baseline.
    nested-pin admits complete layer bubbles, fixed using immutable metadata.
    The latter avoids cross-token LRU thrash in a dense sequential decoder.
    """

    def __init__(self, checkpoint: LocalCheckpoint, *, budget_bytes: int, policy: str):
        if type(budget_bytes) is not int or not 1 <= budget_bytes <= 64 * 1024**3:
            raise ValueError("Cache budget must be an integer 1..64 GiB")
        if policy not in POLICIES:
            raise ValueError("Unknown pocket residency policy")
        self.budget_bytes = budget_bytes
        self.policy = policy
        self.inventory = pocket_inventory(checkpoint)
        self.by_name = {entry["name"]: entry for entry in self.inventory}
        self.resident: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.bytes_resident = 0
        self.peak_bytes_resident = 0
        self.hits = 0
        self.misses = 0
        self.admissions = 0
        self.evictions = 0
        self.bypasses = 0

        # Only whole layers are pinned in E0. Selection is deterministic and
        # charged at actual FP32 execution precision, not checkpoint dtype.
        self.pinned_layers: list[int] = []
        self.pinned_names: set[str] = set()
        self.pinned_planned_bytes = 0
        if policy == "nested-pin":
            groups: dict[int, list[dict]] = {}
            for item in self.inventory:
                groups.setdefault(item["layer"], []).append(item)
            for layer, items in sorted(groups.items()):
                size = sum(item["bytes_fp32"] for item in items)
                if size <= budget_bytes - self.pinned_planned_bytes:
                    self.pinned_layers.append(layer)
                    self.pinned_names.update(item["name"] for item in items)
                    self.pinned_planned_bytes += size

    def load(self, name: str, loader: Callable[[], torch.Tensor]) -> torch.Tensor:
        if name not in self.by_name:
            raise ValueError("Only canonical layer tensor pockets are cacheable")
        if name in self.resident:
            self.hits += 1
            self.resident.move_to_end(name)
            return self.resident[name]
        self.misses += 1
        # Never run loaders for known uncacheable pin candidates just to cache.
        admit = self.policy == "flat-lru" or name in self.pinned_names
        tensor = loader()
        size = tensor.numel() * tensor.element_size()
        if not admit or size > self.budget_bytes:
            self.bypasses += 1
            return tensor
        if tensor.dtype != torch.float32 or size != self.by_name[name]["bytes_fp32"]:
            raise ValueError("Tensor precision or shape differs from immutable pocket manifest")
        if self.policy == "flat-lru":
            while self.bytes_resident + size > self.budget_bytes:
                _, old = self.resident.popitem(last=False)
                self.bytes_resident -= old.numel() * old.element_size()
                self.evictions += 1
        elif self.bytes_resident + size > self.budget_bytes:
            raise MemoryError("Pinned pocket accounting exceeded declared cache budget")
        self.resident[name] = tensor
        self.bytes_resident += size
        self.peak_bytes_resident = max(self.peak_bytes_resident, self.bytes_resident)
        self.admissions += 1
        return tensor

    def stats(self) -> dict:
        return {
            "schema": SCHEMA,
            "policy": self.policy,
            "budget_bytes": self.budget_bytes,
            "bytes_resident": self.bytes_resident,
            "peak_bytes_resident": self.peak_bytes_resident,
            "resident_pockets": len(self.resident),
            "pinned_layer_ids": self.pinned_layers,
            "pinned_planned_bytes": self.pinned_planned_bytes,
            "hits": self.hits, "misses": self.misses,
            "admissions": self.admissions, "evictions": self.evictions,
            "bypasses": self.bypasses,
            "addressing": "layer-quadtree/tensor-quadtree; 0=NW,1=NE,2=SW,3=SE",
            "prefetch_executed": False,
            "training_or_weight_modification": False,
        }


class PocketRotorLlama(StreamingLlama):
    """Separate experimental subclass; StreamingLlama remains the default."""

    def __init__(self, checkpoint: LocalCheckpoint, *, cache_bytes: int,
                 policy: str = "nested-pin", device: str = "cpu",
                 max_tensor_mib: int = 512, head_rows: int = 128):
        if type(cache_bytes) is not int or not 1 <= cache_bytes <= 64 * 1024**3:
            raise ValueError("cache_bytes must be an integer 1..64 GiB")
        if policy not in POLICIES:
            raise ValueError("Unknown cache policy")
        super().__init__(checkpoint, device=device, max_tensor_mib=max_tensor_mib,
                         head_rows=head_rows)
        self.pocket_cache = PocketRotorCache(checkpoint, budget_bytes=cache_bytes,
                                             policy=policy)

    def _weight(self, name: str) -> torch.Tensor:
        # Base __init__ loads final norm before pocket_cache exists.
        if not name.startswith("model.layers.") or not hasattr(self, "pocket_cache"):
            return super()._weight(name)
        return self.pocket_cache.load(
            name, lambda: super(PocketRotorLlama, self)._weight(name),
        )
