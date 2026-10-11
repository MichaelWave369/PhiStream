"""Independent Llama-style one-token decoder with stream / resident parity modes.

No model downloads, remote execution, pickle deserialization, or borrowed engines.
This intentionally supports only the PhiStream V1 experimental architecture.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import time

import torch
import torch.nn.functional as F

from .format import ModelStore


BLOCK_NAMES = ("attn_norm", "ffn_norm", "wq", "wk", "wv", "wo", "gate", "up", "down")


def rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return (x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)).to(x.dtype) * weight


def rope(heads: torch.Tensor, pos: int, theta: float) -> torch.Tensor:
    """Rotate neighboring even/odd feature pairs, never mutate KV history."""
    head_dim = heads.shape[-1]
    exponent = torch.arange(0, head_dim, 2, device=heads.device, dtype=torch.float32) / head_dim
    angle = pos / torch.pow(torch.tensor(theta, device=heads.device), exponent)
    c, s = angle.cos(), angle.sin()
    a, b = heads[..., 0::2], heads[..., 1::2]
    return torch.stack((a*c - b*s, a*s + b*c), dim=-1).flatten(-2)


@dataclass
class Counters:
    layer_loads: int = 0
    loaded_parameter_bytes: int = 0
    load_seconds: float = 0.0
    peak_cuda_allocated_bytes: int | None = None


class PhiStreamEngine:
    def __init__(self, store: ModelStore, mode: str = "stream", device: str = "auto",
                 pin_budget_mib: float = 0.0):
        if mode not in {"stream", "resident"}:
            raise ValueError("mode must be 'stream' or 'resident'")
        if not math.isfinite(pin_budget_mib) or pin_budget_mib < 0:
            raise ValueError("pin_budget_mib must be a nonnegative finite number")
        self.store = store
        self.config = store.config
        self.mode = mode
        device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.device = torch.device(device)
        self.counters = Counters()
        self.pinned: dict[int,dict[str,torch.Tensor]] = {}
        self.embedding = self._load_tensor("embed")
        self.final_norm = self._load_tensor("final_norm")
        self.lm_head = self._load_tensor("lm_head")
        # A deterministic *static* hot set avoids LRU thrash on sequential token passes.
        remaining = int(pin_budget_mib * 1024 * 1024) if mode == "stream" else math.inf
        for i in range(self.config.n_layers):
            size = sum(store.entries[f"layer.{i}.{n}"]["nbytes"] for n in BLOCK_NAMES)
            if size <= remaining:
                self.pinned[i] = self._load_block(i)
                remaining -= size
        self.pinned_parameter_bytes = sum(sum(t.numel()*t.element_size() for t in b.values())
                                         for b in self.pinned.values())
        self.keys: list[list[torch.Tensor]] = [[] for _ in range(self.config.n_layers)]
        self.values: list[list[torch.Tensor]] = [[] for _ in range(self.config.n_layers)]
        self.position = 0

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _load_tensor(self, name: str) -> torch.Tensor:
        start = time.perf_counter()
        source = torch.from_numpy(self.store.array(name))
        # resident mode gets owned copies, stream mode references mapped CPU pages.
        if self.device.type == "cpu" and self.mode == "resident":
            result = source.clone()
        else:
            result = source.to(self.device)
        self._sync()
        self.counters.layer_loads += 1
        self.counters.loaded_parameter_bytes += result.numel() * result.element_size()
        self.counters.load_seconds += time.perf_counter() - start
        return result

    def _load_block(self, i: int) -> dict[str, torch.Tensor]:
        return {name: self._load_tensor(f"layer.{i}.{name}") for name in BLOCK_NAMES}

    def reset(self) -> None:
        self.keys = [[] for _ in range(self.config.n_layers)]
        self.values = [[] for _ in range(self.config.n_layers)]
        self.position = 0

    @torch.inference_mode()
    def step(self, token_id: int) -> torch.Tensor:
        c = self.config
        if not 0 <= token_id < c.vocab_size:
            raise ValueError("Token out of range")
        if self.position >= c.max_context:
            raise ValueError("Context limit reached; reset engine to restart")
        if self.device.type == "cuda" and self.position == 0:
            torch.cuda.reset_peak_memory_stats(self.device)
        pos = self.position
        x = self.embedding[token_id]
        head_dim = c.dim // c.n_heads
        for i in range(c.n_layers):
            persistent = i in self.pinned
            w = self.pinned[i] if persistent else self._load_block(i)
            h = rms(x, w["attn_norm"], c.rms_eps)
            q = rope((w["wq"] @ h).reshape(c.n_heads, head_dim), pos, c.rope_theta)
            k = rope((w["wk"] @ h).reshape(c.n_kv_heads, head_dim), pos, c.rope_theta)
            v = (w["wv"] @ h).reshape(c.n_kv_heads, head_dim)
            self.keys[i].append(k)
            self.values[i].append(v)
            keys = torch.stack(self.keys[i], dim=1).repeat_interleave(c.n_heads // c.n_kv_heads, dim=0)
            values = torch.stack(self.values[i], dim=1).repeat_interleave(c.n_heads // c.n_kv_heads, dim=0)
            probs = torch.softmax(torch.einsum("hd,htd->ht", q, keys) / math.sqrt(head_dim), dim=-1)
            context = torch.einsum("ht,htd->hd", probs, values).reshape(c.dim)
            x = x + (w["wo"] @ context)
            z = rms(x, w["ffn_norm"], c.rms_eps)
            x = x + (w["down"] @ (F.silu(w["gate"] @ z) * (w["up"] @ z)))
            # No hidden GPU residency for a non-pinned block after use.
            del w, h, q, k, v, keys, values, probs, context, z
        self.position += 1
        logits = self.lm_head @ rms(x, self.final_norm, c.rms_eps)
        self._sync()
        if self.device.type == "cuda":
            self.counters.peak_cuda_allocated_bytes = torch.cuda.max_memory_allocated(self.device)
        return logits

    @torch.inference_mode()
    def generate(self, prompt: list[int], n_tokens: int = 4) -> list[int]:
        if not prompt or n_tokens < 0:
            raise ValueError("Prompt must be nonempty and n_tokens nonnegative")
        if self.position or len(prompt) + max(0, n_tokens-1) > self.config.max_context:
            raise ValueError("Need a fresh engine and enough context budget")
        logits = None
        for token in prompt:
            logits = self.step(token)
        generated: list[int] = []
        for i in range(n_tokens):
            token = int(torch.argmax(logits).item())
            generated.append(token)
            if i < n_tokens-1:
                logits = self.step(token)
        return generated
