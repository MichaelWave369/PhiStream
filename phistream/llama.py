"""Strict, local, layer-streamed inference for a narrow canonical Llama layout.

No tokenizer or model downloads. FP32 math for a CPU correctness baseline.
This deliberately refuses variant architectures, unsupported RoPE schemes,
unknown tensor layouts, and oversized per-tensor staging requests.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import platform
import time

import torch
import torch.nn.functional as F

from .safeweights import LocalCheckpoint

LAYER_NAMES = (
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
)
RECEIPT_SCHEMA = "PHISTREAM_LLAMA_TOKEN_IDS_V1"
EXPERIMENTAL_PROVENANCE = "LOCAL_LLAMA_STREAM_EXECUTED_UNVERIFIED_REFERENCE"


def _number(value: object, fallback: float) -> float:
    value = fallback if value is None else value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Invalid numeric architecture setting")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Non-finite or nonpositive architecture setting")
    return result


def _int(value: object, *, default: int | None = None) -> int:
    value = default if value is None else value
    if type(value) is not int or value < 1:
        raise ValueError("Invalid positive architecture integer")
    return value


@dataclass(frozen=True)
class LlamaSettings:
    vocab: int
    dim: int
    heads: int
    kv_heads: int
    layers: int
    intermediate: int
    max_context: int
    rope_theta: float
    rms_eps: float
    tied_head: bool

    @classmethod
    def from_checkpoint(cls, checkpoint: LocalCheckpoint) -> "LlamaSettings":
        config = checkpoint.config
        if not isinstance(config, dict) or config.get("model_type") != "llama":
            raise ValueError("Only model_type 'llama' is supported")
        layout = checkpoint.validate_llama_layout()
        if not layout["passed"]:
            raise ValueError("Llama layout mismatch: " + str(layout["reason"]))
        if layout["extra_tensors"]:
            raise ValueError("Unrecognized tensors found; refusing potentially unsupported model layout")
        if config.get("architectures") not in (None, ["LlamaForCausalLM"]):
            raise ValueError("Unsupported architectures; require LlamaForCausalLM")
        for field in ("rope_scaling", "rope_parameters", "sliding_window", "attention_chunk_size"):
            if config.get(field) not in (None, False):
                raise ValueError(f"Unsupported Llama feature: {field}")
        for field in ("attention_bias", "mlp_bias", "qk_norm"):
            if config.get(field, False) is not False:
                raise ValueError(f"Unsupported Llama feature: {field}")
        if config.get("hidden_act", "silu") != "silu":
            raise ValueError("Only SwiGLU / silu is supported")
        if config.get("pretraining_tp", 1) != 1:
            raise ValueError("Tensor-parallel checkpoints are not supported")
        if config.get("partial_rotary_factor", 1.0) != 1.0:
            raise ValueError("Partial rotary embeddings unsupported")
        dim, heads = _int(config.get("hidden_size")), _int(config.get("num_attention_heads"))
        if config.get("head_dim") not in (None, dim // heads):
            raise ValueError("Nonstandard attention head_dim unsupported")
        settings = cls(
            vocab=_int(config.get("vocab_size")),
            dim=dim,
            heads=heads,
            kv_heads=_int(config.get("num_key_value_heads"), default=heads),
            layers=_int(config.get("num_hidden_layers")),
            intermediate=_int(config.get("intermediate_size")),
            max_context=_int(config.get("max_position_embeddings"), default=2048),
            rope_theta=_number(config.get("rope_theta"), 10000.0),
            rms_eps=_number(config.get("rms_norm_eps"), 1e-6),
            tied_head=bool(config.get("tie_word_embeddings", False)),
        )
        return settings


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(torch.mean(x.square(), dim=-1, keepdim=True) + eps) * weight


def _llama_rope(heads: torch.Tensor, pos: int, theta: float) -> torch.Tensor:
    """Canonical HF-Llama rotate_half: split halves, NOT adjacent even/odd pairs."""
    half = heads.shape[-1] // 2
    freq = torch.arange(half, dtype=torch.float32, device=heads.device)
    freq = theta ** (-freq / half)
    angle = pos * freq
    cos = torch.cat((angle.cos(), angle.cos()))
    sin = torch.cat((angle.sin(), angle.sin()))
    return heads * cos + torch.cat((-heads[..., half:], heads[..., :half]), dim=-1) * sin


class StreamingLlama:
    """Single-sequence, single-token, deterministic greedy inference.

    Embeddings and output projection stream contiguous axis-0 slices.
    Decoder layer parameters are staged a layer at a time in FP32, which
    avoids permanently allocating the whole model but *does not* guarantee
    a particular VRAM ceiling or useful throughput for large checkpoints.
    """

    def __init__(
        self,
        checkpoint: LocalCheckpoint,
        *,
        device: str = "cpu",
        max_tensor_mib: int = 512,
        head_rows: int = 128,
    ):
        if device not in ("cpu", "cuda"):
            raise ValueError("Explicit device must be cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
        if type(max_tensor_mib) is not int or not 1 <= max_tensor_mib <= 4096:
            raise ValueError("max_tensor_mib must be between 1 and 4096")
        if type(head_rows) is not int or not 1 <= head_rows <= 16384:
            raise ValueError("head_rows must be between 1 and 16384")
        self.ckpt = checkpoint
        self.cfg = LlamaSettings.from_checkpoint(checkpoint)
        self.device = torch.device(device)
        self.max_tensor_bytes = max_tensor_mib * 1024 * 1024
        self.head_rows = head_rows
        self.position = 0
        self.keys: list[list[torch.Tensor]] = [[] for _ in range(self.cfg.layers)]
        self.values: list[list[torch.Tensor]] = [[] for _ in range(self.cfg.layers)]
        self.weight_load_count = 0
        self.weight_bytes_staged = 0
        self.weight_load_seconds = 0.0
        self.last_peak_cuda_allocated = None
        self._final_norm = self._weight("model.norm.weight")

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _weight(self, name: str) -> torch.Tensor:
        started = time.perf_counter()
        t = self.ckpt.tensor(name, max_bytes=self.max_tensor_bytes)
        out = t.to(device=self.device, dtype=torch.float32)
        self._sync()
        self.weight_load_count += 1
        self.weight_bytes_staged += t.numel() * t.element_size()
        self.weight_load_seconds += time.perf_counter() - started
        return out

    def _rows(self, name: str, start: int, end: int) -> torch.Tensor:
        entry = self.ckpt.entries[name]
        file = self.ckpt.files[entry.file]
        started = time.perf_counter()
        tensor = file.read_axis0_rows(name, start, end, max_bytes=self.max_tensor_bytes)
        out = tensor.to(device=self.device, dtype=torch.float32)
        self._sync()
        self.weight_load_count += 1
        self.weight_bytes_staged += tensor.numel() * tensor.element_size()
        self.weight_load_seconds += time.perf_counter() - started
        return out

    def reset(self):
        self.position = 0
        self.keys = [[] for _ in range(self.cfg.layers)]
        self.values = [[] for _ in range(self.cfg.layers)]

    @torch.inference_mode()
    def step(self, token_id: int) -> torch.Tensor:
        cfg = self.cfg
        if type(token_id) is not int or not 0 <= token_id < cfg.vocab:
            raise ValueError("Token ID outside vocabulary")
        if self.position >= cfg.max_context:
            raise ValueError("Context limit reached")
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        pos = self.position
        x = self._rows("model.embed_tokens.weight", token_id, token_id + 1).reshape(cfg.dim)
        head_dim = cfg.dim // cfg.heads
        for i in range(cfg.layers):
            prefix = f"model.layers.{i}."
            # Maximum one layer of weights staged at a time. These are
            # explicit loads from local validated safetensors only.
            w = {n: self._weight(prefix + n) for n in LAYER_NAMES}
            z = _rms(x, w["input_layernorm.weight"], cfg.rms_eps)
            q = _llama_rope((w["self_attn.q_proj.weight"] @ z).reshape(cfg.heads, head_dim),
                            pos, cfg.rope_theta)
            k = _llama_rope((w["self_attn.k_proj.weight"] @ z).reshape(cfg.kv_heads, head_dim),
                            pos, cfg.rope_theta)
            v = (w["self_attn.v_proj.weight"] @ z).reshape(cfg.kv_heads, head_dim)
            self.keys[i].append(k)
            self.values[i].append(v)
            keys = torch.stack(self.keys[i], dim=1).repeat_interleave(cfg.heads // cfg.kv_heads, dim=0)
            values = torch.stack(self.values[i], dim=1).repeat_interleave(cfg.heads // cfg.kv_heads, dim=0)
            attn = torch.softmax(torch.einsum("hd,htd->ht", q, keys) / math.sqrt(head_dim), dim=-1)
            mix = torch.einsum("ht,htd->hd", attn, values).reshape(cfg.dim)
            x = x + w["self_attn.o_proj.weight"] @ mix
            z2 = _rms(x, w["post_attention_layernorm.weight"], cfg.rms_eps)
            x = x + w["mlp.down_proj.weight"] @ (
                F.silu(w["mlp.gate_proj.weight"] @ z2) * (w["mlp.up_proj.weight"] @ z2))
            # Do not retain a reference to the weight dictionary between layers.
            del w, z, z2, q, k, v, keys, values, attn, mix
        x = _rms(x, self._final_norm, cfg.rms_eps)
        head_name = ("model.embed_tokens.weight" if cfg.tied_head else "lm_head.weight")
        parts = []
        for start in range(0, cfg.vocab, self.head_rows):
            end = min(cfg.vocab, start + self.head_rows)
            rows = self._rows(head_name, start, end)
            parts.append(rows @ x)
        result = torch.cat(parts)
        self.position += 1
        self._sync()
        if self.device.type == "cuda":
            self.last_peak_cuda_allocated = int(torch.cuda.max_memory_allocated(self.device))
        return result

    @torch.inference_mode()
    def generate_ids(self, prompt: list[int], n_tokens: int = 4) -> list[int]:
        if not prompt or type(n_tokens) is not int or n_tokens < 0:
            raise ValueError("Prompt must be nonempty and token count nonnegative")
        if self.position != 0 or len(prompt) + max(0, n_tokens - 1) > self.cfg.max_context:
            raise ValueError("Fresh decoder required and prompt must fit context")
        logits = None
        for token in prompt:
            logits = self.step(token)
        generated: list[int] = []
        for i in range(n_tokens):
            next_id = int(torch.argmax(logits).item())
            generated.append(next_id)
            if i != n_tokens - 1:
                logits = self.step(next_id)
        return generated


def execute_local_tokens(
    checkpoint: LocalCheckpoint,
    prompt: list[int],
    tokens: int,
    *,
    device: str = "cpu",
    max_tensor_mib: int = 512,
    head_rows: int = 128,
    capacity_mib: float | None = None,
    reserve_mib: float = 512.0,
) -> dict:
    """Receipt with no claim of independent reference agreement."""
    if not isinstance(prompt, list) or not prompt or len(prompt) > 2048:
        raise ValueError("Specify a nonempty prompt of at most 2048 token IDs")
    if type(tokens) is not int or tokens < 1 or tokens > 128:
        raise ValueError("Token count must be 1 to 128")
    plan = None
    if capacity_mib is not None:
        from .memory import memory_preflight, refuse_blocked
        observed_free = None
        if device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
            observed_free = int(torch.cuda.mem_get_info()[0])
        plan = memory_preflight(
            checkpoint, context_tokens=len(prompt) + tokens - 1,
            capacity_mib=capacity_mib, reserve_mib=reserve_mib,
            max_tensor_mib=max_tensor_mib, head_rows=head_rows,
            observed_free_bytes=observed_free,
        )
        refuse_blocked(plan)
    engine = StreamingLlama(checkpoint, device=device, max_tensor_mib=max_tensor_mib,
                            head_rows=head_rows)
    started = time.perf_counter()
    output = engine.generate_ids(prompt, tokens)
    elapsed = time.perf_counter() - started
    return {
        "schema": RECEIPT_SCHEMA,
        "provenance": EXPERIMENTAL_PROVENANCE,
        "status": "EXECUTED_UNVERIFIED_REFERENCE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "device": str(engine.device),
        "gpu_name": torch.cuda.get_device_name(engine.device) if engine.device.type == "cuda" else None,
        "checkpoint_metadata": {
            "model_type": "llama",
            "layer_count": engine.cfg.layers,
            "parameter_weight_bytes": sum(t.nbytes for t in checkpoint.entries.values()),
            "tensor_count": len(checkpoint.entries),
            "dtype_set": sorted({t.dtype for t in checkpoint.entries.values()}),
        },
        "prompt_token_ids": prompt,
        "generated_token_ids": output,
        "generated_token_count": len(output),
        "decode_with_prefill_seconds": round(elapsed, 6),
        "generated_tokens_per_second_including_prefill": round(tokens / elapsed, 6) if elapsed else None,
        "weight_load_events": engine.weight_load_count,
        "weight_bytes_staged_total": engine.weight_bytes_staged,
        "weight_load_seconds_including_transfers": round(engine.weight_load_seconds, 6),
        "peak_cuda_allocated_bytes": engine.last_peak_cuda_allocated,
        "reference_parity": "NOT_ESTABLISHED_FOR_THIS_CHECKPOINT",
        "memory_preflight": plan,
        "interpretation": (
            "Locally executed token IDs with narrow Llama FP32 math. "
            "No tokenizer, EOS policy, license validation, model-specific reference parity, "
            "large-model speed comparison or end-to-end VRAM guarantee."
        ),
    }
