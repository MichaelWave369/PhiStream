"""Opt-in, bounded causal batched-prefill *candidate* for canonical Llama.

Does not change StreamingLlama.step(), realbridge, the CLI or SINGLE defaults.
Only called by an explicitly approved experiment. No downloads or weights
are modified. Not a measured speedup or independently authenticated parity.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .llama import LAYER_NAMES, LlamaSettings, StreamingLlama, _llama_rope, _rms
from .memory import memory_preflight, refuse_blocked
from .safeweights import LocalCheckpoint

MAX_BATCH_PREFILL_TOKENS = 24


def estimated_extra_workspace_bytes(cfg, tokens: int) -> int:
    """Conservative policy *estimate*, not a maximum allocator bound."""
    if type(tokens) is not int or not 2 <= tokens <= MAX_BATCH_PREFILL_TOKENS:
        raise ValueError("Candidate requires 2..24 prompt tokens")
    hdim = cfg.dim // cfg.heads
    # Multiple copies of score/probability, QKV, MLP temporaries and saved KV.
    # Actual PyTorch/CUDA temporaries, allocator state and context are unbounded.
    return 4 * (
        8 * cfg.heads * tokens * tokens
        + 12 * cfg.dim * tokens
        + 8 * cfg.intermediate * tokens
        + 8 * cfg.kv_heads * hdim * tokens
    )


@torch.inference_mode()
def approved_batched_prefill(
    checkpoint: LocalCheckpoint,
    prompt: list[int],
    *,
    approve_local_execution: bool = False,
    device: str = "cpu",
    capacity_mib: float = 8192,
    reserve_mib: float = 1024,
    max_tensor_mib: int = 512,
    head_rows: int = 128,
) -> tuple[StreamingLlama, torch.Tensor, dict]:
    """Prefill a NEW engine layer-by-layer, return last-token logits and KV.

    No real model execution occurs unless the caller explicitly supplies
    approve_local_execution=True. This function is not wired into the CLI.
    Caller must separately verify the full model reference trace and timings.
    """
    if type(approve_local_execution) is not bool or not approve_local_execution:
        raise PermissionError("Batched prefill requires explicit local execution approval")
    if device not in ("cpu", "cuda"):
        raise ValueError("Only explicit cpu or cuda device is accepted")
    if type(prompt) is not list or not 2 <= len(prompt) <= MAX_BATCH_PREFILL_TOKENS:
        raise ValueError("Batched prefill requires a list of 2..24 token IDs")
    if any(type(token) is not int for token in prompt):
        raise ValueError("Prompt must contain integer token IDs only")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")

    # No model inference and no executor initialization before policy checks.
    observed_free = int(torch.cuda.mem_get_info()[0]) if device == "cuda" else None
    plan = memory_preflight(
        checkpoint, context_tokens=len(prompt), capacity_mib=capacity_mib,
        reserve_mib=reserve_mib, max_tensor_mib=max_tensor_mib,
        head_rows=head_rows, observed_free_bytes=observed_free,
    )
    refuse_blocked(plan)
    cfg = LlamaSettings.from_checkpoint(checkpoint)
    if len(prompt) > cfg.max_context or any(not 0 <= token < cfg.vocab for token in prompt):
        raise ValueError("Prompt exceeds vocabulary or context bounds")
    extra = estimated_extra_workspace_bytes(cfg, len(prompt))
    if plan["estimated_working_set_bytes"] + extra > plan["available_policy_bytes"]:
        raise MemoryError("Batched prompt workspace estimate exceeds selected policy budget")

    engine = StreamingLlama(
        checkpoint, device=device, max_tensor_mib=max_tensor_mib,
        head_rows=head_rows,
    )
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.device)

    n = len(prompt)
    head_dim = cfg.dim // cfg.heads
    x = torch.stack([
        engine._rows("model.embed_tokens.weight", token, token + 1).reshape(cfg.dim)
        for token in prompt
    ], dim=0)

    causal_mask = torch.ones((n, n), dtype=torch.bool, device=engine.device).triu(1)
    for layer in range(cfg.layers):
        prefix = f"model.layers.{layer}."
        w = {name: engine._weight(prefix + name) for name in LAYER_NAMES}
        norm = _rms(x, w["input_layernorm.weight"], cfg.rms_eps)
        q = (norm @ w["self_attn.q_proj.weight"].T).reshape(n, cfg.heads, head_dim).transpose(0, 1)
        k = (norm @ w["self_attn.k_proj.weight"].T).reshape(n, cfg.kv_heads, head_dim).transpose(0, 1)
        v = (norm @ w["self_attn.v_proj.weight"].T).reshape(n, cfg.kv_heads, head_dim).transpose(0, 1)

        # Same split-half RoPE as the immutable single-token baseline, with
        # absolute position per prompt token and a strictly causal mask.
        q = torch.stack([_llama_rope(q[:, t, :], t, cfg.rope_theta) for t in range(n)], dim=1)
        k = torch.stack([_llama_rope(k[:, t, :], t, cfg.rope_theta) for t in range(n)], dim=1)
        expanded_k = k.repeat_interleave(cfg.heads // cfg.kv_heads, dim=0)
        expanded_v = v.repeat_interleave(cfg.heads // cfg.kv_heads, dim=0)
        scores = (q @ expanded_k.transpose(-1, -2)) / math.sqrt(head_dim)
        scores = scores.masked_fill(causal_mask.unsqueeze(0), -float("inf"))
        attended = torch.softmax(scores, dim=-1) @ expanded_v
        mix = attended.transpose(0, 1).reshape(n, cfg.dim)
        x = x + mix @ w["self_attn.o_proj.weight"].T
        mlp_input = _rms(x, w["post_attention_layernorm.weight"], cfg.rms_eps)
        x = x + (
            F.silu(mlp_input @ w["mlp.gate_proj.weight"].T)
            * (mlp_input @ w["mlp.up_proj.weight"].T)
        ) @ w["mlp.down_proj.weight"].T

        # Preserve the existing per-position KV shape expected by step().
        engine.keys[layer] = [part.clone() for part in k.unbind(dim=1)]
        engine.values[layer] = [part.clone() for part in v.unbind(dim=1)]
        del w, norm, q, k, v, expanded_k, expanded_v, scores, attended, mix, mlp_input

    head_input = _rms(x[-1], engine._final_norm, cfg.rms_eps)
    head_name = "model.embed_tokens.weight" if cfg.tied_head else "lm_head.weight"
    logits = torch.cat([
        engine._rows(head_name, first, min(cfg.vocab, first + engine.head_rows)) @ head_input
        for first in range(0, cfg.vocab, engine.head_rows)
    ])
    engine.position = n
    engine._sync()
    if engine.device.type == "cuda":
        engine.last_peak_cuda_allocated = int(torch.cuda.max_memory_allocated(engine.device))
    evidence = {
        "status": "UNVERIFIED_OPT_IN_BATCHED_PREFILL_CANDIDATE",
        "mode": "BATCHED_CAUSAL_PREFILL_ONLY",
        "reference_parity": "NOT_ESTABLISHED_FOR_THIS_EXECUTION",
        "production_promoted": False,
        "single_baseline_modified": False,
        "prompt_tokens": n,
        "memory_policy": plan,
        "extra_workspace_estimated_bytes_not_hard_limit": extra,
        "weight_staging_count_observed": engine.weight_load_count,
        "weight_staged_bytes_observed": engine.weight_bytes_staged,
        "peak_cuda_allocated_bytes": engine.last_peak_cuda_allocated,
    }
    return engine, logits, evidence
