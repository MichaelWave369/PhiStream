"""Independent batched causal reference for PhiStream Llama-style token streaming.

Creates only tiny untrained canonical Llama safetensors with hand-written file
format; no network, model hub, transformers dependency or real-model claims.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from phistream.safeweights import LocalCheckpoint
from phistream.llama import StreamingLlama, LlamaSettings, execute_local_tokens


def local_llama(root: Path, *, dtype: str = "F32", tied: bool = False, seed: int = 369):
    cfg = {
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "vocab_size": 32, "max_position_embeddings": 64,
        "rope_theta": 10000.0, "rms_norm_eps": 1e-5,
        "hidden_act": "silu", "tie_word_embeddings": tied,
        "attention_bias": False,
    }
    (root / "config.json").write_text(json.dumps(cfg))
    g = torch.Generator().manual_seed(seed)
    shapes = {
        "model.embed_tokens.weight": (32, 8),
        "model.norm.weight": (8,),
    }
    if not tied:
        shapes["lm_head.weight"] = (32, 8)
    for i in range(2):
        p = f"model.layers.{i}."
        shapes.update({
            p + "input_layernorm.weight": (8,),
            p + "post_attention_layernorm.weight": (8,),
            p + "self_attn.q_proj.weight": (8, 8),
            p + "self_attn.k_proj.weight": (4, 8),
            p + "self_attn.v_proj.weight": (4, 8),
            p + "self_attn.o_proj.weight": (8, 8),
            p + "mlp.gate_proj.weight": (16, 8),
            p + "mlp.up_proj.weight": (16, 8),
            p + "mlp.down_proj.weight": (8, 16),
        })
    dtype_map = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
    entries, weights, payload, offset = {}, {}, bytearray(), 0
    for name, shape in shapes.items():
        if "norm.weight" in name or "layernorm.weight" in name:
            tensor = torch.ones(shape, dtype=torch.float32) + 0.015 * torch.randn(shape, generator=g)
        else:
            tensor = torch.randn(shape, generator=g) * 0.08
        tensor = tensor.to(dtype_map[dtype]).contiguous()
        weights[name] = tensor.float()
        raw = tensor.view(torch.uint8).numpy().tobytes()
        entries[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        payload.extend(raw)
        offset += len(raw)
    header = json.dumps(entries, separators=(",", ":")).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
    return cfg, weights


def independent_batched_logits(weights, config, ids):
    """Whole-sequence causal reference with split-half RoPE and GQA."""
    hidden = weights["model.embed_tokens.weight"][ids].clone()
    t, d = hidden.shape
    heads, kvh = config["num_attention_heads"], config["num_key_value_heads"]
    head_dim = d // heads
    half = head_dim // 2
    theta = config["rope_theta"]
    inverse = torch.pow(torch.tensor(theta, dtype=torch.float32),
                        -torch.arange(half, dtype=torch.float32) / half)
    frequencies = torch.arange(t, dtype=torch.float32).unsqueeze(1) * inverse
    angles = torch.cat([frequencies, frequencies], dim=1)
    cs, sn = angles.cos().unsqueeze(0), angles.sin().unsqueeze(0)
    mask = torch.triu(torch.ones((t, t), dtype=torch.bool), diagonal=1)
    eps = config["rms_norm_eps"]

    def norm(x, scale):
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)) * scale

    def rotate(x):
        return x * cs + torch.cat([-x[..., half:], x[..., :half]], dim=-1) * sn

    for layer in range(config["num_hidden_layers"]):
        p = f"model.layers.{layer}."
        pre = norm(hidden, weights[p + "input_layernorm.weight"])
        q = (pre @ weights[p + "self_attn.q_proj.weight"].T).reshape(t, heads, head_dim).transpose(0, 1)
        k = (pre @ weights[p + "self_attn.k_proj.weight"].T).reshape(t, kvh, head_dim).transpose(0, 1)
        v = (pre @ weights[p + "self_attn.v_proj.weight"].T).reshape(t, kvh, head_dim).transpose(0, 1)
        q, k = rotate(q), rotate(k)
        k = k.repeat_interleave(heads // kvh, dim=0)
        v = v.repeat_interleave(heads // kvh, dim=0)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
        scores = scores.masked_fill(mask, -float("inf"))
        probs = torch.softmax(scores, dim=-1)
        attention_out = torch.matmul(probs, v).transpose(0, 1).reshape(t, d)
        hidden = hidden + attention_out @ weights[p + "self_attn.o_proj.weight"].T
        ff = norm(hidden, weights[p + "post_attention_layernorm.weight"])
        gate = F.silu(ff @ weights[p + "mlp.gate_proj.weight"].T)
        up = ff @ weights[p + "mlp.up_proj.weight"].T
        hidden = hidden + (gate * up) @ weights[p + "mlp.down_proj.weight"].T
    hidden = norm(hidden, weights["model.norm.weight"])
    head = weights["model.embed_tokens.weight"] if config["tie_word_embeddings"] else weights["lm_head.weight"]
    return hidden @ head.T


class StreamingLlamaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config, self.weights = local_llama(self.root)
        self.ckpt = LocalCheckpoint(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_nonadjacent_rope_parity_against_independent_batched_reference(self):
        engine = StreamingLlama(self.ckpt, device="cpu", head_rows=7, max_tensor_mib=1)
        ids = [1, 7, 3, 8, 4, 6]
        baseline = independent_batched_logits(self.weights, self.config, ids)
        for index, token in enumerate(ids):
            logits = engine.step(token)
            torch.testing.assert_close(logits, baseline[index], atol=1e-5, rtol=1e-5)
        self.assertEqual(engine.position, len(ids))
        self.assertGreater(engine.weight_load_count, 0)
        self.assertGreater(engine.weight_bytes_staged, 0)

    def test_greedy_tokens_match_independent_full_sequence(self):
        engine = StreamingLlama(self.ckpt, head_rows=11, max_tensor_mib=1)
        prompt = [1, 4, 3]
        generated = engine.generate_ids(prompt, 6)
        expected = []
        seq = prompt.copy()
        for _ in range(6):
            next_id = int(independent_batched_logits(self.weights, self.config, seq)[-1].argmax())
            expected.append(next_id)
            seq.append(next_id)
        self.assertEqual(generated, expected)

    def test_tied_embedding_and_bfloat16_weights(self):
        with tempfile.TemporaryDirectory() as name:
            cfg, w = local_llama(Path(name), dtype="BF16", tied=True)
            engine = StreamingLlama(LocalCheckpoint(name), head_rows=3, max_tensor_mib=1)
            ids = [2, 5, 1, 7]
            reference = independent_batched_logits(w, cfg, ids)
            for idx, token in enumerate(ids):
                torch.testing.assert_close(engine.step(token), reference[idx], atol=1e-5, rtol=1e-5)

    def test_f16_tensor_staging(self):
        with tempfile.TemporaryDirectory() as name:
            cfg, w = local_llama(Path(name), dtype="F16")
            engine = StreamingLlama(LocalCheckpoint(name), head_rows=5, max_tensor_mib=1)
            ids = [1, 2, 3]
            reference = independent_batched_logits(w, cfg, ids)
            for idx, token in enumerate(ids):
                torch.testing.assert_close(engine.step(token), reference[idx], atol=1e-5, rtol=1e-5)

    def test_row_range_and_owned_buffer(self):
        tensor_file = self.ckpt.files["model.safetensors"]
        part = tensor_file.read_axis0_rows("model.embed_tokens.weight", 4, 8, max_bytes=4096)
        torch.testing.assert_close(part, self.weights["model.embed_tokens.weight"][4:8])
        part[0, 0] = 123
        fresh = tensor_file.read_axis0_rows("model.embed_tokens.weight", 4, 5)
        self.assertNotEqual(fresh[0, 0].item(), 123)

    def test_row_reader_rejects_invalid_requests(self):
        f = self.ckpt.files["model.safetensors"]
        with self.assertRaises(ValueError):
            f.read_axis0_rows("model.embed_tokens.weight", 0, 33)
        with self.assertRaises(ValueError):
            f.read_axis0_rows("model.embed_tokens.weight", 1, 0)
        with self.assertRaises(ValueError):
            f.read_axis0_rows("model.embed_tokens.weight", 0, 10, max_bytes=1)
        with self.assertRaises(ValueError):
            f.read_axis0_rows("model.norm.weight", 0, 1)

    def test_invalid_variant_rejected(self):
        config_file = self.root / "config.json"
        for k, bad in [
            ("rope_scaling", {"type": "linear", "factor": 2}),
            ("attention_bias", True),
            ("hidden_act", "gelu"),
            ("head_dim", 99),
            ("sliding_window", 128),
        ]:
            cfg = dict(self.config)
            cfg[k] = bad
            config_file.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError, msg=f"configuration {k} must be rejected"):
                StreamingLlama(LocalCheckpoint(self.root))
        config_file.write_text(json.dumps(self.config))

    def test_unknown_tensor_rejected_as_unsupported(self):
        cfg = dict(self.config)
        cfg["architectures"] = ["SomeOtherArchitecture"]
        (self.root / "config.json").write_text(json.dumps(cfg))
        with self.assertRaises(ValueError):
            LlamaSettings.from_checkpoint(LocalCheckpoint(self.root))

    def test_cuda_never_silently_falls_back(self):
        with patch("phistream.llama.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                StreamingLlama(self.ckpt, device="cuda")

    def test_context_guards_and_reset(self):
        e = StreamingLlama(self.ckpt)
        with self.assertRaises(ValueError):
            e.step(-1)
        with self.assertRaises(ValueError):
            e.generate_ids([], 2)
        with self.assertRaises(ValueError):
            e.generate_ids([1] * 64, 3)
        first = e.generate_ids([1, 2], 3)
        e.reset()
        self.assertEqual(first, e.generate_ids([1, 2], 3))

    def test_receipt_not_reference_verified(self):
        receipt = execute_local_tokens(self.ckpt, [1, 2, 3], 3, max_tensor_mib=1)
        self.assertEqual(receipt["status"], "EXECUTED_UNVERIFIED_REFERENCE")
        self.assertEqual(receipt["reference_parity"], "NOT_ESTABLISHED_FOR_THIS_CHECKPOINT")
        self.assertEqual(len(receipt["generated_token_ids"]), 3)
        self.assertEqual(receipt["device"], "cpu")
        self.assertIsNone(receipt["peak_cuda_allocated_bytes"])
        json.dumps(receipt, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
