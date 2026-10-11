"""Offline, handcrafted safetensors format tests with no external checkpoints."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import torch

from phistream.safeweights import LocalCheckpoint, SafeTensorFile


def write_st(path, tensors, metadata=None, *, trailing=b"", raw_header=None):
    """Create a format-test file manually, not using a safetensors dependency."""
    offset, data, header = 0, bytearray(), {}
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + len(raw)]}
        data.extend(raw)
        offset += len(raw)
    if metadata is not None:
        header["__metadata__"] = metadata
    encoded = raw_header if raw_header is not None else json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data + trailing)


def floats(*numbers):
    return struct.pack("<" + "f" * len(numbers), *numbers)


def llama_shapes():
    return {
        "model.embed_tokens.weight": (4, 4),
        "model.norm.weight": (4,),
        "lm_head.weight": (4, 4),
        "model.layers.0.input_layernorm.weight": (4,),
        "model.layers.0.post_attention_layernorm.weight": (4,),
        "model.layers.0.self_attn.q_proj.weight": (4, 4),
        "model.layers.0.self_attn.k_proj.weight": (2, 4),
        "model.layers.0.self_attn.v_proj.weight": (2, 4),
        "model.layers.0.self_attn.o_proj.weight": (4, 4),
        "model.layers.0.mlp.gate_proj.weight": (6, 4),
        "model.layers.0.mlp.up_proj.weight": (6, 4),
        "model.layers.0.mlp.down_proj.weight": (4, 6),
    }


class SafetensorsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.file = self.root / "model.safetensors"

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_owned_float32_and_sha(self):
        payload = floats(1, 2, 3, 4)
        write_st(self.file, {"x": ("F32", [2, 2], payload)}, {"format": "pt"})
        reader = SafeTensorFile(self.file)
        self.assertEqual(reader.metadata, {"format": "pt"})
        tensor = reader.read_tensor("x")
        self.assertEqual(tensor.tolist(), [[1.0, 2.0], [3.0, 4.0]])
        tensor[0, 0] = 999
        self.assertEqual(SafeTensorFile(self.file).read_tensor("x")[0, 0].item(), 1)
        self.assertEqual(reader.fingerprint("x"), __import__("hashlib").sha256(payload).hexdigest())
        self.assertEqual(LocalCheckpoint(self.root).inspect()["status"], "STRUCTURE_VALID")

    def test_bfloat16_can_be_materialized(self):
        payload = struct.pack("<HH", 0x3f80, 0x4000)
        write_st(self.file, {"x": ("BF16", [2], payload)})
        a = LocalCheckpoint(self.root).tensor("x")
        self.assertEqual(a.dtype, torch.bfloat16)
        self.assertEqual(a.float().tolist(), [1, 2])

    def test_read_budget_denies_large_tensor(self):
        write_st(self.file, {"x": ("F32", [4], floats(1, 2, 3, 4))})
        with self.assertRaisesRegex(ValueError, "budget"):
            LocalCheckpoint(self.root).tensor("x", max_bytes=4)

    def test_rejects_header_length_over_limit(self):
        self.file.write_bytes(struct.pack("<Q", 16 * 1024 * 1024 + 1) + b"{}")
        with self.assertRaisesRegex(ValueError, "header"):
            SafeTensorFile(self.file)

    def test_rejects_duplicate_json_keys(self):
        write_st(self.file, {}, raw_header=b'{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            SafeTensorFile(self.file)

    def test_rejects_wrong_tensor_byte_size(self):
        write_st(self.file, {"x": ("F32", [2], floats(1))})
        with self.assertRaisesRegex(ValueError, "bounds"):
            SafeTensorFile(self.file)

    def test_rejects_unindexed_trailing_bytes(self):
        write_st(self.file, {"x": ("F32", [1], floats(1))}, trailing=b"bad")
        with self.assertRaisesRegex(ValueError, "trailing"):
            SafeTensorFile(self.file)

    def test_rejects_overlap(self):
        obj = {
            "a": {"dtype":"F32","shape":[1],"data_offsets":[0,4]},
            "b": {"dtype":"F32","shape":[1],"data_offsets":[0,4]},
        }
        h = json.dumps(obj).encode()
        self.file.write_bytes(struct.pack("<Q",len(h))+h+floats(1))
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            SafeTensorFile(self.file)

    def test_rejects_symlinked_weights(self):
        source = self.root / "real.safetensors"
        write_st(source, {"x": ("F32", [1], floats(1))})
        try:
            self.file.symlink_to(source)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not supported by test environment")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            SafeTensorFile(self.file)

    def test_zero_length_tensor_is_allowed(self):
        write_st(self.file, {"empty": ("F32", [0, 4], b"")})
        item = LocalCheckpoint(self.root).tensor("empty")
        self.assertEqual(item.shape, (0, 4))
        self.assertEqual(item.numel(), 0)

    def test_malformed_dtype_is_rejected_cleanly(self):
        header = b'{"x":{"dtype":[],"shape":[1],"data_offsets":[0,4]}}'
        self.file.write_bytes(struct.pack("<Q", len(header)) + header + floats(1))
        with self.assertRaisesRegex(ValueError, "dtype"):
            SafeTensorFile(self.file)

    def test_rejects_boolean_dimension(self):
        encoded = json.dumps({"x":{"dtype":"F32","shape":[True],"data_offsets":[0,4]}}).encode()
        self.file.write_bytes(struct.pack("<Q",len(encoded))+encoded+floats(1))
        with self.assertRaisesRegex(ValueError, "shape"):
            SafeTensorFile(self.file)

    def test_two_shards_with_strict_index(self):
        a = self.root / "model-00001-of-00002.safetensors"
        b = self.root / "model-00002-of-00002.safetensors"
        write_st(a, {"a": ("F16", [1], bytes.fromhex("003c"))})
        write_st(b, {"b": ("F32", [1], floats(1))})
        index = {"weight_map": {"a": a.name, "b": b.name}, "metadata": {"total_size": 6}}
        (self.root / "model.safetensors.index.json").write_text(json.dumps(index))
        reader = LocalCheckpoint(self.root)
        self.assertEqual(reader.inspect()["file_count"], 2)
        self.assertEqual(reader.inspect(hash_all=True)["tensor_count"], 2)
        self.assertEqual(reader.tensor("a").item(), 1)

    def test_denies_path_traversal_in_index(self):
        index = {"weight_map": {"a": "../outside.safetensors"}}
        (self.root / "model.safetensors.index.json").write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "basename"):
            LocalCheckpoint(self.root)

    def test_rejects_index_mismatch(self):
        shard = self.root / "a.safetensors"
        write_st(shard, {"x": ("F32", [1], floats(1))})
        (self.root / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {"missing": "a.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "does not match"):
            LocalCheckpoint(self.root)

    def test_canonical_llama_shape_preflight(self):
        c = dict(model_type="llama", vocab_size=4, hidden_size=4,
                 num_hidden_layers=1, num_attention_heads=2,
                 num_key_value_heads=1, intermediate_size=6,
                 tie_word_embeddings=False)
        (self.root / "config.json").write_text(json.dumps(c))
        tensors = {}
        for name, shape in llama_shapes().items():
            n = __import__("math").prod(shape)
            tensors[name] = ("F32", list(shape), floats(*([0.0] * n)))
        write_st(self.file, tensors)
        reader = LocalCheckpoint(self.root)
        report = reader.inspect(llama_layout=True)
        self.assertEqual(report["status"], "STRUCTURE_VALID")
        self.assertTrue(report["llama_layout"]["passed"])
        self.assertEqual(report["llama_layout"]["expected_core_tensors"], 12)

    def test_llama_shape_mismatch_is_not_compatible(self):
        c = dict(model_type="llama", vocab_size=4, hidden_size=4,
                 num_hidden_layers=1, num_attention_heads=2,
                 num_key_value_heads=1, intermediate_size=6)
        (self.root / "config.json").write_text(json.dumps(c))
        tensors = {}
        for name, shape in llama_shapes().items():
            shape = (1,) if name.endswith("q_proj.weight") else shape
            tensors[name] = ("F32", list(shape), floats(*([0.0] * __import__("math").prod(shape))))
        write_st(self.file, tensors)
        report = LocalCheckpoint(self.root).inspect(llama_layout=True)
        self.assertEqual(report["status"], "LAYOUT_MISMATCH")
        self.assertEqual(report["llama_layout"]["mismatch_count"], 1)

    def test_rejects_non_llama_layout(self):
        write_st(self.file, {"x": ("F32", [1], floats(1))})
        report = LocalCheckpoint(self.root).inspect(llama_layout=True)
        self.assertEqual(report["status"], "LAYOUT_MISMATCH")


if __name__ == "__main__":
    unittest.main()
