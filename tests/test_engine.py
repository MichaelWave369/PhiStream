import json
import tempfile
import unittest
from pathlib import Path

import torch

from phistream import ModelConfig, ModelStore, PhiStreamEngine, build_fixture
from phistream.cli import benchmark


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        build_fixture(self.root, config=ModelConfig(vocab_size=32, dim=32,
                        n_layers=3, n_heads=4, n_kv_heads=2,
                        ffn_dim=48, max_context=12), seed=369)
        self.store = ModelStore(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_manifest_hashes(self):
        self.store.verify()

    def test_stream_vs_resident_logits(self):
        stream = PhiStreamEngine(self.store, mode="stream", device="cpu")
        resident = PhiStreamEngine(self.store, mode="resident", device="cpu")
        for token in [1, 7, 3, 8, 2]:
            torch.testing.assert_close(stream.step(token), resident.step(token), atol=1e-6, rtol=1e-6)
        self.assertEqual(stream.position, resident.position)
        self.assertFalse(stream.pinned)
        self.assertEqual(len(resident.pinned), self.store.config.n_layers)

    def test_seed_reproducible(self):
        s1 = PhiStreamEngine(self.store, mode="stream", device="cpu")
        s2 = PhiStreamEngine(self.store, mode="stream", device="cpu")
        self.assertEqual(s1.generate([1, 2, 3], 4), s2.generate([1, 2, 3], 4))

    def test_pin_budget_respected(self):
        small = PhiStreamEngine(self.store, pin_budget_mib=0.00001, device="cpu")
        self.assertEqual(len(small.pinned), 0)
        large = PhiStreamEngine(self.store, pin_budget_mib=10, device="cpu")
        self.assertEqual(len(large.pinned), self.store.config.n_layers)
        self.assertLessEqual(large.pinned_parameter_bytes, 10 * 1024 * 1024)

    def test_pinned_stream_still_matches_resident(self):
        pinned = PhiStreamEngine(self.store, mode="stream", device="cpu", pin_budget_mib=0.04)
        resident = PhiStreamEngine(self.store, mode="resident", device="cpu")
        self.assertGreater(len(pinned.pinned), 0)
        self.assertLess(len(pinned.pinned), self.store.config.n_layers)
        for token in [2, 7, 5, 1]:
            torch.testing.assert_close(pinned.step(token), resident.step(token), atol=1e-6, rtol=1e-6)

    def test_reset_replays_identically(self):
        engine = PhiStreamEngine(self.store, mode="stream", device="cpu")
        a = engine.generate([1, 2], 3)
        engine.reset()
        b = engine.generate([1, 2], 3)
        self.assertEqual(a, b)

    def test_context_guard(self):
        engine = PhiStreamEngine(self.store, device="cpu")
        with self.assertRaises(ValueError):
            engine.generate([1] * 10, 5)
        with self.assertRaises(ValueError):
            engine.step(-1)

    def test_corrupt_detected(self):
        entry = self.store.entries["embed"]
        with (self.root / "weights.bin").open("r+b") as f:
            f.seek(entry["offset"])
            original = f.read(1)
            f.seek(entry["offset"])
            f.write(bytes([original[0] ^ 0xFF]))
        with self.assertRaises(ValueError):
            ModelStore(self.root).verify()

    def test_manifest_overlap_detected(self):
        p = self.root / "manifest.json"
        obj = json.loads(p.read_text())
        obj["tensors"]["lm_head"]["offset"] = obj["tensors"]["embed"]["offset"]
        p.write_text(json.dumps(obj))
        with self.assertRaises(ValueError):
            ModelStore(self.root)

    def test_receipt(self):
        result = benchmark(self.store, "stream", "cpu", [1, 2], 2, 0)
        self.assertEqual(result["provenance"], "MEASURED_ON_THIS_MACHINE_SYNTHETIC_UNTRAINED")
        self.assertEqual(len(result["generated_token_ids"]), 2)
        self.assertGreater(result["weight_load_events"], 0)


if __name__ == "__main__":
    unittest.main()
