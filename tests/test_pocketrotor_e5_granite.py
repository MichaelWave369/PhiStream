"""E5 Granite native routing adapter tests: CPU mock classes only."""
from __future__ import annotations

import unittest
import torch
from torch import nn

from scripts.pocketrotor_e5_granite import GraniteNativeRouterCapture
from scripts.pocketrotor_e2_route_bridge import convert_capture, validate_capture
from scripts.pocketrotor_e2 import analyze


class GraniteMoeTopKRouter(nn.Module):
    __module__ = "transformers.models.granitemoe.modeling_granitemoe"
    def __init__(self):
        super().__init__()
        self.num_experts = 32
        self.top_k = 8
    def forward(self, x):
        start = int(abs(x[0, 0, 0].item())) % 24
        ids = torch.arange(start, start + 8, dtype=torch.int64).reshape(1, 8)
        return torch.zeros((1, 32)), torch.ones((1, 8)), ids


class GraniteMoeExperts(nn.Module):
    __module__ = "transformers.models.granitemoe.modeling_granitemoe"
    def forward(self, x, chosen, weights):
        return x + .125


class MiniGraniteBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = GraniteMoeTopKRouter()
        self.experts = GraniteMoeExperts()
        self.mutate = False
        self.skip = False
    def forward(self, x):
        _, w, ids = self.router(x)
        if self.skip:
            return x
        if self.mutate:
            ids = (ids + 1) % 32
        return self.experts(x.reshape(-1, 2), ids, w).reshape(1, 1, 2)


class MiniLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_sparse_moe = MiniGraniteBlock()
    def forward(self, x):
        return self.block_sparse_moe(x)


class MiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([MiniLayer(), MiniLayer()])
        self.eval()
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def source():
    return {
        "kind": "synthetic", "runtime": "E5-native-shaped-mock",
        "model_id": "no-pretrained-weights",
        "checkpoint_sha256": "0" * 64,
        "route_semantics": "POST_NATIVE_ROUTER_SELECTED_EXPERT_IDS",
        "token_text_recorded": False,
    }


def capture(model):
    return GraniteNativeRouterCapture(
        model, source=source(), expected_layers=2,
        expected_experts=32, expected_top_k=8)


class GraniteE5Tests(unittest.TestCase):
    def test_captured_native_id_pass_through_and_e2(self):
        model = MiniModel()
        x = torch.tensor([[[3.0, 1.0]]])
        ref = model(x)
        with capture(model) as rec:
            observed = []
            for split, size in (("train", 4), ("same_family", 2), ("shifted", 2)):
                for step in range(size):
                    observed.append(rec.run_step(split, step, lambda: model(x)))
        for out in observed:
            torch.testing.assert_close(ref, out, atol=0, rtol=0)
        native = rec.export()
        data = validate_capture(native)
        self.assertEqual(data["layers"], [0, 1])
        self.assertEqual(len(native["events"]), 16)
        trace, manifest = convert_capture(
            native, layer=0, expert_bytes=1024,
            capacity_bytes=12 * 1024, max_prefetch=1)
        self.assertEqual(trace["expert_count"], 32)
        self.assertEqual(analyze(trace)["schema"],
                         "PHISTREAM_POCKETROTOR_E2_RESULT_V1")
        self.assertEqual(manifest["source_authentication"],
                         "OPERATOR_ASSERTED_NOT_VERIFIED")
        self.assertTrue(manifest["synthetic_source"])

    def test_differing_consumed_ids_fail_closed(self):
        model = MiniModel()
        model.layers[1].block_sparse_moe.mutate = True
        with capture(model) as rec:
            with self.assertRaisesRegex(ValueError, "did not consume"):
                rec.run_step("train", 0, lambda: model(torch.zeros((1, 1, 2))))
        with self.assertRaises(RuntimeError):
            rec.export()

    def test_skipped_experts_fail_closed(self):
        model = MiniModel()
        model.layers[1].block_sparse_moe.skip = True
        with capture(model) as rec:
            with self.assertRaises(ValueError):
                rec.run_step("train", 0, lambda: model(torch.zeros((1, 1, 2))))
        with self.assertRaises(RuntimeError):
            rec.export()

    def test_wrong_module_wrong_mode_and_wrong_contract_refused(self):
        model = MiniModel()
        with self.assertRaisesRegex(ValueError, "coverage"):
            GraniteNativeRouterCapture(model, source=source())
        model.layers[0].block_sparse_moe.router = nn.Linear(2, 32)
        with self.assertRaisesRegex(ValueError, "router class"):
            capture(model)
        model = MiniModel()
        model.train()
        with self.assertRaisesRegex(ValueError, "eval mode"):
            capture(model)
        model = MiniModel()
        model.layers[0].block_sparse_moe.router.top_k = 4
        with self.assertRaisesRegex(ValueError, "count/top-k"):
            capture(model)

    def test_rejects_parallel_tokens_and_invalid_close(self):
        model = MiniModel()
        with capture(model) as rec:
            with self.assertRaisesRegex(ValueError, "one native token"):
                rec.run_step(
                    "train", 0,
                    lambda: model(torch.ones((2, 1, 2))))
        with self.assertRaises(RuntimeError):
            rec.export()
        model = MiniModel()
        c = capture(model)
        c.close()
        with self.assertRaises(RuntimeError):
            c.run_step("train", 0, lambda: model(torch.zeros((1, 1, 2))))

    def test_malformed_source_metadata_denied(self):
        model = MiniModel()
        bad = source()
        bad["token_text_recorded"] = True
        with self.assertRaises(ValueError):
            GraniteNativeRouterCapture(
                model, source=bad, expected_layers=2)
        for key, val in (("expected_layers", True),
                         ("expected_experts", 300),
                         ("expected_top_k", 9)):
            with self.subTest(key=key):
                opts = {"expected_layers": 2,
                        "expected_experts": 32,
                        "expected_top_k": 8}
                opts[key] = val
                with self.assertRaises(ValueError):
                    GraniteNativeRouterCapture(model, source=source(), **opts)


if __name__ == "__main__":
    unittest.main()
