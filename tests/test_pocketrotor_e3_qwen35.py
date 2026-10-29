"""E3 Qwen3.5 contract tests. CPU mock modules, no Transformers or model weights."""
from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from scripts.pocketrotor_e2 import analyze, validate
from scripts.pocketrotor_e2_route_bridge import convert_capture, validate_capture
from scripts.pocketrotor_e3_qwen35 import (
    Qwen35NativeRouterCapture, adapter_report,
)


class Qwen3_5MoeTopKRouter(nn.Module):
    __module__ = "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"

    def __init__(self):
        super().__init__()
        self.num_experts = 256
        self.top_k = 8
        self.bad_shape = False

    def forward(self, x):
        value = int(x[0, 0].item()) % 249
        ids = torch.arange(value, value + 8, dtype=torch.long).reshape(1, 8)
        if self.bad_shape:
            ids = ids.repeat(2, 1)
        return torch.zeros((1, 256)), torch.ones((1, 8)), ids


class Qwen3_5MoeExperts(nn.Module):
    __module__ = "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"

    def forward(self, hidden, selected_ids, weights):
        return hidden + 1


class TinyMoEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = Qwen3_5MoeTopKRouter()
        self.experts = Qwen3_5MoeExperts()
        self.corrupt = False
        self.skip_experts = False
        self.skip_gate = False

    def forward(self, x):
        if self.skip_gate:
            _, weights, indices = (None, torch.ones((1, 8)),
                                    torch.arange(8, dtype=torch.long).reshape(1, 8))
        else:
            _, weights, indices = self.gate(x)
        if self.skip_experts:
            return x + 1
        if self.corrupt:
            indices = (indices + 1) % 256
        return self.experts(x, indices, weights)


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMoEBlock()

    def forward(self, x):
        return self.mlp(x)


class TinyNativeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer(), TinyLayer()])
        self.eval()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def metadata():
    return {
        "kind": "synthetic",
        "runtime": "mock-qwen3.5-shape-only",
        "model_id": "untrained-no-reference-weights",
        "checkpoint_sha256": "0" * 64,
        "route_semantics": "POST_NATIVE_ROUTER_SELECTED_EXPERT_IDS",
        "token_text_recorded": False,
    }


def wrapper(model, *, source=None):
    return Qwen35NativeRouterCapture(
        model, source=metadata() if source is None else source,
        expected_experts=256, expected_top_k=8, expected_layers=2,
    )


class Qwen35NativeHookTests(unittest.TestCase):
    def test_native_gate_experts_parity_preserves_outputs_and_exports_256(self):
        model = TinyNativeModel()
        inputs = [torch.tensor([[float(i)]]) for i in range(6)]
        original = [model(x) for x in inputs]
        with wrapper(model) as session:
            observed = []
            for split, offset in (("train", 0), ("same_family", 2),
                                  ("shifted", 4)):
                for step in range(2):
                    observed.append(session.run_step(
                        split, step, lambda x=inputs[offset + step]: model(x)
                    ))
            with self.assertRaises(RuntimeError):
                session.export()
        for a, b in zip(original, observed):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        capture = session.export()
        self.assertEqual(capture["expert_count"], 256)
        self.assertEqual(len(capture["events"]), 12)
        checked = validate_capture(capture)
        self.assertEqual(checked["layers"], [0, 1])
        self.assertEqual(checked["counts"], {
            "train": 2, "same_family": 2, "shifted": 2,
        })
        self.assertFalse(adapter_report(capture)["runtime_provenance_attested"])
        converted, manifest = convert_capture(
            capture, layer=1, expert_bytes=4096,
            capacity_bytes=8 * 4096, max_prefetch=1,
        )
        self.assertEqual(validate(converted)["expert_count"], 256)
        self.assertEqual(manifest["source_authentication"],
                         "OPERATOR_ASSERTED_NOT_VERIFIED")
        replay = analyze(converted)
        self.assertFalse(replay["model_execution"])
        self.assertTrue(replay["simulated_expert_routing_only"])

    def test_gate_ids_are_never_substituted(self):
        model = TinyNativeModel()
        model.layers[1].mlp.corrupt = True
        with wrapper(model) as session:
            with self.assertRaisesRegex(ValueError, "did not consume"):
                session.run_step("train", 0, lambda: model(torch.tensor([[1.0]])))
        with self.assertRaises(RuntimeError):
            session.export()

    def test_missing_gate_or_native_experts_fails_closed(self):
        for flag in ("skip_gate", "skip_experts"):
            model = TinyNativeModel()
            setattr(model.layers[1].mlp, flag, True)
            with self.subTest(flag=flag):
                with wrapper(model) as session:
                    with self.assertRaises(ValueError):
                        session.run_step(
                            "train", 0, lambda: model(torch.tensor([[1.0]]))
                        )
                with self.assertRaises(RuntimeError):
                    session.export()

    def test_rejects_multitoken_batch_prefill_without_guessing_alignment(self):
        model = TinyNativeModel()
        model.layers[1].mlp.gate.bad_shape = True
        with wrapper(model) as session:
            with self.assertRaisesRegex(ValueError, "one native token"):
                session.run_step("train", 0, lambda: model(torch.tensor([[2.0]])))
        with self.assertRaises(RuntimeError):
            session.export()

    def test_constructor_requires_known_classes_and_complete_coverage(self):
        model = TinyNativeModel()
        with self.assertRaisesRegex(ValueError, "coverage"):
            Qwen35NativeRouterCapture(
                model, source=metadata(), expected_layers=40,
            )
        with self.assertRaisesRegex(ValueError, "architecture mismatch"):
            Qwen35NativeRouterCapture(
                model, source=metadata(), expected_layers=2, expected_top_k=7,
            )
        model.layers[1].mlp.gate = nn.Linear(1, 8)
        with self.assertRaisesRegex(ValueError, "native Qwen3.5 gate"):
            wrapper(model)
        model = TinyNativeModel()
        model.train()
        with self.assertRaisesRegex(ValueError, "eval mode"):
            wrapper(model)

    def test_no_fabricated_source_authority_or_auto_run(self):
        model = TinyNativeModel()
        source = copy.deepcopy(metadata())
        source["kind"] = "operator_asserted_native"
        with wrapper(model, source=source) as session:
            for split in ("train", "same_family", "shifted"):
                for step in range(2):
                    session.run_step(
                        split, step, lambda: model(torch.tensor([[1.0]]))
                    )
        capture = session.export()
        report = adapter_report(capture)
        self.assertEqual(report["source_kind"], "operator_asserted_native")
        self.assertFalse(report["checkpoint_hash_independently_verified"])
        self.assertFalse(report["production_promoted"])

    def test_reuse_after_close_or_duplicate_execution_denied(self):
        model = TinyNativeModel()
        session = wrapper(model)
        session.close()
        with self.assertRaises(RuntimeError):
            session.run_step("train", 0, lambda: model(torch.tensor([[1.0]])))
        with self.assertRaises(RuntimeError):
            session.export()
        with self.assertRaises(RuntimeError):
            session.__enter__()

    def test_overspecified_topk_and_shape_refused(self):
        model = TinyNativeModel()
        for value in (0, True, 9):
            with self.subTest(topk=value):
                with self.assertRaises(ValueError):
                    Qwen35NativeRouterCapture(
                        model, source=metadata(), expected_top_k=value,
                        expected_layers=2,
                    )


if __name__ == "__main__":
    unittest.main()
