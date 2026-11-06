"""Offline synthetic safetensors tests; zero pretrained models, zero network."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from phistream import weight_surgery as ws


class WeightSurgeryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.base = self.d / "original.safetensors"
        entries = {"layer.weight": {"dtype": "F32", "shape": [3], "data_offsets": [0, 12]},
                   "half.weight": {"dtype": "F16", "shape": [2], "data_offsets": [12, 16]},
                   "bf.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [16, 20]}}
        header = json.dumps(entries, separators=(",", ":")).encode()
        header += b" " * ((8 - len(header) % 8) % 8)
        data = struct.pack("<fff", 1.0, 2.0, -3.0) + struct.pack("<ee", 0.5, 1.0) + struct.pack("<HH", 0x3f80, 0x4000)
        self.base.write_bytes(struct.pack("<Q", len(header)) + header + data)

    def path(self, name):
        return self.d / name

    def test_draft_apply_verify_revert_byte_identical(self):
        original_sha = ws._sha256(self.base)
        plan = ws.draft(self.base, "layer.weight", 1, 8.0, self.path("plan.json"))
        self.assertEqual(plan["checkpoint_sha256"], original_sha)
        receipt = ws.apply(self.base, self.path("plan.json"), self.path("patched.safetensors"), self.path("receipt.json"))
        self.assertNotEqual(original_sha, receipt["patched_sha256"])
        ws.verify(self.path("patched.safetensors"), self.path("receipt.json"))
        outcome = ws.revert(self.path("patched.safetensors"), self.path("receipt.json"), self.path("restored.safetensors"))
        self.assertEqual(outcome["sha256"], original_sha)
        self.assertEqual(self.base.read_bytes(), self.path("restored.safetensors").read_bytes())
        self.assertEqual(original_sha, ws._sha256(self.base))

    def test_f16_and_bf16(self):
        for tensor in ("half.weight", "bf.weight"):
            with self.subTest(tensor=tensor):
                plan = self.path(tensor + ".json")
                ws.draft(self.base, tensor, 0, 1.5, plan)
                patched = self.path(tensor + ".safetensors")
                receipt = self.path(tensor + ".receipt")
                ws.apply(self.base, plan, patched, receipt)
                ws.verify(patched, receipt)
                restored = self.path(tensor + ".restored")
                ws.revert(patched, receipt, restored)
                self.assertEqual(self.base.read_bytes(), restored.read_bytes())

    def test_baseline_hash_mismatch(self):
        ws.draft(self.base, "layer.weight", 0, 6.0, self.path("plan.json"))
        plan = json.loads(self.path("plan.json").read_text())
        plan["checkpoint_sha256"] = "0" * 64
        self.path("plan.json").write_text(json.dumps(plan))
        with self.assertRaisesRegex(ws.SurgeryError, "SHA-256"):
            ws.apply(self.base, self.path("plan.json"), self.path("edited"), self.path("receipt"))
        self.assertFalse(self.path("edited").exists())

    def test_reject_tampered_expected_and_duplicate(self):
        plan = ws.draft(self.base, "layer.weight", 0, 5.0, self.path("plan.json"))
        plan["edits"][0]["expected_hex"] = "00000000"
        self.path("plan.json").write_text(json.dumps(plan))
        with self.assertRaisesRegex(ws.SurgeryError, "expected scalar"):
            ws.apply(self.base, self.path("plan.json"), self.path("edited"), self.path("receipt"))
        plan["edits"][0]["expected_hex"] = "0000803f"
        plan["edits"].append(dict(plan["edits"][0]))
        self.path("plan.json").write_text(json.dumps(plan))
        with self.assertRaisesRegex(ws.SurgeryError, "duplicate tensor"):
            ws.apply(self.base, self.path("plan.json"), self.path("edited"), self.path("receipt"))

    def test_reject_existing_output_and_wrong_receipt(self):
        ws.draft(self.base, "layer.weight", 0, 9.0, self.path("plan.json"))
        self.path("edited").write_text("leave me alone")
        with self.assertRaisesRegex(ws.SurgeryError, "new distinct"):
            ws.apply(self.base, self.path("plan.json"), self.path("edited"), self.path("receipt"))
        self.assertEqual(self.path("edited").read_text(), "leave me alone")
        ws.apply(self.base, self.path("plan.json"), self.path("patched"), self.path("receipt"))
        self.path("patched").write_bytes(self.path("patched").read_bytes() + b"!")
        with self.assertRaisesRegex(ws.SurgeryError, "hash or size"):
            ws.verify(self.path("patched"), self.path("receipt"))

    def test_nonfinite_rejected(self):
        with self.assertRaisesRegex(ws.SurgeryError, "non-finite"):
            ws.draft(self.base, "layer.weight", 0, float("nan"), self.path("plan.json"))
        with self.assertRaisesRegex(ws.SurgeryError, "non-finite"):
            ws.draft(self.base, "bf.weight", 0, float("inf"), self.path("plan.json"))

    def test_bad_shape_and_overlap(self):
        data = bytearray(self.base.read_bytes())
        header_len = struct.unpack("<Q", data[:8])[0]
        head = json.loads(data[8:8+header_len])
        head["half.weight"]["data_offsets"] = [8, 12]
        new_header = json.dumps(head).encode()
        new_header += b" " * ((8-len(new_header)%8)%8)
        self.path("corrupt").write_bytes(struct.pack("<Q", len(new_header)) + new_header + data[8+header_len:])
        with self.assertRaisesRegex(ws.SurgeryError, "overlapping"):
            ws._tensor_table(self.path("corrupt"))

    def test_invalid_index_and_nop(self):
        with self.assertRaisesRegex(ws.SurgeryError, "out of bounds"):
            ws.draft(self.base, "layer.weight", 99, 3.0, self.path("p"))
        with self.assertRaisesRegex(ws.SurgeryError, "no-op"):
            ws.draft(self.base, "layer.weight", 0, 1.0, self.path("p"))

    def test_receipt_tampering(self):
        ws.draft(self.base, "layer.weight", 0, 2.0, self.path("plan"))
        ws.apply(self.base, self.path("plan"), self.path("out"), self.path("receipt"))
        r = json.loads(self.path("receipt").read_text())
        r["plan"]["edits"][0]["index"] = 2
        self.path("receipt").write_text(json.dumps(r))
        with self.assertRaisesRegex(ws.SurgeryError, "plan digest"):
            ws.verify(self.path("out"), self.path("receipt"))

    def test_cli_help_and_inspect(self):
        import contextlib
        import io
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(ws.main(["inspect", str(self.base), "--tensor", "bf.weight", "--index", "1"]), 0)
        data = json.loads(stdout.getvalue())
        self.assertEqual(data["scalar"]["value"], 2.0)
        self.assertEqual(data["scalar"]["raw_hex"], "0040")

    def test_multi_edit_plan(self):
        plan = ws.draft(self.base, "layer.weight", 0, 7.0, self.path("plan.json"))
        plan["edits"].append({"tensor":"layer.weight", "index":2,"expected_hex":struct.pack("<f", -3).hex(),"new_hex":struct.pack("<f", 4).hex()})
        self.path("plan.json").write_text(json.dumps(plan))
        ws.apply(self.base, self.path("plan.json"), self.path("patched"), self.path("receipt"))
        ws.revert(self.path("patched"), self.path("receipt"), self.path("original2"))
        self.assertEqual(self.base.read_bytes(), self.path("original2").read_bytes())


if __name__ == "__main__":
    unittest.main()
