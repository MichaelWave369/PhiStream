"""Guarded scheduler: receipt provenance gates, fail-closed and opt-in tests.

No real GPU or model weights are required by CI; CUDA execution is mocked.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phistream.guarded import (
    SCHEMA, _inspect_receipt, find_receipts, review_receipts,
    run_synthetic_guarded,
)
from phistream.overlap import _order, recommend


def example_receipt(index: int, *, single=0.014, double=0.0125, rounds=12):
    started = datetime(2026, 10, 11, 18, tzinfo=timezone.utc) + timedelta(minutes=index*3)
    a = [single] * rounds
    b = [double] * rounds
    policy = recommend(a, b, gpu=True)
    return {
        "schema": "PHISTREAM_OVERLAP_LAB_V1",
        "provenance":"MEASURED_LOCAL_SYNTHETIC_HOST_TIMELINE_NO_MODEL",
        "status":"PASS_CUDA_OVERLAP_LAB", "gpu_requested":"cuda",
        "chunk_mib":8, "chunks":8,
        "model_execution":False, "automatic_layer_placement":False,
        "platform":"Windows-11-10.0.26200-SP0",
        "started_utc":started.isoformat(),
        "completed_utc":(started+timedelta(seconds=2)).isoformat(),
        "design":{
            "paired_rounds":rounds,
            "balanced_orders":_order(rounds) if rounds in (6,12) else [],
            "host_buffers_reused_only_after_cuda_completion_event":True,
            "same_preallocated_host_device_buffer_budget":True,
            "scratch_cache_controlled":False, "gpu_compute_overlap_tested":False,
            "gpu_clock_aligned_with_cpu":False,
        },
        "fixture":{
            "synthetic":True,"scratch_deleted":True,
            "total_bytes":64*1024*1024, "sha256":"a"*64,
        },
        "validation":{
            "full_sha256_host_and_gpu_audit":True,
            "audit_is_outside_timing":True,"audit_chunks":8,
        },
        "hardware":{
            "gpu_name":"NVIDIA GeForce RTX 5070",
            "torch_version":"2.10.0+cu128",
            "cuda_build":"12.8",
            "free_vram_before_bytes":11571036160,
        },
        "methods":{
            "single":{"end_to_end":{"samples_seconds":a,"median_seconds":single}},
            "double_reuse_worker":{"end_to_end":{"samples_seconds":b,"median_seconds":double}},
        },
        "recommendation":policy,
    }


class GuardedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, index, *, changes=None):
        obj = example_receipt(index)
        if changes is not None:
            changes(obj)
        p = self.root / f"overlap-8m8-repeat-{index}.json"
        p.write_text(json.dumps(obj, indent=2), encoding="utf8")
        return str(p)

    def cohort(self):
        return [self.write(i) for i in (1,2,3)]

    def test_three_full_receipts_pass_only_unverified_candidate_gate(self):
        data = review_receipts(self.cohort())
        self.assertEqual(data["schema"],SCHEMA)
        self.assertEqual(data["status"], "ELIGIBLE_FOR_SYNTHETIC_OPT_IN")
        self.assertEqual(data["validated_receipt_count"],3)
        self.assertEqual(data["aggregate"]["double_wins"],36)
        self.assertEqual(data["aggregate"]["paired_trials"],36)
        self.assertEqual(data["guards"],[])
        self.assertEqual(data["default"],"SINGLE")
        self.assertFalse(data["production_policy_applied"])
        self.assertTrue(data["unverified_external_inputs"])
        self.assertFalse(data["model_execution"])

    def test_four_repeats_positive_nine_of_twelve_each(self):
        paths=self.cohort()
        def make_partial(obj):
            d=[0.0124]*9 + [0.0143]*3
            obj["methods"]["double_reuse_worker"]["end_to_end"]={
                "samples_seconds":d,
                "median_seconds":0.0124,
            }
            obj["recommendation"]=recommend([0.014]*12,d,gpu=True)
        for i, path in enumerate(paths):
            item=json.loads(Path(path).read_text())
            make_partial(item)
            Path(path).write_text(json.dumps(item))
        data=review_receipts(paths)
        self.assertEqual(data["status"],"ELIGIBLE_FOR_SYNTHETIC_OPT_IN")
        self.assertEqual(data["aggregate"]["double_wins"],27)

    def test_six_round_and_mixed_sizes_do_not_promote(self):
        paths = self.cohort()
        changed=json.loads(Path(paths[0]).read_text())
        changed["design"]["paired_rounds"]=6
        Path(paths[0]).write_text(json.dumps(changed))
        data=review_receipts(paths)
        self.assertEqual(data["status"],"HELD_SINGLE_DEFAULT")
        self.assertIn("INVALID_OR_DUPLICATE_RECEIPTS",data["guards"])
        self.assertIn("INSUFFICIENT_VALID_RUNS",data["guards"])

    def test_duplicate_and_mismatched_hardware_refuse(self):
        paths=self.cohort()
        duplicate=self.root/"duplicate.json"
        duplicate.write_bytes(Path(paths[0]).read_bytes())
        data=review_receipts(paths+[str(duplicate)])
        self.assertIn("INVALID_OR_DUPLICATE_RECEIPTS",data["guards"])
        changed=json.loads(Path(paths[1]).read_text())
        changed["hardware"]["torch_version"]="9.9-faked"
        Path(paths[1]).write_text(json.dumps(changed))
        data=review_receipts(paths)
        self.assertIn("INCONSISTENT_HARDWARE_OR_SOFTWARE",data["guards"])

    def test_tampered_checksum_and_policy_cannot_pass(self):
        paths=self.cohort()
        changed=json.loads(Path(paths[0]).read_text())
        changed["validation"]["full_sha256_host_and_gpu_audit"]=False
        Path(paths[0]).write_text(json.dumps(changed))
        self.assertEqual(review_receipts(paths)["status"],"HELD_SINGLE_DEFAULT")
        changed["validation"]["full_sha256_host_and_gpu_audit"]=True
        changed["recommendation"]["decision"]="DOUBLE_CANDIDATE"  # Now lie about raw trials.
        changed["methods"]["double_reuse_worker"]["end_to_end"]={
            "samples_seconds":[0.019]*12,"median_seconds":0.019
        }
        Path(paths[0]).write_text(json.dumps(changed))
        self.assertEqual(review_receipts(paths)["status"],"HELD_SINGLE_DEFAULT")

    def test_overlapping_capture_times_are_rejected(self):
        paths=self.cohort()
        changed=json.loads(Path(paths[1]).read_text())
        changed["started_utc"]=example_receipt(1)["started_utc"]
        changed["completed_utc"]=example_receipt(1)["completed_utc"]
        Path(paths[1]).write_text(json.dumps(changed))
        self.assertEqual(review_receipts(paths)["status"],"HELD_SINGLE_DEFAULT")

    def test_glob_is_local_only_bounded_and_nonrecursive(self):
        self.cohort()
        items=find_receipts(str(self.root / "overlap-*.json"))
        self.assertEqual(len(items),3)
        with self.assertRaises(ValueError):
            find_receipts(str(self.root / "**" / "*.json"))
        with self.assertRaises(ValueError):
            find_receipts(str(self.root / "not-found*.json"))

    def test_hardware_capacity_failure_falls_back_or_blocks(self):
        evidence=review_receipts(self.cohort())
        class FakeCuda:
            name="GPU"
            def replay_audit(self,*a): return {"full_sha256_host_and_gpu_audit":True}
        with patch("phistream.guarded._local_cuda_check",return_value=(True,"MOCK_MATCHED")):
            with patch("phistream.guarded._CudaBackend",return_value=FakeCuda()):
                with patch("phistream.guarded._fixture") as mock_fixture:
                    mock_fixture.side_effect=lambda path,*args:(hashlib.sha256(b"").hexdigest(),["a"]*8,path.touch() or 0)
                    def fake_trial(path,chunk_bytes,chunks,method,backend,worker,serialize_worker=False):
                        return {"elapsed_seconds":0.01 if method=="single" else 0.009}
                    with patch("phistream.guarded._trial",side_effect=fake_trial) as trial:
                        a=run_synthetic_guarded(self.root,evidence,request_double=True,
                                                approve_synthetic_double=False)
                        self.assertEqual(a["selected_mode"],"single")
                        self.assertIn("OPERATOR_NOT_APPROVED",a["fallback_reasons"])
                        self.assertFalse(a["production_policy_applied"])
                        self.assertEqual(trial.call_count,7)
                        trial.reset_mock()
                        b=run_synthetic_guarded(self.root,evidence,request_double=True,
                                                approve_synthetic_double=True)
                        self.assertEqual(b["selected_mode"],"double_reuse_worker")
                        self.assertIsNotNone(b["serialized_same_worker_negative_control"])
                        self.assertTrue(any(call.kwargs.get("serialize_worker") is True
                                            for call in trial.call_args_list))
        # No scratch file survives.
        self.assertEqual(sorted(p.name for p in self.root.glob("phistream-guarded-*")), [])

    def test_cli_review_only_does_not_call_cuda(self):
        from phistream.cli import main
        self.cohort()
        destination=self.root/"review.json"
        args=["phistream","guarded-scheduler","--receipt-glob",
              str(self.root/"overlap-8m8-repeat-*.json"),
              "--output",str(destination)]
        with patch("sys.argv",args),redirect_stdout(StringIO()) as output:
            with patch("phistream.guarded._local_cuda_check",side_effect=AssertionError("never GPU")):
                main()
        self.assertEqual(json.loads(output.getvalue())["status"],"ELIGIBLE_FOR_SYNTHETIC_OPT_IN")
        self.assertTrue(destination.exists())
        with patch("sys.argv",args),redirect_stdout(StringIO()):
            with self.assertRaises(FileExistsError):
                main()


if __name__=="__main__":
    unittest.main()
