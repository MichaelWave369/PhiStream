"""Offline synthetic qualification rules. No CUDA hardware is required in CI."""
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

from phistream.qualification import (
    SCHEMA, _inspect, find_guarded_receipts, qualify_guarded,
)


def receipt(index, *, ratio=0.08, randomized=False, identity="RTX 5070"):
    now = datetime(2026, 10, 12, tzinfo=timezone.utc) + timedelta(minutes=index*2)
    control = [0.014] * 6
    double = [0.014 * (1 - ratio)] * 6
    def series(samples):
        return {"samples_seconds": samples, "median_seconds":statistics.median(samples),
                "min_seconds":min(samples), "max_seconds":max(samples), "repetitions":6}
    return {
        "schema": "PHISTREAM_GUARDED_SYNTHETIC_RUN_V1",
        "provenance": "LOCALLY_EXECUTED_SYNTHETIC_GPU_SCRATCH_NO_MODEL",
        "status": "PASS_GUARDED_CUDA_SYNTHETIC",
        "started_utc": now.isoformat(),
        "completed_utc": (now + timedelta(seconds=1)).isoformat(),
        "requested_double": True, "explicit_operator_approval": True,
        "selected_mode": "double_reuse_worker", "fallback_reasons": [],
        "original_gate_status": "ELIGIBLE_FOR_SYNTHETIC_OPT_IN",
        "live_hardware_check": "MATCHED_UNAUTHENTICATED_RECEIPT_COHORT",
        "chunk_mib": 8, "chunks": 8,
        "selected_end_to_end": series(double),
        "serialized_same_worker_negative_control": series(control),
        "control_is_deliberately_nonoverlapping": True,
        "order_randomization": ({
            "mode": "BALANCED_SHUFFLE", "seed": index,
            "control_first_per_trial": [False,True,False,True,True,False],
        } if randomized else None),
        "integrity": {"full_sha256_host_and_gpu_audit":True,
                      "audit_chunks":8, "audit_is_outside_timing":True},
        "temporary_scratch_deleted":True,
        "model_execution":False, "production_policy_applied":False,
        "automatic_layer_placement":False,
        "gate_summary":{
            "schema":"PHISTREAM_GUARDED_SCHEDULER_V1",
            "status":"ELIGIBLE_FOR_SYNTHETIC_OPT_IN",
            "unverified_external_inputs":True,
            "production_policy_applied":False, "model_execution":False,
            "default":"SINGLE", "guards":[], "validated_receipt_count":3,
            "conditions":{"rounds_per_run":12,"chunk_mib":8,"chunks":8},
            "hardware_claim":{
                "platform":"Windows-11-10.0.26200-SP0",
                "gpu_name":identity, "torch_version":"2.10.0+cu128",
                "cuda_build":"12.8",
            },
        },
    }


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.root = Path(self.t.name)

    def tearDown(self):
        self.t.cleanup()

    def write(self, index, **kw):
        p = self.root / f"guarded-double-repeat-{index}.json"
        p.write_text(json.dumps(receipt(index,**kw)), encoding="utf-8")
        return str(p)

    def cohort(self, ratios=(0.08,0.08,0.08), randomized=False):
        return [self.write(i+1, ratio=g, randomized=randomized)
                for i,g in enumerate(ratios)]

    def test_four_legacy_receipts_preserve_borderline_not_cherry_picked(self):
        files=self.cohort((0.1127,0.0497,0.1146),randomized=False)
        files.append(self.write(4,ratio=0.0746,randomized=False))
        summary=qualify_guarded(files)
        self.assertEqual(summary["status"],"INCONCLUSIVE")
        self.assertEqual(summary["input_receipt_count"],4)
        self.assertEqual(summary["accepted_count"],4)
        self.assertEqual(summary["qualified_run_count"],3)
        self.assertEqual(summary["randomized_run_count"],0)
        self.assertIn("AT_LEAST_ONE_RUN_BELOW_PREDECLARED_5PCT_OR_75PCT_GATE",summary["qualification_notes"])
        self.assertIn("FEWER_THAN_THREE_BALANCED_RANDOMIZED_ORDER_RUNS",summary["qualification_notes"])
        self.assertEqual(len(summary["ledger"]),4)
        self.assertTrue(all(len(r["sha256"])==64 for r in summary["ledger"]))
        self.assertEqual(summary["default"],"SINGLE")
        self.assertFalse(summary["production_policy_applied"])

    def test_legacy_fast_without_random_order_remains_inconclusive(self):
        output=qualify_guarded(self.cohort(randomized=False))
        self.assertEqual(output["status"],"INCONCLUSIVE")
        self.assertEqual(output["qualified_run_count"],3)

    def test_three_randomized_full_control_runs_qualify_synthetic_only(self):
        summary=qualify_guarded(self.cohort(randomized=True))
        self.assertEqual(summary["status"],"QUALIFIED_SYNTHETIC")
        self.assertEqual(summary["randomized_run_count"],3)
        self.assertEqual(summary["aggregate"]["double_wins"],18)
        self.assertFalse(summary["model_execution"])
        self.assertFalse(summary["production_policy_applied"])
        self.assertEqual(summary["default"],"SINGLE")
        self.assertTrue(summary["review_only"])
        self.assertTrue(summary["unverified_external_inputs"])

    def test_partial_pair_consistency_blocks_strong_median(self):
        files=self.cohort(randomized=True)
        entry=json.loads(Path(files[0]).read_text())
        old=entry["serialized_same_worker_negative_control"]["samples_seconds"]
        # 4 wins only (less than ceil(.75*6)=5), yet large median benefit.
        vals=[0.011,0.011,0.011,0.011,0.016,0.016]
        entry["selected_end_to_end"]={
            "samples_seconds":vals, "median_seconds":statistics.median(vals),
            "min_seconds":min(vals), "max_seconds":max(vals), "repetitions":6
        }
        Path(files[0]).write_text(json.dumps(entry))
        out=qualify_guarded(files)
        self.assertEqual(out["status"],"INCONCLUSIVE")
        self.assertEqual(out["qualified_run_count"],2)

    def test_bad_attestation_rejected_even_with_other_good(self):
        files=self.cohort(randomized=True)
        data=json.loads(Path(files[0]).read_text())
        data["integrity"]["full_sha256_host_and_gpu_audit"]=False
        Path(files[0]).write_text(json.dumps(data))
        summary=qualify_guarded(files)
        self.assertEqual(summary["status"],"REJECTED")
        self.assertEqual(summary["accepted_count"],2)
        self.assertTrue(summary["rejected_receipts"])

    def test_duplicate_mixed_hardware_and_overlapping_timestamps(self):
        files=self.cohort(randomized=True)
        duplicate=str(self.root/"duplicate.json")
        Path(duplicate).write_bytes(Path(files[0]).read_bytes())
        self.assertEqual(qualify_guarded(files+[duplicate])["status"],"REJECTED")
        item=json.loads(Path(files[1]).read_text())
        item["gate_summary"]["hardware_claim"]["cuda_build"]="fake"
        Path(files[1]).write_text(json.dumps(item))
        summary=qualify_guarded(files)
        self.assertIn("MIXED_HARDWARE_OR_SOFTWARE",summary["blockers"])
        item["gate_summary"]["hardware_claim"]["cuda_build"]="12.8"
        item["started_utc"]=json.loads(Path(files[0]).read_text())["started_utc"]
        Path(files[1]).write_text(json.dumps(item))
        self.assertEqual(qualify_guarded(files)["status"],"REJECTED")

    def test_reject_bad_raw_medians_order_or_missing_negative_control(self):
        files=self.cohort(randomized=True)
        for mutate in [
            lambda v:v["selected_end_to_end"].update(median_seconds=999),
            lambda v:v.update(serialized_same_worker_negative_control=None),
            lambda v:v["order_randomization"].update(control_first_per_trial=[True]*6),
            lambda v:v.update(requested_double=False),
        ]:
            data=receipt(1,randomized=True)
            mutate(data)
            Path(files[0]).write_text(json.dumps(data))
            self.assertEqual(qualify_guarded(files)["status"],"REJECTED")
        # Other valid files stay unchanged, no hidden deletion.
        self.assertTrue(Path(files[1]).exists())

    def test_local_glob_and_cli_no_cuda(self):
        from phistream.cli import main
        files=self.cohort(randomized=True)
        self.assertEqual(find_guarded_receipts(str(self.root/"guarded-double-*.json")),files)
        with self.assertRaises(ValueError):
            find_guarded_receipts(str(self.root/"**"/"*.json"))
        dest=self.root/"qualification.json"
        args=["phistream","qualify-scheduler","--receipt-glob",
              str(self.root/"guarded-double-*.json"),
              "--output",str(dest)]
        with patch("sys.argv",args),redirect_stdout(StringIO()) as output:
            with patch("torch.cuda.is_available",side_effect=AssertionError("No CUDA review")):
                main()
        self.assertEqual(json.loads(output.getvalue())["schema"],SCHEMA)
        self.assertEqual(json.loads(dest.read_text())["status"],"QUALIFIED_SYNTHETIC")
        with patch("sys.argv",args),redirect_stdout(StringIO()):
            with self.assertRaises(FileExistsError):
                main()


if __name__ == "__main__":
    unittest.main()
