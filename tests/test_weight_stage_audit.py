"""PR22 offline staging-counter audit, synthetic receipt fixtures only.

No real checkpoint download, GPU execution or HF Transformers import.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.weight_stage_audit import audit, main
from scripts.real_weight_matrix import CASES
from test_real_weight_matrix import RealMatrixTests

WEIGHT_BYTES = 269030016
EXTRA_BYTES = 1152


class WeightStageAuditTests(unittest.TestCase):
    def setUp(self):
        # Reuse PR21's mock builder; NEVER run external or PhiStream models.
        self.fixture = RealMatrixTests("test_plan_only_and_frozen_cases")
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.fixture.make_pair()
        self.patcher = patch("scripts.weight_stage_audit.pinned_checkpoint",
                             return_value=({"vocab_size":49152},
                                           self.fixture.fp,self.fixture.decl))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.root=self.fixture.root
        self.evidence=self.fixture.evidence
        for i,(cid,prompt) in enumerate(CASES):
            path=self.evidence/("parity_"+cid+".json")
            result=json.loads(path.read_text())
            perf=result["execution"]["performance"]
            steps = len(prompt)+31
            perf.update({
                "measurement_scope":"DECODER_ONLY_EXCLUDES_CHECKPOINT_SHA_AND_TRACE_VALIDATION",
                "no_reference_speedup_claim":True,
                "weight_load_count":1+655*steps,
                "weight_bytes_staged_total":WEIGHT_BYTES*steps+EXTRA_BYTES,
                "weight_load_seconds_including_transfers":round(8.0+0.23*(len(prompt)-1),6),
                "total_with_engine_setup_seconds":round(9.8+0.27*(len(prompt)-1),6),
            })
            result["checkpoint"]={"weight_bytes":WEIGHT_BYTES}
            path.write_text(json.dumps(result),encoding="utf-8")
        self.summary = self.root/"review.json"
        self.fixture.evaluate()
        self.output=self.root/"stage-audit.json"

    def run_audit(self):
        return audit(self.fixture.model,self.fixture.preflight,
                     self.fixture.manifest,self.evidence,self.summary,
                     self.output)

    def test_verified_mock_receipts_exact_bytes_and_operation_count(self):
        a=self.run_audit()
        self.assertEqual(a["status"],"EXACT_STAGE_COUNTER_LINEARITY_OBSERVED")
        self.assertEqual(a["per_run_fixed_staged_bytes_outside_single_token_steps"],
                         EXTRA_BYTES)
        self.assertTrue(a["exact_bytes_step_pattern"])
        self.assertEqual(a["weight_load_operations_per_decoder_step"],655)
        self.assertEqual(a["fixed_initial_weight_load_operations"],1)
        self.assertEqual([c["single_token_decoder_steps"] for c in a["cases"]],
                         [32,38,43,55])
        self.assertEqual([c["recorded_staging_bytes"] for c in a["cases"]],
                         [WEIGHT_BYTES*x+EXTRA_BYTES for x in (32,38,43,55)])
        self.assertFalse(a["model_execution"])
        self.assertFalse(a["real_model_optimization_speedup_measured"])
        self.assertFalse(a["scheduler_policy_changed"])
        self.assertFalse(a["production_ready"])
        self.assertEqual(a["matrix_status"],"MATCHED_LIMITED_FOUR_CASES")
        self.assertTrue(self.output.exists())

    def test_output_exclusive(self):
        self.run_audit()
        with self.assertRaisesRegex(ValueError,"never overwrite"):
            self.run_audit()

    def test_missing_case_denied_before_output(self):
        (self.evidence/"parity_mixed_7.json").unlink()
        with self.assertRaisesRegex(ValueError,"missing"):
            self.run_audit()
        self.assertFalse(self.output.exists())

    def test_changed_counter_after_matrix_review_detected(self):
        path=self.evidence/"parity_short_1.json"
        result=json.loads(path.read_text())
        result["execution"]["performance"]["weight_bytes_staged_total"] += 269030016
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError,"differ from revalidated"):
            self.run_audit()

    def test_changed_trace_rejected(self):
        path=self.evidence/"reference_repeat_12.json"
        text=path.read_text()
        path.write_text(text+" ")
        with self.assertRaisesRegex(ValueError,"REFERENCE SHA256|reference|trace"):
            self.run_audit()

    def test_existing_match_not_implicitly_production_ready(self):
        a=self.run_audit()
        self.assertTrue(a["performance_scope"]["each_case_only_one_run"])
        self.assertTrue(a["performance_scope"]["cannot_infer_transfer_bandwidth_or_gpu_speedup"])
        self.assertEqual(a["four_cases"],4)

    def test_counter_pattern_failure_reported_not_hidden(self):
        path=self.evidence/"parity_spread_24.json"
        result=json.loads(path.read_text())
        result["execution"]["performance"]["weight_bytes_staged_total"] += 100
        path.write_text(json.dumps(result))
        self.summary.unlink()
        self.fixture.evaluate()
        a=self.run_audit()
        self.assertEqual(a["status"],"COUNTER_PATTERN_NOT_FULLY_LINEAR")
        self.assertFalse(a["exact_bytes_step_pattern"])
        self.assertTrue(self.output.exists())

    def test_not_execute_during_review(self):
        with patch("scripts.weight_stage_audit._case_pair",
                   wraps=__import__("scripts.weight_stage_audit",
                                    fromlist=["_case_pair"])._case_pair) as pair:
            self.run_audit()
        self.assertEqual(pair.call_count,4)

    def test_summary_manipulation_denied(self):
        d=json.loads(self.summary.read_text())
        d["passing_cases"]=3
        self.summary.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,"complete, matching"):
            self.run_audit()
        self.assertFalse(self.output.exists())


if __name__=="__main__":
    unittest.main()
