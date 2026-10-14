import { test } from 'node:test';
import assert from 'node:assert/strict';
import { REPLICATION, validateReplicationReview } from './replication-evidence.mjs';

const SHA = 'a'.repeat(64);
function fixture() {
  return {
    schema: 'PHISTREAM_REPLICATION_REVIEW_V1',
    status: 'REPLICATION_REVIEW_CANDIDATE_NOT_SPEEDUP_CERTIFIED',
    model_execution: false,
    source_reference_authenticated: false,
    statistical_significance_established: false,
    cross_day_calendar_verified_independently: false,
    production_promoted: false,
    production_scheduler_unchanged: true,
    general_speedup_proven: false,
    original_seed: 369,
    original_census_sha256: SHA,
    replication_plan_sha256: SHA,
    replication_run_sha256: SHA,
    reference_sha256: SHA,
    original_spread_24: {
      case: 'spread_24',
      pairs: 4,
      candidate_faster_pair_count: 4,
      median_paired_relative_gain: REPLICATION.originalGain,
      review_thresholds_met: true,
    },
    replication: {
      case: REPLICATION.case,
      seed: REPLICATION.seed,
      pairs: 8,
      measured_executions: 16,
      warmups: 2,
      model_executions_recorded: 18,
      greedy_id_matches: 576,
      sampled_logit_matches: 5184,
      candidate_faster_pair_count: 8,
      median_paired_relative_gain: REPLICATION.gain,
      median_relative_ttft_gain: REPLICATION.ttft,
      median_cumulative_weight_bytes_reduction: REPLICATION.stagedBytesSaved,
      review_thresholds_met: true,
      qualification: 'CANDIDATE_FOR_REPLICATED_REVIEW_NOT_PROVEN_SPEEDUP',
    },
    combined: {
      paired_comparisons: 20,
      batched_faster_pairs: 20,
      recorded_executions_including_warmups: 48,
      greedy_id_matches: 1536,
      sparse_logit_matches: 13774,
      cohort_count: 4,
      replication_minus_original_spread_median_gain_percentage_points:
        100 * (REPLICATION.gain - REPLICATION.originalGain),
    },
  };
}
test('operator snapshot captures eight pairs and explicitly no independent SHA claim', () => {
  assert.equal(REPLICATION.pairs, 8);
  assert.equal(REPLICATION.wins, 8);
  assert.equal(REPLICATION.status, 'OPERATOR_REPORTED_LOCAL_SHA_AUDIT_PASSED');
  assert.equal(REPLICATION.auditReceiptStatus, 'REPLICATION_REVIEW_CANDIDATE_NOT_SPEEDUP_CERTIFIED');
  assert.equal(REPLICATION.combinedPairs, 20);
  assert.equal(REPLICATION.combinedWins, 20);
  assert.equal(REPLICATION.checkedExecutions, 48);
  assert.equal(REPLICATION.checkedGreedy, 1536);
  assert.equal(REPLICATION.checkedSparseLogits, 13774);
  assert.equal(REPLICATION.medianGainDifferencePoints, 0.9021);
});
test('positive local review is accepted as structurally consistent, not authenticated', () => {
  const input = fixture();
  assert.equal(validateReplicationReview(input), input);
});
test('negative replication accepted only when marked as such', () => {
  const input = fixture();
  input.status = 'REPLICATION_MATCHED_BUT_PERFORMANCE_NOT_QUALIFIED';
  input.replication.review_thresholds_met = false;
  input.replication.qualification = 'INCONCLUSIVE_OR_NO_PERFORMANCE_BENEFIT';
  input.replication.candidate_faster_pair_count = 0;
  input.replication.median_paired_relative_gain = -0.1;
  input.combined.batched_faster_pairs = 12;
  input.combined.replication_minus_original_spread_median_gain_percentage_points =
    100 * (-0.1 - REPLICATION.originalGain);
  assert.equal(validateReplicationReview(input), input);
});
test('altered hashes, totals, summary or promotion flags fail closed', () => {
  const changes = [
    r => { r.original_census_sha256 = 'not-a-hash'; },
    r => { r.production_promoted = true; },
    r => { r.source_reference_authenticated = true; },
    r => { r.cross_day_calendar_verified_independently = true; },
    r => { r.combined.batched_faster_pairs = 21; },
    r => { r.combined.sparse_logit_matches = 10; },
    r => { r.replication.pairs = 4; },
    r => { r.replication.median_paired_relative_gain = NaN; },
    r => { r.status = 'PROVEN_SPEEDUP'; },
    r => { r.original_spread_24.median_paired_relative_gain = 0.6; },
    r => { r.combined.replication_minus_original_spread_median_gain_percentage_points = 15; },
  ];
  for (const change of changes) {
    const input = fixture();
    change(input);
    assert.throws(() => validateReplicationReview(input));
  }
});
