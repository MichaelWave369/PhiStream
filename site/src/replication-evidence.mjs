// PR28 presentation-only replication snapshot and receipt-check helper.
// The snapshot is operator-reported; NEVER treat browser checks as SHA attestation.
export const REPLICATION = Object.freeze({
  case: "spread_24", seed: 70421, pairs: 8, wins: 8,
  gain: 0.405861, ttft: 0.924930, stagedBytesSaved: 6187663872,
  originalSeed: 369, originalPairs: 4, originalWins: 4, originalGain: 0.396840,
  originalTtft: 0.925016, originalProbes: 288,
  status: "OPERATOR_REPORTED_OFFLINE_AUDIT_PENDING",
});

const obj = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const finite = x => typeof x === 'number' && Number.isFinite(x);
const digest = x => typeof x === 'string' && /^[0-9a-f]{64}$/.test(x);
const near = (x, y, tol = 0.000003) => finite(x) && Math.abs(x - y) <= tol;

// The offline Python audit remains the only SHA-bound contract reviewer.
// This checks local JSON's shape and internal arithmetic to avoid fake UI statuses.
export function validateReplicationReview(r) {
  if (!obj(r) || r.schema !== "PHISTREAM_REPLICATION_REVIEW_V1" ||
    !["REPLICATION_REVIEW_CANDIDATE_NOT_SPEEDUP_CERTIFIED",
      "REPLICATION_MATCHED_BUT_PERFORMANCE_NOT_QUALIFIED"].includes(r.status) ||
    r.model_execution !== false || r.source_reference_authenticated !== false ||
    r.statistical_significance_established !== false ||
    r.cross_day_calendar_verified_independently !== false ||
    r.production_promoted !== false || r.production_scheduler_unchanged !== true ||
    r.general_speedup_proven !== false || r.original_seed !== 369 ||
    !digest(r.original_census_sha256) || !digest(r.replication_plan_sha256) ||
    !digest(r.replication_run_sha256) || !digest(r.reference_sha256) ||
    !obj(r.original_spread_24) || !obj(r.replication) || !obj(r.combined)) {
    throw Error('Not a complete PR28 local research-only replication receipt.');
  }
  const a = r.original_spread_24;
  const b = r.replication;
  const c = r.combined;
  if (a.case !== 'spread_24' || a.pairs !== 4 ||
    a.candidate_faster_pair_count !== 4 ||
    a.review_thresholds_met !== true ||
    !near(a.median_paired_relative_gain, REPLICATION.originalGain) ||
    b.case !== REPLICATION.case || b.seed !== REPLICATION.seed ||
    b.pairs !== 8 || b.measured_executions !== 16 ||
    b.warmups !== 2 || b.model_executions_recorded !== 18 ||
    b.greedy_id_matches !== 576 || b.sampled_logit_matches !== 5184 ||
    !Number.isInteger(b.candidate_faster_pair_count) ||
    b.candidate_faster_pair_count < 0 || b.candidate_faster_pair_count > 8 ||
    !finite(b.median_paired_relative_gain) ||
    !finite(b.median_relative_ttft_gain) ||
    !finite(b.median_cumulative_weight_bytes_reduction) ||
    typeof b.review_thresholds_met !== 'boolean' ||
    b.qualification !== (b.review_thresholds_met
      ? 'CANDIDATE_FOR_REPLICATED_REVIEW_NOT_PROVEN_SPEEDUP'
      : 'INCONCLUSIVE_OR_NO_PERFORMANCE_BENEFIT') ||
    c.paired_comparisons !== 20 ||
    c.batched_faster_pairs !== 12 + b.candidate_faster_pair_count ||
    c.recorded_executions_including_warmups !== 48 ||
    c.greedy_id_matches !== 1536 || c.sparse_logit_matches !== 13774 ||
    c.cohort_count !== 4 ||
    !near(c.replication_minus_original_spread_median_gain_percentage_points,
      100 * (b.median_paired_relative_gain - a.median_paired_relative_gain), 0.00011) ||
    r.status !== (b.review_thresholds_met
      ? 'REPLICATION_REVIEW_CANDIDATE_NOT_SPEEDUP_CERTIFIED'
      : 'REPLICATION_MATCHED_BUT_PERFORMANCE_NOT_QUALIFIED')) {
    throw Error('Replicated performance or combined ledger is inconsistent.');
  }
  return r;
}
