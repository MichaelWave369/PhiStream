import { test } from 'node:test';
import assert from 'node:assert/strict';
import { CASES, LEDGER, MILESTONES, validateCensusReceipt, validatePairedRunReceipt } from './paired-evidence.mjs';

const copy = value => structuredClone(value);
const makeCensus = () => ({
  schema: 'PHISTREAM_THREE_CASE_PAIRED_AB_CENSUS_V1',
  status: 'THREE_CASES_REVIEW_CANDIDATE_NOT_SPEEDUP_CERTIFIED',
  model_execution: false, production_promoted: false,
  general_speedup_proven: false, source_reference_authenticated: false,
  statistical_significance_established: false,
  case_count: 3, total_pairs: 12, total_measured_executions: 24,
  total_warmup_executions: 6, total_checked_model_executions_in_original_receipts: 30,
  total_matching_greedy_ids_in_original_receipts: 960,
  total_matching_sparse_logits_in_original_receipts: 8590,
  total_candidate_faster_pairs: 12,
  cases: CASES.map(c => ({
    case: c.id, pairs: 4, measured_executions: 8, warmup_executions: 2,
    all_executions_with_complete_parity: 10, greedy_id_matches: 320,
    sampled_logit_matches: c.probes * 10, candidate_faster_pair_count: 4,
    review_thresholds_met: true, qualification: 'CANDIDATE_FOR_REPLICATED_REVIEW_NOT_PROVEN_SPEEDUP',
    median_paired_relative_gain: c.gain, median_relative_ttft_gain: c.ttft,
    median_cumulative_weight_bytes_reduction: c.bytes,
  })),
});
const makeTrial = (method, seconds) => ({
  method, status: 'PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED',
  greedy_matched: 32, greedy_total: 32,
  logit_matched: 285, logit_total: 285,
  decoder_seconds: seconds,
  comparisons: Array.from({ length: 32 }, (_, step) => ({
    step, token_match: true, probe_match: true,
  })),
});
const makeRun = () => {
  const orders = [
    ['BATCHED', 'SINGLE'], ['SINGLE', 'BATCHED'],
    ['SINGLE', 'BATCHED'], ['BATCHED', 'SINGLE'],
  ];
  return {
    schema: 'PHISTREAM_PAIRED_REAL_PREFILL_AB_V1',
    status: 'MATCHED_CANDIDATE_FOR_FURTHER_REVIEW',
    case: 'mixed_7', model_execution: true, run_device: 'cuda',
    production_promoted: false, speedup_proven: false,
    source_reference_authenticated: false,
    design: { pairs: 4, warmups_per_method: 1, seed: 369, orders },
    warmup_observations: [
      { warmup: 1, ...makeTrial('SINGLE', 10) },
      { warmup: 1, ...makeTrial('BATCHED', 8.4) },
    ],
    paired_observations: orders.map((order, i) => ({
      pair: i + 1, order, complete: true,
      methods: {
        SINGLE: makeTrial('SINGLE', 10),
        BATCHED: makeTrial('BATCHED', 8.5),
      },
    })),
    summary: {
      pairs: 4, candidate_faster_pair_count: 4, candidate_faster_fraction: 1,
      median_paired_relative_gain: 0.15, thresholds_met_for_human_review: true,
    },
  };
};

test('public snapshot matches frozen user-local ledger totals', () => {
  assert.equal(CASES.length, 3);
  assert.equal(MILESTONES.length, 8);
  assert.equal(LEDGER.greedy, 30 * 32);
  assert.equal(LEDGER.probes, CASES.reduce((n, c) => n + c.probes * 10, 0));
  assert.equal(LEDGER.wins, CASES.reduce((n, c) => n + c.wins, 0));
});

test('valid local PR26 census renders without external authentication', () => {
  const input = makeCensus();
  assert.equal(validateCensusReceipt(input), input);
});
test('census fails closed for tampered totals, duplicate cases, fake verification', () => {
  const mutations = [
    c => { c.total_matching_sparse_logits_in_original_receipts += 1; },
    c => { c.cases[1] = c.cases[0]; },
    c => { c.cases[0].median_paired_relative_gain += 0.01; },
    c => { c.cases[0].greedy_id_matches -= 1; },
    c => { c.production_promoted = true; },
    c => { c.source_reference_authenticated = true; },
    c => { c.status = 'SPEEDUP_PROVEN'; },
  ];
  for (const mutate of mutations) {
    const c = makeCensus(); mutate(c);
    assert.throws(() => validateCensusReceipt(c));
  }
});

test('complete local paired run displays recomputed trial timing, not certified speedup', () => {
  const result = validatePairedRunReceipt(makeRun());
  assert.equal(result.wins, 4);
  assert.ok(Math.abs(result.medianGain - 0.15) < 1e-8);
  assert.equal(result.gains.length, 4);
});

test('failed parity, forged gain, wrong order and unauthorized promotion are rejected', () => {
  const mutations = [
    r => { r.summary.median_paired_relative_gain = 0.5; },
    r => { r.paired_observations[0].methods.SINGLE.comparisons[3].token_match = false; },
    r => { r.paired_observations[1].order.reverse(); },
    r => { r.warmup_observations.pop(); },
    r => { r.paired_observations[0].methods.BATCHED.greedy_matched = 31; },
    r => { r.production_promoted = true; },
    r => { r.source_reference_authenticated = true; },
    r => { r.paired_observations[0].methods.SINGLE.decoder_seconds = NaN; },
  ];
  for (const mutate of mutations) {
    const r = makeRun(); mutate(r);
    assert.throws(() => validatePairedRunReceipt(r));
  }
});
test('nonqualifying runs are allowed but must be labeled honestly', () => {
  const run = makeRun();
  for (const row of run.paired_observations) {
    row.methods.BATCHED.decoder_seconds = 11;
  }
  run.summary.median_paired_relative_gain = -0.1;
  run.summary.candidate_faster_pair_count = 0;
  run.summary.candidate_faster_fraction = 0;
  run.summary.thresholds_met_for_human_review = false;
  run.status = 'MATCHED_BUT_PERFORMANCE_NOT_QUALIFIED';
  const parsed = validatePairedRunReceipt(run);
  assert.equal(parsed.wins, 0);
  assert.equal(parsed.medianGain, -0.1);
});
