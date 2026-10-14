// Research-dashboard presentation data and local, display-only receipt validators.
// Numeric snapshots below are user-reported, NOT published authenticated data.
export const CASES = Object.freeze([
  Object.freeze({ id: 'mixed_7', tokens: 7, gain: 0.155304, ttft: 0.825707, bytes: 1614173184, probes: 285, wins: 4 }),
  Object.freeze({ id: 'repeat_12', tokens: 12, gain: 0.247696, ttft: 0.886926, bytes: 2959317504, probes: 286, wins: 4 }),
  Object.freeze({ id: 'spread_24', tokens: 24, gain: 0.39684, ttft: 0.925016, bytes: 6187663872, probes: 288, wins: 4 }),
]);

export const LEDGER = Object.freeze({
  pairs: 12,
  measured: 24,
  warmups: 6,
  checked: 30,
  greedy: 960,
  probes: 8590,
  wins: 12,
  defaultDecoder: 'SINGLE',
  nextSeed: 70421,
  nextPairs: 8,
  nextCase: 'spread_24',
});

export const MILESTONES = Object.freeze([
  Object.freeze({ pr: 21, name: 'Frozen real-weight matrix', note: 'Four reference cases bound to one local checkpoint and declared source.' }),
  Object.freeze({ pr: 22, name: 'Weight-staging audit', note: 'Repeated per-token staging quantified; no acceleration claimed.' }),
  Object.freeze({ pr: 23, name: 'Batched causal prefill', note: 'Isolated experimental implementation, original SINGLE decoder untouched.' }),
  Object.freeze({ pr: 24, name: 'Real-model parity gate', note: 'Three opt-in BATCHED cases matched reference tokens and sparse logit probes.' }),
  Object.freeze({ pr: 25, name: 'Balanced A/B trials', note: 'Four paired rounds per workload, each step checked against original references.' }),
  Object.freeze({ pr: 26, name: 'Offline evidence census', note: 'Local SHA-bound reviewer recalculated and checked the entire saved evidence series.' }),
  Object.freeze({ pr: 27, name: 'Public research dashboard', note: 'Published bounded real-model observations, local-only receipt readers and provenance limits.' }),
  Object.freeze({ pr: 28, name: 'Cross-day replication review', note: 'Operator reports successful SHA-bound offline audit of the eight-pair seed-70421 replication; general speedup remains unproven.' }),
]);

const isObject = v => v !== null && typeof v === 'object' && !Array.isArray(v);
const finite = v => typeof v === 'number' && Number.isFinite(v);
const equal = (v, expected) => v === expected;
const nearly = (a, b, tol = 0.000003) => Math.abs(a - b) <= tol;

export function validateCensusReceipt(c) {
  if (!isObject(c) ||
    c.schema !== 'PHISTREAM_THREE_CASE_PAIRED_AB_CENSUS_V1' ||
    c.status !== 'THREE_CASES_REVIEW_CANDIDATE_NOT_SPEEDUP_CERTIFIED' ||
    c.model_execution !== false || c.production_promoted !== false ||
    c.general_speedup_proven !== false || c.source_reference_authenticated !== false ||
    c.statistical_significance_established !== false ||
    !Array.isArray(c.cases) || c.cases.length !== 3 ||
    c.case_count !== 3 || c.total_pairs !== 12 ||
    c.total_measured_executions !== 24 || c.total_warmup_executions !== 6 ||
    c.total_checked_model_executions_in_original_receipts !== 30 ||
    c.total_matching_greedy_ids_in_original_receipts !== 960 ||
    c.total_matching_sparse_logits_in_original_receipts !== 8590 ||
    c.total_candidate_faster_pairs !== 12) {
    throw Error('Not a complete three-case PR26 census for the published seed-369 series.');
  }
  const ids = new Set(c.cases.map(x => x?.case));
  if (ids.size !== 3 || CASES.some(x => !ids.has(x.id))) {
    throw Error('Unexpected or repeated workload in local census.');
  }
  for (const known of CASES) {
    const item = c.cases.find(x => x.case === known.id);
    if (!isObject(item) || item.pairs !== 4 ||
      item.measured_executions !== 8 || item.warmup_executions !== 2 ||
      item.all_executions_with_complete_parity !== 10 ||
      item.greedy_id_matches !== 320 ||
      item.sampled_logit_matches !== known.probes * 10 ||
      item.candidate_faster_pair_count !== 4 ||
      item.review_thresholds_met !== true ||
      item.qualification !== 'CANDIDATE_FOR_REPLICATED_REVIEW_NOT_PROVEN_SPEEDUP' ||
      !finite(item.median_paired_relative_gain) ||
      !finite(item.median_relative_ttft_gain) ||
      !finite(item.median_cumulative_weight_bytes_reduction) ||
      !nearly(item.median_paired_relative_gain, known.gain) ||
      !nearly(item.median_relative_ttft_gain, known.ttft) ||
      !nearly(item.median_cumulative_weight_bytes_reduction, known.bytes, 0.5)) {
      throw Error('Census differs from the archived three-case presentation snapshot.');
    }
  }
  return c;
}

function validatedTrial(t, method) {
  return isObject(t) &&
    t.method === method &&
    t.status === 'PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED' &&
    t.greedy_matched === 32 && t.greedy_total === 32 &&
    Number.isInteger(t.logit_total) && t.logit_total > 0 &&
    t.logit_matched === t.logit_total &&
    finite(t.decoder_seconds) && t.decoder_seconds > 0 &&
    Array.isArray(t.comparisons) && t.comparisons.length === 32 &&
    t.comparisons.every((step, i) => step?.step === i &&
      step.token_match === true && step.probe_match === true);
}

export function validatePairedRunReceipt(r) {
  if (!isObject(r) || r.schema !== 'PHISTREAM_PAIRED_REAL_PREFILL_AB_V1' ||
    !['MATCHED_CANDIDATE_FOR_FURTHER_REVIEW',
      'MATCHED_BUT_PERFORMANCE_NOT_QUALIFIED'].includes(r.status) ||
    r.model_execution !== true || r.run_device !== 'cuda' ||
    r.production_promoted !== false || r.source_reference_authenticated !== false ||
    r.speedup_proven !== false ||
    !CASES.some(c => c.id === r.case) ||
    !isObject(r.design) || !Array.isArray(r.design.orders) ||
    !Array.isArray(r.paired_observations) ||
    r.design.pairs !== r.paired_observations.length ||
    r.design.pairs < 4 || r.design.pairs > 20 ||
    r.design.pairs % 2 !== 0 ||
    !Array.isArray(r.warmup_observations) ||
    r.warmup_observations.length !== r.design.warmups_per_method * 2 ||
    !isObject(r.summary) || r.summary.pairs !== r.design.pairs) {
    throw Error('Not a complete, reviewed PR25 paired CUDA run receipt.');
  }
  const forward = r.design.orders.filter(o => o?.[0] === 'SINGLE' && o?.[1] === 'BATCHED').length;
  const reverse = r.design.orders.filter(o => o?.[0] === 'BATCHED' && o?.[1] === 'SINGLE').length;
  if (forward !== r.design.pairs / 2 || reverse !== r.design.pairs / 2) {
    throw Error('Run pair order is not balanced.');
  }
  if (r.warmup_observations.some((t, i) =>
    t?.warmup !== Math.floor(i / 2) + 1 ||
    !validatedTrial(t, i % 2 === 0 ? 'SINGLE' : 'BATCHED'))) {
    throw Error('One or more warmups failed full reference-parity checks.');
  }
  const gains = [];
  let wins = 0;
  for (let i = 0; i < r.paired_observations.length; i++) {
    const pair = r.paired_observations[i];
    const methods = pair?.methods;
    if (pair?.pair !== i + 1 || pair?.complete !== true ||
      !Array.isArray(pair.order) ||
      pair.order.join(',') !== r.design.orders[i].join(',') ||
      !validatedTrial(methods?.SINGLE, 'SINGLE') ||
      !validatedTrial(methods?.BATCHED, 'BATCHED')) {
      throw Error('One or more paired observations failed their reference or ordering checks.');
    }
    const a = methods.SINGLE.decoder_seconds;
    const b = methods.BATCHED.decoder_seconds;
    gains.push((a - b) / a);
    if (b < a) wins++;
  }
  const sorted = [...gains].sort((a, b) => a - b);
  const mid = sorted.length / 2;
  const median = (sorted[mid - 1] + sorted[mid]) / 2;
  const qualified = median >= 0.05 && wins / gains.length >= 0.75;
  if (!finite(r.summary.median_paired_relative_gain) ||
    !nearly(r.summary.median_paired_relative_gain, median) ||
    r.summary.candidate_faster_pair_count !== wins ||
    !nearly(r.summary.candidate_faster_fraction, wins / gains.length) ||
    r.summary.thresholds_met_for_human_review !== qualified ||
    r.status !== (qualified ? 'MATCHED_CANDIDATE_FOR_FURTHER_REVIEW'
      : 'MATCHED_BUT_PERFORMANCE_NOT_QUALIFIED')) {
    throw Error('Recorded summary contradicts the individual paired trial timings.');
  }
  return { receipt: r, medianGain: median, wins, gains };
}
