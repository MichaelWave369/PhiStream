// PR30: browser-local structural interpretation of a held context plan.
// Never trusts, uploads or claims to authenticate user-selected files.
export const LONG_CONTEXT_TARGETS = Object.freeze([
  Object.freeze({ prompt: 128, required: 159 }),
  Object.freeze({ prompt: 512, required: 543 }),
  Object.freeze({ prompt: 2048, required: 2079 }),
]);

const isObject = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const positiveInteger = value => Number.isSafeInteger(value) && value > 0;

export function validateLongContextPlan(p) {
  if (!isObject(p) || p.schema !== 'PHISTREAM_LONG_CONTEXT_PROTOCOL_V1' ||
    p.status !== 'ALL_LONG_CONTEXT_TARGETS_HELD_PROTOCOL_ONLY' ||
    p.model_execution !== false || p.cuda_queried !== false ||
    p.reference_generated !== false || p.checkpoint_mutated !== false ||
    p.scheduler_policy_changed !== false || p.production_promoted !== false ||
    p.speedup_proven !== false || p.source_reference_authenticated !== false ||
    p.checkpoint_binding_scope !== 'METADATA_LAYOUT_ONLY_NOT_SHA_BOUND_TO_WEIGHT_BYTES' ||
    p.existing_batched_prefill_max_prompt_tokens !== 24 ||
    p.continuation_steps !== 32 ||
    !isObject(p.model) || !isObject(p.policy) ||
    !positiveInteger(p.model.attention_heads) ||
    !positiveInteger(p.model.max_context_tokens) ||
    p.policy.observed_cuda_free_bytes !== null ||
    p.policy.no_hardware_fit_guarantee !== true ||
    !Array.isArray(p.targets) || p.targets.length !== 3 ||
    !Array.isArray(p.separate_future_gates)) {
    throw Error('Not a non-executable PhiStream long-context protocol.');
  }
  for (const [i, target] of LONG_CONTEXT_TARGETS.entries()) {
    const row = p.targets[i];
    if (!isObject(row) || row.id !== 'prompt_' + target.prompt ||
      row.prompt_tokens !== target.prompt ||
      row.teacher_forced_continuation_steps !== 32 ||
      row.full_context_tokens_required !== target.required ||
      row.status !== 'HELD_PROTOCOL_ONLY_NO_EXECUTION' ||
      row.execution_approved !== false ||
      row.existing_batched_kernel_supports_prompt !== false ||
      row.candidate_implementation_ready !== false ||
      row.reference_trace_ready !== false ||
      row.benchmark_measured !== false ||
      row.full_context_fits_checkpoint_limit !==
        (target.required <= p.model.max_context_tokens) ||
      !Array.isArray(row.hold_reasons) ||
      !row.hold_reasons.includes('NO_LONG_CONTEXT_BATCHED_IMPLEMENTATION') ||
      !row.hold_reasons.includes('NO_LONG_CONTEXT_EXECUTION_APPROVAL') ||
      row.theoretical_one_fp32_attention_score_matrix_bytes !==
        p.model.attention_heads * target.prompt * target.prompt * 4) {
      throw Error('Long-context case contradicts blocked protocol or context arithmetic.');
    }
    if (!row.full_context_fits_checkpoint_limit) {
      if (row.memory_preflight !== null ||
        !row.hold_reasons.includes('FULL_CONTEXT_EXCEEDS_CHECKPOINT_CONTEXT_LIMIT')) {
        throw Error('Over-context case must be held without a memory-fit estimate.');
      }
    } else if (!isObject(row.memory_preflight) ||
      !['BLOCKED_BY_POLICY', 'REVIEW_REQUIRED'].includes(row.memory_preflight.status) ||
      row.memory_preflight.no_fit_guarantee !== true ||
      row.memory_preflight.context_tokens !== target.required) {
      throw Error('Missing non-guaranteeing policy estimate for within-context case.');
    }
  }
  return p;
}
