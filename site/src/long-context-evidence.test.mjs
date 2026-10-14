import { test } from 'node:test';
import assert from 'node:assert/strict';
import { LONG_CONTEXT_TARGETS, validateLongContextPlan } from './long-context-evidence.mjs';

function fixture(maxContext = 2048) {
  return {
    schema: 'PHISTREAM_LONG_CONTEXT_PROTOCOL_V1',
    status: 'ALL_LONG_CONTEXT_TARGETS_HELD_PROTOCOL_ONLY',
    model_execution: false, cuda_queried: false, reference_generated: false,
    checkpoint_mutated: false, scheduler_policy_changed: false,
    production_promoted: false, speedup_proven: false,
    source_reference_authenticated: false,
    checkpoint_binding_scope: 'METADATA_LAYOUT_ONLY_NOT_SHA_BOUND_TO_WEIGHT_BYTES',
    existing_batched_prefill_max_prompt_tokens: 24,
    continuation_steps: 32,
    model: {attention_heads: 4, max_context_tokens: maxContext},
    policy: {observed_cuda_free_bytes: null, no_hardware_fit_guarantee: true},
    targets: LONG_CONTEXT_TARGETS.map(t => {
      const fits = t.required <= maxContext;
      return {
        id: 'prompt_' + t.prompt,
        prompt_tokens: t.prompt,
        teacher_forced_continuation_steps: 32,
        full_context_tokens_required: t.required,
        status: 'HELD_PROTOCOL_ONLY_NO_EXECUTION',
        execution_approved: false,
        candidate_implementation_ready: false,
        existing_batched_kernel_supports_prompt: false,
        reference_trace_ready: false, benchmark_measured: false,
        full_context_fits_checkpoint_limit: fits,
        hold_reasons: ['NO_LONG_CONTEXT_BATCHED_IMPLEMENTATION', 'NO_LONG_CONTEXT_EXECUTION_APPROVAL',
          ...(!fits ? ['FULL_CONTEXT_EXCEEDS_CHECKPOINT_CONTEXT_LIMIT'] : [])],
        theoretical_one_fp32_attention_score_matrix_bytes: 4 * t.prompt * t.prompt * 4,
        memory_preflight: fits ? {
          status: 'REVIEW_REQUIRED', no_fit_guarantee: true, context_tokens: t.required
        } : null,
      };
    }),
    separate_future_gates: ['SEPARATE_APPROVAL'],
  };
}

test('128/512/2048 cases start held; 2048+31 exceeds 2048 context', () => {
  assert.deepEqual(LONG_CONTEXT_TARGETS.map(x => x.required), [159, 543, 2079]);
  const value = fixture();
  assert.equal(validateLongContextPlan(value), value);
  assert.equal(value.targets[2].full_context_fits_checkpoint_limit, false);
});
test('even a 4096 context checkpoint does NOT authorize batched inference', () => {
  const value = fixture(4096);
  assert.equal(validateLongContextPlan(value), value);
  assert.ok(value.targets.every(x => x.execution_approved === false));
});
test('reject forged approval, model execution, incorrect context, and manufactured benchmark evidence', () => {
  const mutators = [
    r => { r.model_execution = true; },
    r => { r.cuda_queried = true; },
    r => { r.production_promoted = true; },
    r => { r.source_reference_authenticated = true; },
    r => { r.existing_batched_prefill_max_prompt_tokens = 2048; },
    r => { r.targets[0].execution_approved = true; },
    r => { r.targets[1].benchmark_measured = true; },
    r => { r.targets[2].memory_preflight = {status:'REVIEW_REQUIRED'}; },
    r => { r.targets[2].full_context_tokens_required = 2048; },
    r => { r.targets[0].theoretical_one_fp32_attention_score_matrix_bytes = 1; },
    r => { r.targets[0].hold_reasons = []; },
    r => { r.policy.observed_cuda_free_bytes = 123; },
  ];
  for (const mutator of mutators) {
    const value = fixture();
    mutator(value);
    assert.throws(() => validateLongContextPlan(value));
  }
});
