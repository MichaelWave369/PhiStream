import React, { useState } from 'react';
import './transfer-reader.css';

const SCHEMA = 'PHISTREAM_REAL_MODEL_BRIDGE_V1';
const ORIGIN = 'LOCAL_USER_DECLARED_CHECKPOINT_AND_REFERENCE_UNAUTHENTICATED';
const fmtMs = value => Number.isFinite(value) && value >= 0 ?
  (value*1000).toFixed(2) + ' ms' : '—';
const fmtRate = value => Number.isFinite(value) && value >= 0 ?
  value.toFixed(2) + ' token/s' : '—';
const number = n => Number.isFinite(n) ? Math.round(n).toLocaleString('en-US') : '—';

function validate(data) {
  if (!data || typeof data !== 'object' || Array.isArray(data) ||
      data.schema !== SCHEMA || data.provenance !== ORIGIN ||
      !['PREFLIGHT_REVIEW_REQUIRED', 'PREFLIGHT_TRACE_REVIEW_REQUIRED',
        'HELD_MEMORY_POLICY', 'PARITY_MATCH_USER_REFERENCE_UNAUTHENTICATED',
        'PARITY_MISMATCH','HELD_REFERENCE_LENGTH'].includes(data.status) ||
      data.scheduler_policy_applied !== false ||
      data.double_buffer_enabled !== false ||
      data.checkpoint?.path_withheld !== true ||
      data.user_asserted_source?.verified_by_phistream !== false ||
      data.reference_trace?.independence_authenticated !== false ||
      data.memory_preflight?.no_fit_guarantee !== true ||
      !['BLOCKED_BY_POLICY', 'REVIEW_REQUIRED'].includes(data.memory_preflight?.status)) {
    throw Error('Not a Phase-1 local real-checkpoint gate report.');
  }
  if (data.model_execution === true) {
    if (!data.real_model_run_authorized ||
        data.execution?.independent_reference_authenticated !== false ||
        !Number.isInteger(data.execution?.steps) ||
        data.execution.steps < 32 ||
        !data.execution.checkpoint_fingerprint?.files) {
      throw Error('Invalid model-execution or reference evidence.');
    }
  } else if (data.model_execution !== false || data.execution !== null) {
    throw Error('Invalid non-execution status.');
  }
  return data;
}

export default function RealBridgeReader() {
  const [data, setData] = useState(null);
  const [filename, setFilename] = useState('');
  const [error, setError] = useState('');
  const load = async event => {
    setData(null); setFilename(''); setError('');
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      if (file.size > 512*1024) throw Error('Receipt exceeds 512 KiB.');
      setData(validate(JSON.parse(await file.text())));
      setFilename(file.name);
    } catch(e) { setError(e?.message || 'Unable to read local real-model receipt.'); }
  };
  const execution = data?.execution;
  const perf = execution?.performance;
  return <section className="transfer-lab" aria-label="Local real-model bridge gate">
    <div className="transfer-intro">
      <span className="reader-tag">REAL-CHECKPOINT BRIDGE / PHASE 1</span>
      <h3>Real weights. Strict boundaries.</h3>
      <p>Review explicitly selected local canonical Llama safetensors, memory-policy estimates and complete reference-bound SHA fingerprints. Actual model execution requires a separate operator approval and at least 32 reference greedy tokens.</p>
      <a className="reader-docs" href="https://github.com/MichaelWave369/PhiStream/blob/main/docs/REAL_MODEL_BRIDGE.md" target="_blank" rel="noreferrer">Read the opt-in protocol ↗</a>
    </div>
    <div className="transfer-panel">
      <div className="transfer-panel-top"><span>LOCAL CHECKPOINT GATE RECEIPT</span><span>SELF-REPORTED · NOT AUTHENTICATED</span></div>
      <label htmlFor="real-bridge-json">Choose real-model-gate JSON file</label>
      <input id="real-bridge-json" type="file" accept=".json,application/json" onChange={load}/>
      {error && <p role="alert" className="transfer-error">{error}</p>}
      {!data && !error && <p className="transfer-empty">No local checkpoint report loaded</p>}
      {data && <>
        <div className="transfer-result-head"><strong>{data.status}</strong><span>{data.model_execution?'REAL WEIGHTS EXECUTED':'PREFLIGHT ONLY'}</span></div>
        <p className="transfer-file-name">{filename} · {data.checkpoint.layers} decoder layers · {data.checkpoint.dtype_set.join(', ')}</p>
        <div className="transfer-stats">
          <div><small>Weight bytes (metadata)</small><strong>{number(data.checkpoint.weight_bytes)}</strong></div>
          <div><small>FP32 working-set estimate</small><strong>{number(data.memory_preflight.estimated_working_set_bytes)}</strong></div>
          <div><small>Reference steps</small><strong>{data.reference_trace.steps ?? 'Not supplied'}</strong></div>
          <div><small>Greedy token-ID matches</small><strong>{execution ? (execution.exact_greedy_tokens_matched + '/' + execution.steps) : 'Not executed'}</strong></div>
          {perf && <>
            <div><small>First logits incl. setup + prompt</small><strong>{fmtMs(perf.time_to_first_logits_including_prompt_seconds)}</strong></div>
            <div><small>Subsequent teacher-forced rate</small><strong>{fmtRate(perf.subsequent_decode_tokens_per_second)}</strong></div>
          </>}
        </div>
        <p className="transfer-pinned">Memory: <strong>{data.memory_preflight.status}</strong>. User-declared source/license verified: <strong>NO</strong>. Double buffering: <strong>OFF</strong>.</p>
        <p className="transfer-caveat">The receipt is user-provided and the reference's independence is not independently authenticated. Memory estimates cannot guarantee a hard VRAM ceiling. Any measured rate is not a speed comparison with Ollama or other runtimes. No model files upload to this page.</p>
      </>}
    </div>
  </section>;
}
