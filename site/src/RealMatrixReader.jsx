import React, { useState } from 'react';
import './transfer-reader.css';

const expected = new Set(['short_1','mixed_7','repeat_12','spread_24']);
const statuses = new Set([
  'MATCHED_LIMITED_FOUR_CASES',
  'MATCHED_LIMITED_LOW_OUTPUT_DIVERSITY',
  'INCOMPLETE_EVIDENCE','FAIL_TRUST_CONTRACT','FAIL_PARITY'
]);
function validate(payload) {
  if (!payload || typeof payload !== 'object' ||
      payload.schema !== 'PHISTREAM_REAL_WEIGHT_MATRIX_REVIEW_V1' ||
      !statuses.has(payload.status) ||
      payload.required_cases !== 4 ||
      !Array.isArray(payload.case_results) ||
      !Array.isArray(payload.missing_or_invalid) ||
      payload.model_execution !== false ||
      payload.real_model_double_buffer_promoted !== false ||
      payload.production_readiness !== 'NOT_ESTABLISHED' ||
      payload.model_speedup !== 'NOT_MEASURED' ||
      payload.all_reference_origins_authenticated !== false ||
      typeof payload.passing_cases !== 'number' ||
      payload.passing_cases > 4 || payload.passing_cases < 0 ||
      payload.case_results.some(c => !expected.has(c.id)) ||
      new Set(payload.case_results.map(c=>c.id)).size !== payload.case_results.length) {
    throw Error('Not an expected four-case PhiStream matrix summary.');
  }
  return payload;
}

export default function RealMatrixReader() {
  const [matrix,setMatrix]=useState(null);
  const [filename,setFilename]=useState('');
  const [error,setError]=useState('');
  const onFile=async event=>{
    setMatrix(null);setFilename('');setError('');
    const file=event.target.files?.[0];
    if (!file) return;
    try {
      if (file.size > 256*1024) throw Error('Matrix JSON too large.');
      setMatrix(validate(JSON.parse(await file.text())));
      setFilename(file.name);
    } catch(e) { setError(e?.message || 'Cannot read local matrix JSON'); }
  };
  const fmt=n=>Number.isFinite(n)?n.toFixed(2):'—';
  return <section className="transfer-lab" aria-label="Real weight four-case validation matrix">
    <div className="transfer-intro">
      <span className="reader-tag">REAL WEIGHTS / FOUR-CASE VALIDATION</span>
      <h3>Four prompts. Four receipts. No honorary green checks.</h3>
      <p>Review the offline result of four independent 32-step external traces and four separately approved PhiStream parity runs. The Python reviewer binds each reference file by SHA256 before reporting any match.</p>
      <a className="reader-docs" href="https://github.com/MichaelWave369/PhiStream/blob/main/docs/REAL_WEIGHT_MATRIX.md" target="_blank" rel="noreferrer">Operator protocol and limits ↗</a>
    </div>
    <div className="transfer-panel">
      <div className="transfer-panel-top"><span>FOUR-CASE RECEIPT SUMMARY</span><span>LOCAL FILE ONLY</span></div>
      <label htmlFor="real-weight-matrix-file">Choose matrix-review.json</label>
      <input id="real-weight-matrix-file" type="file" accept=".json,application/json" onChange={onFile}/>
      {error && <p className="transfer-error" role="alert">{error}</p>}
      {!matrix && !error && <p className="transfer-empty">No real-weight matrix summary loaded</p>}
      {matrix && <>
        <div className="transfer-result-head">
          <strong>{matrix.status}</strong>
          <span>USER-LOCAL JSON / NOT AUTHENTICATED</span>
        </div>
        <p className="transfer-file-name">{filename}</p>
        <div className="transfer-stats">
          <div><small>Matching cases</small><strong>{matrix.passing_cases}/4</strong></div>
          <div><small>Unique predicted IDs</small><strong>{matrix.unique_reference_greedy_ids_across_cases ?? '—'}</strong></div>
          <div><small>Distinct five-token prefixes</small><strong>{matrix.distinct_first_five_greedy_sequences ?? '—'}</strong></div>
          <div><small>Median decoder seconds</small><strong>{fmt(matrix.median_decoder_seconds_descriptive_only)}</strong></div>
        </div>
        <ul className="matrix-results-list">
          {matrix.case_results.map(c=><li key={c.id}><strong>{c.id}</strong> · {c.status} · {c.tokens_matched}/32 greedy, {c.logit_probes_matched}/{c.logit_probes_total} sparse logits</li>)}
          {matrix.missing_or_invalid.map(c=><li key={'missing-'+c.id}><strong>{c.id}</strong> · {c.reason}</li>)}
        </ul>
        <p className="transfer-caveat">This UI reads but does not authenticate or recompute the JSON evidence. Run the Python offline reviewer for SHA-bound checks. No model execution, internet upload, speedup proof or production promotion occurs here.</p>
      </>}
    </div>
  </section>;
}
