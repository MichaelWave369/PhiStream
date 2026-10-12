import React, { useState } from 'react';
import './transfer-reader.css';

const validStatuses = new Set(['EXACT_STAGE_COUNTER_LINEARITY_OBSERVED','COUNTER_PATTERN_NOT_FULLY_LINEAR']);
function validate(report) {
  if (!report || report.schema !== 'PHISTREAM_WEIGHT_STAGE_AUDIT_V1' ||
      !validStatuses.has(report.status) || report.four_cases !== 4 ||
      !Array.isArray(report.cases) || report.cases.length !== 4 ||
      report.model_execution !== false || report.scheduler_policy_changed !== false ||
      report.real_model_optimization_speedup_measured !== false ||
      report.production_ready !== false || report.source_reference_authenticated !== false ||
      !Number.isFinite(report.weight_payload_bytes) ||
      new Set(report.cases.map(c=>c.id)).size !== 4 ||
      report.cases.some(c=>!['short_1','mixed_7','repeat_12','spread_24'].includes(c.id)
                           || !Number.isFinite(c.recorded_staging_fraction)
                           || !Number.isFinite(c.recorded_staging_bytes)
                           || !Number.isFinite(c.recorded_decoder_seconds))) {
    throw Error('Not a recognized four-case PhiStream weight-stage audit.');
  }
  return report;
}

export default function WeightStageReader(){
  const [report,setReport] = useState(null);
  const [filename,setFilename] = useState('');
  const [error,setError] = useState('');
  const change=async event=>{
    setReport(null);setFilename('');setError('');
    const file=event.target.files?.[0];
    if(!file)return;
    try {
      if(file.size>256*1024)throw Error('Audit JSON exceeds 256 KiB.');
      setReport(validate(JSON.parse(await file.text())));
      setFilename(file.name);
    }catch(e){setError(e?.message||'Cannot inspect local audit JSON.');}
  };
  const f=n=>Number.isFinite(n)?n.toFixed(2):'—';
  return <section className="transfer-lab" aria-label="Real model weight staging counter audit">
    <div className="transfer-intro">
      <span className="reader-tag">PERFORMANCE / MEASURE BEFORE OPTIMIZING</span>
      <h3>Our weights go around again. And again.</h3>
      <p>Inspect the offline SHA-bound staging audit for four previously validated real-weight cases. Counter identity is not an achieved speedup, and the weight-load timer includes more than PCIe transfer.</p>
      <a className="reader-docs" href="https://github.com/MichaelWave369/PhiStream/blob/main/docs/WEIGHT_STAGE_AUDIT.md" target="_blank" rel="noreferrer">Audit method and falsifiable prefill hypothesis ↗</a>
    </div>
    <div className="transfer-panel">
      <div className="transfer-panel-top"><span>FOUR-CASE STAGING AUDIT</span><span>LOCAL JSON / NO INFERENCE</span></div>
      <label htmlFor="weight-stage-audit">Choose weight-stage-audit.json</label>
      <input id="weight-stage-audit" type="file" accept=".json,application/json" onChange={change}/>
      {error&&<p role="alert" className="transfer-error">{error}</p>}
      {!report&&!error&&<p className="transfer-empty">No staging audit loaded</p>}
      {report&&<>
        <div className="transfer-result-head"><strong>{report.status}</strong><span>USER-LOCAL / NOT AUTHENTICATED</span></div>
        <p className="transfer-file-name">{filename}</p>
        <div className="transfer-stats">
          <div><small>Tensor bytes per decoder step</small><strong>{report.weight_payload_bytes.toLocaleString('en-US')}</strong></div>
          <div><small>One-off bytes (all cases)</small><strong>{report.per_run_fixed_staged_bytes_outside_single_token_steps??'nonlinear'}</strong></div>
          <div><small>Weight loads per decoder step</small><strong>{report.weight_load_operations_per_decoder_step??'nonlinear'}</strong></div>
          <div><small>Median stage timer share</small><strong>{f(report.median_staging_time_fraction_observed*100)}%</strong></div>
        </div>
        <ul className="matrix-results-list">
          {report.cases.map(c=><li key={c.id}>
            <strong>{c.id}</strong> · {c.single_token_decoder_steps} decoder steps · {(c.recorded_staging_bytes/1e9).toFixed(3)} GB staged · {f(c.recorded_staging_seconds)}/{f(c.recorded_decoder_seconds)} s load/total
          </li>)}
        </ul>
        <p className="transfer-caveat">Offline reviewer rechecks hash-bound receipts. This browser only displays supplied JSON, with no independent authentication. No inference, activation, batch-prefill speedup, bandwidth or production-readiness claim is made here.</p>
      </>}
    </div>
  </section>;
}
