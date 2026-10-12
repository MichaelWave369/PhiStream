import React, { useState } from 'react';
import './transfer-reader.css';

const SHA=/^[0-9a-f]{64}$/;
function validate(trace) {
  if (!trace || typeof trace !== 'object' || Array.isArray(trace) ||
    trace.schema !== 'PHISTREAM_REFERENCE_TRACE_V1' ||
    trace.reference?.asserted_independent !== true ||
    typeof trace.reference.runtime !== 'string' ||
    typeof trace.reference.version !== 'string' ||
    !['cpu','cuda'].includes(trace.reference.execution_device) ||
    trace.reference.algorithm !== 'FULL_SEQUENCE_GREEDY_NO_CACHE' ||
    trace.checkpoint_fingerprint?.algorithm !== 'sha256' ||
    !SHA.test(trace.checkpoint_fingerprint.files?.['config.json'] || '') ||
    !SHA.test(trace.checkpoint_fingerprint.files?.['model.safetensors'] || '') ||
    !Array.isArray(trace.prompt_token_ids) ||
    !Array.isArray(trace.expected_generated_token_ids) ||
    trace.expected_generated_token_ids.length !== 32 ||
    !Array.isArray(trace.logit_probes) ||
    trace.logit_probes.length !== 32 ||
    trace.experiment?.status !== 'REFERENCE_OBSERVED_NOT_AUTHENTICATED' ||
    trace.experiment?.checkpoint_bytes_rehashed_after_execution !== true ||
    trace.experiment?.phistream_decoder_not_imported !== true) {
    throw Error('Not an expected, 32-step external HF reference trace.');
  }
  return trace;
}

export default function HFReferenceReader() {
  const [trace,setTrace]=useState(null);
  const [name,setName]=useState('');
  const [error,setError]=useState('');
  const change=async event=>{
    setTrace(null);setName('');setError('');
    const file=event.target.files?.[0];
    if(!file)return;
    try {
      if(file.size>512*1024)throw Error('Local reference file exceeds 512 KiB.');
      setTrace(validate(JSON.parse(await file.text())));
      setName(file.name);
    } catch(e){setError(e?.message||'Could not read reference trace.');}
  };
  return <section className="transfer-lab" aria-label="Independent external reference trace">
    <div className="transfer-intro">
      <span className="reader-tag">REAL-WEIGHT REFERENCE / VALIDATION ONLY</span>
      <h3>The second decoder gets a vote.</h3>
      <p>External Hugging Face Transformers generates 32 greedy token IDs and sampled logits from a SHA-bound local checkpoint. Import a trace here for read-only inspection. This is not yet proof of PhiStream parity.</p>
      <a className="reader-docs" href="https://github.com/MichaelWave369/PhiStream/blob/main/docs/EXTERNAL_REFERENCE_LAB.md" target="_blank" rel="noreferrer">Offline reference protocol ↗</a>
    </div>
    <div className="transfer-panel">
      <div className="transfer-panel-top"><span>REFERENCE TRACE INSPECTION</span><span>LOCAL FILE ONLY</span></div>
      <label htmlFor="hf-reference-file">Choose external reference JSON</label>
      <input id="hf-reference-file" type="file" accept=".json,application/json" onChange={change}/>
      {error&&<p role="alert" className="transfer-error">{error}</p>}
      {!trace&&!error&&<p className="transfer-empty">No reference trace loaded</p>}
      {trace&&<>
        <div className="transfer-result-head"><strong>EXTERNAL TRACE PRESENT</strong><span>PARITY NOT CHECKED HERE</span></div>
        <p className="transfer-file-name">{name}</p>
        <div className="transfer-stats">
          <div><small>Greedy IDs</small><strong>{trace.expected_generated_token_ids.length}/32</strong></div>
          <div><small>Logit probe maps</small><strong>{trace.logit_probes.length}/32</strong></div>
          <div><small>External execution device</small><strong>{trace.reference.execution_device}</strong></div>
          <div><small>Reference model load</small><strong>USER ASSERTED</strong></div>
        </div>
        <p className="transfer-pinned">Runtime: <strong>{trace.reference.runtime} {trace.reference.version}</strong></p>
        <p className="transfer-caveat">Local JSON is editable and not signed. Checkpoint hashes identify bytes, not source authenticity. Importing this file does not run PhiStream or verify token/logit equivalence. Run the separately approved real-model-gate comparator for that evidence. No uploads.</p>
      </>}
    </div>
  </section>;
}
