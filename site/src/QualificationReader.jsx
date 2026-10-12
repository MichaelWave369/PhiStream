import React, { useState } from 'react';
import './transfer-reader.css';

const SCHEMA='PHISTREAM_SCHEDULER_QUALIFICATION_V1';
const PROVENANCE='REVIEWED_UNAUTHENTICATED_GUARDED_SYNTHETIC_COHORT';
const percent=n=>Number.isFinite(n)?(n*100).toFixed(2)+'%':'—';

function check(report){
  if(!report||typeof report!=='object'||Array.isArray(report)||
     report.schema!==SCHEMA||report.provenance!==PROVENANCE||
     !['QUALIFIED_SYNTHETIC','INCONCLUSIVE','REJECTED'].includes(report.status)||
     report.default!=='SINGLE'||report.review_only!==true||
     report.unverified_external_inputs!==true||
     report.model_execution!==false||
     report.automatic_layer_placement!==false||
     report.production_policy_applied!==false||
     !Number.isInteger(report.input_receipt_count)||
     !Array.isArray(report.ledger)||
     !Array.isArray(report.rejected_receipts)||
     !Array.isArray(report.qualification_notes)||
     !Array.isArray(report.blockers)||
     report.ledger.length+report.rejected_receipts.length!==report.input_receipt_count||
     report.input_receipt_count>16){
    throw Error('Not a valid read-only PhiStream qualification report.');
  }
  return report;
}

export default function QualificationReader(){
  const [data,setData]=useState(null);
  const [filename,setFilename]=useState('');
  const [error,setError]=useState('');
  const open=async event=>{
    setData(null);setFilename('');setError('');
    const file=event.target.files?.[0];
    if(!file)return;
    try{
      if(file.size>512*1024)throw Error('Report exceeds 512 KiB');
      setData(check(JSON.parse(await file.text())));
      setFilename(file.name);
    }catch(e){setError(e?.message||'Cannot read local qualification report.');}
  };
  return <section className="transfer-lab" aria-label="Scheduler qualification evidence">
    <div className="transfer-intro">
      <span className="reader-tag">SCHEDULER QUALIFICATION / RESEARCH ONLY</span>
      <h3>Include the awkward runs.</h3>
      <p>Review every matched guarded CUDA receipt. A borderline 4.97% run remains visible and prevents synthetic qualification under the strict all-run threshold.</p>
      <a className="reader-docs" href="https://github.com/MichaelWave369/PhiStream/blob/main/docs/QUALIFICATION_LAB.md" target="_blank" rel="noreferrer">Qualification protocol ↗</a>
    </div>
    <div className="transfer-panel">
      <div className="transfer-panel-top"><span>LOCAL QUALIFICATION LEDGER</span><span>UNAUTHENTICATED INPUTS</span></div>
      <label htmlFor="qualification-json">Choose qualify-scheduler JSON report</label>
      <input id="qualification-json" type="file" accept=".json,application/json" onChange={open}/>
      {error&&<p role="alert" className="transfer-error">{error}</p>}
      {!data&&!error&&<p className="transfer-empty">No qualification review loaded</p>}
      {data&&<>
        <div className="transfer-result-head"><strong>{data.status}</strong><span>DEFAULT: SINGLE</span></div>
        <p className="transfer-file-name">{filename}</p>
        <div className="transfer-stats">
          <div><small>Included receipts</small><strong>{data.accepted_count}/{data.input_receipt_count}</strong></div>
          <div><small>Per-run gate passes</small><strong>{data.qualified_run_count}</strong></div>
          <div><small>Randomized-order runs</small><strong>{data.randomized_run_count}</strong></div>
          <div><small>Pooled paired gain</small><strong>{percent(data.aggregate?.median_paired_gain)}</strong></div>
        </div>
        <p className="transfer-pinned">Reasons: <strong>{[...data.blockers,...data.qualification_notes].join('; ')||'All synthetic research checks passed'}</strong></p>
        <p className="transfer-pinned">Receipt census (SHA-256 and local filenames are in the JSON):</p>
        <div style={{overflowX:'auto'}}>
          <table style={{fontSize:12,width:'100%',borderCollapse:'collapse'}}>
            <thead><tr><th scope="col">File</th><th scope="col">Control gain</th><th scope="col">Pairs won</th><th scope="col">Order</th><th scope="col">Gate</th></tr></thead>
            <tbody>{data.ledger.map((item,i)=><tr key={item.sha256||i}>
              <td style={{maxWidth:180,overflowWrap:'anywhere'}}>{item.filename}</td>
              <td>{percent(item.gain_from_method_medians)}</td>
              <td>{item.pair_wins}/6</td>
              <td>{item.randomized_order?'Shuffled':'Legacy'}</td>
              <td>{item.passes_individual_gate?'PASS':'HELD'}</td>
            </tr>)}</tbody>
          </table>
        </div>
        {!!data.rejected_receipts.length&&<p className="transfer-error">Rejected input: {data.rejected_receipts.map(r=>r.filename+': '+r.reason).join('; ')}</p>}
        <p className="transfer-caveat">Synthetic benchmark classification only. External JSON can be edited; SHA-256 detects duplicate content, not authenticated hardware. This does not run a model, change GPU scheduling or establish physical SSD/GPU overlap.</p>
      </>}
    </div>
  </section>;
}
