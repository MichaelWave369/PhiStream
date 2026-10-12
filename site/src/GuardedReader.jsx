import React, { useState } from 'react';
import './transfer-reader.css';

const REVIEW='PHISTREAM_GUARDED_SCHEDULER_V1';
const RUN='PHISTREAM_GUARDED_SYNTHETIC_RUN_V1';
const fmt=n=>Number.isFinite(n)&&n>=0?(n*1000).toFixed(3)+' ms':'—';

function verify(data){
  if(!data||typeof data!=='object'||Array.isArray(data)||
     data.model_execution!==false||data.production_policy_applied!==false||
     data.automatic_layer_placement!==false)throw Error('Not a PhiStream guarded local receipt.');
  if(data.schema===REVIEW){
    if(data.provenance!=='LOCAL_UNAUTHENTICATED_SYNTHETIC_RECEIPT_REVIEW'||
       !['ELIGIBLE_FOR_SYNTHETIC_OPT_IN','HELD_SINGLE_DEFAULT'].includes(data.status)||
       data.default!=='SINGLE'||data.unverified_external_inputs!==true||
       !Number.isInteger(data.validated_receipt_count)||
       !Array.isArray(data.guards))throw Error('Invalid guarded evidence review.');
    return data;
  }
  if(data.schema===RUN){
    if(data.provenance!=='LOCALLY_EXECUTED_SYNTHETIC_GPU_SCRATCH_NO_MODEL'||
       data.status!=='PASS_GUARDED_CUDA_SYNTHETIC'||
       !['single','double_reuse_worker'].includes(data.selected_mode)||
       data.temporary_scratch_deleted!==true||
       data.integrity?.full_sha256_host_and_gpu_audit!==true||
       !Array.isArray(data.selected_end_to_end?.samples_seconds)||
       data.selected_end_to_end.samples_seconds.length!==6)throw Error('Incomplete guarded synthetic run.');
    return data;
  }
  throw Error('Unknown guarded receipt schema.');
}
export default function GuardedReader(){
  const [data,setData]=useState(null);
  const [filename,setFilename]=useState('');
  const [error,setError]=useState('');
  async function change(event){
    setData(null);setFilename('');setError('');
    const file=event.target.files?.[0];
    if(!file)return;
    try{
      if(file.size>512*1024)throw Error('Receipt exceeds 512 KiB.');
      setData(verify(JSON.parse(await file.text())));
      setFilename(file.name);
    }catch(e){setError(e?.message||'Unable to load receipt.');}
  }
  const reviewed=data?.schema===REVIEW;
  return <section className="transfer-lab" aria-label="Guarded scheduler receipt review">
    <div className="transfer-intro">
      <span className="reader-tag">GUARDED SCHEDULER / LOCAL ONLY</span>
      <h3>Evidence before activation.</h3>
      <p>Three independent CUDA receipts, twelve paired trials each, strict provenance checks and an explicit synthetic-only operator approval. Single buffering remains the production default.</p>
      <a className="reader-docs" href="https://github.com/MichaelWave369/PhiStream/blob/main/docs/GUARDED_SCHEDULER.md" target="_blank" rel="noreferrer">Read the approval and fallback protocol ↗</a>
    </div>
    <div className="transfer-panel">
      <div className="transfer-panel-top"><span>LOCAL GUARD RECEIPT</span><span>USER-PROVIDED · UNAUTHENTICATED</span></div>
      <label htmlFor="guarded-json">Choose guarded review or synthetic-run JSON</label>
      <input id="guarded-json" type="file" accept=".json,application/json" onChange={change}/>
      {error&&<p className="transfer-error" role="alert">{error}</p>}
      {!data&&!error&&<p className="transfer-empty">No guarded evidence report loaded</p>}
      {data&&<>
        <div className="transfer-result-head"><strong>{reviewed?'OFFLINE EVIDENCE REVIEW':'EXPLICIT SYNTHETIC CUDA TEST'}</strong><span>{data.status}</span></div>
        <p className="transfer-file-name">{filename}</p>
        {reviewed?<>
          <div className="transfer-stats">
            <div><small>Accepted independent runs</small><strong>{data.validated_receipt_count}</strong></div>
            <div><small>Paired trials</small><strong>{data.aggregate?.paired_trials??'—'}</strong></div>
            <div><small>Double wins</small><strong>{data.aggregate?.double_wins??'—'}</strong></div>
            <div><small>Median paired gain</small><strong>{Number.isFinite(data.aggregate?.median_paired_gain_double)?(data.aggregate.median_paired_gain_double*100).toFixed(2)+'%':'—'}</strong></div>
          </div>
          <p className="transfer-pinned">Gate: <strong>{data.status}</strong>. Default: <strong>SINGLE</strong>. {data.guards.length?'Hold reasons: '+data.guards.join(', ')+'.':'Eligible for an explicit scratch-only opt-in, NOT production.'}</p>
        </>:<>
          <div className="transfer-stats">
            <div><small>Selected synthetic strategy</small><strong>{data.selected_mode==='single'?'SINGLE':'DOUBLE'}</strong></div>
            <div><small>Median selected elapsed</small><strong>{fmt(data.selected_end_to_end.median_seconds)}</strong></div>
            <div><small>Serialized negative control</small><strong>{fmt(data.serialized_same_worker_negative_control?.median_seconds)}</strong></div>
            <div><small>GPU environment result</small><strong>{data.live_hardware_check||'—'}</strong></div>
          </div>
          <p className="transfer-pinned">Explicit operator approval: {data.explicit_operator_approval?'YES':'NO'}. Fallbacks: {data.fallback_reasons?.length?data.fallback_reasons.join(', '):'None'}. No production scheduler changes.</p>
        </>}
        <p className="transfer-caveat">These JSON files are local, user-provided and not independently authenticated. A passing review is advisory only. No real model execution, calibrated GPU/I/O overlap or automatic activation is evidenced.</p>
      </>}
    </div>
  </section>;
}
