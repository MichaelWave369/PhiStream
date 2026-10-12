import React, { useState } from 'react';
import './transfer-reader.css';

const SCHEMA = 'PHISTREAM_OVERLAP_LAB_V1';
const PROVENANCE = 'MEASURED_LOCAL_SYNTHETIC_HOST_TIMELINE_NO_MODEL';
const NAMES = [
  ['single', 'Single buffer'],
  ['double_new_worker', 'New worker each trial'],
  ['double_reuse_worker', 'Persistent worker'],
];
const milli = s => Number.isFinite(s) && s >= 0 ? (s * 1000).toFixed(3) + ' ms' : '—';

function validate(data) {
  if (data?.schema !== SCHEMA || data.provenance !== PROVENANCE ||
      !['PASS_CUDA_OVERLAP_LAB', 'PASS_CPU_OVERLAP_SCHEDULER'].includes(data.status) ||
      data.model_execution !== false || data.automatic_layer_placement !== false ||
      data.fixture?.scratch_deleted !== true || data.fixture?.synthetic !== true ||
      data.design?.scratch_cache_controlled !== false ||
      data.design?.gpu_clock_aligned_with_cpu !== false ||
      data.design?.gpu_compute_overlap_tested !== false ||
      data.recommendation?.default !== 'SINGLE' ||
      !['NO_GPU_EVIDENCE', 'INCONCLUSIVE', 'DOUBLE_CANDIDATE', 'SINGLE_CANDIDATE'].includes(data.recommendation?.decision) ||
      ![6, 12].includes(data.design?.paired_rounds) ||
      !Array.isArray(data.design?.balanced_orders) ||
      data.design.balanced_orders.length !== data.design.paired_rounds ||
      data.validation?.audit_is_outside_timing !== true) {
    throw Error('Not a valid PhiStream synthetic overlap receipt.');
  }
  for (const [key] of NAMES) {
    const row = data.methods?.[key];
    if (!Number.isFinite(row?.end_to_end?.median_seconds) ||
        row.end_to_end.median_seconds <= 0 ||
        !Array.isArray(row.end_to_end.samples_seconds) ||
        row.end_to_end.samples_seconds.length !== data.design.paired_rounds ||
        !Array.isArray(row.first_measured_trial_host_timeline) ||
        row.first_measured_trial_host_timeline.length !== data.chunks) {
      throw Error('Missing paired wall-clock samples or host timeline.');
    }
  }
  if (data.status === 'PASS_CUDA_OVERLAP_LAB' &&
      (data.validation.full_sha256_host_and_gpu_audit !== true || !data.hardware?.gpu_name)) {
    throw Error('CUDA receipt lacks full GPU replay integrity evidence.');
  }
  return data;
}

function HostTimeline({ items }) {
  const shown = items.slice(0, 8);
  const max = Math.max(0.001, ...items.map(item => Number(item.completion_observed_ms) || 0));
  const bar = (from, to, color) => {
    const start = Math.max(0, Number(from) || 0);
    const end = Math.max(start, Number(to) || start);
    const left = Math.min(100, 100 * start / max);
    return {
      position: 'absolute', top: 0, bottom: 0, borderRadius: 3,
      left: left + '%',
      width: Math.max(0.4, Math.min(100 - left, 100 * (end - start) / max)) + '%',
      background: color,
    };
  };
  return <div style={{margin:'14px 0',display:'grid',gap:7}}>
    <small style={{color:'#9dbdb2'}}>First measured persistent-worker trial · host-clock intervals · first {shown.length} chunks</small>
    {shown.map(item => <div key={item.chunk} style={{display:'grid',gridTemplateColumns:'28px 1fr',gap:8,alignItems:'center'}}>
      <span style={{fontSize:10,color:'#b2d5c6'}}>#{item.chunk + 1}</span>
      <div style={{height:16,background:'#173336',position:'relative',borderRadius:3}}>
        <span style={bar(item.enqueue_start_ms, item.completion_observed_ms, '#667f97')} title="Host enqueue to observed completion bracket, NOT device execution time"/>
        <span style={bar(item.read_start_ms, item.read_end_ms, '#37be9f')} title="Measured CPU host read"/>
      </div>
    </div>)}
    <small style={{fontSize:10,color:'#a8bdb6'}}>Teal: CPU read. Grey: host enqueue-to-completion-observation bracket. CUDA and CPU clocks are NOT calibrated, so overlap is not proved.</small>
  </div>;
}

export default function OverlapReader() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [filename, setFilename] = useState('');
  const load = async event => {
    setData(null); setError(''); setFilename('');
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      if (file.size > 512 * 1024) throw Error('Receipt exceeds 512 KiB.');
      setData(validate(JSON.parse(await file.text())));
      setFilename(file.name);
    } catch (e) { setError(e?.message || 'Could not read local overlap JSON.'); }
  };
  const cuda = data?.status === 'PASS_CUDA_OVERLAP_LAB';
  return <section className="transfer-lab" aria-label="Overlap Intelligence experiment">
    <div className="transfer-intro">
      <span className="reader-tag">OVERLAP INTELLIGENCE / EXPERIMENTAL</span>
      <h3>When does another worker help?</h3>
      <p>Compare serial transfers, a fresh background worker and a reused worker. Measure host reads, queue delays and CUDA completion observations without mixing CPU and GPU clocks.</p>
      <a className="reader-docs" href="https://github.com/MichaelWave369/PhiStream/blob/main/docs/OVERLAP_LAB.md" target="_blank" rel="noreferrer">Read the protocol and thresholds ↗</a>
    </div>
    <div className="transfer-panel">
      <div className="transfer-panel-top"><span>LOCAL OVERLAP RECEIPT</span><span>USER-PROVIDED · NOT AUTHENTICATED</span></div>
      <label htmlFor="overlap-json">Choose overlap-lab JSON file</label>
      <input id="overlap-json" type="file" accept=".json,application/json" onChange={load}/>
      {error && <p role="alert" className="transfer-error">{error}</p>}
      {!data && !error && <p className="transfer-empty">No host timeline evidence loaded</p>}
      {data && <>
        <div className="transfer-result-head"><strong>{cuda ? 'RTX CUDA OVERLAP LAB' : 'CPU SCHEDULER CHECK'}</strong><span>{data.status}</span></div>
        <p className="transfer-file-name">{filename} · {data.chunks} × {data.chunk_mib} MiB · {data.design.paired_rounds} rounds</p>
        <div className="transfer-stats">
          {NAMES.map(([key, title]) => <div key={key}>
            <small>{title} · whole trial</small>
            <strong>{milli(data.methods[key].end_to_end.median_seconds)}</strong>
          </div>)}
          <div><small>Persistent worker setup, excluded</small><strong>{milli(data.design.persistent_worker_startup_seconds_excluded)}</strong></div>
        </div>
        <p className="transfer-pinned">Evidence gate: <strong>{data.recommendation.decision}</strong>. Default remains SINGLE. No policy applied. {data.recommendation.reason}</p>
        <p className="transfer-pinned">Reused-worker median queue delay total: {milli(data.methods.double_reuse_worker.worker_queue_sum.median_seconds)}. CUDA wait total: {milli(data.methods.double_reuse_worker.cuda_wait_sum.median_seconds)}.</p>
        <HostTimeline items={data.methods.double_reuse_worker.first_measured_trial_host_timeline}/>
        <p className="transfer-caveat">{cuda ? 'Full host/GPU SHA256 checked in an independent replay.' : 'CPU-only structural check, not CUDA measurements.'} Synthetic bytes, OS filesystem cache uncontrolled, no real model or calibrated device/CPU overlap evidence.</p>
      </>}
    </div>
  </section>;
}
