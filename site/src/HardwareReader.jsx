import React, { useState } from 'react';
import './hardware-reader.css';

const MIB = 1048576;
const fmt = n => Number.isFinite(n) && n >= 0 ? (n / MIB).toFixed(1) + ' MiB' : '—';
const numeric = (o, key) => Number.isFinite(o?.[key]) && o[key] >= 0;

function validate(data) {
  if (data?.schema === 'PHISTREAM_HARDWARE_SNAPSHOT_V1') {
    if (data.status !== 'OBSERVED_READ_ONLY' ||
        data.provenance !== 'LOCAL_HARDWARE_COUNTERS_NOT_A_BENCHMARK' ||
        !numeric(data.ram, 'available_bytes') || !numeric(data.ram, 'total_bytes') ||
        data.ram.available_bytes > data.ram.total_bytes ||
        (data.gpu != null && (!numeric(data.gpu, 'free_bytes') ||
          !numeric(data.gpu, 'total_bytes') || data.gpu.free_bytes > data.gpu.total_bytes)) ||
        !Array.isArray(data.drives)) throw Error('Invalid local hardware receipt.');
    return { kind: 'snapshot', data };
  }
  if (data?.schema === 'PHISTREAM_RESIDENCY_ADVICE_V1') {
    if (!['REVIEW_REQUIRED', 'BLOCKED_BY_POLICY'].includes(data.status) ||
        data.execution_approved !== false || data.auto_placement_enabled !== false ||
        data.no_fit_guarantee !== true ||
        data.provenance !== 'LIVE_MEMORY_COUNTERS_PLUS_METADATA_ESTIMATE_NOT_BENCHMARKED' ||
        !numeric(data.observations, 'gpu_free_bytes_at_snapshot') ||
        !numeric(data.observations, 'ram_available_bytes_at_snapshot') ||
        !numeric(data.proposed_gpu, 'estimated_current_executor_working_set_bytes') ||
        !Array.isArray(data.blocked_reasons)) throw Error('Invalid unapproved residency advice.');
    return { kind: 'advice', data };
  }
  throw Error('Select PhiStream hardware-snapshot or recommend-residency JSON.');
}

export default function HardwareReader() {
  const [selection, setSelection] = useState(null);
  const [error, setError] = useState('');
  const [name, setName] = useState('');
  const open = async event => {
    setSelection(null); setError(''); setName('');
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      if (file.size > 512 * 1024) throw Error('Local JSON exceeds 512 KiB.');
      setSelection(validate(JSON.parse(await file.text())));
      setName(file.name);
    } catch (e) {
      setError(e?.message || 'Unable to read JSON file.');
    }
  };
  const data = selection?.data;
  const advice = selection?.kind === 'advice';
  const gpuFree = advice ? data?.observations?.gpu_free_bytes_at_snapshot : data?.gpu?.free_bytes;
  const ramFree = advice ? data?.observations?.ram_available_bytes_at_snapshot : data?.ram?.available_bytes;
  return <section className="hardware-reader" aria-label="Read local hardware reports">
    <div className="hardware-intro">
      <span className="reader-tag">LOCAL HARDWARE INTELLIGENCE / OPT-IN</span>
      <h3>Observe first. Schedule later.</h3>
      <p>Read your real PC's available RAM, free CUDA memory, and optional model-placement proposals. Files stay in your browser. Neither this viewer nor the advisor moves any model layers.</p>
      <a className="reader-docs" href="https://github.com/MichaelWave369/PhiStream/blob/main/docs/HARDWARE_ADVISOR.md" target="_blank" rel="noreferrer">Windows instructions ↗</a>
    </div>
    <div className="hardware-panel">
      <div className="hardware-panel-top"><span>LOCAL HARDWARE RECEIPT</span><span>USER-PROVIDED · NOT AUTHENTICATED</span></div>
      <label htmlFor="hardware-json">Choose a hardware or residency JSON file</label>
      <input id="hardware-json" type="file" accept=".json,application/json" onChange={open}/>
      {error && <p role="alert" className="hardware-error">{error}</p>}
      {!selection && !error && <p className="hardware-empty">Awaiting local hardware snapshot · No automatic device access</p>}
      {selection && <>
        <div className="hardware-result-head"><strong>{advice ? 'RESIDENCY ADVICE' : 'HARDWARE SNAPSHOT'}</strong><span>{data.status}</span></div>
        <p className="hardware-file">{name} · {advice ? data.snapshot_observed_utc : data.observed_utc}</p>
        <div className="hardware-metrics">
          <div><small>Observed available RAM</small><strong>{fmt(ramFree)}</strong></div>
          <div><small>Observed free CUDA VRAM</small><strong>{fmt(gpuFree)}</strong></div>
          {advice ? <>
            <div><small>Estimated active working set</small><strong>{fmt(data.proposed_gpu.estimated_current_executor_working_set_bytes)}</strong></div>
            <div><small>Candidate pinned layers (NOT APPLIED)</small><strong>{data.proposed_gpu.candidate_pinned_layer_indices?.length ?? 0}</strong></div>
          </> : <>
            <div><small>GPU model</small><strong>{data.gpu?.name || 'Not queried'}</strong></div>
            <div><small>Observed volumes</small><strong>{data.drives.length}</strong></div>
          </>}
        </div>
        {!advice && data.drives.map((disk, i) => <div className="hardware-drive" key={i}>
          <span>{disk.requested_path}</span><span>{fmt(disk.free_bytes)} free</span>
        </div>)}
        {advice && data.blocked_reasons.length > 0 &&
          <p className="hardware-error">Blocked: {data.blocked_reasons.join(', ')}</p>}
        <p className="hardware-caveat">{advice
          ? 'Proposal only. No GPU residency was changed; no execution is approved and memory fit remains unverified.'
          : 'Counters are a momentary snapshot. SSD throughput and actual model fit have not been measured.'}</p>
      </>}
    </div>
  </section>;
}
