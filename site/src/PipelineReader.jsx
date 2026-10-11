import React, { useState } from 'react';
import './transfer-reader.css';

const SCHEMA = 'PHISTREAM_PIPELINE_LAB_V1';
const PROVENANCE = 'MEASURED_LOCAL_PAIRED_SYNTHETIC_PIPELINE_NO_MODEL';
const ms = seconds => Number.isFinite(seconds) && seconds > 0 ? (seconds * 1000).toFixed(3) + ' ms' : '—';
const rate = value => Number.isFinite(value) && value > 0 ? value.toFixed(1) + ' MiB/s' : '—';

function validate(receipt) {
  if (receipt?.schema !== SCHEMA || receipt.provenance !== PROVENANCE ||
      !['PASS_CUDA_PIPELINE','PASS_CPU_PIPELINE'].includes(receipt.status) ||
      receipt.fixture?.synthetic !== true ||
      receipt.fixture?.owned_scratch_deleted !== true ||
      receipt.trial_design?.os_file_cache_controlled !== false ||
      receipt.trial_design?.warmup_excluded_from_statistics !== true ||
      receipt.model_execution !== false ||
      receipt.automatic_layer_placement !== false ||
      !Array.isArray(receipt.trial_design?.orders) ||
      receipt.trial_design.orders.length < 4 ||
      !Number.isFinite(receipt.paired_outcome?.median_pageable_total_over_pinned_total)) {
    throw Error('Not a valid PhiStream paired pipeline lab receipt.');
  }
  const methods = receipt.methods;
  for (const name of ['pageable','pinned']) {
    const measurement = methods?.[name]?.measurements;
    if (!measurement || !Number.isFinite(measurement.end_to_end?.median_seconds) ||
        !Number.isFinite(measurement.end_to_end?.median_mib_per_second) ||
        !Array.isArray(measurement.end_to_end?.samples_seconds) ||
        measurement.end_to_end.samples_seconds.length !== receipt.trial_design.orders.length) {
      throw Error('Missing repeated end-to-end pipeline timings.');
    }
  }
  if (receipt.status === 'PASS_CUDA_PIPELINE' &&
      (!receipt.hardware?.name || !methods.pinned.measurements.pinned_host_stage ||
        !methods.pageable.measurements.h2d)) {
    throw Error('CUDA pipeline evidence is missing GPU measurements.');
  }
  return receipt;
}

export default function PipelineReader() {
  const [receipt, setReceipt] = useState(null);
  const [error, setError] = useState('');
  const [name, setName] = useState('');
  const load = async event => {
    setReceipt(null); setError(''); setName('');
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      if (file.size > 256 * 1024) throw Error('Local pipeline JSON exceeds 256 KiB.');
      const data = validate(JSON.parse(await file.text()));
      setReceipt(data); setName(file.name);
    } catch (err) { setError(err?.message || 'Unable to open pipeline file.'); }
  };
  const pageable = receipt?.methods?.pageable?.measurements;
  const pinned = receipt?.methods?.pinned?.measurements;
  return <section className="transfer-lab" aria-label="Paired end-to-end transfer evidence">
    <div className="transfer-intro">
      <span className="reader-tag">PIPELINE INTELLIGENCE / PAIRED TEST</span>
      <h3>Time the whole journey.</h3>
      <p>Compare fresh pageable buffers and reusable pinned-host staging from a synthetic file read through synchronized CUDA transfer. Controlled warmups, balanced method ordering and full-pipeline wall time.</p>
      <a href="https://github.com/MichaelWave369/PhiStream/blob/main/docs/PIPELINE_LAB.md" target="_blank" rel="noreferrer" className="reader-docs">Read the paired-test protocol ↗</a>
    </div>
    <div className="transfer-panel">
      <div className="transfer-panel-top"><span>LOCAL PIPELINE RECEIPT</span><span>USER-PROVIDED · UNAUTHENTICATED</span></div>
      <label htmlFor="pipeline-json">Choose paired pipeline JSON</label>
      <input id="pipeline-json" type="file" accept=".json,application/json" onChange={load}/>
      {error && <p className="transfer-error" role="alert">{error}</p>}
      {!receipt && !error && <p className="transfer-empty">No paired pipeline measurement loaded</p>}
      {receipt && <>
        <div className="transfer-result-head"><strong>{receipt.status === 'PASS_CUDA_PIPELINE' ? 'CUDA PIPELINE' : 'CPU PIPELINE'}</strong><span>{receipt.status}</span></div>
        <p className="transfer-file-name">{name} · {receipt.trial_design.orders.length} paired trials · {receipt.trial_design.warmups_per_mode} warmups/method</p>
        <div className="transfer-stats">
          <div><small>Pageable · complete path</small><strong>{ms(pageable.end_to_end.median_seconds)}</strong></div>
          <div><small>Staged · complete path</small><strong>{ms(pinned.end_to_end.median_seconds)}</strong></div>
          <div><small>Pageable · median rate</small><strong>{rate(pageable.end_to_end.median_mib_per_second)}</strong></div>
          <div><small>Staged · median rate</small><strong>{rate(pinned.end_to_end.median_mib_per_second)}</strong></div>
        </div>
        <p className="transfer-pinned">Median paired pageable/staged time ratio: <strong>{receipt.paired_outcome.median_pageable_total_over_pinned_total.toFixed(3)}×</strong>. {receipt.paired_outcome.median_pageable_total_over_pinned_total < 1 ? 'Pageable favored' : receipt.paired_outcome.median_pageable_total_over_pinned_total > 1 ? 'Pinned favored' : 'Tie'} in this synthetic run. Staged path won {receipt.paired_outcome.pinned_wins} of {receipt.trial_design.orders.length} pairs.</p>
        {pinned.unattributed_wall_time && <p className="transfer-pinned">Extra timing diagnostics · pinned pre-H2D synchronization: <strong>{ms(pinned.pre_h2d_sync?.median_seconds)}</strong>; pinned unattributed remainder: <strong>{ms(pinned.unattributed_wall_time.median_seconds)}</strong>; pageable unattributed remainder: <strong>{ms(pageable.unattributed_wall_time?.median_seconds)}</strong>. These are measured diagnostics, not a speedup claim.</p>}
        <p className="transfer-caveat">Local synthetic data, not real-model inference. Storage reads may come from Windows file cache. Warmup and paired ordering reduce noise but cannot eliminate it. No automatic layer placement or prefetch is enabled.</p>
      </>}
    </div>
  </section>;
}
