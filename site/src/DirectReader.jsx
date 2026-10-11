import React, { useState } from 'react';
import './transfer-reader.css';

const SCHEMA = 'PHISTREAM_DIRECT_READ_LAB_V1';
const ORIGIN = 'MEASURED_LOCAL_SYNTHETIC_READINTO_NO_MODEL';
const ms = n => Number.isFinite(n) && n >= 0 ? (n * 1000).toFixed(3) + ' ms' : '—';

function validate(data) {
  if (data?.schema !== SCHEMA || data.provenance !== ORIGIN ||
      !['PASS_CUDA_DIRECT_READ', 'PASS_CPU_DIRECT_READ'].includes(data.status) ||
      data.model_execution !== false || data.automatic_layer_placement !== false ||
      data.design?.filesystem_cache_controlled !== false ||
      data.design?.checksums_checked_outside_timing !== true ||
      !Array.isArray(data.size_reports) ||
      ![6,12].includes(data.design?.rounds_per_size)) {
    throw Error('Not a valid synthetic PhiStream direct-read receipt.');
  }
  const methods = data.status === 'PASS_CUDA_DIRECT_READ'
    ? ['legacy_copy', 'direct_pageable', 'direct_pinned']
    : ['legacy_host_copy', 'direct_host_readinto'];
  if (data.design.methods?.join(',') !== methods.join(',') ||
      data.size_reports.length < 1 || data.size_reports.length > 3) {
    throw Error('Unexpected experiment methods or number of test sizes.');
  }
  for (const report of data.size_reports) {
    if (report.temporary_scratch_deleted !== true ||
        !Number.isFinite(report.size_mib) || report.size_mib < 1 ||
        !Array.isArray(report.orders) ||
        report.orders.length !== data.design.rounds_per_size) {
      throw Error('Scratch or balanced-order evidence missing.');
    }
    for (const mode of methods) {
      const series = report.methods?.[mode]?.total;
      if (!Number.isFinite(series?.median_seconds) ||
          !Array.isArray(series.samples_seconds) ||
          series.samples_seconds.length !== report.orders.length) {
        throw Error('Incomplete measured direct-read timings.');
      }
    }
  }
  return data;
}

export default function DirectReader() {
  const [receipt, setReceipt] = useState(null);
  const [error, setError] = useState('');
  const [filename, setFilename] = useState('');
  async function change(event) {
    setReceipt(null); setError(''); setFilename('');
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      if (file.size > 512 * 1024) throw Error('Receipt exceeds 512 KiB.');
      setReceipt(validate(JSON.parse(await file.text())));
      setFilename(file.name);
    } catch (e) {
      setError(e?.message || 'Unable to read local direct-read JSON.');
    }
  }
  const isGpu = receipt?.status === 'PASS_CUDA_DIRECT_READ';
  return <section className="transfer-lab" aria-label="Direct to pinned file read experiment">
    <div className="transfer-intro">
      <span className="reader-tag">DIRECT STAGING RESEARCH / OPT-IN</span>
      <h3>Remove the unnecessary copy.</h3>
      <p>Compare our multi-copy Python path with direct file reads into preallocated pageable and pinned host tensors. Every trial verifies the synthetic bytes, and this page never uploads your receipt.</p>
      <a className="reader-docs" href="https://github.com/MichaelWave369/PhiStream/blob/main/docs/DIRECT_READ_LAB.md" target="_blank" rel="noreferrer">Experiment protocol ↗</a>
    </div>
    <div className="transfer-panel">
      <div className="transfer-panel-top"><span>DIRECT READ RECEIPT</span><span>USER-PROVIDED · NOT AUTHENTICATED</span></div>
      <label htmlFor="direct-json">Choose direct-read-lab JSON</label>
      <input id="direct-json" type="file" accept=".json,application/json" onChange={change}/>
      {error && <p role="alert" className="transfer-error">{error}</p>}
      {!receipt && !error && <p className="transfer-empty">No direct-to-host-buffer receipt loaded</p>}
      {receipt && <>
        <div className="transfer-result-head"><strong>{isGpu ? 'CUDA DIRECT STAGING' : 'CPU DIRECT STAGING'}</strong><span>{receipt.status}</span></div>
        <p className="transfer-file-name">{filename} · {receipt.design.rounds_per_size} balanced rounds/size · {receipt.design.warmups_per_method} warmups/method</p>
        {receipt.size_reports.map(report => {
          const legacy = report.methods[isGpu ? 'legacy_copy' : 'legacy_host_copy'].total.median_seconds;
          const pinned = isGpu ? report.methods.direct_pinned.total.median_seconds : null;
          return <div key={report.size_mib} style={{marginTop: 16, paddingTop: 8, borderTop: '1px solid #36524e'}}>
            <div className="transfer-result-head"><strong>{report.size_mib} MiB / SYNTHETIC</strong><span>Integrity checked</span></div>
            <div className="transfer-stats">
              {receipt.design.methods.map(key => <div key={key}>
                <small>{key.replaceAll('_', ' ')}</small>
                <strong>{ms(report.methods[key].total.median_seconds)}</strong>
              </div>)}
            </div>
            <p className="transfer-pinned">{isGpu ? 'Direct pinned vs legacy' : 'Direct host vs legacy'}:
              {' '}{legacy > 0 && (isGpu ? pinned : report.methods.direct_host_readinto.total.median_seconds) > 0
                ? ((legacy / (isGpu ? pinned : report.methods.direct_host_readinto.total.median_seconds))).toFixed(3) + '× time ratio'
                : '—'}.
              {' '}Round wins: {receipt.design.methods.map(key => key + ': ' + (report.round_winner_counts[key] ?? 0)).join(' · ')}.
            </p>
          </div>;
        })}
        <p className="transfer-caveat">Synthetic scratch files only. Windows filesystem caching is uncontrolled. Direct-to-pinned means disk read into pinned CPU memory, then a separate GPU copy, not GPUDirect or verified model inference.</p>
      </>}
    </div>
  </section>;
}
