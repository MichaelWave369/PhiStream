import React, { useState } from 'react';
import './transfer-reader.css';

const SCHEMA = 'PHISTREAM_REUSE_LAB_V1';
const ORIGIN = 'MEASURED_LOCAL_SYNTHETIC_HOST_BUFFER_REUSE_NO_MODEL';
const milli = n => Number.isFinite(n) && n >= 0 ? (n * 1000).toFixed(3) + ' ms' : '—';

function inspect(json) {
  if (json?.schema !== SCHEMA || json.provenance !== ORIGIN ||
      !['PASS_CPU_REUSE_BASELINE', 'PASS_CUDA_REUSE_LAB'].includes(json.status) ||
      json.model_execution !== false || json.automatic_layer_placement !== false ||
      !Array.isArray(json.size_reports) || ![6,12].includes(json.design?.rounds_per_size) ||
      !Array.isArray(json.design?.methods)) {
    throw Error('Not a valid synthetic PhiStream Reuse Lab V1 receipt.');
  }
  const methods = json.status === 'PASS_CUDA_REUSE_LAB'
    ? ['pageable', 'pinned_new', 'pinned_reuse'] : ['fresh_host', 'reused_host'];
  if (json.design.methods.join(',') !== methods.join(',')) {
    throw Error('Reuse Lab method provenance does not match device status.');
  }
  for (const report of json.size_reports) {
    if (report.scratch_deleted !== true ||
        !Number.isFinite(report.size_mib) || report.size_mib <= 0 ||
        !Array.isArray(report.orders) ||
        report.orders.length !== json.design.rounds_per_size) {
      throw Error('Missing local scratch integrity or balanced order evidence.');
    }
    for (const key of methods) {
      const series = report.methods?.[key]?.total;
      if (!Number.isFinite(series?.median_seconds) ||
          !Array.isArray(series.samples_seconds) ||
          series.samples_seconds.length !== json.design.rounds_per_size) {
        throw Error('Missing measured method samples.');
      }
    }
  }
  return json;
}

export default function ReuseReader() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [name, setName] = useState('');
  const changed = async event => {
    setData(null); setError(''); setName('');
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      if (file.size > 512 * 1024) throw Error('Receipt is larger than 512 KiB.');
      setData(inspect(JSON.parse(await file.text())));
      setName(file.name);
    } catch (e) { setError(e?.message || 'Could not read the receipt.'); }
  };
  const gpu = data?.status === 'PASS_CUDA_REUSE_LAB';
  return <section className="transfer-lab" aria-label="Reusable host buffer evidence">
    <div className="transfer-intro">
      <span className="reader-tag">TRANSFER RESEARCH / REUSABLE BUFFERS</span>
      <h3>Does keeping the buffer pay?</h3>
      <p>Compare fresh pageable transfers, freshly allocated pinned staging, and reused pinned staging across multiple synthetic transfer sizes. Results are local, not real model performance.</p>
      <a className="reader-docs" href="https://github.com/MichaelWave369/PhiStream/blob/main/docs/REUSE_LAB.md" target="_blank" rel="noreferrer">Read the buffer-reuse protocol ↗</a>
    </div>
    <div className="transfer-panel">
      <div className="transfer-panel-top"><span>LOCAL BUFFER REUSE RECEIPT</span><span>USER-PROVIDED · UNAUTHENTICATED</span></div>
      <label htmlFor="reuse-json">Choose reuse-lab JSON</label>
      <input id="reuse-json" type="file" accept=".json,application/json" onChange={changed}/>
      {error && <p className="transfer-error" role="alert">{error}</p>}
      {!data && !error && <p className="transfer-empty">No reusable buffer evidence loaded</p>}
      {data && <>
        <div className="transfer-result-head"><strong>{gpu ? 'RTX / CUDA REUSE LAB' : 'CPU BUFFER BASELINE'}</strong><span>{data.status}</span></div>
        <p className="transfer-file-name">{name} · {data.design.rounds_per_size} rounds per size · {data.design.warmups_per_method} warmups/method</p>
        {data.size_reports.map(report => <div key={report.size_mib} style={{ marginTop: 16, paddingTop: 8, borderTop: '1px solid #36524e' }}>
          <div className="transfer-result-head"><strong>{report.size_mib} MiB · synthetic bytes</strong><span>Temp removed</span></div>
          <div className="transfer-stats">
            {data.design.methods.map(key => <div key={key}>
              <small>{key.replaceAll('_',' ')}</small>
              <strong>{milli(report.methods[key].total.median_seconds)}</strong>
            </div>)}
          </div>
          <p className="transfer-pinned">Fastest in {data.design.rounds_per_size} balanced rounds: {data.design.methods.map(key => key + ' ' + (report.round_winner_counts[key] || 0)).join(' · ')}</p>
        </div>)}
        <p className="transfer-caveat">Synthetic file-to-device measurements with uncontrolled Windows caching. Pinned-new allocation is timed; final buffer cleanup and program startup are excluded. No model execution or autonomous layer placement was performed.</p>
      </>}
    </div>
  </section>;
}
