import React, { useState } from 'react';
import './transfer-reader.css';

const SCHEMA = 'PHISTREAM_TRANSFER_LAB_V1';
const PROVENANCE = 'MEASURED_LOCAL_SYNTHETIC_BYTES_NO_MODEL';
const rate = n => Number.isFinite(n) && n > 0 ? n.toFixed(1) + ' MiB/s' : '—';
const validStats = s => s && Array.isArray(s.samples_seconds) && s.samples_seconds.length >= 3 &&
  Number.isFinite(s.median_mib_per_second) && s.median_mib_per_second > 0;
function inspect(data) {
  if (data?.schema !== SCHEMA || data.provenance !== PROVENANCE ||
    !['PASS_CPU_BASELINE', 'PASS_CUDA_TRANSFERS'].includes(data.status) ||
    data.fixture?.type !== 'SYNTHETIC_DETERMINISTIC_BYTES' ||
    !Number.isFinite(data.fixture?.size_bytes) || data.fixture.size_bytes <= 0 ||
    data.fixture?.temporary_owned_file_deleted !== true ||
    data.storage?.first_read_is_cold_cache !== false ||
    data.model_execution !== false || data.automatic_layer_placement !== false ||
    !validStats(data.storage?.warm_reads) || !validStats(data.host?.bytearray_copy)) {
    throw Error('Invalid PhiStream V1 synthetic transfer receipt.');
  }
  if (data.status === 'PASS_CUDA_TRANSFERS' &&
    (!data.cuda?.gpu_name || !validStats(data.cuda.pageable_h2d) ||
      !validStats(data.cuda.pinned_h2d) ||
      !validStats(data.cuda.pageable_to_pinned_host_stage))) {
    throw Error('CUDA transfer receipt is missing expected transfer measurements.');
  }
  if (data.status === 'PASS_CPU_BASELINE' && data.cuda != null) {
    throw Error('CPU-only report contains inconsistent GPU fields.');
  }
  return data;
}

export default function TransferReader() {
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [name, setName] = useState('');
  const change = async event => {
    setData(null); setError(''); setName('');
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      if (file.size > 128 * 1024) throw Error('Transfer JSON exceeds 128 KiB.');
      setData(inspect(JSON.parse(await file.text())));
      setName(file.name);
    } catch (e) {
      setError(e?.message || 'Unable to read transfer receipt.');
    }
  };
  const gpu = data?.cuda;
  return <section className="transfer-lab">
    <div className="transfer-intro">
      <span className="reader-tag">TRANSFER LAB / LOCAL EXPERIMENT</span>
      <h3>Measure the space between tiers.</h3>
      <p>Bench synthetic SSD reads, RAM copies and RTX transfers. The file is read inside this browser. No uploads, GPU access or model execution happens on this page.</p>
      <a href="https://github.com/MichaelWave369/PhiStream/blob/main/docs/TRANSFER_LAB.md"
         target="_blank" rel="noreferrer" className="reader-docs">Run the transfer laboratory ↗</a>
    </div>
    <div className="transfer-panel">
      <div className="transfer-panel-top"><span>TRANSFER RECEIPT IMPORT</span><span>USER-PROVIDED · NOT VERIFIED</span></div>
      <label htmlFor="transfer-file">Choose a local transfer-lab JSON file</label>
      <input id="transfer-file" type="file" accept=".json,application/json" onChange={change}/>
      {error && <p role="alert" className="transfer-error">{error}</p>}
      {!data && !error && <p className="transfer-empty">No synthetic transfer measurements loaded</p>}
      {data && <>
        <div className="transfer-result-head"><strong>{gpu ? 'CUDA TRANSFER LAB' : 'CPU / STORAGE LAB'}</strong><span>{data.status}</span></div>
        <p className="transfer-file-name">{name} · {data.started_utc || 'Local run'} · {data.repetitions} repetitions</p>
        <div className="transfer-stats">
          <div><small>Warm read (OS cache uncontrolled)</small><strong>{rate(data.storage.warm_reads.median_mib_per_second)}</strong></div>
          <div><small>RAM bytearray copy</small><strong>{rate(data.host.bytearray_copy.median_mib_per_second)}</strong></div>
          {gpu ? <>
            <div><small>Pageable → GPU, wall-clock</small><strong>{rate(gpu.pageable_h2d.median_mib_per_second)}</strong></div>
            <div><small>Pinned → GPU, excludes staging</small><strong>{rate(gpu.pinned_h2d.median_mib_per_second)}</strong></div>
          </> : <>
            <div><small>Fixture size</small><strong>{(data.fixture.size_bytes / 1048576).toFixed(0)} MiB</strong></div>
            <div><small>Device copy</small><strong>Not requested</strong></div>
          </>}
        </div>
        {gpu && <p className="transfer-pinned">Pinned CPU staging + H2D: approximately {Number(gpu.pinned_total_median_seconds_approx).toFixed(6)} seconds (sum of stage and transfer medians).</p>}
        <p className="transfer-caveat">Synthetic bytes only. A first read after writing is not cold-cache I/O. OS caching is uncontrolled, no real model or cache scheduler was benchmarked, and this JSON is user-provided rather than independently authenticated.</p>
      </>}
    </div>
  </section>;
}
