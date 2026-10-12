import React, { useState } from 'react';
import './transfer-reader.css';

const SCHEMA = 'PHISTREAM_CHUNK_PIPELINE_V1';
const ORIGIN = 'MEASURED_LOCAL_SYNTHETIC_CHUNK_STREAM_NO_MODEL';
const milli = seconds => Number.isFinite(seconds) && seconds >= 0 ? (seconds * 1000).toFixed(3) + ' ms' : '—';
function check(data) {
  if (data?.schema !== SCHEMA || data.provenance !== ORIGIN ||
      !['PASS_CUDA_CHUNK_LAB', 'PASS_CPU_CHUNK_SCHEDULER'].includes(data.status) ||
      data.model_execution !== false || data.automatic_layer_placement !== false ||
      data.fixture?.scratch_deleted !== true || data.fixture?.synthetic !== true ||
      data.design?.os_cache_controlled !== false ||
      data.design?.gpu_compute_overlap_tested !== false ||
      data.design?.wait_for_completion_before_host_buffer_reuse !== true ||
      data.validation?.audit_is_outside_timing !== true ||
      ![4, 6, 8].includes(data.design?.paired_rounds)) {
    throw Error('Not a valid synthetic ΦStream chunk pipeline receipt.');
  }
  for (const method of ['single','double']) {
    const stats = data.methods?.[method]?.end_to_end;
    if (!Number.isFinite(stats?.median_seconds) || stats.median_seconds <= 0 ||
        !Array.isArray(stats.samples_seconds) ||
        stats.samples_seconds.length !== data.design.paired_rounds) {
      throw Error('Missing complete repeated wall-clock evidence.');
    }
  }
  if (data.status === 'PASS_CUDA_CHUNK_LAB' &&
      (data.validation.full_sha256_host_and_gpu_audit !== true ||
       !data.hardware?.gpu_name)) {
    throw Error('CUDA receipt lacks full independent replay integrity audit.');
  }
  return data;
}
export default function ChunkReader() {
  const [data,setData] = useState(null);
  const [name,setName] = useState('');
  const [error,setError] = useState('');
  const load = async event => {
    setData(null);setName('');setError('');
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      if (file.size > 512 * 1024) throw Error('Receipt exceeds 512 KiB.');
      setData(check(JSON.parse(await file.text())));
      setName(file.name);
    } catch (e) {
      setError(e?.message || 'Could not open local chunk JSON.');
    }
  };
  const cuda = data?.status === 'PASS_CUDA_CHUNK_LAB';
  return <section className="transfer-lab" aria-label="Chunked pinned CUDA transfer experiment">
    <div className="transfer-intro">
      <span className="reader-tag">CHUNKED STREAMING / DOUBLE BUFFER</span>
      <h3>Read ahead. Prove ownership.</h3>
      <p>Compare serial pinned transfers with two alternating pinned buffers and CUDA completion events. We measure wall-clock throughput and audit every synthetic chunk in a separate GPU replay.</p>
      <a className="reader-docs" target="_blank" rel="noreferrer"
         href="https://github.com/MichaelWave369/PhiStream/blob/main/docs/CHUNK_LAB.md">Experimental protocol ↗</a>
    </div>
    <div className="transfer-panel">
      <div className="transfer-panel-top"><span>LOCAL CHUNK TRANSFER RECEIPT</span><span>USER-PROVIDED · NOT VERIFIED</span></div>
      <label htmlFor="chunk-json">Choose chunk-lab JSON file</label>
      <input id="chunk-json" type="file" accept=".json,application/json" onChange={load}/>
      {error && <p role="alert" className="transfer-error">{error}</p>}
      {!data && !error && <p className="transfer-empty">No chunk scheduling results loaded</p>}
      {data && <>
        <div className="transfer-result-head"><strong>{cuda ? 'CUDA TWO-BUFFER LAB' : 'CPU SCHEDULER CHECK'}</strong><span>{data.status}</span></div>
        <p className="transfer-file-name">{name} · {data.chunks} chunks × {data.chunk_mib} MiB · {data.design.paired_rounds} paired rounds</p>
        <div className="transfer-stats">
          <div><small>Single-buffer wall time</small><strong>{milli(data.methods.single.end_to_end.median_seconds)}</strong></div>
          <div><small>Double-buffer wall time</small><strong>{milli(data.methods.double.end_to_end.median_seconds)}</strong></div>
          <div><small>Single / double time ratio</small><strong>{Number(data.paired_outcome.median_single_over_double_time).toFixed(3)}×</strong></div>
          <div><small>Double wins</small><strong>{data.paired_outcome.double_wins} / {data.design.paired_rounds}</strong></div>
        </div>
        <p className="transfer-pinned">Paired result favors: <strong>{data.paired_outcome.time_ratio_favors}</strong>. {cuda ? 'All chunks replay-audited against full host and returned GPU SHA-256 outside timing.' : 'CPU-only scheduling test; no CUDA measured.'}</p>
        <p className="transfer-caveat">Scratch-only research. Windows disk cache uncontrolled. A pending CUDA event at read scheduling is not proof that device transfer and file I/O physically overlapped. No compute kernels or real-model inference were benchmarked.</p>
      </>}
    </div>
  </section>;
}
