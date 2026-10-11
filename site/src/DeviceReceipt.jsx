import React, { useState } from 'react';
import './device-receipt.css';

const SCHEMA = 'PHISTREAM_DEVICE_VALIDATION_V1';
const PROVENANCE = 'MEASURED_ON_THIS_MACHINE_SYNTHETIC_UNTRAINED';
const gb = bytes => Number.isFinite(bytes) ? (bytes / 1024 ** 3).toFixed(2) : '—';
const rate = n => Number.isFinite(n) && n >= 0 ? n.toFixed(2) : '—';

function inspect(receipt) {
  if (receipt?.schema !== SCHEMA || receipt?.provenance !== PROVENANCE) {
    throw Error('This is not a PhiStream V1 local device-validation receipt.');
  }
  if (receipt.hardware?.device !== 'cuda' || typeof receipt.hardware.gpu_name !== 'string' ||
      !receipt.hardware.gpu_name.trim()) {
    throw Error('The selected receipt is not a CUDA measurement. CPU receipts cannot be shown as GPU results.');
  }
  if (!['PASS', 'FAIL'].includes(receipt.status) ||
      receipt.fixture?.type !== 'SYNTHETIC_UNTRAINED' ||
      receipt.fixture?.verified_sha256 !== true || typeof receipt.parity?.passed !== 'boolean') {
    throw Error('The receipt has incomplete validation metadata.');
  }
  if (receipt.status === 'PASS' && receipt.parity.passed !== true) {
    throw Error('Inconsistent PASS receipt: numerical parity did not pass.');
  }
  if (receipt.status === 'PASS') {
    for (const mode of ['stream', 'pinned', 'resident']) {
      if (receipt.runs?.[mode]?.device !== 'cuda' ||
          !Number.isFinite(receipt.runs[mode].generated_tokens_per_second_end_to_end) ||
          receipt.runs[mode].generated_tokens_per_second_end_to_end < 0) {
        throw Error('The receipt does not contain valid CUDA runs for all three modes.');
      }
    }
  }
  return receipt;
}

export default function DeviceReceipt() {
  const [receipt, setReceipt] = useState(null);
  const [fileName, setFileName] = useState('');
  const [error, setError] = useState('');
  const open = async event => {
    const file = event.target.files?.[0];
    setReceipt(null); setError(''); setFileName('');
    if (!file) return;
    try {
      if (file.size > 128 * 1024) throw Error('Receipt is over the 128 KiB local review limit.');
      const parsed = inspect(JSON.parse(await file.text()));
      setReceipt(parsed); setFileName(file.name);
    } catch (err) {
      setError(err?.message || 'Unable to parse the local receipt.');
    }
  };
  return <div className="device-reader">
    <div className="reader-copy">
      <span className="reader-tag">LOCAL DEVICE EVIDENCE / OPT-IN</span>
      <h3>Read your RTX validation receipt.</h3>
      <p>Run PhiStream on your Windows PC, then select the resulting JSON file. The dashboard reads it <b>only inside this browser</b>. No uploads, accounts, model execution, or automatic publication.</p>
      <a href="https://github.com/MichaelWave369/PhiStream/blob/main/docs/DEVICE_VALIDATION.md" target="_blank" rel="noreferrer" className="reader-docs">Read Windows validation instructions ↗</a>
    </div>
    <div className="reader-card">
      <div className="reader-header"><span>GPU RECEIPT IMPORT</span><span className="reader-state">USER-PROVIDED · NOT VERIFIED</span></div>
      <label htmlFor="receipt-file" className="reader-file-label">Choose a local CUDA receipt (.json)</label>
      <input id="receipt-file" className="reader-input" type="file" accept=".json,application/json" onChange={open}/>
      {error && <p role="alert" className="reader-error">{error}</p>}
      {!receipt && !error && <div className="reader-empty"><span>NO LOCAL CUDA RECEIPT LOADED</span><small>Default research status: GPU validation pending</small></div>}
      {receipt && <div className="reader-results">
        <div className="reader-subhead"><strong>{receipt.hardware.gpu_name}</strong><span className={receipt.status === 'PASS' ? 'reader-pass' : 'reader-fail'}>{receipt.status}</span></div>
        <small className="reader-file-name">{fileName} · {receipt.date_utc || 'No timestamp'}</small>
        <div className="reader-stats">
          <div><span>GPU reported capacity</span><strong>{gb(receipt.hardware.gpu_total_bytes)} GiB</strong></div>
          <div><span>Streamed toy throughput</span><strong>{rate(receipt.runs?.stream?.generated_tokens_per_second_end_to_end)} tok/s</strong></div>
          <div><span>Resident toy throughput</span><strong>{rate(receipt.runs?.resident?.generated_tokens_per_second_end_to_end)} tok/s</strong></div>
          <div><span>Peak streamed PyTorch alloc.</span><strong>{gb(receipt.runs?.stream?.peak_cuda_allocated_bytes)} GiB</strong></div>
        </div>
        <p className="reader-footnote">Synthetic, untrained toy model only. These file-provided numbers are not independently authenticated or evidence of real 8B/70B model support. Timing excludes startup; memory excludes driver and allocator overhead.</p>
      </div>}
    </div>
  </div>;
}
