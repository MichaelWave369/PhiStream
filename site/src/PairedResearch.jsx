import React, { useState } from 'react';
import { CASES, LEDGER, MILESTONES, validateCensusReceipt, validatePairedRunReceipt } from './paired-evidence.mjs';
import './paired-research.css';
import ReplicationResearch from './ReplicationResearch.jsx';
import { REPLICATION } from './replication-evidence.mjs';

const REPO = 'https://github.com/MichaelWave369/PhiStream';
const percent = number => (100 * number).toFixed(2) + '%';
const gb = bytes => (bytes / 1e9).toFixed(2) + ' GB';
const integer = number => number.toLocaleString('en-US');

async function localJson(file, maxBytes) {
  if (!file) return null;
  if (file.size > maxBytes) throw Error('Receipt exceeds the local browser size limit.');
  if (!/\.json$/i.test(file.name)) throw Error('Choose a local JSON receipt.');
  return JSON.parse(await file.text());
}

function Metric({ label, value, detail }) {
  return <div className="pr-metric">
    <span>{label}</span><strong>{value}</strong><small>{detail}</small>
  </div>;
}

function PairDiagram({ row }) {
  return <div className="pr-pair-diagram">
    <div className="pr-diagram-top"><span>{row.tokens}-TOKEN PROMPT</span><strong>{percent(row.gain)} <small>median time reduction</small></strong></div>
    <div className="pr-track" role="img" aria-label={row.id + ': median paired decoder-time reduction ' + percent(row.gain)}>
      <div className="pr-track-fill" style={{ width: percent(row.gain) }}/>
    </div>
  </div>;
}

function Caution() {
  return <p className="pr-limits">
    <strong>Scientific scope:</strong> one local SmolLM2-135M-Instruct checkpoint and one RTX 5070.
    Four original pairs per prompt case plus an eight-pair spread_24 replication; 32 generated steps each. The operator reports successful local SHA-bound auditing,
    offline SHA-bound evidence, <strong>not independently authenticated model
    provenance, statistical significance, or general speedup certification</strong>.
    The decoder still defaults to SINGLE. The browser does not verify SHA256 chains.
  </p>;
}

export default function PairedResearch() {
  const [selected, setSelected] = useState('spread_24');
  const [census, setCensus] = useState(null);
  const [censusName, setCensusName] = useState('');
  const [censusError, setCensusError] = useState('');
  const [run, setRun] = useState(null);
  const [runName, setRunName] = useState('');
  const [runError, setRunError] = useState('');
  const selectedCase = CASES.find(c => c.id === selected);

  const readCensus = async event => {
    setCensus(null); setCensusError(''); setCensusName('');
    try {
      const file = event.target.files?.[0];
      if (!file) return;
      const parsed = validateCensusReceipt(await localJson(file, 2 * 1024 * 1024));
      setCensus(parsed); setCensusName(file.name);
    } catch (error) {
      setCensusError(error?.message || 'Unable to inspect census receipt.');
    }
  };

  const readRun = async event => {
    setRun(null); setRunError(''); setRunName('');
    try {
      const file = event.target.files?.[0];
      if (!file) return;
      const parsed = validatePairedRunReceipt(await localJson(file, 4 * 1024 * 1024));
      setRun(parsed); setRunName(file.name);
    } catch (error) {
      setRunError(error?.message || 'Unable to inspect paired-run receipt.');
    }
  };

  return <div className="pr-wrap" aria-label="Real-model paired A/B research and local evidence explorer">
    <div className="pr-lead">
      <div>
        <span className="pr-kicker"><span className="pr-live-dot"/> REAL MODEL LAB / OCTOBER 2026</span>
        <h3>From weight movement<br/><em>to measured improvement.</em></h3>
        <p>Experimental batched causal prefill versus the original SINGLE decoder.
          A frozen reference, balanced paired CUDA trials and an offline evidence census
          make this a testable performance hypothesis rather than a product claim.</p>
      </div>
      <div className="pr-stage">
        <span>RESEARCH STAGE</span>
        <strong>Further review</strong>
        <small>All four cohorts locally audited in the operator's terminal, including the new cross-day replication. General speedup remains unproven.</small>
      </div>
    </div>

    <div className="pr-metric-grid">
      <Metric label="BATCHED FASTER PAIRS" value={REPLICATION.combinedWins + '/' + REPLICATION.combinedPairs} detail="Across four operator-reported SHA-audited local cohorts"/>
      <Metric label="GREEDY ID CHECKS" value={integer(REPLICATION.checkedGreedy) + '/' + integer(REPLICATION.checkedGreedy)} detail="Across 48 recorded local executions"/>
      <Metric label="SAMPLED LOGIT CHECKS" value={integer(REPLICATION.checkedSparseLogits) + '/' + integer(REPLICATION.checkedSparseLogits)} detail="Sparse coordinates, frozen reference tolerances"/>
      <Metric label="SCHEDULER DEFAULT" value="SINGLE" detail="No experimental production promotion"/>
    </div>

    <div className="pr-main-grid">
      <div className="pr-plot-card">
        <div className="pr-card-head">
          <span>01 / PERFORMANCE BY PROMPT LENGTH</span>
          <span className="pr-quiet">LOWER DECODER TIME IS BETTER</span>
        </div>
        <h4>Same checkpoint. Different prompt patterns.</h4>
        <div className="pr-bars">
          {CASES.map(c => <button type="button"
            key={c.id} className={'pr-case ' + (selected === c.id ? 'is-current' : '')}
            aria-pressed={selected === c.id} onClick={() => setSelected(c.id)}>
            <span className="pr-case-info"><strong>{c.id}</strong><small>{c.tokens} input tokens · {c.wins}/4 BATCHED wins</small></span>
            <span className="pr-case-gauge"><i style={{ width: percent(c.gain) }}/></span>
            <strong className="pr-case-num">{percent(c.gain)}</strong>
          </button>)}
        </div>
        <div className="pr-case-detail" aria-live="polite">
          <div className="pr-detail-row"><span>SELECTED EXPERIMENT</span><strong>{selectedCase.id}</strong></div>
          <div className="pr-detail-grid">
            <div><small>Median paired decoder-time improvement</small><strong>{percent(selectedCase.gain)}</strong></div>
            <div><small>Median time-to-first-logits improvement</small><strong>{percent(selectedCase.ttft)}</strong></div>
            <div><small>Median cumulative weight staging reduction</small><strong>{gb(selectedCase.bytes)}</strong></div>
          </div>
          <PairDiagram row={selectedCase}/>
        </div>
      </div>

      <div className="pr-method-card">
        <div className="pr-card-head"><span>02 / EXPERIMENT CONTRACT</span><span>FROZEN</span></div>
        <h4>Nothing gets a free green check.</h4>
        <div className="pr-method-step"><span>01</span><p><b>Freeze and fingerprint.</b> One exact checkpoint, the original token prompts and externally generated user-declared reference traces.</p></div>
        <div className="pr-method-step"><span>02</span><p><b>Balance the order.</b> Two SINGLE-first and two BATCHED-first pairs per original case, seed 369.</p></div>
        <div className="pr-method-step"><span>03</span><p><b>Verify every execution.</b> Token and sampled-logit checks on six warmups plus 24 measured model executions.</p></div>
        <div className="pr-method-step"><span>04</span><p><b>Audit the receipts.</b> PR #26 recalculates individual paired results against unchanged local SHA-bound evidence.</p></div>
        <a className="pr-link" href={REPO + '/blob/main/docs/THREE_CASE_AB_CENSUS.md'} target="_blank" rel="noreferrer">Read audit procedure ↗</a>
      </div>
    </div>

    <Caution/>

    <ReplicationResearch/>

    <div className="pr-bottom-grid">
      <div className="pr-timeline">
        <div className="pr-card-head"><span>03 / EXPERIMENT MILESTONES</span><span>PR #21–28</span></div>
        <h4>A traceable development history.</h4>
        <ol>{MILESTONES.map(m => <li key={m.pr}>
          <span className="pr-timeline-index">#{m.pr}</span>
          <div><a href={REPO + '/pull/' + m.pr} target="_blank" rel="noreferrer">{m.name} ↗</a><small>{m.note}</small></div>
        </li>)}</ol>
      </div>
      <div className="pr-next">
        <div className="pr-card-head"><span>04 / NEXT TEST</span><span>UNAUTHORIZED / NOT EXECUTED</span></div>
        <h4>Held-out prompts and VRAM safety.</h4>
        <p>With the operator's local SHA audit of the eight-pair spread_24 replication complete, the next research design can investigate longer held-out prompts, a second GPU and real memory pressure. These are proposals, not approved runs or expected results.</p>
        <div className="pr-next-stats"><div><strong>2K</strong><small>token target for future design</small></div><div><strong>3060</strong><small>alternative GPU to evaluate</small></div><div><strong>0</strong><small>new runs authorized</small></div></div>
        <p className="pr-next-note">The operator reports PR #28 SHA-bound review success. That confirms local receipt consistency, not independent provenance or speedup certification. No separate longer-context experiment has been approved.</p>
      </div>
    </div>

    <div className="pr-explorer">
      <div className="pr-card-head"><span>05 / OFFLINE EVIDENCE EXPLORER</span><span>YOUR FILES STAY ON THIS DEVICE</span></div>
      <h4>Open the receipts. Inspect the claims.</h4>
      <p className="pr-explorer-lead">Select your own PR #26 census or PR #25 paired-run JSON.
        Files are parsed in-browser, never submitted to GitHub or a server. These display checks
        catch obvious inconsistencies; they <strong>do not</strong> cryptographically verify
        the original model, references or SHA256 evidence chain.</p>
      <div className="pr-upload-grid">
        <div className="pr-upload">
          <label htmlFor="paired-census-import">Three-case evidence census</label>
          <input id="paired-census-import" type="file" accept=".json,application/json" onChange={readCensus}/>
          {censusError && <p role="alert" className="pr-alert">{censusError}</p>}
          {census ? <div className="pr-import-result" aria-live="polite">
            <span>LOCAL FILE / STRUCTURE CONSISTENT</span>
            <strong>{census.status}</strong>
            <small>{censusName} · {census.total_pairs} pairs · {integer(census.total_matching_sparse_logits_in_original_receipts)} sparse probes</small>
            <p>PR26 recorded hashes and counts are present. Run the offline Python SHA-bound auditor to authenticate evidence consistency.</p>
          </div> : !censusError && <p className="pr-upload-empty">No local census loaded. The public chart above is the user-reported research snapshot.</p>}
        </div>
        <div className="pr-upload">
          <label htmlFor="paired-run-import">Individual paired CUDA run</label>
          <input id="paired-run-import" type="file" accept=".json,application/json" onChange={readRun}/>
          {runError && <p role="alert" className="pr-alert">{runError}</p>}
          {run ? <div className="pr-import-result" aria-live="polite">
            <span>LOCAL FILE / PAIRED TIMINGS RECOMPUTED</span>
            <strong>{run.receipt.case} · {percent(run.medianGain)} paired improvement</strong>
            <small>{runName} · {run.wins}/{run.receipt.design.pairs} faster BATCHED pairs · seed {run.receipt.design.seed}</small>
            <div className="pr-mini-pairs">{run.gains.map((gain, i) => <span key={i} title={'Pair ' + (i + 1) + ': ' + percent(gain)}>{i + 1}: {percent(gain)}</span>)}</div>
            <p>Parsed and recomputed from file-provided trials. This is not external proof or automatic promotion.</p>
          </div> : !runError && <p className="pr-upload-empty">No local paired-run receipt loaded. Import one to inspect its individual timing results.</p>}
        </div>
      </div>
      <div className="pr-explorer-footer"><span>Static website · No inference · No local storage · No uploads</span>
        <a href={REPO + '/blob/main/docs/PAIRED_REAL_PREFILL_AB.md'} target="_blank" rel="noreferrer">Method and limitations ↗</a></div>
    </div>
  </div>;
}
