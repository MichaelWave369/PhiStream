import React, { useState } from 'react';
import { REPLICATION, validateReplicationReview } from './replication-evidence.mjs';
import './replication-research.css';

const percent = value => (100 * value).toFixed(2) + '%';

export default function ReplicationResearch() {
  const [review, setReview] = useState(null);
  const [filename, setFilename] = useState('');
  const [error, setError] = useState('');

  async function load(event) {
    setReview(null);
    setFilename('');
    setError('');
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      if (file.size > 2 * 1024 * 1024) throw Error('Review receipt exceeds 2 MiB.');
      if (!/\.json$/i.test(file.name)) throw Error('Choose a local JSON file.');
      const data = validateReplicationReview(JSON.parse(await file.text()));
      setReview(data);
      setFilename(file.name);
    } catch (e) {
      setError(e?.message || 'Could not inspect local replication-review JSON.');
    }
  }

  return <section className="pr-replication" aria-label="Operator-reported cross-day replication research">
    <div className="pr-replication-head">
      <div><span className="pr-replication-kicker">CROSS-DAY EXPERIMENT / REPLICATION #1</span>
        <h4>Same frozen prompt.<br/>A new seed and a new day.</h4>
        <p>A second local RTX 5070 spread_24 cohort was reported on October 13,
          following the October 12 original. Eight balanced pairs, seed 70421.
          The performance pattern appeared again. The operator subsequently ran
          the PR #28 local SHA-bound receipt audit and reported that it passed.
          This remains a limited, locally audited experiment, not external validation.</p>
      </div>
      <div className="pr-replication-status">
        <span>OPERATOR-REPORTED / LOCAL AUDIT PASSED</span>
        <strong>{percent(REPLICATION.gain)}</strong>
        <small>Median paired decoder-time reduction</small>
        <b>8/8 BATCHED faster pairs</b>
      </div>
    </div>
    <div className="pr-compare">
      <div>
        <span>ORIGINAL · SEED 369</span>
        <strong>{percent(REPLICATION.originalGain)}</strong>
        <small>4/4 pairs · October 12</small>
        <div className="pr-compare-bar"><i style={{width:percent(REPLICATION.originalGain)}}/></div>
      </div>
      <div>
        <span>REPLICATION · SEED 70421</span>
        <strong>{percent(REPLICATION.gain)}</strong>
        <small>8/8 pairs · October 13</small>
        <div className="pr-compare-bar"><i style={{width:percent(REPLICATION.gain)}}/></div>
      </div>
    </div>
    <div className="pr-replication-ledger" aria-label="Operator-reported complete four-cohort local audit counts">
      <div><small>BATCHED-faster comparisons</small><strong>{REPLICATION.combinedWins}/{REPLICATION.combinedPairs}</strong></div>
      <div><small>Recorded executions audited</small><strong>{REPLICATION.checkedExecutions}</strong></div>
      <div><small>Matching greedy-token checks</small><strong>{REPLICATION.checkedGreedy.toLocaleString('en-US')}</strong></div>
      <div><small>Matching sampled-logit checks</small><strong>{REPLICATION.checkedSparseLogits.toLocaleString('en-US')}</strong></div>
    </div>
    <div className="pr-replication-detail">
      <span>TTFT medians: {percent(REPLICATION.originalTtft)} → {percent(REPLICATION.ttft)}</span>
      <span>Median paired-gain difference: +{REPLICATION.medianGainDifferencePoints.toFixed(4)} percentage points</span>
      <span>Staging reduction in new cohort: {(REPLICATION.stagedBytesSaved / 1e9).toFixed(2)} GB</span>
      <span>New cohort: 576/576 token checks, 5,184/5,184 sparse probes in the local audit</span>
    </div>
    <div className="pr-replication-limit">
      <b>Claim boundary.</b> The operator reported successfully running the PR #28
      SHA-bound offline receipt auditor. Its recorded verdict is
      REPLICATION_REVIEW_CANDIDATE_NOT_SPEEDUP_CERTIFIED, not independent
      checkpoint-publisher authentication or reproduction by another laboratory.
      Calendar separation remains operator-reported, not externally time-attested.
      No statistical proof or scheduler promotion. The original SINGLE decoder remains the default.
    </div>
    <div className="pr-replication-reader">
      <div>
        <h5>Open your new offline replication review</h5>
        <p>The operator has completed the PR #28 Python audit locally. Select the saved
          replication-review-seed70421.json receipt here to inspect it. This static browser
          checks internal structure and arithmetic, <strong>not original SHA256 chains</strong>.
          Files remain in your browser and are not uploaded or stored.</p>
        <a href="https://github.com/MichaelWave369/PhiStream/blob/main/docs/REPLICATION_REVIEW.md"
          target="_blank" rel="noreferrer">Offline audit procedure ↗</a>
      </div>
      <div>
        <label htmlFor="replication-review-file">Choose local PR28 review JSON</label>
        <input id="replication-review-file" type="file"
          accept=".json,application/json" onChange={load}/>
        {error && <p role="alert" className="pr-replication-error">{error}</p>}
        {review ? <div className="pr-replication-loaded" aria-live="polite">
          <span>LOCAL FILE / STRUCTURE CONSISTENT / NO SHA ATTESTATION</span>
          <strong>{review.status}</strong>
          <small>{filename} · {review.combined.batched_faster_pairs}/20 BATCHED faster
            · {percent(review.replication.median_paired_relative_gain)} median new paired gain</small>
          <p>This browser checks structure only. The operator-reported successful offline Python SHA-bound review is the evidence-consistency check, not an external certification.</p>
        </div> : !error && <p className="pr-replication-empty">No local PR28 JSON selected. The chart reports the operator's completed offline audit; import the file to inspect its structure.</p>}
      </div>
    </div>
  </section>;
}
