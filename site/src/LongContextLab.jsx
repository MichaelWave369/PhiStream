import React, { useState } from 'react';
import { LONG_CONTEXT_TARGETS, validateLongContextPlan } from './long-context-evidence.mjs';
import './long-context-lab.css';

const stages = [
  { number: '01', name: 'Protocol first', detail: 'Freeze inputs, full context, seed and a bounded memory policy. No model execution.' },
  { number: '02', name: 'Implement chunked causal prefill', detail: 'Separate research implementation, explicit per-chunk causal masking and correct absolute KV positions.' },
  { number: '03', name: 'Check reference parity', detail: 'A separately approved reference trace for each held-out prompt and all 32 continuation steps.' },
  { number: '04', name: 'Authorize CUDA later', detail: 'Monitor real device memory, bound allocation, then compare SINGLE and candidate with paired controls.' },
];

export default function LongContextLab() {
  const [plan, setPlan] = useState(null);
  const [filename, setFilename] = useState('');
  const [error, setError] = useState('');
  const open = async event => {
    setPlan(null); setFilename(''); setError('');
    const file = event.target.files?.[0];
    if (!file) return;
    try {
      if (file.size > 1024 * 1024) throw Error('Local plan exceeds the 1 MiB display limit.');
      if (!/\.json$/i.test(file.name)) throw Error('Choose a JSON planning receipt.');
      const result = validateLongContextPlan(JSON.parse(await file.text()));
      setPlan(result); setFilename(file.name);
    } catch (e) {
      setError(e?.message || 'Could not read the local planning receipt.');
    }
  };

  return <section className="lc-lab" aria-label="Held 128 512 and 2048 token long context research">
    <div className="lc-head">
      <div>
        <span className="lc-kicker">PR #30 / LONG CONTEXT LABORATORY</span>
        <h3>Longer context.<br/><em>Stronger guardrails.</em></h3>
        <p>Our current experimental BATCHED prefill accepts only <strong>2–24 prompt tokens</strong>.
          Here are three <strong>proposed</strong> next workloads.
          All are explicitly held until a separate bounded implementation, new frozen references,
          resource checks and human execution approval exist.</p>
      </div>
      <div className="lc-banner"><span>EXPERIMENT STATUS</span><strong>PLANNING ONLY</strong><small>0 new CUDA runs authorized</small></div>
    </div>
    <div className="lc-targets">
      {LONG_CONTEXT_TARGETS.map((item, index) => {
        const row = plan?.targets[index];
        const limitReached = row && !row.full_context_fits_checkpoint_limit;
        return <article className="lc-target" key={item.prompt}>
          <div className="lc-target-top"><span>PROPOSED INPUT</span><strong>HELD</strong></div>
          <h4>{item.prompt.toLocaleString('en-US')}<small> prompt tokens</small></h4>
          <p>{item.required.toLocaleString('en-US')} full-context positions with 32 continuation steps</p>
          <div className="lc-target-meta">
            <span>Expected attention score matrix*</span>
            <strong>{row ? (row.theoretical_one_fp32_attention_score_matrix_bytes / (1024 * 1024)).toFixed(2) + ' MiB' : 'Checkpoint-dependent'}</strong>
          </div>
          <div className="lc-target-meta"><span>Metadata context check</span>
            <strong>{row ? (limitReached ? 'HELD: context ceiling' : 'Within declared limit only') : 'Import local plan'}</strong>
          </div>
          <div className="lc-target-meta"><span>Current BATCHED kernel</span><strong>Not supported</strong></div>
        </article>;
      })}
    </div>
    <p className="lc-caveat">*One theoretical dense FP32 attention score matrix, not an allocator peak, hardware-fit guarantee or proposed implementation cost. A 2,048-token input with 32 generated steps needs <strong>2,079</strong> model context positions, and can exceed a checkpoint advertised at exactly 2,048.</p>
    <div className="lc-grid">
      <div className="lc-path">
        <div className="lc-strap">DELIBERATELY SEPARATE FUTURE GATES</div>
        {stages.map(step => <div className="lc-step" key={step.number}>
          <span>{step.number}</span><div><b>{step.name}</b><p>{step.detail}</p></div>
        </div>)}
      </div>
      <div className="lc-reader">
        <div className="lc-strap">BROWSER-LOCAL RECEIPT VIEWER</div>
        <h4>Inspect the held plan.</h4>
        <p>Use the read-only Python planner from the repo to generate a local metadata receipt.
          The website reads that JSON only on your computer. It never uploads the file,
          executes models, probes your GPU or authenticates checkpoint weight hashes.</p>
        <a href="https://github.com/MichaelWave369/PhiStream/blob/main/docs/LONG_CONTEXT_LAB.md" target="_blank" rel="noreferrer">Planning protocol and CLI ↗</a>
        <label htmlFor="long-context-plan-input">Open local long-context-plan JSON</label>
        <input id="long-context-plan-input" type="file" accept=".json,application/json" onChange={open}/>
        {error && <p role="alert" className="lc-error">{error}</p>}
        {plan && <div className="lc-readout" aria-live="polite">
          <span>LOCAL STRUCTURE CONSISTENT · NO SHA ATTESTATION</span>
          <strong>{plan.status}</strong>
          <small>{filename} · Model context {plan.model.max_context_tokens.toLocaleString('en-US')} tokens · All three targets held</small>
          <p>Memory policy: {plan.policy.user_declared_capacity_mib.toLocaleString('en-US')} MiB capacity, {plan.policy.user_declared_reserve_mib.toLocaleString('en-US')} MiB reserve. No hardware fit certified.</p>
        </div>}
        {!error && !plan && <p className="lc-empty">No planning receipt selected. All experiments remain held.</p>}
      </div>
    </div>
    <p className="lc-footer">SINGLE remains the default. No checkpoint weights, tokens, traces, reference or benchmark measurements are generated by this page or PR #30.</p>
  </section>;
}
