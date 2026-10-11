import React, { useEffect, useMemo, useState } from 'react';

const REPO = 'https://github.com/MichaelWave369/PhiStream';
const MODELS = [
  { label: '8B', params: 8, layers: 'Small dense' },
  { label: '32B', params: 32, layers: 'Medium dense' },
  { label: '70B', params: 70, layers: 'Large dense' },
  { label: '405B', params: 405, layers: 'Very large dense' }
];
const PRECISIONS = [
  { label: 'FP16', bytes: 2, caption: '16-bit weights' },
  { label: 'INT8', bytes: 1, caption: '8-bit estimate' },
  { label: 'INT4', bytes: 0.5, caption: '4-bit estimate' }
];
const pretty = (n, digits = 1) => Number.isFinite(n) ? Number(n).toFixed(digits) : '—';
const fmt = (n) => typeof n === 'number' ? n.toLocaleString('en-US') : '—';

function Icon({ name, size = 18, strokeWidth = 1.8 }) {
  const icons = {
    arrow: <><path d="M4 12h16M13 5l7 7-7 7"/></>,
    external: <><path d="M14 4h6v6M10 14 20 4"/><path d="M20 13v7H4V4h7"/></>,
    github: <><path d="M9 19c-4 1.5-4-2-6-2m12 4v-3.9c0-1.1-.3-1.7-.9-2.1 3-.3 6.2-1.4 6.2-6.6 0-1.4-.5-2.5-1.3-3.4.1-.4.6-2-.2-4-1.4 0-2.3.9-3.1 1.4a11 11 0 0 0-7.4 0C7.5 1.9 6.6 1 5.2 1c-.8 2-.3 3.6-.2 4-.8.9-1.3 2-1.3 3.4 0 5.2 3.2 6.3 6.2 6.6-.6.4-.9 1-.9 2.1V21"/></>,
    cpu: <><rect x="5" y="5" width="14" height="14" rx="2"/><path d="M9 9h6v6H9zM9 1v4m6-4v4M9 19v4m6-4v4M1 9h4m-4 6h4m14-6h4m-4 6h4"/></>,
    layers: <><path d="m12 3 9 5-9 5-9-5 9-5Zm-9 9 9 5 9-5M3 16l9 5 9-5"/></>,
    activity: <><path d="M2 12h4l3-8 5 16 3-8h5"/></>,
    box: <><path d="m12 2 9 5v10l-9 5-9-5V7l9-5ZM3 7l9 5 9-5m-9 5v10"/></>,
    database: <><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v7c0 2 4 4 9 4s9-2 9-4V5M3 12v7c0 2 4 3 9 3s9-1 9-3v-7"/></>,
    shield: <><path d="m12 2 9 4v6c0 5-4 8-9 10-5-2-9-5-9-10V6l9-4Z"/><path d="m8 12 3 3 5-6"/></>,
    check: <><path d="m4 12 5 5L20 6"/></>,
    chevron: <><path d="m8 4 8 8-8 8"/></>,
    sliders: <><path d="M4 7h16M4 17h16"/><circle cx="9" cy="7" r="3"/><circle cx="16" cy="17" r="3"/></>,
    document: <><path d="M6 2h8l5 5v15H6zM14 2v5h5M9 12h7M9 16h7"/></>
  };
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" stroke="currentColor" strokeWidth={strokeWidth} fill="none" strokeLinecap="round" strokeLinejoin="round">{icons[name] || icons.box}</svg>;
}

function Logo() {
  return <div className="brand"><span className="brand-mark">Φ<span className="brand-dot"/></span><span className="brand-type">PHI<span>STREAM</span><small>INFERENCE RESEARCH ENGINE</small></span></div>;
}

function SectionTitle({ eyebrow, title, description, id }) {
  return <div className="section-title" id={id}><div className="eyebrow"><span className="eyebrow-line"/>{eyebrow}</div><h2>{title}</h2><p>{description}</p></div>;
}

function StatusPill({ children, variant = 'teal' }) {
  return <span className={'pill pill-' + variant}><span className="pill-dot"/>{children}</span>;
}

function ReceiptCard({ label, value, foot, icon, accent = '' }) {
  return <div className={'receipt-card ' + accent}><div className="receipt-head"><span>{label}</span><Icon name={icon} size={19}/></div><strong>{value}</strong><small>{foot}</small></div>;
}

function ArchitectureGraphic() {
  return <div className="architecture-graphic">
    <div className="arch-lane arch-lane-storage"><span className="lane-tag">01 / STORAGE</span><Icon name="database" size={24}/><strong>On-disk weights</strong><small>Memory-mapped file</small></div>
    <span className="arch-connector"><span/><Icon name="arrow" size={17}/></span>
    <div className="arch-lane arch-lane-memory"><span className="lane-tag">02 / HOST</span><Icon name="layers" size={24}/><strong>Selected layer</strong><small>Stage to compute device</small></div>
    <span className="arch-connector"><span/><Icon name="arrow" size={17}/></span>
    <div className="arch-lane arch-lane-gpu"><span className="lane-tag">03 / COMPUTE</span><Icon name="cpu" size={24}/><strong>Active transformer</strong><small>Execute, release, advance</small></div>
  </div>;
}

function Planner() {
  const [model, setModel] = useState(70);
  const [precision, setPrecision] = useState(2);
  const [vram, setVram] = useState(12);
  const [reserve, setReserve] = useState(3);
  const weightsGB = useMemo(() => model * precision, [model, precision]);
  const persistentBudget = Math.max(0, vram - reserve);
  const storageMultiple = weightsGB / Math.max(vram, 1);
  const modelDetails = MODELS.find(m => m.params === model);
  return <div className="planner-card">
    <div className="planner-top"><div><div className="micro-label"><Icon name="sliders" size={14}/> CAPACITY EXPLORER</div><h3>Size up the impossible.</h3><p>Explore weight storage versus available GPU memory. These are arithmetic estimates, <b>not running-model benchmarks or fit guarantees.</b></p></div><StatusPill variant="amber">ESTIMATE ONLY</StatusPill></div>
    <div className="planner-layout">
      <div className="planner-controls">
        <div className="control"><div className="control-label"><span>Model parameter count</span><span>{modelDetails.layers}</span></div><div className="segment-row">{MODELS.map(m => <button type="button" key={m.params} className={model === m.params ? 'segment active' : 'segment'} onClick={() => setModel(m.params)} aria-pressed={model === m.params}>{m.label}</button>)}</div></div>
        <div className="control"><div className="control-label"><span>Weight precision</span><span>Storage approximation</span></div><div className="segment-row">{PRECISIONS.map(p => <button type="button" key={p.label} className={precision === p.bytes ? 'segment active' : 'segment'} onClick={() => setPrecision(p.bytes)} aria-pressed={precision === p.bytes}>{p.label}</button>)}</div></div>
        <div className="control"><div className="control-label"><span>Available GPU VRAM</span><b>{vram} GB</b></div><input aria-label="Available GPU VRAM in gigabytes" type="range" min="4" max="24" step="4" value={vram} onChange={e => { const value = Number(e.target.value); setVram(value); setReserve(Math.min(reserve, Math.max(1, value - 1))); }}/><div className="range-labels"><span>4 GB</span><span>12 GB</span><span>24 GB</span></div></div>
        <div className="control"><div className="control-label"><span>Illustrative runtime reserve</span><b>{reserve} GB</b></div><input aria-label="Illustrative runtime reserve in gigabytes" type="range" min="1" max={Math.max(1, vram - 1)} value={Math.min(reserve, vram - 1)} step="1" onChange={e => setReserve(Number(e.target.value))}/><div className="control-caption">For activations, KV cache, CUDA context and temporary tensors. Actual requirements vary.</div></div>
      </div>
      <div className="planner-output">
        <div className="output-kicker">THEORETICAL MODEL WEIGHT STORAGE</div>
        <div className="output-number">{pretty(weightsGB, weightsGB < 50 ? 1 : 0)}<span>GB</span></div>
        <div className="output-sub">Approx. {pretty(storageMultiple, 1)}× your selected VRAM capacity</div>
        <div className="capacity-chart">
          <div className="capacity-label"><span>GPU capacity</span><b>{vram} GB</b></div>
          <div className="capacity-track"><div className="capacity-reserve" style={{ width: pretty(Math.min(100, reserve / vram * 100), 2) + '%' }} /><div className="capacity-cache" style={{ width: pretty(Math.min(100, persistentBudget / vram * 100), 2) + '%' }}/></div>
          <div className="legend"><span><i className="legend-available"/>Illustrative available {persistentBudget} GB</span><span><i className="legend-reserved"/>Reserved {reserve} GB</span></div>
        </div>
        <div className="output-note"><Icon name="shield" size={18}/><span>Layer streaming may reduce <b>resident weight memory</b>. It does not eliminate active-layer, KV-cache, bandwidth, RAM or disk requirements.</span></div>
      </div>
    </div>
  </div>;
}

function Benchmarks() {
  const [receipts, setReceipts] = useState(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    let active = true;
    const base = import.meta.env.BASE_URL || '/';
    Promise.all(['stream', 'resident'].map(mode =>
      fetch(base + 'benchmarks/build-cpu-' + mode + '.json').then(r => { if (!r.ok) throw Error('Receipt unavailable'); return r.json(); })
    )).then(([stream, resident]) => { if (active) setReceipts({ stream, resident }); }).catch(() => { if (active) setError(true); });
    return () => { active = false; };
  }, []);
  const max = Math.max(receipts?.stream?.generated_tokens_per_second_end_to_end || 0, receipts?.resident?.generated_tokens_per_second_end_to_end || 0, 1);
  return <div className="evidence-grid">
    <div className="evidence-panel">
      <div className="panel-head"><span className="micro-label"><Icon name="activity" size={15}/> RECORDED EVIDENCE</span><StatusPill>CPU · SYNTHETIC</StatusPill></div>
      <h3>Stream vs resident</h3><p className="panel-sub">End-to-end tokens/second, from committed CPU fixture receipts. Higher is faster, but this tiny test cannot predict real-model performance.</p>
      {receipts ? <div className="benchmark-bars">{[['stream','STREAMED'],['resident','RESIDENT']].map(([key,label]) => <div className="bar-row" key={key}><div className="bar-heading"><span>{label}</span><b>{pretty(receipts[key].generated_tokens_per_second_end_to_end, 1)} tok/s</b></div><div className="bar-track"><div className={'bar-fill ' + key} style={{width: pretty(receipts[key].generated_tokens_per_second_end_to_end / max * 100, 2) + '%'}}/></div></div>)}</div> : <div className="receipt-loading">{error ? 'Receipt data unavailable. View the originals on GitHub.' : 'Loading verified repository receipts…'}</div>}
      <div className="evidence-footer"><span>Source: Oct 10–11, 2026 · Linux CPU · untrained 4-layer fixture</span><a href={REPO + '/tree/main/benchmarks'} target="_blank" rel="noreferrer">View receipts <Icon name="external" size={14}/></a></div>
    </div>
    <div className="evidence-stats">
      <ReceiptCard label="STREAMED WEIGHT LOADS" value={receipts ? fmt(receipts.stream.weight_load_events) : '—'} foot="Load events in synthetic CPU fixture" icon="layers" />
      <ReceiptCard label="RESIDENT WEIGHT LOADS" value={receipts ? fmt(receipts.resident.weight_load_events) : '—'} foot="Same fixture, resident execution" icon="database" />
      <ReceiptCard label="PARITY CHECK" value="0.0" foot="Reported maximum logit diff on CPU fixture" icon="check" accent="accent-green" />
      <ReceiptCard label="GPU BENCHMARK" value="PENDING" foot="RTX 5070 validation not yet measured" icon="cpu" accent="accent-amber" />
    </div>
  </div>;
}

const STEPS = [
  { index: '01', state: 'COMPLETE', title: 'Independent inference core', text: 'Toy decoder, memory-mapped weights, layer streaming, resident reference, integrity receipts.', status: 'complete' },
  { index: '02', state: 'NEXT', title: 'CUDA verification', text: 'Test numerics, allocator use, and real host-to-device transfers on the RTX 5070.', status: 'next' },
  { index: '03', state: 'PLANNED', title: 'Checkpoint compatibility', text: 'Safetensors mapping and real model validation with explicit format and license checks.', status: 'future' },
  { index: '04', state: 'RESEARCH', title: 'Adaptive layer residency', text: 'Benchmark-driven GPU/RAM/SSD placement with transparent budgets and fallbacks.', status: 'future' },
  { index: '05', state: 'RESEARCH', title: 'Vessie / BrainC integration', text: 'Optional audited inference routing with BudgetGenius cost and speed receipts.', status: 'future' }
];

export default function App() {
  const [menu, setMenu] = useState('overview');
  const navigate = id => { setMenu(id); document.getElementById(id)?.scrollIntoView({behavior:'smooth',block:'start'}); };
  return <div className="app">
    <div className="ambient ambient-a"/><div className="ambient ambient-b"/>
    <header className="site-header"><div className="container header-inner">
      <a href={import.meta.env.BASE_URL} aria-label="PhiStream homepage"><Logo/></a>
      <nav aria-label="Main navigation">{[['overview','Overview'],['architecture','Architecture'],['evidence','Evidence'],['roadmap','Roadmap']].map(([id,label])=><button key={id} type="button" className={menu===id ? 'nav-link selected':'nav-link'} onClick={()=>navigate(id)}>{label}</button>)}</nav>
      <a className="header-github" href={REPO} target="_blank" rel="noreferrer"><Icon name="github" size={18}/><span>GitHub</span><Icon name="external" size={13}/></a>
    </div></header>

    <main>
      <section className="hero container" id="overview">
        <div className="hero-copy">
          <div className="hero-label"><span className="pulse-indicator"/> OPEN RESEARCH <span className="slash">/</span> V0.1.0 EXPERIMENTAL</div>
          <h1>Big models.<br/><span>Smaller footprint.</span><br/>Real receipts.</h1>
          <p>ΦStream explores memory-tiered inference: moving model layers through limited GPU memory, measuring every tradeoff, and keeping claims accountable to evidence.</p>
          <div className="hero-buttons"><button type="button" className="primary-button" onClick={()=>navigate('planner')}>Explore capacity <Icon name="arrow" size={18}/></button><a href={REPO} target="_blank" rel="noreferrer" className="secondary-button"><Icon name="github" size={18}/> View source</a></div>
          <div className="hero-disclaimer"><Icon name="shield" size={16}/><span>CPU prototype verified · Real-model support not yet available · No 70B performance claim</span></div>
        </div>
        <div className="hero-visual" aria-label="Conceptual layer streaming visualization">
          <div className="orbit orbit-outer"/><div className="orbit orbit-mid"/><div className="orbit orbit-inner"/>
          <div className="orbit-crosshair crosshair-a"/><div className="orbit-crosshair crosshair-b"/>
          <div className="core"><span className="core-letter">Φ</span><span>STREAM</span></div>
          <div className="orbital-node node-top"><i/><span>SSD / WEIGHTS</span></div>
          <div className="orbital-node node-right"><i/><span>GPU / COMPUTE</span></div>
          <div className="orbital-node node-bottom"><i/><span>CPU / STAGING</span></div>
          <div className="visual-corner visual-corner-one">PHI / 01</div><div className="visual-corner visual-corner-two">LAYER-WISE ENGINE</div>
        </div>
      </section>

      <div className="stripe-band"><div className="container stripe-inner"><span><i className="stripe-square"/> ORIGINAL ENGINE</span><span><i className="stripe-square"/> NO MODEL WEIGHTS SHIPPED</span><span><i className="stripe-square"/> JSON RECEIPTS</span><span><i className="stripe-square"/> LOCAL-FIRST RESEARCH</span></div></div>

      <section className="section container" id="architecture">
        <SectionTitle eyebrow="HOW IT WORKS / 01" title="Move the layers. Keep the proof." description="A deliberately small, independently written inference core built to study the cost of moving weights between memory tiers." />
        <ArchitectureGraphic/>
        <div className="feature-grid">
          <article className="feature"><div className="feature-icon"><Icon name="layers" size={21}/></div><h3>One layer at a time</h3><p>Load weights, execute attention and feed-forward operations, release unpinned parameters, advance.</p></article>
          <article className="feature"><div className="feature-icon"><Icon name="database" size={21}/></div><h3>Budgeted residency</h3><p>Pin selected parameters while tracking weight-load events. A pin budget is not a total VRAM guarantee.</p></article>
          <article className="feature"><div className="feature-icon"><Icon name="shield" size={21}/></div><h3>Evidence first</h3><p>Compare against resident reference mode and record machine, configuration, throughput, and provenance.</p></article>
        </div>
      </section>

      <section className="section container" id="planner">
        <SectionTitle eyebrow="CAPACITY / 02" title="What if the model is bigger than the GPU?" description="Experiment with memory arithmetic before touching your hardware."/>
        <Planner/>
      </section>

      <section className="section container" id="evidence">
        <SectionTitle eyebrow="BENCHMARKS / 03" title="Numbers with a paper trail." description="Real receipts from the repository, and no made-up CUDA results."/>
        <Benchmarks/>
      </section>

      <section className="section container roadmap-section" id="roadmap">
        <SectionTitle eyebrow="WHAT'S NEXT / 04" title="Built in stages. Verified at each one." description="Experimental milestones with deliberately honest status labels."/>
        <div className="roadmap-list">{STEPS.map(step=><div className="roadmap-item" key={step.index}><span className={'step-index '+ step.status}>{step.index}</span><div className="step-content"><div className="step-title-row"><h3>{step.title}</h3><span className={'step-state '+step.status}>{step.state}</span></div><p>{step.text}</p></div><Icon name="chevron" size={17}/></div>)}</div>
      </section>

      <section className="cta container"><div className="cta-panel"><span className="eyebrow">OPEN SOURCE / OPEN EVIDENCE</span><h2>Build the engine.<br/><span>Measure the truth.</span></h2><p>Explore the independently developed Python source, reproduce the synthetic fixture tests, and follow the road toward real CUDA validation.</p><div className="cta-links"><a href={REPO} target="_blank" rel="noreferrer" className="primary-button">Explore on GitHub <Icon name="external" size={17}/></a><a href={REPO + '/blob/main/docs/ROADMAP.md'} target="_blank" rel="noreferrer" className="text-link">Read research roadmap <Icon name="arrow" size={16}/></a></div></div></section>
    </main>
    <footer className="footer"><div className="container footer-inner"><Logo/><div><p>© 2026 ΦStream · MichaelWave369 · MIT Licensed</p><p>Research prototype. All capacity forecasts are illustrative, not measured GPU performance.</p></div><a href={REPO} target="_blank" rel="noreferrer"><Icon name="github" size={20}/></a></div></footer>
  </div>;
}
