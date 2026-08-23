'use client';

import { useEffect, useState } from 'react';
import { useEvidenceUrl } from './EvidenceMode';

type Variant = {
  runId: string;
  factor: string;
  label: string;
  parameterCount: number;
  elapsedSeconds: number;
  peakMetalMemoryBytes: number;
  evaluation: { validationLossMean: number; validationLossStd: number; perplexity: number; continuation: string };
};
type Evidence = { status: string; selectedRunId: string; variants: Variant[] };

export default function ScalingPanel() {
  const evidenceUrl = useEvidenceUrl('shakespeare-scaling.json');
  const [loaded, setLoaded] = useState<{ url: string; data: Evidence | null }>({ url: '', data: null });
  useEffect(() => {
    fetch(`${evidenceUrl}?t=${Date.now()}`, { cache: 'no-store' })
      .then(response => response.ok ? response.json() : null)
      .then(data => setLoaded({ url: evidenceUrl, data }))
      .catch(() => setLoaded({ url: evidenceUrl, data: null }));
  }, [evidenceUrl]);
  const evidence = loaded.url === evidenceUrl ? loaded.data : null;

  return <section className="recipe-panel">
    <div className="evaluation-head"><div><p className="kicker">SCALING DASHBOARD</p><h2>{evidence ? 'Width, depth, and context expose different cost–quality trade-offs' : 'Compare width, depth, and context under one protocol'}</h2></div><span>{evidence?.status ?? (loaded.url === evidenceUrl ? 'Not run yet' : 'Loading evidence…')}</span></div>
    <div className="scaling-grid">{evidence?.variants.map(variant => {
      const selected = variant.runId === evidence.selectedRunId;
      return <article className={selected ? 'recipe-winner' : ''} key={variant.runId}>
        <span>{selected ? 'SELECTED · ' : ''}{variant.factor}</span><h3>{variant.label}</h3>
        <dl><div><dt>Parameters</dt><dd>{variant.parameterCount.toLocaleString()}</dd></div><div><dt>Validation loss</dt><dd>{variant.evaluation.validationLossMean.toFixed(4)} ± {variant.evaluation.validationLossStd.toFixed(4)}</dd></div><div><dt>Perplexity</dt><dd>{variant.evaluation.perplexity.toFixed(2)}</dd></div><div><dt>Peak accelerator memory</dt><dd>{(variant.peakMetalMemoryBytes / 1e6).toFixed(0)} MB</dd></div><div><dt>Run time</dt><dd>{variant.elapsedSeconds.toFixed(1)} s</dd></div></dl>
        <pre aria-label="Continuation from the shared frozen prompt and sampling seed">{variant.evaluation.continuation}</pre>
      </article>;
    })}</div>
    <p className="evaluation-foot">The selected highlight comes from the recorded <code>selectedRunId</code>, not card order. Samples use the same frozen generation protocol; validation loss, memory, runtime, and parameter count expose different parts of the trade-off.</p>
  </section>;
}
