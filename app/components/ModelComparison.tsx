'use client';

import { useEffect, useState } from 'react';
import { useEvidenceMode } from './EvidenceMode';

type Checkpoint = { label: string; step: number; trainLoss: number | null; validationLoss: number | null; sample: string };
type Run = { status: string; runId: string; model: string; parameters: number | null; contextLength?: number; elapsedSeconds?: number; checkpoints: Checkpoint[] };

const sources = [
  { key: 'bigram', label: 'Bigram', file: 'shakespeare-metrics.json', context: 1 },
  { key: 'context', label: 'Fixed context', file: 'shakespeare-context-metrics.json', context: 8 },
  { key: 'attention', label: 'Self-attention', file: 'shakespeare-attention-metrics.json', context: 64 },
  { key: 'transformer', label: 'Tiny transformer', file: 'shakespeare-transformer-metrics.json', context: 64 },
];

export default function ModelComparison() {
  const { mode } = useEvidenceMode();
  const [loaded, setLoaded] = useState<{ mode: string; runs: Record<string, Run> }>({ mode: '', runs: {} });
  useEffect(() => {
    let active = true;
    const refresh = async () => {
      const loaded = await Promise.all(sources.map(async source => {
        try { const response = await fetch(`/data/${mode}/${source.file}?t=${Date.now()}`, { cache: 'no-store' }); return [source.key, response.ok ? await response.json() : null] as const; }
        catch { return [source.key, null] as const; }
      }));
      if (active) setLoaded({ mode, runs: Object.fromEntries(loaded.filter(([, run]) => run)) });
    };
    refresh(); const timer = window.setInterval(refresh, 2000); return () => { active = false; window.clearInterval(timer); };
  }, [mode]);
  const runs = loaded.mode === mode ? loaded.runs : {};

  return <section className="comparison-panel">
    <div className="comparison-heading"><div><p className="kicker">ARCHITECTURE COMPARISON</p><h2>What changed when memory grew?</h2></div><p>Same corpus and character vocabulary. Lower validation loss is better; parameter count and training recipe also changed, so this is a learning comparison rather than a controlled benchmark.</p></div>
    <div className="comparison-grid">{sources.map(source => {
      const run = runs[source.key]; const first = run?.checkpoints[0]; const final = run?.checkpoints.at(-1);
      return <article key={source.key} className={`comparison-card comparison-${source.key}`}>
        <div className="comparison-card-head"><span>{source.label}</span><strong>{run?.status ?? (loaded.mode === mode ? 'Not run yet' : 'Loading')}</strong></div>
        <dl><div><dt>Context</dt><dd>{run?.contextLength ?? source.context} character{source.context === 1 ? '' : 's'}</dd></div><div><dt>Parameters</dt><dd>{run?.parameters?.toLocaleString() ?? '—'}</dd></div><div><dt>Final validation loss</dt><dd>{final?.validationLoss?.toFixed(4) ?? '—'}</dd></div><div><dt>Loss reduction</dt><dd>{first?.validationLoss != null && final?.validationLoss != null ? (first.validationLoss - final.validationLoss).toFixed(4) : '—'}</dd></div><div><dt>Recorded training</dt><dd>{run?.elapsedSeconds != null ? `${run.elapsedSeconds.toFixed(2)} s` : 'Not recorded'}</dd></div></dl>
        <pre>{final?.sample ?? 'Waiting for a checkpoint…'}</pre>
      </article>;
    })}</div>
    <aside className="comparison-conclusion"><strong>Architecture result</strong><p>The original complete transformer won this architecture comparison at 1.7267. A later controlled training-recipe experiment kept that architecture fixed and improved frozen-protocol validation loss to 1.7051 with warmup plus cosine decay.</p></aside>
  </section>;
}
