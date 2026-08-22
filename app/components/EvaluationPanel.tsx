'use client';

import { useEffect, useState } from 'react';
import { useEvidenceUrl } from './EvidenceMode';

type Checkpoint = { step: number; trainLossMean: number; validationLossMean: number; validationLossStd: number; generalisationGap: number; perplexity: number };
type Evaluation = { status: string; evaluationId: string; elapsedSeconds: number; protocol: { repeats: number; predictionsPerRepeat: number }; checkpoints: Checkpoint[] };

export default function EvaluationPanel() {
  const [loaded, setLoaded] = useState<{ url: string; data: Evaluation | null }>({ url: '', data: null });
  const evidenceUrl = useEvidenceUrl('shakespeare-evaluation.json');
  useEffect(() => {
    fetch(`${evidenceUrl}?t=${Date.now()}`, { cache: 'no-store' })
      .then(response => response.ok ? response.json() : null).then(data => setLoaded({ url: evidenceUrl, data })).catch(() => setLoaded({ url: evidenceUrl, data: null }));
  }, [evidenceUrl]);
  const evaluation = loaded.url === evidenceUrl ? loaded.data : null;
  const first = evaluation?.checkpoints[0];
  const final = evaluation?.checkpoints.at(-1);
  return <section className="evaluation-panel" id="evaluation-dashboard">
    <div className="evaluation-head"><div><p className="kicker">FROZEN EVALUATION DASHBOARD</p><h2>Learning improved prediction—and widened the gap</h2></div><span>{evaluation?.status ?? (loaded.url === evidenceUrl ? 'Not run yet' : 'Loading evidence…')}</span></div>
    <div className="evaluation-summary"><div><small>FINAL VALIDATION LOSS</small><strong>{final?.validationLossMean.toFixed(4) ?? '—'}</strong></div><div><small>FINAL PERPLEXITY</small><strong>{final?.perplexity.toFixed(2) ?? '—'}</strong></div><div><small>GENERALISATION GAP</small><strong>{final?.generalisationGap.toFixed(4) ?? '—'}</strong></div><div><small>EVALUATION TIME</small><strong>{evaluation ? `${evaluation.elapsedSeconds.toFixed(2)} s` : '—'}</strong></div></div>
    <div className="evaluation-rows">{evaluation?.checkpoints.map(checkpoint => <article key={checkpoint.step}><div><strong>step {checkpoint.step.toLocaleString()}</strong><span>validation {checkpoint.validationLossMean.toFixed(4)} ± {checkpoint.validationLossStd.toFixed(4)}</span></div><div className="evaluation-track"><i style={{ width: `${Math.max(5, ((first!.validationLossMean - checkpoint.validationLossMean) / (first!.validationLossMean - final!.validationLossMean)) * 100)}%` }} /></div><small>perplexity {checkpoint.perplexity.toFixed(2)} · gap {checkpoint.generalisationGap.toFixed(4)}</small></article>)}</div>
    <p className="evaluation-foot">Five repeats × {evaluation?.protocol.predictionsPerRepeat.toLocaleString() ?? '40,960'} predictions per repeat and split. Bars show relative validation-loss improvement from random weights to the final checkpoint.</p>
  </section>;
}
