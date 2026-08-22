'use client';

import { useEffect, useState } from 'react';
import { useEvidenceUrl } from './EvidenceMode';

type Checkpoint = { label: string; step: number; trainLoss: number | null; validationLoss: number | null; sample: string };
type Metrics = { status: string; runId: string; model: string; parameters: number | null; updatedAt: string; checkpoints: Checkpoint[] };

const empty: Metrics = { status: 'Not run yet', runId: 'shakespeare-bigram-001', model: 'Character bigram', parameters: null, updatedAt: '', checkpoints: [] };

export default function LiveMetrics() {
  const [loaded, setLoaded] = useState<{ url: string; data: Metrics }>({ url: '', data: empty });
  const evidenceUrl = useEvidenceUrl('shakespeare-metrics.json');
  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try { const response = await fetch(`${evidenceUrl}?t=${Date.now()}`, { cache: 'no-store' }); if (active) setLoaded({ url: evidenceUrl, data: response.ok ? await response.json() : empty }); } catch { if (active) setLoaded({ url: evidenceUrl, data: empty }); }
    };
    refresh(); const timer = window.setInterval(refresh, 2000); return () => { active = false; window.clearInterval(timer); };
  }, [evidenceUrl]);
  const metrics = loaded.url === evidenceUrl ? loaded.data : empty;
  const losses = metrics.checkpoints.map(c => c.validationLoss).filter((v): v is number => v !== null);
  const max = Math.max(...losses, 1); const min = Math.min(...losses, 0);
  return <section className="metrics-panel" id="live-results">
    <div className="metrics-head"><div><p className="kicker">LIVE EXPERIMENT VIEW</p><h2>What the model says as it learns</h2></div><span className={`run-state ${metrics.status.toLowerCase().replaceAll(' ', '-')}`}><span className="live-dot" />{metrics.status}</span></div>
    <div className="metric-summary"><div><span>RUN</span><strong>{metrics.runId}</strong></div><div><span>MODEL</span><strong>{metrics.model}</strong></div><div><span>PARAMETERS</span><strong>{metrics.parameters?.toLocaleString() ?? 'Not built yet'}</strong></div><div><span>CHECKPOINTS</span><strong>{metrics.checkpoints.length}</strong></div></div>
    {losses.length > 0 && <div className="loss-chart" aria-label="Validation loss by checkpoint"><div className="chart-label">VALIDATION LOSS · LOWER IS BETTER</div><div className="loss-bars">{metrics.checkpoints.map((checkpoint) => checkpoint.validationLoss === null ? null : <div className="loss-bar-wrap" key={checkpoint.label}><span className="loss-value">{checkpoint.validationLoss.toFixed(3)}</span><span className="loss-bar" style={{ height: `${26 + ((max - checkpoint.validationLoss) / Math.max(max - min, .001)) * 74}%` }} /><small>{checkpoint.step}</small></div>)}</div></div>}
    <div className="checkpoint-grid">{metrics.checkpoints.length ? metrics.checkpoints.map(checkpoint => <article className="checkpoint" key={checkpoint.label}><div><span>{checkpoint.label}</span><strong>step {checkpoint.step}</strong></div><pre>{checkpoint.sample}</pre><p>{checkpoint.validationLoss === null ? 'Random weights: no training has happened.' : `Validation loss ${checkpoint.validationLoss.toFixed(4)}`}</p></article>) : <div className="empty-state"><strong>The experiment has not run yet.</strong><p>When training begins, this panel refreshes every two seconds. The first saved sample will come from random weights.</p></div>}</div>
  </section>;
}
