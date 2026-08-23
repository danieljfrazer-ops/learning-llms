'use client';

import { useEffect, useState } from 'react';
import { useEvidenceUrl } from './EvidenceMode';

type SeedResult = { seed: number; runId: string; validationLossMean: number; validationLossStd: number; perplexity: number; continuation: string };
type Evidence = { status: string; validationLossAcrossSeedsMean: number; validationLossAcrossSeedsStd: number; perplexityFromMeanLoss: number; selectedRunId: string; seeds: SeedResult[] };

export default function FinalModelPanel() {
  const evidenceUrl = useEvidenceUrl('shakespeare-final.json');
  const [loaded, setLoaded] = useState<{ url: string; data: Evidence | null }>({ url: '', data: null });
  useEffect(() => {
    fetch(`${evidenceUrl}?t=${Date.now()}`, { cache: 'no-store' })
      .then(response => response.ok ? response.json() : null)
      .then(data => setLoaded({ url: evidenceUrl, data }))
      .catch(() => setLoaded({ url: evidenceUrl, data: null }));
  }, [evidenceUrl]);
  const evidence = loaded.url === evidenceUrl ? loaded.data : null;
  const selected = evidence?.seeds.find(seed => seed.runId === evidence.selectedRunId);
  const lossRange = evidence?.seeds.length
    ? Math.max(...evidence.seeds.map(seed => seed.validationLossMean)) - Math.min(...evidence.seeds.map(seed => seed.validationLossMean))
    : null;

  return <section className="evaluation-panel">
    <div className="evaluation-head"><div><p className="kicker">FINAL THREE-SEED CONFIRMATION</p><h2>{lossRange == null ? 'Repeat the selected architecture across independent training paths' : `Three independent paths span ${lossRange.toFixed(4)} validation loss`}</h2></div><span>{evidence?.status ?? (loaded.url === evidenceUrl ? 'Not run yet' : 'Loading evidence…')}</span></div>
    <div className="evaluation-summary"><div><small>MEAN VALIDATION LOSS</small><strong>{evidence?.validationLossAcrossSeedsMean.toFixed(4) ?? '—'}</strong></div><div><small>BETWEEN-SEED STD</small><strong>{evidence?.validationLossAcrossSeedsStd.toFixed(4) ?? '—'}</strong></div><div><small>MEAN-LOSS PERPLEXITY</small><strong>{evidence?.perplexityFromMeanLoss.toFixed(2) ?? '—'}</strong></div><div><small>SELECTED SEED</small><strong>{selected?.seed ?? '—'}</strong></div></div>
    <div className="recipe-grid">{evidence?.seeds.map(seed => <article className={seed.runId === evidence.selectedRunId ? 'recipe-winner' : ''} key={seed.seed}><div><span>{seed.runId === evidence.selectedRunId ? 'SELECTED CHECKPOINT' : 'CONFIRMATION RUN'}</span><h3>Training seed {seed.seed}</h3></div><dl><div><dt>Validation loss</dt><dd>{seed.validationLossMean.toFixed(4)} ± {seed.validationLossStd.toFixed(4)}</dd></div><div><dt>Perplexity</dt><dd>{seed.perplexity.toFixed(2)}</dd></div></dl><pre aria-label="Continuation from the shared frozen prompt and sampling seed">{seed.continuation}</pre></article>)}</div>
    <p className="evaluation-foot">The selected-seed label is read from the evidence file. Every card shows the same prompt and sampling setup, so visible wording differences come from independently trained weights rather than a changed demonstration.</p>
  </section>;
}
