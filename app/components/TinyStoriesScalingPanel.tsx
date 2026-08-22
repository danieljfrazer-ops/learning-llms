'use client';

import { useEffect, useState } from 'react';
import { useEvidenceUrl } from './EvidenceMode';
import styles from './TinyStoriesScalingPanel.module.css';

type Probe = {
  candidate: { id: string; label: string; axis: string; contextSize: number };
  parameterCount: number; trainingRows: number; feasible?: boolean;
  finalValidation: { crossEntropyLoss: number };
  performance: { meanUpdateMilliseconds: number; peakMlxAllocationBytes: number; validTargetsPerSecond: number };
};
type Evidence = {
  status: string;
  progress: { phase: string; update: number; plannedUpdates: number; latestTrainLoss?: number };
  configuration: { resourceCeilings: { peakMlxAllocationBytes: number; meanUpdateMilliseconds: number }; selectionRule: { minimumImprovementOverReference: number } };
  probes: Probe[];
  selection?: { selectedCandidateId: string; selectedLabel: string; reason: string; selectedCheckpointDirectory: string };
  longRun?: { finalValidation: { crossEntropyLoss: number }; performance: { meanUpdateMilliseconds: number; peakMlxAllocationBytes: number }; samples: { prompt: string; continuation: string }[] };
  performance?: { wholeExperimentElapsedSeconds: number };
};

const mib = (bytes: number) => `${(bytes / 1024 / 1024).toFixed(0)} MiB`;

export default function TinyStoriesScalingPanel() {
  const evidenceUrl = useEvidenceUrl('tinystories-scaling-budget.json');
  const [loaded, setLoaded] = useState<{ url: string; data: Evidence | null }>({ url: '', data: null });

  useEffect(() => {
    let active = true;
    const load = () => fetch(`${evidenceUrl}?t=${Date.now()}`, { cache: 'no-store' })
      .then(response => response.ok ? response.json() : null)
      .then(data => { if (active) setLoaded({ url: evidenceUrl, data }); })
      .catch(() => { if (active) setLoaded({ url: evidenceUrl, data: null }); });
    load();
    const timer = window.setInterval(load, 2_000);
    return () => { active = false; window.clearInterval(timer); };
  }, [evidenceUrl]);

  const evidence = loaded.url === evidenceUrl ? loaded.data : null;
  return <section className="evaluation-panel scale-budget-panel" aria-label="TinyStories laptop scaling evidence">
    <div className="evaluation-head"><div><p className="kicker">LESSON 09 · LAPTOP BUDGET</p><h2>{evidence?.selection ? 'The existing 5.82M model earned the longer run' : 'Spend compute only after measuring the trade-off'}</h2></div><span>{evidence?.status ?? (loaded.url === evidenceUrl ? 'Not run yet' : 'Loading evidence…')}</span></div>
    <div className="evaluation-summary">
      <div><small>PHASE</small><strong>{evidence?.progress.phase ?? '—'}</strong></div>
      <div><small>UPDATE</small><strong>{evidence ? `${evidence.progress.update} / ${evidence.progress.plannedUpdates}` : '—'}</strong></div>
      <div><small>SELECTED</small><strong>{evidence?.selection?.selectedCandidateId ?? '—'}</strong></div>
      <div><small>LONG-RUN LOSS</small><strong>{evidence?.longRun?.finalValidation.crossEntropyLoss.toFixed(4) ?? '—'}</strong></div>
    </div>
    {evidence ? <>
      <div className={styles.probes}>{evidence.probes.map(probe => {
        const selected = evidence.selection?.selectedCandidateId === probe.candidate.id;
        const overSpeed = probe.performance.meanUpdateMilliseconds > evidence.configuration.resourceCeilings.meanUpdateMilliseconds;
        return <article key={probe.candidate.id} className={selected ? styles.selected : ''}>
          <small>{selected ? 'SELECTED CONTROL' : probe.candidate.axis.toUpperCase()}</small>
          <h3>{probe.candidate.label}</h3>
          <dl>
            <div><dt>Parameters</dt><dd>{(probe.parameterCount / 1_000_000).toFixed(2)}M</dd></div>
            <div><dt>Stories</dt><dd>{probe.trainingRows.toLocaleString()}</dd></div>
            <div><dt>Loss after 100</dt><dd>{probe.finalValidation.crossEntropyLoss.toFixed(4)}</dd></div>
            <div><dt>Mean update</dt><dd>{probe.performance.meanUpdateMilliseconds.toFixed(0)} ms{overSpeed ? ' · over ceiling' : ''}</dd></div>
            <div><dt>Peak MLX</dt><dd>{mib(probe.performance.peakMlxAllocationBytes)}</dd></div>
          </dl>
        </article>;
      })}</div>
      {evidence.selection && <div className={styles.verdict}>
        <article><small>DECISION RULE</small><strong>{evidence.selection.selectedLabel}</strong><p>{evidence.selection.reason}. An alternative needed at least {evidence.configuration.selectionRule.minimumImprovementOverReference.toFixed(2)} lower loss and had to stay below both resource ceilings.</p></article>
        <article><small>LONGER CONFIRMATION</small><strong>{evidence.longRun?.finalValidation.crossEntropyLoss.toFixed(4) ?? 'RUNNING'}</strong><p>Fresh random weights received 700 updates: constant 3e-4 for 600, then the Lesson 8 cosine decay for 100.</p></article>
      </div>}
      {evidence.longRun && <blockquote className={styles.sample}><strong>{evidence.longRun.samples[0].prompt}</strong>{evidence.longRun.samples[0].continuation}</blockquote>}
      <p className="evaluation-foot">{evidence.performance ? `The complete experiment took ${evidence.performance.wholeExperimentElapsedSeconds.toFixed(1)} seconds.` : 'The dashboard refreshes every ten updates.'} Probe timing is local evidence from this passively cooled MacBook, not a universal hardware benchmark.</p>
    </> : <p className="evaluation-foot">Run the Lesson 9 command in My Lab to populate five fresh-start probes, their cost/quality frontier, the selection record, and one complete long-run checkpoint.</p>}
  </section>;
}
