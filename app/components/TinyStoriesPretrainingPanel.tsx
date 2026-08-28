'use client';

import { useEffect, useMemo, useState } from 'react';
import { useEvidenceUrl } from './EvidenceMode';

type Sample = { prompt: string; continuation: string; generatedTokens: number; endedWithEos: boolean };
type Capture = {
  step: number;
  trainBatchLoss: number | null;
  gradientL2Norm: number | null;
  validTargetsSeen: number;
  equivalentCompleteTrainingPasses: number;
  validation: { crossEntropyLoss: number; perplexity: number; validTargets: number; windows: number };
  samples: Sample[];
};
type Evidence = {
  status: string;
  progress: { completedUpdates: number; plannedUpdates: number; validTargetsSeen: number };
  configuration: {
    parameterCount: number; learningRate: number; weightDecay: number; optimiser: string;
    trainingWindows: number; trainingValidTargetsPerCompletePass: number; validationWindows: number;
  };
  checkpoints: Capture[];
  trace: { step: number; trainBatchLoss: number; gradientL2Norm: number }[];
  memory?: { peakMlxAllocationBytes: number };
  trainingPerformance?: { elapsedSecondsIncludingEvaluationAndGeneration: number; validTargetsPerSecondDuringUpdates: number; peakMlxAllocationBytes: number };
};

const mebibytes = (value: number) => `${(value / 1024 / 1024).toFixed(0)} MiB`;

export default function TinyStoriesPretrainingPanel() {
  const evidenceUrl = useEvidenceUrl('tinystories-first-pretraining.json');
  const [loaded, setLoaded] = useState<{ url: string; data: Evidence | null }>({ url: '', data: null });
  const [promptIndex, setPromptIndex] = useState(0);

  useEffect(() => {
    let active = true;
    const load = () => fetch(`${evidenceUrl}?t=${Date.now()}`, { cache: 'no-store' })
      .then(response => response.ok ? response.json() : null)
      .then(data => { if (active) setLoaded({ url: evidenceUrl, data }); })
      .catch(() => { if (active) setLoaded({ url: evidenceUrl, data: null }); });
    load();
    const interval = window.setInterval(load, 2_000);
    return () => { active = false; window.clearInterval(interval); };
  }, [evidenceUrl]);

  const evidence = loaded.url === evidenceUrl ? loaded.data : null;
  const captures = useMemo(() => evidence?.checkpoints ?? [], [evidence]);
  const latest = captures.at(-1);
  const chart = useMemo(() => {
    if (!captures.length) return '';
    const losses = captures.map(item => item.validation.crossEntropyLoss);
    const min = Math.min(...losses);
    const max = Math.max(...losses);
    return captures.map((item, index) => {
      const x = captures.length === 1 ? 0 : index / (captures.length - 1) * 100;
      const y = max === min ? 50 : (max - item.validation.crossEntropyLoss) / (max - min) * 82 + 8;
      return `${x},${y}`;
    }).join(' ');
  }, [captures]);
  const promptOptions = captures[0]?.samples ?? [];
  const peakMemory = evidence?.trainingPerformance?.peakMlxAllocationBytes ?? evidence?.memory?.peakMlxAllocationBytes;

  return <section className="evaluation-panel pretraining-panel" aria-label="TinyStories first pretraining evidence">
    <div className="evaluation-head"><div><p className="kicker">LESSON 06 · FIRST PRETRAINING</p><h2>{evidence ? 'Watch prediction error turn into weight changes' : 'Run the first complete learning loop'}</h2></div><span>{evidence?.status ?? (loaded.url === evidenceUrl ? 'Not run yet' : 'Loading evidence…')}</span></div>
    <div className="evaluation-summary">
      <div><small>UPDATES</small><strong>{evidence ? `${evidence.progress.completedUpdates} / ${evidence.progress.plannedUpdates}` : '—'}</strong></div>
      <div><small>LATEST VALIDATION LOSS</small><strong>{latest?.validation.crossEntropyLoss.toFixed(4) ?? '—'}</strong></div>
      <div><small>REAL TARGETS SEEN</small><strong>{evidence ? evidence.progress.validTargetsSeen.toLocaleString() : '—'}</strong></div>
      <div><small>PEAK MLX ALLOCATION</small><strong>{peakMemory ? mebibytes(peakMemory) : '—'}</strong></div>
    </div>
    {evidence && captures.length ? <>
      <div className="pretraining-contract">
        <article><small>WHAT STAYED FIXED</small><p>{(evidence.configuration.parameterCount / 1_000_000).toFixed(2)}M parameters · {evidence.configuration.trainingWindows.toLocaleString()} training windows · complete {evidence.configuration.validationWindows}-window validation</p></article>
        <article><small>WHAT WAS ADDED</small><p>{evidence.configuration.optimiser} · learning rate {evidence.configuration.learningRate} · weight decay {evidence.configuration.weightDecay} · masked gradients</p></article>
      </div>
      <figure className="pretraining-curve" aria-labelledby="pretraining-curve-title">
        <figcaption><strong id="pretraining-curve-title">Complete held-out loss at preserved checkpoints</strong><p>Lower is better. Each dot grades every validation target; the noisy per-batch training losses are kept separately in the evidence file.</p></figcaption>
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Validation loss falls across training checkpoints">
          <polyline points={chart} />
          {captures.map((item, index) => {
            const losses = captures.map(point => point.validation.crossEntropyLoss);
            const min = Math.min(...losses); const max = Math.max(...losses);
            const x = captures.length === 1 ? 0 : index / (captures.length - 1) * 100;
            const y = max === min ? 50 : (max - item.validation.crossEntropyLoss) / (max - min) * 82 + 8;
            return <circle key={item.step} cx={x} cy={y} r="1.6" />;
          })}
        </svg>
        <div>{captures.map(item => <span key={item.step}><b>step {item.step}</b><strong>{item.validation.crossEntropyLoss.toFixed(3)}</strong><small>{item.equivalentCompleteTrainingPasses.toFixed(2)} data passes</small></span>)}</div>
      </figure>
      <div className="pretraining-prompt-select" aria-label="Select a fixed prompt">
        {promptOptions.map((sample, index) => <button className={index === promptIndex ? 'selected' : ''} onClick={() => setPromptIndex(index)} key={sample.prompt}>{sample.prompt}</button>)}
      </div>
      <div className="pretraining-samples">{captures.map(item => {
        const sample = item.samples[promptIndex];
        return <article key={item.step}><small>STEP {item.step} · LOSS {item.validation.crossEntropyLoss.toFixed(3)}</small><strong>{sample?.prompt}</strong><p>{sample?.continuation || '〈no visible continuation before EOS〉'}</p><span>{item.step === 0 ? 'Random control' : `${item.validTargetsSeen.toLocaleString()} targets used for updates`}</span></article>;
      })}</div>
      <p className="evaluation-foot">The latest complete validation uses {latest?.validation.validTargets.toLocaleString()} untrained-on targets. A “data pass” is targets seen divided by the frozen training-target count; shuffled batches can end mid-pass. {evidence.trainingPerformance ? `The full run, including evaluation and generation, took ${evidence.trainingPerformance.elapsedSecondsIncludingEvaluationAndGeneration.toFixed(1)} seconds.` : 'This run is still in progress.'}</p>
    </> : <p className="evaluation-foot">Run Lesson 6 in My Lab to populate your staged loss, gradient, memory, checkpoint, and fixed-prompt evidence. Published Reference evidence is maintained separately.</p>}
  </section>;
}
