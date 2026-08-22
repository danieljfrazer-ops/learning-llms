'use client';

import { useEffect, useMemo, useState } from 'react';
import { useEvidenceUrl } from './EvidenceMode';

type Trace = { segmentUpdate: number; trainBatchLoss: number; learningRate: number };
type Candidate = { id: string; label: string; policy: string; startLearningRate: number; endLearningRate: number };
type Variant = {
  candidate: Candidate;
  startingValidation: { crossEntropyLoss: number };
  finalValidation: { crossEntropyLoss: number; validTargets: number };
  validationLossChange: number;
  trace: Trace[];
  samples: { prompt: string; continuation: string }[];
  performance: { elapsedSeconds: number; validTargetsPerSecond: number; peakMlxAllocationBytes: number };
};
type Evidence = {
  status: string;
  progress: { phase: string; candidateUpdate: number; candidatePlannedUpdates: number; latestTrainLoss?: number; latestLearningRate?: number };
  configuration: {
    parameterCount: number; batchSize: number; contextSize: number; updatesPerCandidate: number;
    selectionRule: { minimumImprovementOverControl: number };
    lesson7CrossProcessValidationDifference: number;
  };
  candidates: Candidate[];
  liveTraces: Record<string, Trace[]>;
  variants: Variant[];
  comparison?: {
    identicalBatchSequences: boolean; comparedBatchesPerCandidate: number; startingValidationLossRange: number;
    controlValidationLoss: number; lowestObservedValidationLoss: number; improvementOverControl: number;
    minimumRequiredImprovement: number;
  };
  selection?: { selectedCandidateId: string; selectedLabel: string; reason: string; selectedCheckpointDirectory: string };
  performance?: { wholeExperimentElapsedSeconds: number; peakCandidateMlxAllocationBytes: number };
};

const mebibytes = (bytes: number) => `${(bytes / 1024 / 1024).toFixed(0)} MiB`;

export default function TinyStoriesRecipePanel() {
  const evidenceUrl = useEvidenceUrl('tinystories-training-recipe.json');
  const [loaded, setLoaded] = useState<{ url: string; data: Evidence | null }>({ url: '', data: null });

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
  const variants = useMemo(() => evidence?.variants ?? [], [evidence]);
  const losses = useMemo(() => variants.map(item => item.finalValidation.crossEntropyLoss), [variants]);
  const minLoss = losses.length ? Math.min(...losses) : 0;
  const maxLoss = losses.length ? Math.max(...losses) : 1;

  return <section className="evaluation-panel recipe-comparison" aria-label="TinyStories controlled training recipe evidence">
    <div className="evaluation-head"><div><p className="kicker">LESSON 08 · CONTROLLED RECIPE TEST</p><h2>{evidence?.selection ? `${evidence.selection.selectedLabel} won this bounded comparison` : evidence ? 'Compare one dial while everything else stays fixed' : 'Test learning-rate policies from one saved state'}</h2></div><span>{evidence?.status ?? (loaded.url === evidenceUrl ? 'Not run yet' : 'Loading evidence…')}</span></div>
    <div className="evaluation-summary">
      <div><small>PHASE</small><strong>{evidence?.progress.phase ?? '—'}</strong></div>
      <div><small>CANDIDATE UPDATE</small><strong>{evidence ? `${evidence.progress.candidateUpdate} / ${evidence.progress.candidatePlannedUpdates}` : '—'}</strong></div>
      <div><small>LATEST BATCH LOSS</small><strong>{evidence?.progress.latestTrainLoss?.toFixed(4) ?? '—'}</strong></div>
      <div><small>LATEST LEARNING RATE</small><strong>{evidence?.progress.latestLearningRate?.toExponential(2) ?? '—'}</strong></div>
    </div>
    {evidence ? <>
      <figure className="recipe-branch" aria-labelledby="recipe-branch-title">
        <figcaption><strong id="recipe-branch-title">One complete checkpoint branches into three alternate update histories</strong><p>Every branch reloads the same model, AdamW history and next-batch cursor. Only the multiplier controlling update size follows a different policy.</p></figcaption>
        <div className="recipe-source"><span>STEP 600</span><strong>one saved training state</strong><small>{(evidence.configuration.parameterCount / 1_000_000).toFixed(2)}M parameters · batch {evidence.configuration.batchSize} · context {evidence.configuration.contextSize}</small></div>
        <div className="recipe-arrows" aria-hidden="true">↙ ↓ ↘</div>
        <div className="recipe-paths">{evidence.candidates.map(candidate => <article key={candidate.id} className={evidence.selection?.selectedCandidateId === candidate.id ? 'selected' : ''}><small>{candidate.policy === 'cosine' ? 'CHANGING RATE' : 'FIXED RATE'}</small><strong>{candidate.label}</strong><span>{candidate.startLearningRate.toExponential(1)}{candidate.endLearningRate !== candidate.startLearningRate ? ` → ${candidate.endLearningRate.toExponential(1)}` : ''}</span></article>)}</div>
      </figure>

      {variants.length ? <div className="recipe-results">{variants.map(variant => {
        const selected = evidence.selection?.selectedCandidateId === variant.candidate.id;
        const width = maxLoss === minLoss ? 50 : 20 + (maxLoss - variant.finalValidation.crossEntropyLoss) / (maxLoss - minLoss) * 80;
        return <article key={variant.candidate.id} className={selected ? 'selected' : ''}>
          <div><small>{selected ? 'SELECTED' : variant.candidate.id === 'constant-3e-4' ? 'CONTROL' : 'CANDIDATE'}</small><strong>{variant.candidate.label}</strong></div>
          <div className="recipe-loss"><span style={{ width: `${width}%` }} /><b>{variant.finalValidation.crossEntropyLoss.toFixed(4)}</b></div>
          <p>{variant.validationLossChange > 0 ? '+' : ''}{variant.validationLossChange.toFixed(4)} from the shared start · {Math.round(variant.performance.validTargetsPerSecond).toLocaleString()} targets/s</p>
          <blockquote><strong>{variant.samples[0]?.prompt}</strong>{variant.samples[0]?.continuation || '〈no visible continuation before EOS〉'}</blockquote>
        </article>;
      })}</div> : <p className="resume-wait">The first candidate has not completed its held-out evaluation yet.</p>}

      {evidence.comparison && evidence.selection && <div className="recipe-verdict">
        <article><small>CONTROL → SELECTED</small><strong>{evidence.comparison.controlValidationLoss.toFixed(4)} → {evidence.comparison.lowestObservedValidationLoss.toFixed(4)}</strong><p>Validation loss improved by {evidence.comparison.improvementOverControl.toFixed(4)}; the required margin was {evidence.comparison.minimumRequiredImprovement.toFixed(4)}.</p></article>
        <article><small>FAIRNESS CHECK</small><strong>{evidence.comparison.identicalBatchSequences ? '100 / 100 BATCHES MATCH' : 'BATCHES DIFFER'}</strong><p>Starting validation range: {evidence.comparison.startingValidationLossRange.toExponential(1)}. Lesson 7 process difference: {evidence.configuration.lesson7CrossProcessValidationDifference.toExponential(2)}.</p></article>
        <article><small>NEXT COMPLETE STATE</small><strong>STEP 700</strong><p>{evidence.selection.reason}. Model, optimiser, data cursor and manifest were saved together.</p></article>
      </div>}
      <p className="evaluation-foot">{evidence.performance ? `All three branches, complete evaluations, samples and checkpoints took ${evidence.performance.wholeExperimentElapsedSeconds.toFixed(1)} seconds. Peak candidate MLX allocation was ${mebibytes(evidence.performance.peakCandidateMlxAllocationBytes)}.` : 'The dashboard refreshes every ten updates.'} Training-batch curves monitor operation; complete held-out loss selects the recipe.</p>
    </> : <p className="evaluation-foot">Run the Lesson 8 command in My Lab to populate the three learning-rate branches, matched-batch checks, complete validation, fixed-prompt samples and selected checkpoint. Reference mode remains blank until a separate reviewed promotion.</p>}
  </section>;
}
