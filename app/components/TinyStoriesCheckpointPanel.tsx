'use client';

import { useEffect, useMemo, useState } from 'react';
import { useEvidenceUrl } from './EvidenceMode';

type Trace = { absoluteStep: number; trainBatchLoss: number; gradientL2Norm: number };
type Evidence = {
  status: string;
  progress: { phase: string; absoluteStep: number; segmentCompletedUpdates: number; segmentPlannedUpdates: number; latestTrainLoss: number | null; latestGradientL2Norm: number | null };
  configuration: { parameterCount: number; interruptionStep: number; segmentStartStep: number; segmentEndStep: number; optimiser: string; learningRate: number; weightDecay: number };
  liveTraces: Record<string, Trace[]>;
  checkpointContract?: {
    requiredFiles: string[];
    manifestWrittenLast: boolean;
    checksumVerificationBeforeLoad: boolean;
    interruptionCheckpoint: {
      manifest: { files: Record<string, { bytes: number; sha256: string }> };
      checkpointDirectory: string;
      validTargetsSeenThisSegment: number;
    };
  };
  parity?: {
    comparedUpdates: number;
    maximumTrainLossDifference: number;
    maximumGradientNormDifference: number;
    maximumFinalWeightDifference: number;
    maximumFinalOptimizerStateDifference: number;
    finalValidationLossDifference: number;
    mismatchedBatchCount: number;
    nextBatchIndicesMatch: boolean;
    bitwiseExactAcrossProcesses: boolean;
    withinDeclaredTolerance: boolean;
    resumeValidated: boolean;
  };
  validation?: { lesson6Step500Loss: number; resumedStep600: { crossEntropyLoss: number; validTargets: number } };
  samples?: { prompt: string; continuation: string }[];
  performance?: { wholeExperimentElapsedSeconds: number; control: { validTargetsPerSecond: number; peakMlxAllocationBytes: number } };
};

const mebibytes = (bytes: number) => `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
const fileSize = (bytes: number) => bytes < 1024 * 1024
  ? `${(bytes / 1024).toFixed(1)} KiB`
  : mebibytes(bytes);

export default function TinyStoriesCheckpointPanel() {
  const evidenceUrl = useEvidenceUrl('tinystories-checkpoints-dashboard.json');
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
  const visibleTrace = useMemo(() => evidence?.liveTraces.resume ?? evidence?.liveTraces.interrupt ?? evidence?.liveTraces.control ?? [], [evidence]);
  const chartPoints = useMemo(() => {
    if (visibleTrace.length < 2) return '';
    const losses = visibleTrace.map(item => item.trainBatchLoss);
    const min = Math.min(...losses); const max = Math.max(...losses);
    return visibleTrace.map((item, index) => {
      const x = index / (visibleTrace.length - 1) * 100;
      const y = max === min ? 50 : (max - item.trainBatchLoss) / (max - min) * 76 + 12;
      return `${x},${y}`;
    }).join(' ');
  }, [visibleTrace]);
  const interruption = evidence?.checkpointContract?.interruptionCheckpoint;

  return <section className="evaluation-panel resume-panel" aria-label="TinyStories resumable checkpoint evidence">
    <div className="evaluation-head"><div><p className="kicker">LESSON 07 · RESUMABLE STATE</p><h2>{evidence?.parity?.resumeValidated ? 'The restart preserved state and stayed within tolerance' : evidence ? 'Testing whether disk state preserves the next update' : 'Create and test a complete training save'}</h2></div><span>{evidence?.status ?? (loaded.url === evidenceUrl ? 'Not run yet' : 'Loading evidence…')}</span></div>
    <div className="evaluation-summary">
      <div><small>PHASE</small><strong>{evidence?.progress.phase ?? '—'}</strong></div>
      <div><small>SEGMENT UPDATES</small><strong>{evidence ? `${evidence.progress.segmentCompletedUpdates} / ${evidence.progress.segmentPlannedUpdates}` : '—'}</strong></div>
      <div><small>LATEST BATCH LOSS</small><strong>{evidence?.progress.latestTrainLoss?.toFixed(4) ?? '—'}</strong></div>
      <div><small>LATEST GRADIENT NORM</small><strong>{evidence?.progress.latestGradientL2Norm?.toFixed(4) ?? '—'}</strong></div>
    </div>
    {evidence ? <>
      <figure className="resume-route" aria-labelledby="resume-route-title">
        <figcaption><strong id="resume-route-title">Two routes must preserve the same state and remain numerically equivalent</strong><p>The upper route saves but never exits. The lower route loads those four checked files at step {evidence.configuration.interruptionStep} in a fresh process. Discrete state must match exactly; floating-point results use declared tolerances.</p></figcaption>
        <div><span><b>CONTROL</b><i>step 500</i><em>50 updates → save → 50 updates; no exit</em><i>step 600</i></span><span><b>RESUME TEST</b><i>step 500</i><em>50 updates</em><strong>save → exit → load</strong><em>50 updates</em><i>step 600</i></span></div>
      </figure>
      <figure className="resume-live-chart" aria-labelledby="resume-live-title">
        <figcaption><strong id="resume-live-title">Live batch loss remains noisy while the run advances</strong><p>Every point is one training batch from the currently visible branch. This is operational monitoring, not the complete held-out evaluation.</p></figcaption>
        {chartPoints ? <svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="Training batch loss across the current resume experiment"><polyline points={chartPoints} /></svg> : <div className="resume-wait">Waiting for at least two updates…</div>}
      </figure>
      {evidence.parity && <div className="parity-grid">
        <article className={evidence.parity.resumeValidated ? 'pass' : ''}><small>OVERALL VERDICT</small><strong>{evidence.parity.resumeValidated ? 'RESUME VALIDATED' : 'OUTSIDE TOLERANCE'}</strong><p>{evidence.parity.comparedUpdates} post-reload updates compared; cross-process execution was {evidence.parity.bitwiseExactAcrossProcesses ? 'bitwise exact' : 'not bitwise exact'}.</p></article>
        <article><small>FINAL WEIGHTS</small><strong>{evidence.parity.maximumFinalWeightDifference.toExponential(2)}</strong><p>Maximum absolute difference across all 5.82M learned parameters.</p></article>
        <article><small>ADAMW STATE</small><strong>{evidence.parity.maximumFinalOptimizerStateDifference.toExponential(2)}</strong><p>Maximum difference across saved step, rate, and moving-average tensors.</p></article>
        <article><small>NEXT BATCH</small><strong>{evidence.parity.nextBatchIndicesMatch ? 'MATCH' : 'DIFFERS'}</strong><p>{evidence.parity.mismatchedBatchCount} mismatched training batches in the compared trace.</p></article>
      </div>}
      {interruption && <div className="checkpoint-anatomy">
        <div><small>COMPLETE SAVE AT STEP {evidence.configuration.interruptionStep}</small><strong>Four parts, one loadable state</strong><p>The manifest is written last. A loader rejects the directory if any required file is absent or has the wrong checksum.</p></div>
        <div>{evidence.checkpointContract?.requiredFiles.map(filename => {
          const file = interruption.manifest.files[filename];
          return <article key={filename}><strong>{filename}</strong><span>{file ? fileSize(file.bytes) : 'manifest metadata'}</span><code>{file?.sha256.slice(0, 12) ?? 'written last'}…</code></article>;
        })}</div>
      </div>}
      {evidence.validation && <div className="resume-outcome">
        <article><small>STEP 500 VALIDATION LOSS</small><strong>{evidence.validation.lesson6Step500Loss.toFixed(4)}</strong></article>
        <span aria-hidden="true">→</span>
        <article><small>STEP 600 VALIDATION LOSS</small><strong>{evidence.validation.resumedStep600.crossEntropyLoss.toFixed(4)}</strong></article>
        <p>Quality movement belongs to the extra updates; the parity bounds show that the restart stayed within its declared numerical tolerance.</p>
      </div>}
      {evidence.samples?.[0] && <article className="resume-sample"><small>RESUMED STEP 600 · FIXED PROMPT</small><strong>{evidence.samples[0].prompt}</strong><p>{evidence.samples[0].continuation || '〈no visible continuation before EOS〉'}</p></article>}
      <p className="evaluation-foot">{evidence.performance ? `The complete teaching proof took ${evidence.performance.wholeExperimentElapsedSeconds.toFixed(1)} seconds and the control branch processed ${Math.round(evidence.performance.control.validTargetsPerSecond).toLocaleString()} real targets/s. Peak MLX allocation was ${mebibytes(evidence.performance.control.peakMlxAllocationBytes)}.` : 'The JSON refreshes every ten updates while the multi-process parity test runs.'} My Lab files remain separate from reviewed Reference evidence.</p>
    </> : <p className="evaluation-foot">Run the Lesson 7 command in My Lab to populate checkpoint anatomy, live training metrics, process-restart parity, held-out loss, and resumed prompt evidence.</p>}
  </section>;
}
