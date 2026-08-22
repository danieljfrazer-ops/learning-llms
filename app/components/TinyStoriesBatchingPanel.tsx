'use client';

import { useEffect, useState } from 'react';
import { useEvidenceUrl } from './EvidenceMode';

type SplitAudit = {
  stories: number;
  examples: number;
  validTargetTokens: number;
  paddingTargetPositions: number;
  targetUtilisation: number;
  crossStoryTargetPairs: number;
  pairCoverageMatches: boolean;
};

type NaiveAudit = { targetUtilisation: number; crossStoryTargetPairs: number; windows: number };
type BoundaryPosition = { position: number; inputPiece: string; targetPiece: string; countsForLoss: boolean };

type BatchingEvidence = {
  status: string;
  elapsedSeconds: number;
  configuration: { contextSize: number; batchSize: number };
  isolatedWindows: Record<'train' | 'validation', SplitAudit>;
  naiveConcatenation: Record<'train' | 'validation', NaiveAudit>;
  fixedBatches: Record<'train' | 'validation', { shapes: Record<string, number[]>; tensorStorageBytes: number; validTargets: number; paddingTargets: number }>;
  paddingBoundaryPreview: { sourceRow: number; chunkIndex: number; validTargetCount: number; positions: BoundaryPosition[] };
  causalMask: { allowedCurrentOrEarlierRelationships: number; blockedFutureRelationships: number };
  decision: { selectedMethod: string; reason: string; tradeoff: string; scope: string };
};

const visiblePiece = (piece: string) => piece.replaceAll('Ġ', '␠');

export default function TinyStoriesBatchingPanel() {
  const evidenceUrl = useEvidenceUrl('tinystories-batching.json');
  const [loaded, setLoaded] = useState<{ url: string; data: BatchingEvidence | null }>({ url: '', data: null });

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
  const train = evidence?.isolatedWindows.train;
  const batchShape = evidence?.fixedBatches.train.shapes.inputs;

  return <section className="evaluation-panel batching-panel" aria-label="TinyStories sequence batching evidence">
    <div className="evaluation-head"><div><p className="kicker">LESSON 04 · BATCHING EVIDENCE</p><h2>{evidence ? 'Every real target is preserved without mixing stories' : 'Inspect boundaries, shifted targets, padding, and masks'}</h2></div><span>{evidence?.status ?? (loaded.url === evidenceUrl ? 'Not run yet' : 'Loading evidence…')}</span></div>
    <div className="evaluation-summary">
      <div><small>TRAIN WINDOWS</small><strong>{train?.examples.toLocaleString() ?? '—'}</strong></div>
      <div><small>VALID TARGET UTILISATION</small><strong>{train ? `${(train.targetUtilisation * 100).toFixed(1)}%` : '—'}</strong></div>
      <div><small>CROSS-STORY TARGETS</small><strong>{train?.crossStoryTargetPairs ?? '—'}</strong></div>
      <div><small>INPUT BATCH SHAPE</small><strong>{batchShape ? batchShape.join(' × ') : '—'}</strong></div>
    </div>
    {evidence ? <>
      <figure className="batching-choice" aria-labelledby="batching-choice-title">
        <figcaption><strong id="batching-choice-title">Padding costs positions; naïve packing costs boundary correctness</strong><p>The two methods use the same stories and 128-token context. Utilisation is attractive only when the resulting prediction questions remain legitimate.</p></figcaption>
        <div className="batching-choice-grid">
          <article className="selected"><span>SELECTED · STORY-ISOLATED</span><strong>{(train!.targetUtilisation * 100).toFixed(1)}% utilised</strong><p>{train!.validTargetTokens.toLocaleString()} real targets; {train!.paddingTargetPositions.toLocaleString()} masked padding positions; zero cross-story targets.</p></article>
          <article><span>DEFERRED · NAÏVE CONCATENATION</span><strong>{(evidence.naiveConcatenation.train.targetUtilisation * 100).toFixed(1)}% utilised</strong><p>{evidence.naiveConcatenation.train.crossStoryTargetPairs.toLocaleString()} EOS→BOS cross-story targets, with unrelated earlier-story context still visible.</p></article>
        </div>
      </figure>
      <figure className="shift-preview" aria-labelledby="shift-preview-title">
        <figcaption><strong id="shift-preview-title">The target is the input shifted one position—and padding is excluded</strong><p>Validation source row {evidence.paddingBoundaryPreview.sourceRow.toLocaleString()}, chunk {evidence.paddingBoundaryPreview.chunkIndex}. “␠” marks a leading space stored inside a byte-level token.</p></figcaption>
        <div className="shift-preview-row shift-preview-head"><span>Position</span><span>Input</span><span>Target</span><span>Loss?</span></div>
        {evidence.paddingBoundaryPreview.positions.map(position => <div className={`shift-preview-row ${position.countsForLoss ? 'counts' : 'masked'}`} key={position.position}><span>{position.position}</span><code>{visiblePiece(position.inputPiece)}</code><code>{visiblePiece(position.targetPiece)}</code><strong>{position.countsForLoss ? 'yes · real target' : 'no · padding'}</strong></div>)}
      </figure>
      <div className="transition-evidence-grid">
        <article><strong>Pair coverage</strong><p>{train!.pairCoverageMatches ? 'Every within-story adjacent token pair appears exactly once.' : 'Coverage check failed.'} EOS is learned; BOS is input context, never a target.</p></article>
        <article><strong>Three aligned tensors</strong><p>Inputs, targets, and loss mask each have shape {batchShape?.join(' × ')}. Together they use {(evidence.fixedBatches.train.tensorStorageBytes / 1024).toFixed(0)} KiB before model activations.</p></article>
        <article><strong>Causal visibility</strong><p>Each 128 × 128 attention grid allows {evidence.causalMask.allowedCurrentOrEarlierRelationships.toLocaleString()} current/earlier relationships and blocks {evidence.causalMask.blockedFutureRelationships.toLocaleString()} future relationships.</p></article>
        <article><strong>Frozen boundary</strong><p>{evidence.decision.scope}. {evidence.decision.tradeoff}.</p></article>
      </div>
      <p className="evaluation-foot">Batching audit runtime: {evidence.elapsedSeconds.toFixed(2)} seconds. This stage performed no model forward pass or weight update.</p>
    </> : <p className="evaluation-foot">Run the sequence-batching command in My Lab to populate this panel. Reference mode stays blank until a maintainer separately reviews and promotes course evidence.</p>}
  </section>;
}
