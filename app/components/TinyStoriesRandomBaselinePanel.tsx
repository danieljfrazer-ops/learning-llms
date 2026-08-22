'use client';

import { useEffect, useState } from 'react';
import { useEvidenceUrl } from './EvidenceMode';

type Candidate = {
  label: string;
  modelSize: number;
  attentionHeads: number;
  transformerBlocks: number;
  parameterCount: number;
  decision: string;
};

type Sample = {
  prompt: string;
  seed: number;
  generatedTokens: number;
  endedWithEos: boolean;
  continuation: string;
};

type Evidence = {
  status: string;
  elapsedSeconds: number;
  configuration: {
    parameterCount: number;
    modelSize: number;
    attentionHeads: number;
    headSize: number;
    transformerBlocks: number;
    contextSize: number;
    batchSize: number;
    optimiserUpdates: number;
  };
  architectureDecision: { candidates: Candidate[]; selectedReason: string; parameterBreakdown: Record<string, number> };
  validation: {
    windows: number;
    validTargets: number;
    maskedPaddingTargets: number;
    crossEntropyLoss: number;
    perplexity: number;
    uniformGuessLoss: number;
    validTargetsPerSecond: number;
  };
  randomSamples: Sample[];
  firstPromptTopPredictions: { tokenId: number; tokenPiece: string; probability: number }[];
  checkpoint: { bytes: number; sha256: string; strictReloadMaximumLogitDifference: number };
  memory: { parameterBytes: number; peakMlxAllocationDuringEvaluationBytes: number };
  environment: { mlxDevice: string; mlx: string };
};

const millions = (value: number) => `${(value / 1_000_000).toFixed(2)}M`;
const mebibytes = (value: number) => `${(value / 1024 / 1024).toFixed(1)} MiB`;
const visiblePiece = (piece: string) => piece.replaceAll('Ġ', '␠');

export default function TinyStoriesRandomBaselinePanel() {
  const evidenceUrl = useEvidenceUrl('tinystories-random-baseline.json');
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
  const maxProbability = Math.max(...(evidence?.firstPromptTopPredictions.map(item => item.probability) ?? [1]));

  return <section className="evaluation-panel random-baseline-panel" aria-label="TinyStories random GPT baseline evidence">
    <div className="evaluation-head"><div><p className="kicker">LESSON 05 · CHECKPOINT ZERO</p><h2>{evidence ? 'A complete GPT can run before it has learned language' : 'Create the official before-training control'}</h2></div><span>{evidence?.status ?? (loaded.url === evidenceUrl ? 'Not run yet' : 'Loading evidence…')}</span></div>
    <div className="evaluation-summary">
      <div><small>PARAMETERS</small><strong>{evidence ? millions(evidence.configuration.parameterCount) : '—'}</strong></div>
      <div><small>VALIDATION LOSS</small><strong>{evidence?.validation.crossEntropyLoss.toFixed(4) ?? '—'}</strong></div>
      <div><small>OPTIMISER UPDATES</small><strong>{evidence?.configuration.optimiserUpdates ?? '—'}</strong></div>
      <div><small>RELOAD DIFFERENCE</small><strong>{evidence?.checkpoint.strictReloadMaximumLogitDifference.toFixed(1) ?? '—'}</strong></div>
    </div>
    {evidence ? <>
      <figure className="baseline-candidates" aria-labelledby="baseline-candidates-title">
        <figcaption><strong id="baseline-candidates-title">Choose the smallest architecture inside the declared range</strong><p>These counts come from the architecture formula. Only the selected model was materialised and measured in this lesson.</p></figcaption>
        <div>{evidence.architectureDecision.candidates.map(candidate => <article className={candidate.label === 'Selected baseline' ? 'selected' : ''} key={candidate.label}><span>{candidate.label}</span><strong>{millions(candidate.parameterCount)}</strong><p>width {candidate.modelSize} · {candidate.transformerBlocks} blocks · {candidate.attentionHeads} heads</p><small>{candidate.decision}</small></article>)}</div>
      </figure>
      <div className="baseline-evidence-grid">
        <article><small>COMPLETE HELD-OUT PASS</small><strong>{evidence.validation.validTargets.toLocaleString()} targets</strong><p>{evidence.validation.windows.toLocaleString()} windows; {evidence.validation.maskedPaddingTargets.toLocaleString()} padding targets excluded.</p></article>
        <article><small>RANDOM CONTROL</small><strong>{evidence.validation.crossEntropyLoss.toFixed(4)} loss</strong><p>Uniform guessing gives {evidence.validation.uniformGuessLoss.toFixed(4)}. Random initialisation is not required to match it exactly.</p></article>
        <article><small>FORWARD-ONLY MEMORY</small><strong>{mebibytes(evidence.memory.peakMlxAllocationDuringEvaluationBytes)}</strong><p>{mebibytes(evidence.memory.parameterBytes)} are parameters; the peak includes MLX forward arrays, not total system memory.</p></article>
        <article><small>SAVED STATE</small><strong>{mebibytes(evidence.checkpoint.bytes)}</strong><p>Strict reload reproduced the checked logits exactly. SHA-256 starts <code>{evidence.checkpoint.sha256.slice(0, 12)}…</code></p></article>
      </div>
      <figure className="random-probabilities" aria-labelledby="random-probabilities-title">
        <figcaption><strong id="random-probabilities-title">No next token has a confident lead</strong><p>Top predictions after “Once upon a time, there was”. “␠” marks a leading space encoded inside a token.</p></figcaption>
        {evidence.firstPromptTopPredictions.map(item => <div key={item.tokenId}><code>{visiblePiece(item.tokenPiece)}</code><span><i style={{ width: `${item.probability / maxProbability * 100}%` }} /></span><strong>{(item.probability * 100).toFixed(3)}%</strong></div>)}
      </figure>
      <div className="random-sample-grid">{evidence.randomSamples.map(sample => <article key={sample.seed}><small>SEED {sample.seed} · {sample.generatedTokens} TOKENS</small><strong>{sample.prompt}</strong><p>{sample.continuation || '〈no visible continuation before EOS〉'}</p><span>{sample.endedWithEos ? 'Stopped at EOS' : 'Stopped at the fixed token limit'}</span></article>)}</div>
      <p className="evaluation-foot">Completed in {evidence.elapsedSeconds.toFixed(2)} seconds on {evidence.environment.mlxDevice} with MLX {evidence.environment.mlx}. Forward throughput was {Math.round(evidence.validation.validTargetsPerSecond).toLocaleString()} valid targets/s; this is not training throughput.</p>
    </> : <p className="evaluation-foot">Run the random-baseline command in My Lab to populate architecture, held-out loss, memory, checkpoint-reload, probability, and prompt evidence. Reference mode remains blank until a maintainer separately reviews and promotes a course run.</p>}
  </section>;
}
