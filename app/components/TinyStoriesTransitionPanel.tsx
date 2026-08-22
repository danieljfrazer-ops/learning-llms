'use client';

import { useEffect, useState } from 'react';
import { useEvidenceUrl } from './EvidenceMode';

type TransitionEvidence = {
  status: string;
  elapsedSeconds: number;
  dataset: {
    train: { stories: number; characters: number; exactDuplicateStories: number; storyCharacters: { median: number; p95: number } };
    validation: { stories: number; characters: number; exactDuplicateStories: number };
  };
  provisionalTokenizer: {
    actualVocabularySize: number;
    trainTokens: number;
    trainCharactersPerToken: number;
    promptTokens: string[];
  };
  randomModel: {
    parameterCount: number;
    optimiserUpdates: number;
    fixedValidationBatchLoss: number;
    uniformGuessLoss: number;
    peakAcceleratorMemoryBytes: number;
    prompt: string;
    continuation: string;
  };
};

export default function TinyStoriesTransitionPanel() {
  const evidenceUrl = useEvidenceUrl('tinystories-transition.json');
  const [loaded, setLoaded] = useState<{ url: string; data: TransitionEvidence | null }>({ url: '', data: null });
  useEffect(() => {
    fetch(`${evidenceUrl}?t=${Date.now()}`, { cache: 'no-store' })
      .then(response => response.ok ? response.json() : null)
      .then(data => setLoaded({ url: evidenceUrl, data }))
      .catch(() => setLoaded({ url: evidenceUrl, data: null }));
  }, [evidenceUrl]);
  const evidence = loaded.url === evidenceUrl ? loaded.data : null;

  return <section className="evaluation-panel tinystories-transition-panel" aria-label="TinyStories transition evidence">
    <div className="evaluation-head"><div><p className="kicker">LESSON 01 · MEASURED EVIDENCE</p><h2>One complete pass through the new pipeline</h2></div><span>{evidence?.status ?? (loaded.url === evidenceUrl ? 'Not run yet' : 'Loading evidence…')}</span></div>
    <div className="evaluation-summary">
      <div><small>TRAINING STORIES</small><strong>{evidence?.dataset.train.stories.toLocaleString() ?? '—'}</strong></div>
      <div><small>BPE VOCABULARY</small><strong>{evidence?.provisionalTokenizer.actualVocabularySize.toLocaleString() ?? '—'}</strong></div>
      <div><small>RANDOM PARAMETERS</small><strong>{evidence?.randomModel.parameterCount.toLocaleString() ?? '—'}</strong></div>
      <div><small>OPTIMISER UPDATES</small><strong>{evidence?.randomModel.optimiserUpdates ?? '—'}</strong></div>
    </div>
    {evidence ? <>
      <div className="transition-evidence-grid">
        <article><strong>Bounded data</strong><p>{evidence.dataset.train.characters.toLocaleString()} training characters and {evidence.dataset.validation.characters.toLocaleString()} validation characters. Exact duplicates found within the sampled splits: {evidence.dataset.train.exactDuplicateStories + evidence.dataset.validation.exactDuplicateStories}.</p></article>
        <article><strong>Token compression</strong><p>{evidence.provisionalTokenizer.trainTokens.toLocaleString()} training tokens, averaging {evidence.provisionalTokenizer.trainCharactersPerToken.toFixed(2)} characters per token. The approved prompt became seven pieces.</p></article>
        <article><strong>Random loss check</strong><p>Measured {evidence.randomModel.fixedValidationBatchLoss.toFixed(4)} on one fixed batch, versus {evidence.randomModel.uniformGuessLoss.toFixed(4)} for an exactly uniform vocabulary guess. This is a smoke test, not evaluation.</p></article>
        <article><strong>Resource check</strong><p>{(evidence.randomModel.peakAcceleratorMemoryBytes / 1_000_000).toFixed(1)} MB peak MLX allocation and {evidence.elapsedSeconds.toFixed(2)} seconds for audit, tokenizer, construction and invocation.</p></article>
      </div>
      <div className="token-strip"><small>THE PROMPT AS SUBWORD PIECES</small><div>{evidence.provisionalTokenizer.promptTokens.map((token, index) => <code key={`${token}-${index}`}>{token.replaceAll('Ġ', '␠')}</code>)}</div></div>
      <blockquote className="model-sample"><mark>{evidence.randomModel.prompt}</mark>{evidence.randomModel.continuation}</blockquote>
      <p className="evaluation-foot">The highlighted opening is the human prompt. Everything after it came from random weights. Broken fragments are the expected control result, not a failed training run: training has not started.</p>
    </> : <p className="evaluation-foot">Run the lesson command in My Lab to create this ignored local evidence. Reference mode displays the reviewed course run.</p>}
  </section>;
}
