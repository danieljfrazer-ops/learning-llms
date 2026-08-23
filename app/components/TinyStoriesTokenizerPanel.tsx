'use client';

import { useEffect, useMemo, useState } from 'react';
import { useEvidenceUrl } from './EvidenceMode';

type SplitEvidence = {
  tokens: number;
  charactersPerToken: number;
  tokensPerStory: { median: number; p95: number };
  unknownTokens: number;
  nfkcRoundTripMismatches: number;
};

type Candidate = {
  actualVocabularySize: number;
  trainingSeconds: number;
  passesIntegrityGates: boolean;
  train: SplitEvidence;
  validation: SplitEvidence;
  modelCostProxy: { totalVocabularyDependentParameters: number; assumedModelWidth: number };
  promptPieces: { text: string; tokens: string[] }[];
};

type TokenizerEvidence = {
  status: string;
  elapsedSeconds: number;
  candidates: Candidate[];
  decision: { selectedVocabularySize: number; rationale: string; scope: string };
};

const visiblePiece = (piece: string) => piece.replaceAll('Ġ', '␠');

export default function TinyStoriesTokenizerPanel() {
  const evidenceUrl = useEvidenceUrl('tinystories-tokenizer.json');
  const [loaded, setLoaded] = useState<{ url: string; data: TokenizerEvidence | null }>({ url: '', data: null });

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
  const selected = evidence?.candidates.find(candidate => candidate.actualVocabularySize === evidence.decision.selectedVocabularySize);
  const maximumTokens = Math.max(...(evidence?.candidates.map(candidate => candidate.validation.tokens) ?? [1]));
  const maximumParameters = Math.max(...(evidence?.candidates.map(candidate => candidate.modelCostProxy.totalVocabularyDependentParameters) ?? [1]));
  const promptComparisons = useMemo(() => {
    if (!evidence?.candidates.length) return [];
    const promptCount = Math.min(...evidence.candidates.map(candidate => candidate.promptPieces.length));
    return Array.from({ length: promptCount }, (_, promptIndex) => {
      const rows = evidence.candidates.map(candidate => ({
        vocabularySize: candidate.actualVocabularySize,
        pieces: candidate.promptPieces[promptIndex].tokens,
      }));
      const counts = rows.map(row => row.pieces.length);
      return {
        text: evidence.candidates[0].promptPieces[promptIndex].text,
        rows,
        distinctSegmentations: new Set(rows.map(row => row.pieces.join('\u0000'))).size,
        minimumPieces: Math.min(...counts),
        maximumPieces: Math.max(...counts),
      };
    }).sort((left, right) =>
      right.distinctSegmentations - left.distinctSegmentations
      || (right.maximumPieces - right.minimumPieces) - (left.maximumPieces - left.minimumPieces));
  }, [evidence]);

  return <section className="evaluation-panel tokenizer-panel" aria-label="TinyStories tokenizer comparison evidence">
    <div className="evaluation-head"><div><p className="kicker">LESSON 03 · TOKENIZER EVIDENCE</p><h2>{evidence ? `${evidence.decision.selectedVocabularySize.toLocaleString()} pieces balance this development trade-off` : 'Compare shorter sequences with a larger model interface'}</h2></div><span>{evidence?.status ?? (loaded.url === evidenceUrl ? 'Not run yet' : 'Loading evidence…')}</span></div>
    <div className="evaluation-summary">
      <div><small>SELECTED VOCABULARY</small><strong>{selected?.actualVocabularySize.toLocaleString() ?? '—'}</strong></div>
      <div><small>VALIDATION CHARS / TOKEN</small><strong>{selected?.validation.charactersPerToken.toFixed(3) ?? '—'}</strong></div>
      <div><small>VALIDATION P95 TOKENS / STORY</small><strong>{selected?.validation.tokensPerStory.p95 ?? '—'}</strong></div>
      <div><small>INTEGRITY FAILURES</small><strong>{evidence ? evidence.candidates.filter(candidate => !candidate.passesIntegrityGates).length : '—'}</strong></div>
    </div>
    {evidence ? <>
      <figure className="tokenizer-tradeoff" aria-labelledby="tokenizer-tradeoff-title">
        <figcaption><strong id="tokenizer-tradeoff-title">Every larger vocabulary buys fewer validation tokens at a rising parameter cost</strong><p>Bars are scaled independently within each column. Read the printed values for the comparison; shorter orange bars are fewer sequence tokens, while longer teal bars are more vocabulary-dependent parameters.</p></figcaption>
        <div className="tokenizer-tradeoff-head"><span>Vocabulary</span><span>Validation sequence</span><span>Model interface at width 256</span></div>
        {evidence.candidates.map(candidate => <article className={candidate.actualVocabularySize === evidence.decision.selectedVocabularySize ? 'selected' : ''} key={candidate.actualVocabularySize}>
          <strong>{candidate.actualVocabularySize.toLocaleString()}{candidate.actualVocabularySize === evidence.decision.selectedVocabularySize ? ' · selected' : ''}</strong>
          <div><span className="tradeoff-bar sequence" style={{ width: `${candidate.validation.tokens / maximumTokens * 100}%` }} /><small>{candidate.validation.tokens.toLocaleString()} tokens · {candidate.validation.charactersPerToken.toFixed(2)} chars/token</small></div>
          <div><span className="tradeoff-bar parameters" style={{ width: `${candidate.modelCostProxy.totalVocabularyDependentParameters / maximumParameters * 100}%` }} /><small>{candidate.modelCostProxy.totalVocabularyDependentParameters.toLocaleString()} parameters</small></div>
        </article>)}
      </figure>
      <div className="transition-evidence-grid">
        <article><strong>Integrity gates</strong><p>Every candidate recorded zero unknown tokens and zero NFKC round-trip mismatches on both frozen splits.</p></article>
        <article><strong>Frozen decision rule</strong><p>{evidence.decision.rationale}.</p></article>
        <article><strong>What the proxy means</strong><p>The parameter count covers the token embedding and output projection for the course&apos;s untied, 256-wide design—not a complete model.</p></article>
        <article><strong>Runtime and scope</strong><p>Four tokenizer candidates completed in {evidence.elapsedSeconds.toFixed(2)} seconds. {evidence.decision.scope}.</p></article>
      </div>
      {selected && <div className="token-piece-comparisons">
        <div className="token-piece-legend"><small>SAME TEXT · FOUR VOCABULARIES</small><strong>Lead with the prompt that exposes the largest segmentation difference</strong><p><code>␠</code> means “the token begins with a space”. The tokenizer’s raw diagnostic spelling is <code>Ġ</code>; neither symbol is a letter generated into the decoded story.</p></div>
        {promptComparisons.map((comparison, promptIndex) => <section className="token-piece-comparison" key={comparison.text}>
          <header><small>{comparison.distinctSegmentations === 1 ? 'STABLE COMMON PHRASE' : promptIndex === 0 ? 'MOST REVEALING EXAMPLE' : 'ADDITIONAL CONTRAST'}</small><h3>{comparison.text}</h3><p>{comparison.distinctSegmentations} distinct segmentation{comparison.distinctSegmentations === 1 ? '' : 's'} · {comparison.maximumPieces === comparison.minimumPieces ? `${comparison.maximumPieces} pieces in every vocabulary` : `${comparison.maximumPieces} → ${comparison.minimumPieces} pieces`}</p></header>
          <div>{comparison.rows.map(row => <article key={row.vocabularySize}><strong>{row.vocabularySize.toLocaleString()} vocabulary · {row.pieces.length} pieces</strong><p>{row.pieces.map((token, index) => <code key={`${token}-${index}`}>{visiblePiece(token)}</code>)}</p></article>)}</div>
        </section>)}
      </div>}
    </> : <p className="evaluation-foot">Run the tokenizer experiment in My Lab to populate this comparison. Reference mode remains blank until a maintainer separately reviews and promotes a course result.</p>}
  </section>;
}
