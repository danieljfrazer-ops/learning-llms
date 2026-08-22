'use client';

import { useEffect, useState } from 'react';
import { useEvidenceUrl } from './EvidenceMode';

type SplitStructure = {
  stories: number;
  distinctLowercaseWordForms: number;
  storyWords: { median: number; p95: number; maximum: number };
  topFiveWordOpenings: { opening: string; stories: number }[];
  templatePhraseStories: Record<string, number>;
};

type AuditEvidence = {
  status: string;
  elapsedSeconds: number;
  datasetDecision: { datasetId: string; decision: string; notApprovedFor: string; trainStories: number; validationStories: number };
  integrity: Record<'train' | 'validation', { emptyStories: number; unicodeReplacementCharacters: number; unexpectedControlCharacters: number; storiesOver2000Characters: number }>;
  duplicates: {
    crossSplit: { comparisons: number; rawExactCrossSplitPairs: unknown[]; normalisedExactCrossSplitPairs: unknown[]; nearDuplicatePairsAtOrAboveThreshold: unknown[]; nearDuplicateThreshold: number; strongestPair: { similarity: number; trainSourceRow: number; validationSourceRow: number } };
  };
  structure: Record<'train' | 'validation', SplitStructure>;
  contentTermScreen: { method: string; train: Record<string, number>; validation: Record<string, number> };
  provisionalTokenizer: Record<'train' | 'validation', { tokens: number; tokensPerStoryMedian: number; unknownTokens: number; nfkcRoundTripMismatches: number }> & { vocabularySize: number };
};

export default function TinyStoriesDatasetAuditPanel() {
  const evidenceUrl = useEvidenceUrl('tinystories-dataset-audit.json');
  const [loaded, setLoaded] = useState<{ url: string; data: AuditEvidence | null }>({ url: '', data: null });
  useEffect(() => {
    fetch(`${evidenceUrl}?t=${Date.now()}`, { cache: 'no-store' })
      .then(response => response.ok ? response.json() : null)
      .then(data => setLoaded({ url: evidenceUrl, data }))
      .catch(() => setLoaded({ url: evidenceUrl, data: null }));
  }, [evidenceUrl]);
  const audit = loaded.url === evidenceUrl ? loaded.data : null;
  const cross = audit?.duplicates.crossSplit;
  const commonOpening = audit?.structure.train.topFiveWordOpenings[0];
  const integrityProblems = audit ? (['train', 'validation'] as const).reduce((sum, split) => {
    const item = audit.integrity[split];
    return sum + item.emptyStories + item.unicodeReplacementCharacters + item.unexpectedControlCharacters;
  }, 0) : null;

  return <section className="evaluation-panel dataset-audit-panel" aria-label="TinyStories dataset audit evidence">
    <div className="evaluation-head"><div><p className="kicker">LESSON 02 · DATASET EVIDENCE</p><h2>Low measured leakage, strong template repetition</h2></div><span>{audit?.status ?? (loaded.url === evidenceUrl ? 'Not run yet' : 'Loading evidence…')}</span></div>
    <div className="evaluation-summary">
      <div><small>DEVELOPMENT SPLIT</small><strong>{audit ? `${audit.datasetDecision.trainStories.toLocaleString()} + ${audit.datasetDecision.validationStories}` : '—'}</strong></div>
      <div><small>CROSS-SPLIT COMPARISONS</small><strong>{cross?.comparisons.toLocaleString() ?? '—'}</strong></div>
      <div><small>≥80% SIMILAR PAIRS</small><strong>{cross?.nearDuplicatePairsAtOrAboveThreshold.length ?? '—'}</strong></div>
      <div><small>TEXT INTEGRITY ERRORS</small><strong>{integrityProblems ?? '—'}</strong></div>
    </div>
    {audit ? <>
      <div className="transition-evidence-grid">
        <article><strong>Frozen scope</strong><p><code>{audit.datasetDecision.datasetId}</code> is approved for tokenizer, batching, baseline and short pretraining comparisons—not the final scaling run.</p></article>
        <article><strong>Closest split pair</strong><p>The strongest word five-gram Jaccard match was only {(cross!.strongestPair.similarity * 100).toFixed(1)}%, between source rows {cross!.strongestPair.trainSourceRow.toLocaleString()} and {cross!.strongestPair.validationSourceRow.toLocaleString()}.</p></article>
        <article><strong>Dominant opening</strong><p>“{commonOpening?.opening}” begins {commonOpening?.stories} of {audit.structure.train.stories.toLocaleString()} sampled training stories. Low pairwise leakage does not mean broad stylistic diversity.</p></article>
        <article><strong>Tokenizer integrity</strong><p>{audit.provisionalTokenizer.train.unknownTokens + audit.provisionalTokenizer.validation.unknownTokens} unknown tokens and {audit.provisionalTokenizer.train.nfkcRoundTripMismatches + audit.provisionalTokenizer.validation.nfkcRoundTripMismatches} NFKC round-trip mismatches across both splits.</p></article>
      </div>
      <table className="lesson-table"><thead><tr><th>Split</th><th>Stories</th><th>Distinct word forms</th><th>Median words/story</th><th>95th percentile</th><th>Over 2,000 characters</th></tr></thead><tbody>
        {(['train', 'validation'] as const).map(split => <tr key={split}><td>{split}</td><td>{audit.structure[split].stories.toLocaleString()}</td><td>{audit.structure[split].distinctLowercaseWordForms.toLocaleString()}</td><td>{audit.structure[split].storyWords.median}</td><td>{audit.structure[split].storyWords.p95}</td><td>{audit.integrity[split].storiesOver2000Characters}</td></tr>)}
      </tbody></table>
      <div className="audit-screen-grid"><article><small>TRAIN · TERM-SCREENED STORIES</small>{Object.entries(audit.contentTermScreen.train).map(([name, count]) => <div key={name}><span>{name}</span><strong>{count}</strong></div>)}</article><article><small>VALIDATION · TERM-SCREENED STORIES</small>{Object.entries(audit.contentTermScreen.validation).map(([name, count]) => <div key={name}><span>{name}</span><strong>{count}</strong></div>)}</article></div>
      <p className="evaluation-foot">{audit.contentTermScreen.method}. Audit runtime: {audit.elapsedSeconds.toFixed(2)} seconds.</p>
    </> : <p className="evaluation-foot">Complete Lesson 1 first, then run the dataset-audit command in My Lab. Reference mode displays the reviewed frozen development split.</p>}
  </section>;
}
