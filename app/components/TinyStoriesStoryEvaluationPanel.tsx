'use client';

import { useEffect, useMemo, useState } from 'react';
import { useEvidenceUrl } from './EvidenceMode';
import styles from './TinyStoriesStoryEvaluationPanel.module.css';

type Sample = { scenarioId: string; prompt: string; continuation: string; expectedKeywordHits: string[]; contradictionHits: string[]; longestExactTrainingSpanWordsUpTo12: number };
type Result = {
  candidate: { id: string; label: string; history: string };
  validation: { crossEntropyLoss: number; perplexity: number };
  aggregate: {
    adherenceProxyPassRate: number; contradictionProxyPassRate: number; sentenceClosureRate: number;
    meanDistinctBigramRatio: number; meanRepeatedFourGramFraction: number;
    meanPairwiseThreeGramJaccardAcrossSeeds: number; exactEightWordTrainingSpanRate: number;
    maximumObservedExactTrainingSpanWordsUpTo12: number;
  };
  fixedExamples: Sample[];
};
type Evidence = {
  status: string; elapsedSeconds?: number;
  progress: { phase: string; completedCandidates: number; plannedCandidates: number };
  protocol: { generation: { scenarios: { id: string; intent: string }[]; seeds: number[]; temperature: number; maximumTokens: number }; interpretationBoundary: string };
  results: Result[];
};

const percent = (value: number) => `${(value * 100).toFixed(value < .01 ? 1 : 0)}%`;

export default function TinyStoriesStoryEvaluationPanel() {
  const evidenceUrl = useEvidenceUrl('tinystories-story-evaluation.json');
  const [loaded, setLoaded] = useState<{ url: string; data: Evidence | null }>({ url: '', data: null });
  const [candidateId, setCandidateId] = useState('clean-700');
  const [scenarioId, setScenarioId] = useState('care-for-kitten');

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
  const selectedResult = useMemo(() => evidence?.results.find(item => item.candidate.id === candidateId) ?? evidence?.results.at(-1), [evidence, candidateId]);
  const selectedSample = selectedResult?.fixedExamples.find(item => item.scenarioId === scenarioId) ?? selectedResult?.fixedExamples[0];
  const scenario = evidence?.protocol.generation.scenarios.find(item => item.id === scenarioId);

  return <section className="evaluation-panel" aria-label="TinyStories story behaviour evaluation evidence">
    <div className="evaluation-head"><div><p className="kicker">LESSON 10 · MULTI-LENS EVALUATION</p><h2>{evidence?.status === 'Complete' ? 'Training improved prediction before it produced reliable story following' : 'Apply several lenses without inventing one story score'}</h2></div><span>{evidence?.status ?? (loaded.url === evidenceUrl ? 'Not run yet' : 'Loading evidence…')}</span></div>
    <div className="evaluation-summary">
      <div><small>PHASE</small><strong>{evidence?.progress.phase ?? '—'}</strong></div>
      <div><small>CHECKPOINTS</small><strong>{evidence ? `${evidence.progress.completedCandidates} / ${evidence.progress.plannedCandidates}` : '—'}</strong></div>
      <div><small>SCENARIOS</small><strong>{evidence?.protocol.generation.scenarios.length ?? '—'}</strong></div>
      <div><small>SEEDS EACH</small><strong>{evidence?.protocol.generation.seeds.length ?? '—'}</strong></div>
    </div>
    {evidence?.results.length ? <>
      <div className={styles.cards}>{evidence.results.map(result => <article key={result.candidate.id}>
        <small>{result.candidate.id === 'random' ? 'BEFORE TRAINING' : result.candidate.id === 'inherited-700' ? 'INHERITED HISTORY' : 'LESSON 9 CHECKPOINT'}</small>
        <h3>{result.candidate.label}</h3>
        <dl>
          <div><dt>Validation loss</dt><dd>{result.validation.crossEntropyLoss.toFixed(4)}</dd></div>
          <div><dt>Scenario-keyword proxy</dt><dd>{percent(result.aggregate.adherenceProxyPassRate)}</dd></div>
          <div><dt>Repeated 4-grams</dt><dd>{percent(result.aggregate.meanRepeatedFourGramFraction)}</dd></div>
          <div><dt>Eight-word train copy</dt><dd>{percent(result.aggregate.exactEightWordTrainingSpanRate)}</dd></div>
          <div><dt>Longest train span</dt><dd>{result.aggregate.maximumObservedExactTrainingSpanWordsUpTo12} words</dd></div>
        </dl>
      </article>)}</div>
      <table className={styles.lensTable}><thead><tr><th>Lens</th><th>Random</th><th>Inherited 700</th><th>Clean 700</th></tr></thead><tbody>
        <tr><td>Ends naturally before 64-token cap</td>{evidence.results.map(item => <td key={item.candidate.id}>{percent(item.aggregate.sentenceClosureRate)}</td>)}</tr>
        <tr><td>Distinct generated bigrams</td>{evidence.results.map(item => <td key={item.candidate.id}>{percent(item.aggregate.meanDistinctBigramRatio)}</td>)}</tr>
        <tr><td>Cross-seed 3-gram similarity</td>{evidence.results.map(item => <td key={item.candidate.id}>{percent(item.aggregate.meanPairwiseThreeGramJaccardAcrossSeeds)}</td>)}</tr>
        <tr><td>No narrow contradiction-list hit</td>{evidence.results.map(item => <td key={item.candidate.id}>{percent(item.aggregate.contradictionProxyPassRate)}</td>)}</tr>
      </tbody></table>
      <div className={styles.selectors}>{evidence.results.map(result => <button key={result.candidate.id} className={selectedResult?.candidate.id === result.candidate.id ? styles.selected : ''} onClick={() => setCandidateId(result.candidate.id)}>{result.candidate.id}</button>)}</div>
      <div className={styles.selectors}>{evidence.protocol.generation.scenarios.map(item => <button key={item.id} className={scenarioId === item.id ? styles.selected : ''} onClick={() => setScenarioId(item.id)}>{item.id}</button>)}</div>
      {selectedSample && <article className={styles.sample}><small>FIXED SEED {evidence.protocol.generation.seeds[0]} · {scenario?.intent}</small><strong>{selectedSample.prompt}</strong><p>{selectedSample.continuation || '〈no visible continuation before EOS〉'}</p><span>Related keyword hits: {selectedSample.expectedKeywordHits.join(', ') || 'none'} · contradiction-list hits: {selectedSample.contradictionHits.join(', ') || 'none'} · longest exact training span: {selectedSample.longestExactTrainingSpanWordsUpTo12} words</span></article>}
      <aside className={styles.boundary}><strong>A perfect proxy can still be useless</strong><p>All three checkpoints avoided the tiny contradiction lists, including random noise. That 100% result does not certify consistency; it demonstrates that a deterministic rule answers only the narrow question encoded in its word list. {evidence.protocol.interpretationBoundary}</p></aside>
      <p className="evaluation-foot">{evidence.elapsedSeconds ? `All complete validation passes, 45 generations and overlap checks took ${evidence.elapsedSeconds.toFixed(2)} seconds.` : 'Evaluation is still running.'} Weights were loaded for inference and never updated.</p>
    </> : <p className="evaluation-foot">Run the Lesson 10 command in My Lab to populate checkpoint comparisons, scenario proxies, repetition/diversity measures, overlap checks and fixed outputs.</p>}
  </section>;
}
