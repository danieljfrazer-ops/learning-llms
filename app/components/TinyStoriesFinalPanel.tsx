'use client';

import { useEffect, useState } from 'react';
import { useEvidenceUrl } from './EvidenceMode';

type Validation = { crossEntropyLoss: number; perplexity: number };
type SeedResult = {
  seed: number;
  reusedScalingRun: boolean;
  finalValidation: Validation;
  validTargetsSeen: number;
  samples: { prompt: string; continuation: string }[];
  behaviour: { aggregate: { adherenceProxyPassRate: number; sentenceClosureRate: number; meanRepeatedFourGramFraction: number; maximumObservedExactTrainingSpanWordsUpTo12: number } };
  performance: { elapsedSeconds: number; meanUpdateMilliseconds: number; peakMlxAllocationBytes: number };
};
type Project = { tokenUnit: string; vocabularySize: number; contextTokens: number; approximateContextCharacters: number; parameterCount: number; trainingUpdates: number; trainingExamples: string; randomValidationLoss: number; finalValidationLossMean: number; withinProjectLossReductionFraction: number; observedCapability: string; boundary: string };
type Evidence = {
  status: string;
  progress: { phase: string; seed?: number; completedUpdates?: number; plannedUpdates?: number; completedSeeds?: number; plannedSeeds?: number };
  seeds: SeedResult[];
  validationLossAcrossSeedsMean?: number;
  validationLossAcrossSeedsStd?: number;
  selectedSeed?: number;
  crossProject?: { comparisonBoundary: string; shakespeare: Project; tinyStories: Project; ratios: { tinyStoriesToShakespeareParameters: number; approximateContextCharacters: number } };
  elapsedSeconds?: number;
};

export default function TinyStoriesFinalPanel() {
  const evidenceUrl = useEvidenceUrl('tinystories-final.json');
  const [loaded, setLoaded] = useState<{ url: string; data: Evidence | null }>({ url: '', data: null });
  useEffect(() => {
    let active = true;
    const load = () => fetch(`${evidenceUrl}?t=${Date.now()}`, { cache: 'no-store' })
      .then(response => response.ok ? response.json() : null)
      .then(data => { if (active) setLoaded({ url: evidenceUrl, data }); })
      .catch(() => { if (active) setLoaded({ url: evidenceUrl, data: null }); });
    load();
    const timer = window.setInterval(load, 3000);
    return () => { active = false; window.clearInterval(timer); };
  }, [evidenceUrl]);
  const evidence = loaded.url === evidenceUrl ? loaded.data : null;
  const progress = evidence?.progress;

  return <section className="recipe-panel" id="tinystories-final-dashboard">
    <div className="evaluation-head"><div><p className="kicker">FINAL THREE-SEED CONFIRMATION</p><h2>One recipe, three independent training paths</h2></div><span>{evidence?.status ?? 'Not run yet'}</span></div>
    {evidence ? <>
      {evidence.status !== 'Complete' && <p className="evaluation-foot">{progress?.phase}{progress?.seed ? ` · seed ${progress.seed}` : ''}{progress?.completedUpdates !== undefined ? ` · update ${progress.completedUpdates}/${progress.plannedUpdates}` : ''} · {progress?.completedSeeds ?? evidence.seeds.length}/{progress?.plannedSeeds ?? 3} seeds captured</p>}
      <div className="evaluation-summary"><div><small>MEAN VALIDATION LOSS</small><strong>{evidence.validationLossAcrossSeedsMean?.toFixed(4) ?? 'Running'}</strong></div><div><small>BETWEEN-SEED STD</small><strong>{evidence.validationLossAcrossSeedsStd?.toFixed(4) ?? '—'}</strong></div><div><small>SELECTED SEED</small><strong>{evidence.selectedSeed ?? '—'}</strong></div><div><small>NEW RUN WALL TIME</small><strong>{evidence.elapsedSeconds ? `${evidence.elapsedSeconds.toFixed(1)} s` : '—'}</strong></div></div>
      <div className="recipe-grid">{evidence.seeds.map(seed => <article className={seed.seed === evidence.selectedSeed ? 'recipe-winner' : ''} key={seed.seed}><div><span>{seed.seed === evidence.selectedSeed ? 'SELECTED CHECKPOINT' : seed.reusedScalingRun ? 'VERIFIED REUSE' : 'INDEPENDENT REPLICATE'}</span><h3>Training seed {seed.seed}</h3></div><dl><div><dt>Validation loss</dt><dd>{seed.finalValidation.crossEntropyLoss.toFixed(4)}</dd></div><div><dt>Perplexity</dt><dd>{seed.finalValidation.perplexity.toFixed(2)}</dd></div><div><dt>Prompt proxy</dt><dd>{Math.round(seed.behaviour.aggregate.adherenceProxyPassRate * 15)}/15</dd></div><div><dt>Sentence closure</dt><dd>{Math.round(seed.behaviour.aggregate.sentenceClosureRate * 15)}/15</dd></div><div><dt>Valid targets seen</dt><dd>{seed.validTargetsSeen.toLocaleString()}</dd></div><div><dt>Mean update</dt><dd>{seed.performance.meanUpdateMilliseconds.toFixed(1)} ms</dd></div><div><dt>Peak MLX</dt><dd>{(seed.performance.peakMlxAllocationBytes / 1e9).toFixed(2)} GB</dd></div></dl><pre><mark>{seed.samples[0].prompt}</mark>{seed.samples[0].continuation}</pre></article>)}</div>
      {evidence.crossProject && <div className="final-project-comparison"><h3>Project 1 and Project 2 answer different questions</h3><p>{evidence.crossProject.comparisonBoundary}</p><table className="lesson-table"><thead><tr><th>Property</th><th>Shakespeare</th><th>TinyStories</th></tr></thead><tbody><tr><td>Token unit</td><td>{evidence.crossProject.shakespeare.tokenUnit}</td><td>{evidence.crossProject.tinyStories.tokenUnit}</td></tr><tr><td>Vocabulary</td><td>{evidence.crossProject.shakespeare.vocabularySize.toLocaleString()}</td><td>{evidence.crossProject.tinyStories.vocabularySize.toLocaleString()}</td></tr><tr><td>Context</td><td>{evidence.crossProject.shakespeare.contextTokens} tokens ≈ {evidence.crossProject.shakespeare.approximateContextCharacters.toFixed(0)} characters</td><td>{evidence.crossProject.tinyStories.contextTokens} tokens ≈ {evidence.crossProject.tinyStories.approximateContextCharacters.toFixed(0)} characters</td></tr><tr><td>Parameters</td><td>{evidence.crossProject.shakespeare.parameterCount.toLocaleString()}</td><td>{evidence.crossProject.tinyStories.parameterCount.toLocaleString()} · {evidence.crossProject.ratios.tinyStoriesToShakespeareParameters.toFixed(1)}× Shakespeare</td></tr><tr><td>Within-project loss reduction</td><td>{(evidence.crossProject.shakespeare.withinProjectLossReductionFraction * 100).toFixed(1)}%</td><td>{(evidence.crossProject.tinyStories.withinProjectLossReductionFraction * 100).toFixed(1)}%</td></tr><tr><td>Observed strength</td><td>{evidence.crossProject.shakespeare.observedCapability}</td><td>{evidence.crossProject.tinyStories.observedCapability}</td></tr><tr><td>Boundary</td><td>{evidence.crossProject.shakespeare.boundary}</td><td>{evidence.crossProject.tinyStories.boundary}</td></tr></tbody></table></div>}
    </> : <p className="evaluation-foot">Run the Lesson 12 command in My Lab to create independent seeds, final checkpoint evidence, and the cross-project comparison. A fresh clone remains blank here.</p>}
  </section>;
}
