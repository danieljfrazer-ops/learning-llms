'use client';

import { useEffect, useState } from 'react';

type Variant = { label: string; learningRateRecipe: string; gradientClipNorm: number | null; clippedSteps?: number; elapsedSeconds: number; evaluation: { validationLossMean: number; validationLossStd: number; generalisationGap: number; perplexity: number; continuation: string } };
type Experiment = { status: string; elapsedSeconds: number; variants: Variant[] };

export default function TrainingImprovementsPanel() {
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  useEffect(() => {
    fetch(`/data/shakespeare-training-improvements.json?t=${Date.now()}`, { cache: 'no-store' })
      .then(response => response.ok ? response.json() : null).then(setExperiment).catch(() => undefined);
  }, []);
  const baseline = experiment?.variants[0];
  return <section className="recipe-panel" id="training-recipe-dashboard">
    <div className="evaluation-head"><div><p className="kicker">CONTROLLED TRAINING EXPERIMENT</p><h2>Change the recipe, keep the model fixed</h2></div><span>{experiment?.status ?? 'Loading evidence…'}</span></div>
    <div className="recipe-grid">{experiment?.variants.map((variant, index) => {
      const delta = baseline ? variant.evaluation.validationLossMean - baseline.evaluation.validationLossMean : 0;
      return <article className={index === 1 ? 'recipe-winner' : ''} key={variant.label}><div><span>{index === 0 ? 'REFERENCE' : index === 1 ? 'NEW BEST' : 'CONTROLLED VARIANT'}</span><h3>{variant.label}</h3></div><dl><div><dt>Validation loss</dt><dd>{variant.evaluation.validationLossMean.toFixed(4)} ± {variant.evaluation.validationLossStd.toFixed(4)}</dd></div><div><dt>Versus baseline</dt><dd>{index === 0 ? 'reference' : `${delta > 0 ? '+' : ''}${delta.toFixed(4)}`}</dd></div><div><dt>Perplexity</dt><dd>{variant.evaluation.perplexity.toFixed(2)}</dd></div><div><dt>Generalisation gap</dt><dd>{variant.evaluation.generalisationGap.toFixed(4)}</dd></div></dl><pre>{variant.evaluation.continuation}</pre></article>;
    })}</div>
    <p className="evaluation-foot">All variants use the same 112,065-parameter architecture, corpus split, seed, batch size, context and 3,000 updates. Final metrics use the frozen five-repeat evaluation.</p>
  </section>;
}
