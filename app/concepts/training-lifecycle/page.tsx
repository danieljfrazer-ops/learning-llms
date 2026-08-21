import Link from 'next/link';
import WikiChrome from '@/app/components/WikiChrome';

const steps = [
  ['1', 'Define the question', 'State what the model should learn and how we will tell whether it learned it.'],
  ['2', 'Audit and split data', 'Inspect provenance and quality, then freeze train, validation and test sets before tuning decisions.'],
  ['3', 'Tokenise', 'Map text to integer IDs. Token choice determines vocabulary size and sequence length.'],
  ['4', 'Build a baseline', 'Generate from random weights and train the simplest plausible model before adding transformer complexity.'],
  ['5', 'Train', 'Predict next tokens, measure loss, backpropagate gradients and let an optimiser update weights.'],
  ['6', 'Checkpoint', 'Save comparable snapshots so we can inspect emergence and safely resume interrupted work.'],
  ['7', 'Evaluate', 'Use held-out loss plus task-relevant measures and representative generations.'],
  ['8', 'Change one thing', 'Run a controlled experiment, record the configuration, and compare it with the baseline.'],
];

export default function TrainingLifecycle() {
  return <WikiChrome active="path"><article className="article-page"><div className="breadcrumbs"><Link href="/">Dashboard</Link><span>/</span><strong>Learning path</strong></div><header className="article-hero single"><div><p className="kicker">THE REPEATABLE LOOP</p><h1>From raw text to evidence</h1><p>Every project follows the same scientific shape. The models and datasets change; the discipline of making comparable, reproducible experiments does not.</p></div></header><section className="lifecycle">{steps.map(step => <article key={step[0]}><span>{step[0]}</span><div><h2>{step[1]}</h2><p>{step[2]}</p></div></article>)}</section><aside className="principle-card"><p className="kicker">CORE PRINCIPLE</p><h2>Never improve a model you have not measured.</h2><p>A simple baseline makes progress legible. Without it, added complexity may produce more impressive-looking code without producing a better model.</p></aside></article></WikiChrome>;
}
