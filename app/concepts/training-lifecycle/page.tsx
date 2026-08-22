import Link from 'next/link';
import WikiChrome from '@/app/components/WikiChrome';
import { BeginnerOnly } from '@/app/components/BeginnerMode';

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
  return <WikiChrome active="path"><article className="article-page"><div className="breadcrumbs"><Link href="/">Dashboard</Link><span>/</span><strong>Learning path</strong></div><header className="article-hero single"><div><p className="kicker">THE REPEATABLE LOOP</p><h1>From raw text to evidence</h1><p>Every project follows the same scientific shape. The models and datasets change; the discipline of making comparable, reproducible experiments does not.</p></div></header><BeginnerOnly className="beginner-lesson-intro"><p className="kicker">BEGINNER MODE · THE BIG PICTURE</p><h2>Think like a coach, not a magician</h2><p>We give a student practice material, choose the alphabet it may use, ask it to predict missing answers, score the mistakes, and adjust it. We keep a separate quiz and change one teaching idea at a time.</p><aside><strong>The loop repeats</strong><span>Prepare → practise → score → save → compare → change one thing.</span></aside></BeginnerOnly><section className="lifecycle">{steps.map(step => <article key={step[0]}><span>{step[0]}</span><div><h2>{step[1]}</h2><p>{step[2]}</p><BeginnerOnly className="lifecycle-plain"><strong>In everyday terms:</strong> {({ '1':'Decide what “good” means before building.', '2':'Keep the exam pages away from the practice desk.', '3':'Give every piece of text a number the model can handle.', '4':'Record how the simplest student performs.', '5':'Practise, score mistakes, and make tiny corrections.', '6':'Create save points you can compare or restore.', '7':'Use the same exam and scoring rules for every candidate.', '8':'Alter one ingredient so you know what caused the difference.' } as Record<string,string>)[step[0]]}</BeginnerOnly></div></article>)}</section><aside className="principle-card"><p className="kicker">CORE PRINCIPLE</p><h2>Never improve a model you have not measured.</h2><p>A simple baseline makes progress legible. Without it, added complexity may produce more impressive-looking code without producing a better model.</p></aside></article></WikiChrome>;
}
