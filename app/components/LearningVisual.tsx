import type { ReactNode } from 'react';

type VisualStep = {
  label: string;
  detail?: string;
  tone?: 'input' | 'process' | 'evidence' | 'warning';
};

type LearningVisualProps = {
  eyebrow?: string;
  title: string;
  description: string;
  steps: VisualStep[];
  kind?: 'flow' | 'cycle' | 'compare' | 'stack';
  footer?: ReactNode;
  compact?: boolean;
};

export function LearningVisual({
  eyebrow = 'VISUAL MODEL',
  title,
  description,
  steps,
  kind = 'flow',
  footer,
  compact = false,
}: LearningVisualProps) {
  const titleId = `visual-${title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '')}`;

  return <figure className={`learning-visual learning-visual-${kind}${compact ? ' learning-visual-compact' : ''}`} aria-labelledby={titleId}>
    <figcaption>
      <span>{eyebrow}</span>
      <h2 id={titleId}>{title}</h2>
      <p>{description}</p>
    </figcaption>
    <ol className="visual-steps">
      {steps.map((step, index) => <li className={step.tone ? `visual-${step.tone}` : undefined} key={`${step.label}-${index}`}>
        <span className="visual-index" aria-hidden="true">{String(index + 1).padStart(2, '0')}</span>
        <strong>{step.label}</strong>
        {step.detail && <small>{step.detail}</small>}
      </li>)}
    </ol>
    {footer && <div className="visual-footer">{footer}</div>}
  </figure>;
}

const lessonVisuals: Record<string, Omit<LearningVisualProps, 'compact'>> = {
  'lab-setup': {
    title: 'A result is a chain, not a single file',
    description: 'Reproduction depends on preserving every link between the machine and the learner-visible evidence.',
    steps: [
      { label: 'Machine', detail: 'Apple silicon · memory · operating system', tone: 'input' },
      { label: 'Environment', detail: 'Python · MLX · locked dependencies', tone: 'process' },
      { label: 'Experiment', detail: 'Code · data · seed · configuration', tone: 'process' },
      { label: 'Evidence', detail: 'Metrics · samples · checkpoints', tone: 'evidence' },
    ],
  },
  'data-and-tokenisation': {
    title: 'How visible text becomes a training question',
    description: 'The target sequence is the input shifted one position to the left, so every position asks what comes next.',
    steps: [
      { label: 'Text', detail: '“To be”', tone: 'input' },
      { label: 'Tokens', detail: 'T · o · ␠ · b · e', tone: 'process' },
      { label: 'Integer IDs', detail: 'Stable vocabulary lookup', tone: 'process' },
      { label: 'Input → target', detail: 'T predicts o; o predicts space', tone: 'evidence' },
    ],
  },
  'random-baseline': {
    title: 'Random weights can run without having learned',
    description: 'Architecture determines the route through the model; training determines whether its numerical choices become useful.',
    steps: [
      { label: 'Token ID', detail: 'Select one vocabulary row', tone: 'input' },
      { label: 'Random score row', detail: '65 untrained logits', tone: 'warning' },
      { label: 'Probabilities', detail: 'Softmax rescales the scores', tone: 'process' },
      { label: 'Sampled character', detail: 'Valid output, no learned structure', tone: 'evidence' },
    ],
  },
  'bigram-training': {
    kind: 'cycle',
    title: 'One training step closes the correction loop',
    description: 'A forward calculation produces an error; gradients connect that error back to small weight changes.',
    steps: [
      { label: 'Predict', detail: 'Current weights produce logits', tone: 'input' },
      { label: 'Measure loss', detail: 'Compare with the known next token', tone: 'warning' },
      { label: 'Calculate gradients', detail: 'Estimate each weight’s influence', tone: 'process' },
      { label: 'Update weights', detail: 'AdamW applies a scaled change', tone: 'evidence' },
    ],
    footer: 'Repeat with new batches; checkpoints make the changing behaviour inspectable.',
  },
  'context-windows': {
    kind: 'compare',
    title: 'The question grows from one character to eight',
    description: 'Both models predict one next character, but the second receives a short ordered history.',
    steps: [
      { label: 'Bigram', detail: '… [q] → predict u', tone: 'warning' },
      { label: 'Fixed window', detail: '[the que] → predict e', tone: 'input' },
      { label: 'Concatenate', detail: 'Eight position vectors become one', tone: 'process' },
      { label: 'Next-token scores', detail: 'Earlier characters can change the answer', tone: 'evidence' },
    ],
  },
  'self-attention': {
    title: 'Attention routes information by content',
    description: 'A query compares with visible keys; the resulting weights control how their values are mixed.',
    steps: [
      { label: 'Query', detail: 'What pattern fits this position?', tone: 'input' },
      { label: 'Keys', detail: 'Labels for visible earlier positions', tone: 'process' },
      { label: 'Masked scores', detail: 'Future positions are forbidden', tone: 'warning' },
      { label: 'Weighted values', detail: 'A content-dependent mixture', tone: 'evidence' },
    ],
  },
  'tiny-transformer': {
    kind: 'stack',
    title: 'A transformer block has two different jobs',
    description: 'Attention moves information between positions; the feed-forward network transforms each position independently.',
    steps: [
      { label: 'Token + position', detail: 'Meaning and order enter together', tone: 'input' },
      { label: 'Multi-head attention', detail: 'Route several relationship types', tone: 'process' },
      { label: 'Feed-forward network', detail: 'Transform each position’s features', tone: 'process' },
      { label: 'Residual stream', detail: 'Preserve and accumulate information', tone: 'evidence' },
    ],
  },
  evaluation: {
    kind: 'compare',
    title: 'Fair comparison freezes the measuring instrument',
    description: 'Candidates change; evaluation data, prompts, seeds and generation settings do not.',
    steps: [
      { label: 'Frozen protocol', detail: 'Same held-out batches and prompts', tone: 'input' },
      { label: 'Candidate A', detail: 'Run without weight updates', tone: 'process' },
      { label: 'Candidate B', detail: 'Run under identical conditions', tone: 'process' },
      { label: 'Evidence bundle', detail: 'Loss + gap + perplexity + samples', tone: 'evidence' },
    ],
  },
  'prompt-playground': {
    title: 'The interface and the model are separate processes',
    description: 'The browser sends settings to a local service; only the Python process loads weights and performs tensor operations.',
    steps: [
      { label: 'Browser form', detail: 'Prompt · model · temperature · seed', tone: 'input' },
      { label: 'Local JSON request', detail: 'A defined API boundary', tone: 'process' },
      { label: 'MLX inference service', detail: 'Load checkpoint and generate', tone: 'process' },
      { label: 'Completion', detail: 'Text plus identifying metadata', tone: 'evidence' },
    ],
  },
  'training-improvements': {
    kind: 'compare',
    title: 'Improve the route while holding the vehicle fixed',
    description: 'The architecture and data stay constant so recipe changes can be interpreted separately.',
    steps: [
      { label: 'Same model', detail: '64-wide · two blocks · seed 42', tone: 'input' },
      { label: 'Warmup', detail: 'Begin with smaller updates', tone: 'process' },
      { label: 'Cosine decay + clipping', detail: 'Taper late updates; limit spikes', tone: 'process' },
      { label: 'Frozen evaluation', detail: 'Compare quality and cost fairly', tone: 'evidence' },
    ],
  },
  'scaling-experiment': {
    kind: 'compare',
    title: 'Width, depth and context buy different capabilities',
    description: 'Changing one axis at a time reveals why parameters alone do not predict runtime or memory.',
    steps: [
      { label: 'Wider', detail: 'More features per token; dense matrices grow fast', tone: 'process' },
      { label: 'Deeper', detail: 'More sequential transformation stages', tone: 'process' },
      { label: 'Longer context', detail: 'More visible tokens; larger attention grids', tone: 'warning' },
      { label: 'Measure trade-offs', detail: 'Loss · time · memory · parameters', tone: 'evidence' },
    ],
  },
  'final-evaluation': {
    title: 'A selected model becomes a claim only after replication',
    description: 'Independent initial weights and batch paths test whether one promising run was a lucky draw.',
    steps: [
      { label: 'Freeze specification', detail: 'Architecture · data · recipe · protocol', tone: 'input' },
      { label: 'Train seed 42', detail: 'Independent random path', tone: 'process' },
      { label: 'Train seeds 43 and 44', detail: 'Repeat the whole training process', tone: 'process' },
      { label: 'Report distribution', detail: 'Mean · variation · limitations', tone: 'evidence' },
    ],
  },
  'transition-to-subwords': {
    kind: 'compare',
    title: 'Subwords trade a larger vocabulary for shorter sequences',
    description: 'The underlying text is unchanged; only the pieces used to represent it are different.',
    steps: [
      { label: 'Characters', detail: 'Once upon… → many transparent pieces', tone: 'input' },
      { label: 'BPE merges', detail: 'Frequent adjacent bytes become reusable units', tone: 'process' },
      { label: 'Subwords', detail: 'Once · upon · a · time', tone: 'process' },
      { label: 'Same context budget', detail: 'Now covers more visible text', tone: 'evidence' },
    ],
  },
  'dataset-audit': {
    title: 'A dataset earns the right to become training material',
    description: 'Integrity checks narrow what the eventual model result can honestly mean.',
    steps: [
      { label: 'Source + manifest', detail: 'Provenance · licence · hashes · row IDs', tone: 'input' },
      { label: 'Structural checks', detail: 'Readable rows · lengths · required fields', tone: 'process' },
      { label: 'Leakage checks', detail: 'Exact and near duplicates across splits', tone: 'warning' },
      { label: 'Frozen subset', detail: 'Declared boundary for later experiments', tone: 'evidence' },
    ],
  },
  'tokenizer-experiment': {
    kind: 'compare', title: 'Tokenizer selection is a measured trade-off', description: 'A useful vocabulary must preserve text while balancing sequence length and model cost.',
    steps: [{ label: 'Candidate sizes', detail: 'Train only on the training split', tone: 'input' }, { label: 'Reversibility', detail: 'Decode must recover the source text', tone: 'warning' }, { label: 'Compression', detail: 'Compare tokens per story and coverage', tone: 'process' }, { label: 'Freeze one tokenizer', detail: 'Version its files and evidence', tone: 'evidence' }],
  },
  'sequence-batching': {
    title: 'Stories become causal windows without losing boundaries', description: 'Batch construction decides which tokens may predict which targets.',
    steps: [{ label: 'Separate stories', detail: 'Preserve start and end boundaries', tone: 'input' }, { label: 'Token streams', detail: 'Add structural tokens explicitly', tone: 'process' }, { label: 'Window + mask', detail: 'Hide padding and all future targets', tone: 'warning' }, { label: 'Input/target tensors', detail: 'Shapes are ready for training', tone: 'evidence' }],
  },
  'random-gpt-baseline': {
    title: 'Freeze the starting line before training', description: 'The official random checkpoint proves the complete pipeline works while establishing zero learned capability.',
    steps: [{ label: 'Frozen architecture', detail: 'Width · depth · heads · context', tone: 'input' }, { label: 'Random weights', detail: 'Seeded, materialised parameters', tone: 'warning' }, { label: 'Forward pass', detail: 'Finite loss and valid shapes', tone: 'process' }, { label: 'Checkpoint 0000', detail: 'Metrics and fixed-prompt sample', tone: 'evidence' }],
  },
  'first-pretraining': {
    kind: 'cycle', title: 'Pretraining repeats one next-token exercise', description: 'Millions of local corrections can accumulate into broader statistical regularities.',
    steps: [{ label: 'Sample a batch', detail: 'Known story-token sequences', tone: 'input' }, { label: 'Predict every next token', detail: 'Logits across the vocabulary', tone: 'process' }, { label: 'Score and update', detail: 'Loss → gradients → optimiser', tone: 'warning' }, { label: 'Check held-out text', detail: 'Evidence outside weight updates', tone: 'evidence' }],
  },
  'checkpoints-dashboard': {
    title: 'Snapshots turn a final model into a visible learning path', description: 'Consistent captures reveal when behaviour changes and make interrupted work recoverable.',
    steps: [{ label: 'Fixed intervals', detail: 'Same step schedule for every run', tone: 'input' }, { label: 'Save state', detail: 'Weights · config · optimiser where needed', tone: 'process' }, { label: 'Generate fixed prompts', detail: 'Same seed and sampling settings', tone: 'process' }, { label: 'Dashboard timeline', detail: 'Loss · speed · memory · samples', tone: 'evidence' }],
  },
  'training-recipe': {
    kind: 'compare', title: 'A recipe experiment changes one condition at a time', description: 'A controlled comparison separates useful optimisation changes from mere run-to-run noise.',
    steps: [{ label: 'Reference recipe', detail: 'Frozen baseline configuration', tone: 'input' }, { label: 'One changed factor', detail: 'Schedule, regularisation, or batch choice', tone: 'process' }, { label: 'Same budget', detail: 'Data, model, steps, and evaluation', tone: 'warning' }, { label: 'Decision', detail: 'Quality improvement weighed against cost', tone: 'evidence' }],
  },
  'scaling-budget': {
    kind: 'compare', title: 'Laptop scaling is a constrained allocation problem', description: 'The best experiment is not automatically the largest model the machine can hold.',
    steps: [{ label: 'Quality goal', detail: 'Held-out loss and story behaviour', tone: 'input' }, { label: 'Compute budget', detail: 'Time · memory · energy · iteration speed', tone: 'warning' }, { label: 'Scale one axis', detail: 'Width · depth · context · data', tone: 'process' }, { label: 'Select a frontier', detail: 'Best defensible trade-off', tone: 'evidence' }],
  },
  'story-evaluation': {
    kind: 'compare', title: 'Story quality needs several measuring lenses', description: 'No single number captures prediction quality, prompt adherence, consistency, or memorisation risk.',
    steps: [{ label: 'Held-out loss', detail: 'Next-token prediction quality', tone: 'input' }, { label: 'Behaviour rubric', detail: 'Adherence · consistency · repetition', tone: 'process' }, { label: 'Overlap checks', detail: 'Look for suspicious copying', tone: 'warning' }, { label: 'Fixed examples', detail: 'Representative successes and failures', tone: 'evidence' }],
  },
  'tinystories-playground': {
    title: 'A playground becomes an experiment when controls are visible', description: 'Prompts and sampling settings are inputs, not hidden knobs.',
    steps: [{ label: 'Prompt + checkpoint', detail: 'Identify the starting conditions', tone: 'input' }, { label: 'Sampling controls', detail: 'Temperature · length · seed', tone: 'process' }, { label: 'Local inference', detail: 'Weights stay fixed while tokens are sampled', tone: 'process' }, { label: 'Comparable runs', detail: 'Save settings beside each completion', tone: 'evidence' }],
  },
  'final-story-model': {
    title: 'Close the project with a bounded claim', description: 'Replication and comparison show what the model gained—and what the experiment still cannot establish.',
    steps: [{ label: 'Freeze candidate', detail: 'Data · tokenizer · architecture · recipe', tone: 'input' }, { label: 'Repeat training', detail: 'Independent seeds', tone: 'process' }, { label: 'Compare baselines', detail: 'Random · early · selected · Shakespeare', tone: 'process' }, { label: 'State the boundary', detail: 'Observed result, uncertainty, limitation', tone: 'evidence' }],
  },
  'task-and-dataset-audit': {
    title: 'Executable answers begin with a precise task contract', description: 'The natural-language question, table schema, SQL and returned answer must remain distinct.',
    steps: [{ label: 'Question + schema', detail: 'The model’s permitted input', tone: 'input' }, { label: 'Reference query', detail: 'Inspect structure and dataset risks', tone: 'process' }, { label: 'Sandbox execution', detail: 'Run SQL against the intended table', tone: 'warning' }, { label: 'Returned answer', detail: 'Define what counts as correct', tone: 'evidence' }],
  },
  'base-model-selection': {
    kind: 'compare', title: 'Choose a base model by constraints and evidence', description: 'Size alone does not determine whether a pretrained model is usable for this task.',
    steps: [{ label: 'Candidate models', detail: 'Licence · format · context · tokenizer', tone: 'input' }, { label: 'Local fit', detail: 'Memory and latency on target hardware', tone: 'warning' }, { label: 'Frozen baseline', detail: 'Same prompts and execution set', tone: 'process' }, { label: 'Selection record', detail: 'Quality, cost, and limitations', tone: 'evidence' }],
  },
  'prompt-formatting': {
    title: 'A prompt is a structured interface contract', description: 'Consistent boundaries help the model distinguish instructions, schema and question without exposing the answer.',
    steps: [{ label: 'Instruction', detail: 'Define the requested output', tone: 'input' }, { label: 'Table schema', detail: 'Names and types only', tone: 'process' }, { label: 'User question', detail: 'The natural-language request', tone: 'process' }, { label: 'SQL target', detail: 'No answer leakage in the input', tone: 'evidence' }],
  },
  'lora-fine-tuning': {
    kind: 'stack', title: 'LoRA adds a small trainable route beside frozen weights', description: 'The pretrained matrix remains fixed while two low-rank matrices learn a task-specific update.',
    steps: [{ label: 'Frozen base weight', detail: 'Preserve pretrained parameters', tone: 'input' }, { label: 'Low-rank down projection', detail: 'Compress the update path', tone: 'process' }, { label: 'Low-rank up projection', detail: 'Expand back to the layer shape', tone: 'process' }, { label: 'Combined output', detail: 'Base transformation + learned adapter', tone: 'evidence' }],
  },
  'execution-evaluation': {
    title: 'Judge SQL by what it does, not only how it looks', description: 'Different query strings can be equivalent, while plausible-looking SQL can return the wrong answer.',
    steps: [{ label: 'Generated SQL', detail: 'Treat as untrusted program text', tone: 'input' }, { label: 'Parse + restrict', detail: 'Enforce safe read-only execution', tone: 'warning' }, { label: 'Sandbox database', detail: 'Execute against the frozen table', tone: 'process' }, { label: 'Compare results', detail: 'Returned values determine correctness', tone: 'evidence' }],
  },
};

export function LessonVisual({ lessonSlug }: { lessonSlug: string }) {
  const visual = lessonVisuals[lessonSlug];
  if (!visual) return null;
  return <LearningVisual {...visual} />;
}

export function CourseMapVisual() {
  return <LearningVisual compact eyebrow="COURSE MAP" title="One mechanism, three levels of challenge" description="Each project preserves the prediction-and-evidence loop while changing what the model starts from and what success means." kind="compare" steps={[
    { label: 'Characters', detail: 'Train 4,225 weights from random noise', tone: 'input' },
    { label: 'Subword stories', detail: 'Scale pretraining from random weights', tone: 'process' },
    { label: 'English → SQL', detail: 'Adapt a pretrained base with LoRA', tone: 'process' },
    { label: 'Independent work', detail: 'Design your own evidence standard', tone: 'evidence' },
  ]} />;
}

export function ProjectJourneyVisual({ name, stages }: { name: string; stages: { name: string; state: string }[] }) {
  const sampled = stages.length > 6
    ? [stages[0], stages[Math.floor((stages.length - 1) / 3)], stages[Math.floor((stages.length - 1) * 2 / 3)], stages.at(-1)!]
    : stages;
  return <LearningVisual compact eyebrow="PROJECT JOURNEY" title={`${name}: from starting state to evidence`} description="The diagram samples the project arc; the complete, clickable sequence appears below." steps={sampled.map(stage => ({ label: stage.name, detail: stage.state === 'complete' ? 'Reference lesson complete' : stage.state === 'active' ? 'Current course stage' : 'Planned and evidence-gated', tone: stage.state === 'complete' ? 'evidence' : stage.state === 'active' ? 'process' : 'input' }))} />;
}
