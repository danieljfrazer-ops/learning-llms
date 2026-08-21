export type Project = {
  slug: string;
  number: string;
  name: string;
  shortName: string;
  description: string;
  stage: string;
  progress: number;
  status: string;
  method: string;
  objective: string;
  dataset: { name: string; detail: string; source: string; sourceLabel: string };
  stages: { name: string; state: 'complete' | 'active' | 'planned'; lesson: string }[];
  concepts: { term: string; explanation: string }[];
  furtherReading: { label: string; href: string }[];
};

export const projects: Project[] = [
  {
    slug: 'shakespeare', number: '01', name: 'Tiny Shakespeare', shortName: 'Shakespeare',
    description: 'Build a character-level language model from random weights and watch structure emerge.',
    stage: 'Bigram baseline complete', progress: 28, status: 'ACTIVE', method: 'Train from scratch',
    objective: 'Understand the complete language-model loop with the smallest model that still produces visible learning: data → tokens → predictions → loss → updated weights → generated text.',
    dataset: { name: 'Tiny Shakespeare', detail: 'Approximately 1 MB of dialogue from Shakespeare plays. We use a fixed 90/10 train/validation split and begin with individual characters as tokens.', source: 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', sourceLabel: 'Karpathy char-rnn dataset' },
    stages: [
      { name: 'Lab and wiki setup', state: 'complete', lesson: 'Create a reproducible project, record the hardware, and make results visible.' },
      { name: 'Random baseline', state: 'complete', lesson: 'Generate from random weights before learning so every later improvement has a meaningful comparison.' },
      { name: 'Bigram training', state: 'complete', lesson: 'Learn which character tends to follow another character.' },
      { name: 'Context windows', state: 'active', lesson: 'Let the model use several earlier characters instead of only one.' },
      { name: 'Self-attention', state: 'planned', lesson: 'Allow each position to selectively combine information from its context.' },
      { name: 'Tiny transformer', state: 'planned', lesson: 'Assemble embeddings, attention, feed-forward layers, residual paths and normalisation.' },
      { name: 'Evaluation', state: 'planned', lesson: 'Compare checkpoints on held-out loss and representative generations.' },
    ],
    concepts: [
      { term: 'Token', explanation: 'A unit represented by an integer. In this first project every distinct character—including spaces and line breaks—is a token.' },
      { term: 'Weight', explanation: 'A learned number inside the model. Random initial weights know nothing; training adjusts them to reduce prediction error.' },
      { term: 'Logit', explanation: 'An unnormalised score the model assigns to each possible next token.' },
      { term: 'Loss', explanation: 'A single number measuring prediction error. Lower validation loss generally indicates better next-token predictions on unseen text.' },
    ],
    furtherReading: [
      { label: 'MLX framework', href: 'https://github.com/ml-explore/mlx' },
      { label: 'Neural Networks: Zero to Hero', href: 'https://karpathy.ai/zero-to-hero.html' },
      { label: 'Tiny Shakespeare preparation', href: 'https://github.com/karpathy/llm.c/blob/master/dev/data/tinyshakespeare.py' },
    ],
  },
  {
    slug: 'tinystories', number: '02', name: 'TinyStories', shortName: 'TinyStories',
    description: 'Scale the same ideas into a small GPT that learns simple, coherent English stories.',
    stage: 'Planned', progress: 0, status: 'UP NEXT', method: 'Train from scratch',
    objective: 'Move from character imitation to subword language modelling and train a roughly 5–15M parameter transformer on a deliberately simple language distribution.',
    dataset: { name: 'TinyStories', detail: 'Synthetic short stories written with vocabulary familiar to young children. We will begin with a bounded subset before choosing whether to scale.', source: 'https://www.microsoft.com/en-us/research/publication/tinystories-how-small-can-language-models-be-and-still-speak-coherent-english/', sourceLabel: 'Microsoft Research' },
    stages: [
      { name: 'Dataset audit', state: 'planned', lesson: 'Inspect provenance, format, vocabulary, repetition and licence before training.' },
      { name: 'Subword tokenizer', state: 'planned', lesson: 'Balance vocabulary size against the number of tokens needed to represent text.' },
      { name: 'Scale experiment', state: 'planned', lesson: 'Choose architecture size from measured memory and speed rather than guesswork.' },
      { name: 'Pretraining', state: 'planned', lesson: 'Train next-token prediction while preserving comparable checkpoints.' },
      { name: 'Story evaluation', state: 'planned', lesson: 'Score grammar, consistency, diversity and prompt adherence alongside loss.' },
    ],
    concepts: [
      { term: 'Subword token', explanation: 'A reusable piece of a word, allowing a finite vocabulary to represent unfamiliar words efficiently.' },
      { term: 'Parameter count', explanation: 'The total number of trainable weights; a useful size measure, but not a quality measure by itself.' },
      { term: 'Pretraining', explanation: 'Learning general statistical patterns from a broad text corpus through next-token prediction.' },
    ],
    furtherReading: [
      { label: 'TinyStories paper and overview', href: 'https://www.microsoft.com/en-us/research/publication/tinystories-how-small-can-language-models-be-and-still-speak-coherent-english/' },
      { label: 'TinyStories paper on arXiv', href: 'https://arxiv.org/abs/2305.07759' },
    ],
  },
  {
    slug: 'sql', number: '03', name: 'English → SQL', shortName: 'English → SQL',
    description: 'Fine-tune a pretrained small model and evaluate answers by executing generated SQL.',
    stage: 'Planned', progress: 0, status: 'LATER', method: 'LoRA fine-tuning',
    objective: 'Contrast pretraining with adaptation: keep most pretrained weights fixed, train small adapter matrices, then evaluate generated queries by execution.',
    dataset: { name: 'WikiSQL', detail: 'Natural-language questions paired with table schemas and executable SQL structures. Execution provides a concrete evaluation signal.', source: 'https://github.com/salesforce/WikiSQL', sourceLabel: 'Official Salesforce WikiSQL repository' },
    stages: [
      { name: 'Task and dataset audit', state: 'planned', lesson: 'Define what counts as correct and inspect known limitations of the benchmark.' },
      { name: 'Base-model selection', state: 'planned', lesson: 'Measure candidate models for memory use, licence, context format and baseline accuracy.' },
      { name: 'Prompt formatting', state: 'planned', lesson: 'Represent the question and schema consistently without leaking answers.' },
      { name: 'LoRA fine-tuning', state: 'planned', lesson: 'Train a small number of adapter parameters while preserving base weights.' },
      { name: 'Execution evaluation', state: 'planned', lesson: 'Execute queries in a sandbox and compare returned answers, not just SQL strings.' },
    ],
    concepts: [
      { term: 'Fine-tuning', explanation: 'Continuing training from pretrained weights on a narrower task or style.' },
      { term: 'LoRA', explanation: 'Low-Rank Adaptation: small trainable matrices modify selected layers while the original model weights stay frozen.' },
      { term: 'Execution accuracy', explanation: 'The percentage of generated queries that return the correct result when executed.' },
    ],
    furtherReading: [
      { label: 'WikiSQL repository', href: 'https://github.com/salesforce/WikiSQL' },
      { label: 'MLX-LM fine-tuning', href: 'https://github.com/ml-explore/mlx-lm' },
      { label: 'LoRA paper', href: 'https://arxiv.org/abs/2106.09685' },
    ],
  },
];

export const glossary = [
  ['Autoregressive', 'Generating or predicting one token at a time while conditioning on earlier tokens.'],
  ['Backpropagation', 'Computing how much each weight contributed to the loss so the optimiser can update it.'],
  ['Batch', 'Several training examples processed together before one optimiser update.'],
  ['Checkpoint', 'A saved snapshot of model weights and enough training state to inspect or resume a run.'],
  ['Context window', 'The maximum number of earlier tokens visible when predicting the next token.'],
  ['Epoch', 'One complete pass through the training dataset. Token-based LLM runs are often tracked by steps instead.'],
  ['Gradient', 'The direction and magnitude in which changing a weight would change the loss.'],
  ['Inference', 'Using trained weights to make predictions or generate output without updating them.'],
  ['Learning rate', 'The scale of each optimiser update; too high can destabilise training and too low can make it impractically slow.'],
  ['Overfitting', 'Improving on training examples while getting worse or failing to improve on unseen validation data.'],
  ['Perplexity', 'An exponential transformation of average cross-entropy loss; lower means the correct next tokens are less surprising to the model.'],
  ['Seed', 'A number used to make pseudo-random initialisation and sampling reproducible.'],
  ['Tensor', 'A multidimensional array used to store token batches, activations, weights and gradients.'],
  ['Token', 'A discrete unit of text mapped to an integer before it enters a model.'],
  ['Validation set', 'Held-out examples used to measure generalisation during development but never used for weight updates.'],
  ['Weight', 'A trainable numerical parameter that transforms information inside a neural network.'],
] as const;

export function getProject(slug: string) { return projects.find((project) => project.slug === slug); }
