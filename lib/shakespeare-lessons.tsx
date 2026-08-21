import type { ReactNode } from 'react';
import Link from 'next/link';

export type RichLesson = {
  slug: string;
  title: string;
  summary: string;
  outcome: string;
  evidence: string;
  sections: { id: string; title: string; body: ReactNode }[];
};

function Term({ id, children }: { id: string; children: ReactNode }) {
  return <Link className="inline-term" href={`/glossary#${id}`}>{children}</Link>;
}

function Source({ href, children }: { href: string; children: ReactNode }) {
  return <a className="inline-source" href={href} target="_blank" rel="noreferrer">{children} ↗</a>;
}

function Code({ children }: { children: string }) {
  return <pre className="lesson-code"><code>{children}</code></pre>;
}

export const shakespeareLessons: RichLesson[] = [
  {
    slug: 'lab-setup',
    title: 'Build the local learning lab',
    summary: 'Establish an isolated ML environment, a living wiki, a repeatable experiment structure, and a version-controlled record before touching model code.',
    outcome: 'A Python 3.12 + MLX GPU lab and a hot-refresh wiki were running locally on the M5 MacBook Air.',
    evidence: 'Completed · verified build · Git commit f0b05dd',
    sections: [
      {
        id: 'why-a-lab', title: 'Why environment setup is part of the model', body: <>
          <p>An ML result depends on more than Python source. Runtime versions, numerical libraries, hardware, data and command-line settings can all change behaviour. We therefore made <Term id="reproducibility">reproducibility</Term> a first-class lesson instead of treating setup as invisible plumbing.</p>
          <p>The hardware check recorded an Apple M5 processor, 32 GB of <Term id="unified-memory">unified memory</Term>, and an arm64 environment. MLX reported its default compute device as <code>Device(gpu, 0)</code>, confirming that tensor work would use the Apple GPU.</p>
        </>,
      },
      {
        id: 'tooling-installed', title: 'What was installed and what each tool does', body: <>
          <div className="tool-detail-grid">
            <article><h3>Python 3.12.13</h3><p>The language used for data preparation and model training. We deliberately avoided relying on the machine-wide Python 3.14 environment.</p></article>
            <article><h3>uv</h3><p>An existing Python project manager used to create the isolated <Term id="virtual-environment">virtual environment</Term>, resolve dependencies and lock exact versions.</p></article>
            <article><h3>MLX 0.32.0</h3><p>Apple&apos;s array and machine-learning framework. It provides <Term id="tensor">tensors</Term>, neural-network layers, automatic differentiation and optimisers. See the <Source href="https://github.com/ml-explore/mlx">official MLX repository</Source>.</p></article>
            <article><h3>mlx-metal 0.32.0</h3><p>The Metal support package used by MLX to execute numerical operations on the Apple GPU.</p></article>
            <article><h3>Node.js 24.14.0</h3><p>The JavaScript runtime used for the wiki. The initial scaffold command inherited Node 20 and emitted an engine warning; development and builds were then run with the installed Node 24 runtime required by the site stack.</p></article>
            <article><h3>Vinext + React</h3><p>The local documentation application: Vinext 1.0.0-beta.3, React 19.2.6 and Next-compatible routing. Its <Term id="hot-module-reload">hot module reload</Term> updates open pages as lessons change.</p></article>
          </div>
          <aside className="lesson-caveat"><strong>Dependency note</strong><p>The initial npm install reported 13 known dependency advisories: 1 low, 2 moderate and 10 high across the full development dependency tree. We did not run the suggested forced automatic upgrade because a forced major-version rewrite can break the generated stack. The wiki is local-only; dependency review and targeted upgrades are required before treating it as an internet-facing production application.</p></aside>
        </>,
      },
      {
        id: 'environment-command', title: 'The reproducible Python environment', body: <>
          <p>The project declares MLX in <code>pyproject.toml</code>, while <code>uv.lock</code> records the resolved package versions. This command created <code>.venv/</code> with Python 3.12 and installed MLX plus its Metal package:</p>
          <Code>{`uv sync --python 3.12`}</Code>
          <p>A <Term id="virtual-environment">virtual environment</Term> prevents packages required by this course from silently mixing with packages used by unrelated projects. The lock file makes a future reinstall far more likely to reproduce this environment.</p>
        </>,
      },
      {
        id: 'project-shape', title: 'How the project is organised', body: <>
          <Code>{`LearningLLMs/
├── ml/                 readable training programs
├── data/raw/           downloaded source text (not committed)
├── experiments/        run configuration and model checkpoints
├── public/data/        metrics consumed by the live dashboard
├── app/ and lib/       wiki pages and structured lesson content
├── pyproject.toml      Python dependencies
├── uv.lock             exact Python resolution
└── package.json        wiki dependencies and scripts`}</Code>
          <p>Large generated <Term id="checkpoint">checkpoints</Term>, raw data and virtual-environment files are excluded from Git. Source, lesson text, configurations and the compact metrics record are committed. This keeps the repository reviewable without losing the evidence needed to explain a run.</p>
        </>,
      },
      {
        id: 'live-refresh', title: 'How the open wiki refreshes', body: <>
          <p>Hand-authored lesson changes use hot module reload. During training, the Python process atomically rewrites <code>public/data/shakespeare-metrics.json</code>. The dashboard requests that file every two seconds with caching disabled. This separates model training from presentation: Python only emits a small evidence record, while the wiki decides how to display it.</p>
          <p>The local server is started with <code>npm run dev</code>. The current application requires Node 22.13 or newer; this project used Node 24.14.0.</p>
        </>,
      },
    ],
  },
  {
    slug: 'data-and-tokenisation',
    title: 'Load text and turn it into tensors',
    summary: 'Download Tiny Shakespeare, inspect its size, create a character vocabulary, encode text as integers, and freeze a train/validation split.',
    outcome: '1,115,394 characters became integer tensors: 1,003,854 for training and 111,540 for validation.',
    evidence: 'Observed vocabulary: 65 unique characters · source file: 40,000 lines',
    sections: [
      {
        id: 'download', title: 'Download the source corpus', body: <>
          <p>We downloaded the Tiny Shakespeare text directly from the <Source href="https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare">Karpathy char-rnn dataset repository</Source>. The resulting UTF-8 file contained 40,000 lines and 1,115,394 bytes.</p>
          <Code>{`curl --fail --location \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt \
  --output data/raw/tiny-shakespeare.txt

wc -c -l data/raw/tiny-shakespeare.txt
# 40000 lines · 1115394 bytes`}</Code>
          <p>The raw corpus is not committed to Git because it can always be recovered from its recorded source. Dataset provenance remains linked in the lesson.</p>
        </>,
      },
      {
        id: 'characters-as-tokens', title: 'Choose characters as tokens', body: <>
          <p>A <Term id="token">token</Term> is the discrete unit presented to a language model. For this first experiment we used a <Term id="character-level-tokenizer">character-level tokenizer</Term>: letters, punctuation, spaces and line breaks are all separate tokens.</p>
          <p>This is inefficient for serious language modelling, but pedagogically valuable. There are only 65 possible tokens, encoding is completely transparent, and there is no hidden pretrained tokenizer to confuse what the model learned itself.</p>
          <Code>{`text = args.data.read_text(encoding="utf-8")
vocabulary = sorted(set(text))
char_to_id = {
    character: index
    for index, character in enumerate(vocabulary)
}`}</Code>
          <p><code>sorted(set(text))</code> finds the 65 unique characters and gives them a stable order. <code>char_to_id</code> is the <Term id="vocabulary">vocabulary</Term> mapping used to replace each character with an integer from 0 to 64.</p>
        </>,
      },
      {
        id: 'encode', title: 'Encode text into integer tensors', body: <>
          <Code>{`def encode(text, char_to_id):
    return [char_to_id[character] for character in text]

train_data = mx.array(
    encode(text[:split_index], char_to_id),
    dtype=mx.int32,
)`}</Code>
          <p>The Python list of IDs is converted into an MLX <Term id="tensor">tensor</Term> with 32-bit integer values. Neural networks operate on numbers, so encoding is the bridge between visible text and model input. Decoding applies the inverse mapping to make generated IDs readable again.</p>
        </>,
      },
      {
        id: 'split', title: 'Freeze the 90/10 split', body: <>
          <Code>{`split_index = int(len(text) * 0.9)
train_data = text[:split_index]       # 1,003,854 characters
validation_data = text[split_index:]  #   111,540 characters`}</Code>
          <p>The first 90% is used to update model <Term id="weight">weights</Term>. The final 10% is a <Term id="validation-set">validation set</Term>: it is used for measurement but never for weight updates. This gives us a basic check that the model is learning patterns that transfer beyond its sampled training batches.</p>
          <aside className="lesson-caveat"><strong>Important limitation</strong><p>Both splits contain Shakespeare drawn from the same compiled corpus. A lower validation loss therefore shows in-domain generalisation, not general English ability or factual knowledge. We will use stricter evaluation where the task requires it.</p></aside>
        </>,
      },
      {
        id: 'batches', title: 'Create next-character training batches', body: <>
          <Code>{`starts = mx.random.randint(
    0, data.size - block_size - 1,
    shape=(batch_size,),
)
positions = starts[:, None] + mx.arange(block_size)[None, :]
inputs = data[positions]
targets = data[positions + 1]`}</Code>
          <p>With batch size 64 and block size 64, one update sees 4,096 adjacent input/target pairs. If an input position contains <code>q</code>, its target is the character immediately following that <code>q</code> in the corpus. The current bigram model treats those pairs independently even though they are stored in 64-character blocks; the same batch structure will become useful when the next model gains a larger <Term id="context-window">context window</Term>.</p>
        </>,
      },
    ],
  },
  {
    slug: 'random-baseline',
    title: 'Create and invoke a random-weight model',
    summary: 'Build the smallest next-token neural model, inspect its 4,225 random weights, and generate text before performing a single training update.',
    outcome: 'The untrained model produced uniformly structureless characters, establishing the baseline that later samples must beat.',
    evidence: 'Checkpoint step 0 · seed 42 · 65 × 65 = 4,225 trainable weights',
    sections: [
      {
        id: 'architecture', title: 'The entire architecture is one table', body: <>
          <Code>{`class BigramLanguageModel(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        self.next_token_scores = nn.Embedding(
            vocabulary_size,
            vocabulary_size,
        )

    def __call__(self, token_ids: mx.array) -> mx.array:
        return self.next_token_scores(token_ids)`}</Code>
          <p>The model subclasses MLX <code>nn.Module</code>. Its only layer is an <Term id="embedding">embedding table</Term> with 65 rows and 65 columns. Each row belongs to a current character; each column is the score for one possible next character.</p>
          <p>That gives exactly <code>65 × 65 = 4,225</code> trainable <Term id="weight">weights</Term>. This is a <Term id="bigram-model">bigram model</Term> because its prediction depends on only two positions: the current character and the next character.</p>
        </>,
      },
      {
        id: 'random-weights', title: 'Where the random weights came from', body: <>
          <Code>{`mx.random.seed(42)
model = BigramLanguageModel(len(vocabulary))
mx.eval(model.parameters())`}</Code>
          <p>Constructing <code>nn.Embedding</code> asks MLX to initialise its table with small random values. Setting the <Term id="seed">random seed</Term> immediately beforehand makes that pseudo-random initialisation repeatable.</p>
          <p>No text has entered the model at this point. A random value does not encode a letter, word or rule—it only breaks symmetry so different weights can later receive different <Term id="gradient">gradients</Term>. <code>mx.eval</code> materialises MLX&apos;s lazily constructed parameter arrays.</p>
        </>,
      },
      {
        id: 'invoke', title: 'What “invoking the model” means', body: <>
          <Code>{`current = mx.array([start_id])
logits = model(current)[-1]`}</Code>
          <p>Calling <code>model(current)</code> invokes the model&apos;s <code>__call__</code> method. If <code>current</code> is the integer ID for a line break, the embedding layer selects that character&apos;s row of 65 values. Those raw next-token scores are called <Term id="logit">logits</Term>.</p>
          <p>The model returns scores, not text. Generation code must choose a token from those scores, append it to the output and invoke the model again with the newly selected token.</p>
        </>,
      },
      {
        id: 'sample', title: 'Turn scores into a generated sequence', body: <>
          <Code>{`for _ in range(280):
    logits = model(current)[-1] / 0.9
    next_id = int(mx.random.categorical(logits).item())
    generated.append(next_id)
    current = mx.array([next_id])`}</Code>
          <p><Term id="temperature">Temperature</Term> rescales the logits before sampling. MLX&apos;s categorical sampler converts their relative scores into a probability distribution and randomly selects the next token. This is repeated 280 times, making generation <Term id="autoregressive">autoregressive</Term>.</p>
          <p>Because the untrained logits are random, the output is random-looking too:</p>
          <blockquote className="model-sample">AM?EWAKvDhgtWQMbubo:rG&apos;s3zfXSruc;kJza…</blockquote>
          <p>This sample was captured by calling <code>capture(0)</code> before the training loop. Saving the random baseline prevents us from judging later output only by intuition.</p>
        </>,
      },
      {
        id: 'limits', title: 'What this model can and cannot represent', body: <>
          <p>After training, a row can learn that <code>q</code> is often followed by <code>u</code>, or that a colon is often followed by a line break. It cannot know the earlier word, speaker or sentence because its <Term id="context-window">context window</Term> is exactly one character.</p>
          <p>This deliberate limitation gives the course a clean controlled experiment: when a longer-context model improves, we can attribute the new behaviour to additional context rather than a change of dataset or tokenizer.</p>
        </>,
      },
    ],
  },
  {
    slug: 'bigram-training',
    title: 'Train the bigram model and measure the ceiling',
    summary: 'Optimise the random 65×65 score table against real adjacent characters, save checkpoints, and compare held-out loss and generated text.',
    outcome: 'Validation loss reached approximately 2.489 by step 100, while generated text gained English-like character transitions but no durable meaning.',
    evidence: '400 updates · batch 64 × 64 · AdamW learning rate 0.03 · weight decay 0',
    sections: [
      {
        id: 'command', title: 'The command that ran the experiment', body: <>
          <Code>{`.venv/bin/python ml/shakespeare_bigram.py --steps 400`}</Code>
          <p>The command uses the project&apos;s isolated Python interpreter. Default arguments select seed 42, batch size 64, block size 64, learning rate 0.03, the downloaded dataset and the run identifier <code>shakespeare-bigram-001</code>.</p>
        </>,
      },
      {
        id: 'loss', title: 'Give wrong predictions a numerical cost', body: <>
          <Code>{`def loss_fn(model, inputs, targets):
    logits = model(inputs)
    return nn.losses.cross_entropy(
        logits,
        targets,
        reduction="mean",
    )`}</Code>
          <p><Term id="cross-entropy">Cross-entropy loss</Term> penalises the model when the correct next character receives low probability. Averaging over 4,096 pairs produces one scalar loss for the update. The number is useful for comparison; it is not a percentage or a direct grammar score.</p>
        </>,
      },
      {
        id: 'gradients', title: 'Turn loss into weight updates', body: <>
          <Code>{`optimiser = optim.AdamW(
    learning_rate=0.03,
    weight_decay=0.0,
)
loss_and_grad = nn.value_and_grad(model, loss_fn)

loss, gradients = loss_and_grad(model, inputs, targets)
optimiser.update(model, gradients)
mx.eval(model.parameters(), optimiser.state, loss)`}</Code>
          <p>MLX <Term id="automatic-differentiation">automatic differentiation</Term> calculates the <Term id="gradient">gradient</Term> of the loss with respect to every weight. The <Term id="adamw">AdamW optimiser</Term> uses those gradients to update the table. One calculation and update is a training <Term id="training-step">step</Term>.</p>
          <p>Weight decay was explicitly disabled because this tiny lookup table is a teaching baseline, not a tuned production model. The relatively large learning rate works here because there are only 4,225 independent scores to learn.</p>
        </>,
      },
      {
        id: 'checkpoints', title: 'Capture comparable checkpoints', body: <>
          <p>We saved weights and generated a 280-character sample at steps 0, 1, 25, 100 and 400. Each weight snapshot uses the <Term id="safetensors">safetensors</Term> format. The dashboard&apos;s JSON record stores exact step, train loss, validation loss and generated text, plus the run&apos;s last-updated timestamp.</p>
          <table className="lesson-table"><thead><tr><th>Step</th><th>Train loss</th><th>Validation loss</th><th>Visible behaviour</th></tr></thead><tbody>
            <tr><td>0</td><td>—</td><td>—</td><td>Unstructured characters</td></tr>
            <tr><td>1</td><td>4.1873</td><td>4.0614</td><td>Still effectively random</td></tr>
            <tr><td>25</td><td>2.5049</td><td>2.5499</td><td>Spaces and common letter pairs emerge</td></tr>
            <tr><td>100</td><td>2.4906</td><td>2.4887</td><td>Word-like fragments and speaker-label texture</td></tr>
            <tr><td>400</td><td>2.4387</td><td>2.4889</td><td>No meaningful improvement over step 100</td></tr>
          </tbody></table>
        </>,
      },
      {
        id: 'interpret', title: 'Interpret the result without overclaiming', body: <>
          <p>The large early loss reduction shows that the model learned real next-character statistics. The generated sample now contains patterns such as capitalised speaker-like fragments, punctuation and English-looking syllables.</p>
          <p>The 0.00014 difference between validation loss at steps 100 and 400 is too small to treat as meaningful: validation was estimated from 20 randomly sampled batches, so measurement noise can easily exceed that difference. The honest conclusion is a plateau around 2.49.</p>
          <aside className="lesson-caveat"><strong>The ceiling is architectural</strong><p>More updates cannot give a one-character model memory it does not possess. Our next experiment changes one variable—the <Term id="context-window">context window</Term>—while holding the corpus, tokenizer and evaluation pattern steady.</p></aside>
        </>,
      },
    ],
  },
  {
    slug: 'context-windows',
    title: 'Give the model short-term memory',
    summary: 'Replace the one-character lookup with a model that can condition each prediction on several earlier characters.',
    outcome: 'Planned next experiment: quantify what additional context improves before introducing self-attention.',
    evidence: 'Active lesson · implementation not run yet',
    sections: [
      { id: 'question', title: 'The question this experiment will answer', body: <><p>Does looking at several earlier characters reduce held-out loss and produce more stable word fragments than the bigram baseline? We will keep Tiny Shakespeare, the 65-character vocabulary, the train/validation split and checkpoint prompts unchanged.</p></> },
      { id: 'approach', title: 'The first longer-context architecture', body: <><p>We will embed every token in a short <Term id="context-window">context window</Term>, combine those embeddings, and predict the next character with a small feed-forward network. This is sometimes described as a fixed-window neural language model.</p><p>It deliberately comes before <Term id="self-attention">self-attention</Term>. That lets us separate the benefit of having memory from the benefit of attention&apos;s content-dependent routing.</p></> },
      { id: 'comparison', title: 'What will remain comparable', body: <><ul><li>Same dataset and 90/10 split</li><li>Same character tokenizer and vocabulary</li><li>Same random baseline checkpoint</li><li>Same checkpoint sampling temperature and length</li><li>Validation loss plus side-by-side generated samples</li></ul><p>Parameter count and context length will be recorded explicitly because both increase capacity.</p></> },
    ],
  },
];

export function getShakespeareLesson(slug: string) {
  return shakespeareLessons.find(lesson => lesson.slug === slug);
}
