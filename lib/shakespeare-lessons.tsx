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
    summary: 'Replace the one-character lookup with an eight-character fixed-window network, then measure whether ordered short-term memory improves held-out prediction and generated text.',
    outcome: 'The eight-character MLP reduced validation loss from the bigram plateau of about 2.49 to 1.9456 and generated recognisable speaker labels plus longer word fragments.',
    evidence: 'Completed · 43,361 parameters · 2,000 updates · 2.38 seconds recorded training',
    sections: [
      { id: 'question', title: 'Change one capability: memory', body: <><p>The bigram model could see one current character. This experiment asks whether an eight-character <Term id="context-window">context window</Term> improves prediction while retaining the same Tiny Shakespeare corpus, 65-character tokenizer, 90/10 split, sampling temperature and checkpoint length.</p><p>Parameter count cannot remain fixed because the new architecture needs embeddings and dense layers. We record that increase explicitly rather than pretending context is the only capacity change.</p></> },
      { id: 'batch-shape', title: 'Build one eight-character question per example', body: <>
        <Code>{`starts = mx.random.randint(
    0, data.size - context_size - 1,
    shape=(batch_size,),
)
positions = starts[:, None] + mx.arange(context_size)[None, :]
inputs = data[positions]                 # shape: (128, 8)
targets = data[starts + context_size]    # shape: (128,)`}</Code>
        <p>Each row asks one question: “Given these eight ordered characters, what is the ninth?” The previous bigram batch predicted after every individual position. This fixed-window model instead makes 128 next-character predictions per update.</p>
      </> },
      { id: 'architecture', title: 'Embed, concatenate, transform, predict', body: <>
        <Code>{`self.token_embedding = nn.Embedding(65, 32)
self.hidden = nn.Linear(8 * 32, 128)
self.output = nn.Linear(128, 65)

embeddings = self.token_embedding(token_ids)  # (128, 8, 32)
flattened = embeddings.reshape(128, 8 * 32)   # (128, 256)
hidden = nn.gelu(self.hidden(flattened))       # (128, 128)
logits = self.output(hidden)                   # (128, 65)`}</Code>
        <p>The <Term id="embedding">embedding</Term> turns each token into 32 learned features. Concatenation preserves position: the first character always occupies the first 32 values, the second occupies the next 32, and so on. A <Term id="feed-forward-network">feed-forward network</Term> maps the resulting 256 values through a 128-unit <Term id="hidden-layer">hidden layer</Term>.</p>
        <p><Term id="gelu">GELU</Term> is the nonlinear activation. Without a nonlinearity, two dense layers would collapse mathematically into one linear transformation and gain far less expressive power.</p>
      </> },
      { id: 'parameters', title: 'Account for all 43,361 parameters', body: <>
        <table className="lesson-table"><thead><tr><th>Component</th><th>Calculation</th><th>Parameters</th></tr></thead><tbody><tr><td>Token embedding</td><td>65 × 32</td><td>2,080</td></tr><tr><td>Hidden weights + bias</td><td>256 × 128 + 128</td><td>32,896</td></tr><tr><td>Output weights + bias</td><td>128 × 65 + 65</td><td>8,385</td></tr><tr><td><strong>Total</strong></td><td>sum</td><td><strong>43,361</strong></td></tr></tbody></table>
        <p>The model has roughly ten times the bigram&apos;s 4,225 parameters. Better results therefore demonstrate the capability of this small fixed-context architecture as configured—not a pure context-length ablation.</p>
      </> },
      { id: 'invoke', title: 'Train and invoke the fixed-window model', body: <>
        <Code>{`.venv/bin/python ml/shakespeare_context.py --steps 2000`}</Code>
        <p>Training used AdamW, learning rate 0.003, weight decay 0.01 and batch size 128. At generation time the model starts with eight line-break tokens, predicts one next token, discards the oldest context token and appends the new token:</p>
        <Code>{`context = context[1:] + [next_id]
logits = model(mx.array([context]))[0] / 0.9`}</Code>
        <p>This sliding window keeps the invocation shape fixed at <code>(1, 8)</code>.</p>
      </> },
      { id: 'results', title: 'Observe the gain from short-term memory', body: <>
        <table className="lesson-table"><thead><tr><th>Step</th><th>Train loss</th><th>Validation loss</th><th>Observation</th></tr></thead><tbody><tr><td>0</td><td>—</td><td>4.2043</td><td>Random characters</td></tr><tr><td>50</td><td>2.6702</td><td>2.5289</td><td>Local word texture begins</td></tr><tr><td>250</td><td>2.3753</td><td>2.2096</td><td>Longer fragments and line structure</td></tr><tr><td>1,000</td><td>1.9190</td><td>2.0344</td><td>Speaker-like labels and phrases</td></tr><tr><td>2,000</td><td>1.9498</td><td>1.9456</td><td>Best held-out result so far</td></tr></tbody></table>
        <blockquote className="model-sample">KING HENRY VI:\nHid\nThat upuss sead oftenter wasts\nWhich here, in be a, lay…</blockquote>
        <p>The words remain mostly invented and sentence meaning is unreliable, but the model now maintains enough local structure to form a real speaker label and longer English-like chunks. Recorded training, including checkpoint evaluation and generation, took 2.38 seconds on the Apple GPU.</p>
      </> },
      { id: 'limits', title: 'Why a fixed window is still awkward', body: <><p>Every position has a permanently assigned slot in one large input vector. Increasing context from 8 to 64 would multiply the first dense layer&apos;s input size by eight. More importantly, the model cannot decide dynamically that a nearby colon matters more than an unrelated character.</p><p><Term id="self-attention">Self-attention</Term>, introduced next from the <Source href="https://arxiv.org/abs/1706.03762">Attention Is All You Need paper</Source>, provides content-dependent routing across a longer sequence.</p></> },
    ],
  },
  {
    slug: 'self-attention',
    title: 'Let characters choose what to attend to',
    summary: 'Implement one causal self-attention head over a 64-character window and test the common misconception that attention alone must beat a simpler network.',
    outcome: 'Single-head attention reached validation loss 2.1889: better than the bigram, but worse than the eight-character MLP. Attention alone was not the winning architecture.',
    evidence: 'Completed · 28,993 parameters · 2,000 updates · 5.56 seconds recorded training',
    sections: [
      { id: 'purpose', title: 'Attention is dynamic information routing', body: <><p>The fixed-window MLP always combines its eight positions through the same dense weights. <Term id="self-attention">Self-attention</Term> instead lets each position calculate how strongly it should read from each available earlier position. Our implementation follows the scaled dot-product idea introduced in <Source href="https://arxiv.org/abs/1706.03762">Attention Is All You Need</Source>, but uses only one head and one block.</p><p>The training window grows from 8 to 64 characters. A <Term id="causal-mask">causal mask</Term> prevents information leaking from the future target into its prediction.</p></> },
      { id: 'positions', title: 'Add token meaning and position', body: <>
        <Code>{`positions = mx.arange(token_ids.shape[1])
hidden = (
    self.token_embedding(token_ids)
    + self.position_embedding(positions)
)`}</Code>
        <p>Self-attention does not inherently know order. A learned <Term id="positional-embedding">positional embedding</Term> is added to each 64-dimensional token embedding so the same character at positions 2 and 20 can be represented differently.</p>
      </> },
      { id: 'qkv', title: 'Produce queries, keys and values', body: <>
        <Code>{`queries = self.query(inputs)
keys = self.key(inputs)
values = self.value(inputs)

scores = (
    queries @ keys.transpose(0, 2, 1)
) / math.sqrt(model_size)`}</Code>
        <p>A <Term id="query-vector">query</Term> represents what the current position is looking for. A <Term id="key-vector">key</Term> represents what each candidate position offers. Their dot product becomes an <Term id="attention-score">attention score</Term>. The <Term id="value-vector">value</Term> contains the information that will actually be blended if a position receives attention.</p>
        <p>Division by <code>sqrt(64)</code> keeps score magnitudes controlled so the following softmax does not become unnecessarily saturated.</p>
      </> },
      { id: 'mask', title: 'Block the future, then normalise', body: <>
        <Code>{`future_mask = mx.triu(
    mx.full((sequence_length, sequence_length), -1e9),
    k=1,
)
attention_weights = mx.softmax(
    scores + future_mask,
    axis=-1,
)
mixed_information = attention_weights @ values`}</Code>
        <p>The upper triangle corresponds to future positions and receives a huge negative number. After <Term id="softmax">softmax</Term>, those positions have effectively zero probability. Each remaining row sums to one, producing a weighted average of earlier value vectors.</p>
      </> },
      { id: 'block', title: 'Wrap attention in a minimal residual block', body: <>
        <Code>{`hidden = self.normalisation(
    hidden + self.attention(hidden)
)
logits = self.output(hidden)`}</Code>
        <p>The <Term id="residual-connection">residual connection</Term> adds the original representation back after attention, preserving a direct information path. <Term id="layer-normalisation">Layer normalisation</Term> stabilises the combined features. A final linear layer maps every sequence position to 65 next-character logits.</p>
        <aside className="lesson-caveat"><strong>This is not yet a full transformer</strong><p>There is one head, no feed-forward sublayer and only one attention block. The next lesson will add those missing components deliberately.</p></aside>
      </> },
      { id: 'parameters', title: 'Count the 28,993 parameters', body: <>
        <table className="lesson-table"><thead><tr><th>Component</th><th>Parameters</th></tr></thead><tbody><tr><td>Token embedding: 65 × 64</td><td>4,160</td></tr><tr><td>Position embedding: 64 × 64</td><td>4,096</td></tr><tr><td>Query, key, value and projection: 4 × 64 × 64</td><td>16,384</td></tr><tr><td>Layer normalisation scale + bias</td><td>128</td></tr><tr><td>Output weights + bias: 64 × 65 + 65</td><td>4,225</td></tr><tr><td><strong>Total</strong></td><td><strong>28,993</strong></td></tr></tbody></table>
      </> },
      { id: 'run', title: 'Run the attention experiment', body: <>
        <Code>{`.venv/bin/python ml/shakespeare_attention.py --steps 2000`}</Code>
        <p>The run used batch size 32, context length 64, model width 64, AdamW learning rate 0.003, weight decay 0.01, and the same seed 42. Each batch predicts the next character at all 64 positions, giving 2,048 token predictions per update.</p>
      </> },
      { id: 'results', title: 'Record the surprising result', body: <>
        <table className="lesson-table"><thead><tr><th>Step</th><th>Train loss</th><th>Validation loss</th></tr></thead><tbody><tr><td>0</td><td>—</td><td>4.3256</td></tr><tr><td>50</td><td>2.5211</td><td>2.5114</td></tr><tr><td>250</td><td>2.3072</td><td>2.2973</td></tr><tr><td>1,000</td><td>2.1148</td><td>2.2151</td></tr><tr><td>2,000</td><td>2.0490</td><td>2.1889</td></tr></tbody></table>
        <blockquote className="model-sample">Lenrey: were seall; ain gereseo to,\nThan cre, &apos;ut thane, ind…</blockquote>
        <p>The 64-character attention model beats the bigram&apos;s approximately 2.49 loss but does not beat the simpler eight-character MLP at 1.9456. Recorded training took 5.56 seconds.</p>
      </> },
      { id: 'interpret', title: 'Attention is useful, not sufficient', body: <><p>This is a valuable negative result. A longer context and fashionable mechanism do not guarantee a better model. Our attention head can route information, but it lacks the per-position nonlinear processing supplied by a transformer&apos;s feed-forward sublayer. One head also has only one learned relationship space.</p><p>The controlled next step is a small transformer: multiple attention heads, a feed-forward network, residual connections around both sublayers, layer normalisation and several stacked blocks. We will compare it against all three existing baselines rather than assuming success.</p></> },
    ],
  },
];

export function getShakespeareLesson(slug: string) {
  return shakespeareLessons.find(lesson => lesson.slug === slug);
}
