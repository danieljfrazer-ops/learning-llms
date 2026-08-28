import type { ReactNode } from 'react';
import BeginnerTerm from '@/app/components/BeginnerTerm';

export type RichLesson = {
  slug: string;
  title: string;
  summary: string;
  outcome: string;
  evidence: string;
  sections: { id: string; title: string; body: ReactNode }[];
};

function Term({ id, children }: { id: string; children: ReactNode }) {
  return <BeginnerTerm id={id}>{children}</BeginnerTerm>;
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
          <p>The project declares platform-specific MLX profiles in <code>pyproject.toml</code>, while <code>uv.lock</code> records the resolved package versions. This reference run used the Apple profile, which created <code>.venv/</code> with Python 3.12 and installed MLX plus its Metal backend:</p>
          <Code>{`uv sync --python 3.12 --extra apple`}</Code>
          <p>A <Term id="virtual-environment">virtual environment</Term> prevents packages required by this course from silently mixing with packages used by unrelated projects. The lock file makes a future reinstall far more likely to reproduce this environment.</p>
        </>,
      },
      {
        id: 'project-shape', title: 'How the project is organised', body: <>
          <Code>{`LearningLLMs/
├── ml/                 readable training programs
├── data/raw/           downloaded source text (not committed)
├── experiments/        committed reference configurations
├── work/experiments/   ignored learner checkpoints and configurations
├── public/data/reference/  committed reference dashboard evidence
├── public/data/local/      ignored learner dashboard evidence
├── app/ and lib/       wiki pages and structured lesson content
├── pyproject.toml      Python dependencies
├── uv.lock             exact Python resolution
└── package.json        wiki dependencies and scripts`}</Code>
          <p>Large generated <Term id="checkpoint">checkpoints</Term>, raw data and virtual-environment files are excluded from Git. Source, lesson text, configurations and the compact metrics record are committed. This keeps the repository reviewable without losing the evidence needed to explain a run.</p>
        </>,
      },
      {
        id: 'live-refresh', title: 'How the open wiki refreshes', body: <>
          <p>Hand-authored lesson changes use hot module reload. During training, the Python process atomically rewrites <code>public/data/local/shakespeare-metrics.json</code>. In My Lab mode the dashboard requests that file every two seconds with caching disabled. Reference mode instead reads the committed file under <code>public/data/reference/</code>, so a learner run cannot overwrite the published comparison accidentally.</p>
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
          <Code>{`uv run --no-sync python scripts/download_tiny_shakespeare.py
# Downloaded and verified data/raw/tiny-shakespeare.txt (1,115,394 bytes)`}</Code>
          <p>The downloader writes a temporary file, verifies SHA-256 <code>86c4e6…565ed</code>, then atomically replaces the ignored raw path. An existing file with that hash is reused. This turns the source URL into a reproducible input rather than trusting whichever bytes a direct download happens to return.</p>
          <p>The upstream repository describes Tiny Shakespeare as a subset of Shakespeare&apos;s works and states an <Source href="https://github.com/karpathy/char-rnn#license">MIT licence</Source> in its README, but the dataset directory does not separately explain how that grant applies to the compiled corpus. Keep the raw file uncommitted and review corpus and derived-checkpoint redistribution separately before public release.</p>
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
          <Code>{`uv run --no-sync python ml/shakespeare_bigram.py --steps 400`}</Code>
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
        <Code>{`uv run --no-sync python ml/shakespeare_context.py --steps 2000`}</Code>
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
        <Code>{`uv run --no-sync python ml/shakespeare_attention.py --steps 2000`}</Code>
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
  {
    slug: 'tiny-transformer',
    title: 'Assemble the first complete tiny transformer',
    summary: 'Combine four-head causal attention, nonlinear feed-forward processing, residual connections, pre-layer-normalisation and two stacked blocks into a decoder-only language model.',
    outcome: 'The 112,065-parameter transformer achieved validation loss 1.7267—the best result so far—and produced substantially more sentence-like Shakespearean dialogue.',
    evidence: 'Completed · 3,000 updates · 64-character context · 18.81 seconds recorded training',
    sections: [
      { id: 'complete', title: 'What makes this a transformer', body: <>
        <p>The earlier attention model had only one attention head and no feed-forward sublayer. This model is a small <Term id="decoder-only-transformer">decoder-only transformer</Term>: it uses causal next-token prediction and repeatedly applies the two complementary operations found in a transformer block.</p>
        <ol><li><Term id="multi-head-attention">Multi-head attention</Term> moves information between sequence positions.</li><li>A <Term id="feed-forward-network">feed-forward network</Term> transforms the information within each position.</li></ol>
        <p>Residual paths and layer normalisation wrap both sublayers, and two complete blocks are stacked to create <Term id="model-depth">depth</Term>. The design is a deliberately tiny descendant of the decoder architecture in <Source href="https://arxiv.org/abs/1706.03762">Attention Is All You Need</Source>.</p>
      </> },
      { id: 'flow', title: 'Follow one batch through the model', body: <>
        <Code>{`token_ids                    # (32, 64)
token + position embeddings  # (32, 64, 64)
transformer block 1          # (32, 64, 64)
transformer block 2          # (32, 64, 64)
final layer norm             # (32, 64, 64)
output logits                # (32, 64, 65)`}</Code>
        <p>The <Term id="model-width">model width</Term> is 64 features per position. Sequence length is also 64, but those equal numbers describe different axes: one is the number of positions; the other is the representation size at each position. The output contains 65 logits at every position, so each update evaluates 32 × 64 = 2,048 next-character predictions.</p>
      </> },
      { id: 'heads', title: 'Split attention into four relationship spaces', body: <>
        <Code>{`queries, keys, values = mx.split(
    self.query_key_value(inputs),
    3,
    axis=-1,
)

# (batch, sequence, 64) → (batch, 4 heads, sequence, 16)
queries = queries.reshape(B, T, 4, 16).transpose(0, 2, 1, 3)`}</Code>
        <p>The single-head experiment used one 64-dimensional attention relationship. Here, four <Term id="attention-head">attention heads</Term> each receive a 16-dimensional slice. Every head has its own query, key and value features and can learn a different routing pattern.</p>
        <p>After causal attention, the four outputs are concatenated back into 64 features and passed through a learned projection. Multiple heads do not guarantee interpretable specialisation, but they give the block several independent relationship subspaces.</p>
      </> },
      { id: 'feed-forward', title: 'Add nonlinear per-position processing', body: <>
        <Code>{`class FeedForward(nn.Module):
    def __init__(self, model_size=64):
        self.expand = nn.Linear(64, 256)
        self.contract = nn.Linear(256, 64)

    def __call__(self, inputs):
        return self.contract(nn.gelu(self.expand(inputs)))`}</Code>
        <p>Attention blends information across positions; it does not by itself provide a rich nonlinear computation at each position. The feed-forward network expands 64 features to 256, applies GELU, then contracts back to 64. Its weights are shared across sequence positions.</p>
      </> },
      { id: 'block', title: 'Use pre-normalised residual sublayers', body: <>
        <Code>{`hidden = inputs + self.attention(
    self.attention_norm(inputs)
)
output = hidden + self.feed_forward(
    self.feed_forward_norm(hidden)
)`}</Code>
        <p>This is <Term id="pre-layer-normalisation">pre-layer-normalisation</Term>: each sublayer receives normalised input, then its output is added through a residual connection. A residual path lets a block preserve its input when a learned transformation is unhelpful and provides a short route for gradients.</p>
        <p>Two independently parameterised blocks are chained with <code>nn.Sequential</code>. Block two therefore processes representations that block one has already routed and transformed.</p>
      </> },
      { id: 'parameters', title: 'Account for all 112,065 parameters', body: <>
        <table className="lesson-table"><thead><tr><th>Component</th><th>Calculation</th><th>Parameters</th></tr></thead><tbody>
          <tr><td>Token embedding</td><td>65 × 64</td><td>4,160</td></tr>
          <tr><td>Position embedding</td><td>64 × 64</td><td>4,096</td></tr>
          <tr><td>Attention per block</td><td>QKV 64 × 192 + projection 64 × 64</td><td>16,384</td></tr>
          <tr><td>Feed-forward per block</td><td>64 × 256 + 256 bias + 256 × 64 + 64 bias</td><td>33,088</td></tr>
          <tr><td>Two layer norms per block</td><td>2 × (64 scale + 64 bias)</td><td>256</td></tr>
          <tr><td>Two complete blocks</td><td>2 × 49,728</td><td>99,456</td></tr>
          <tr><td>Final layer norm</td><td>64 scale + 64 bias</td><td>128</td></tr>
          <tr><td>Output projection</td><td>64 × 65 + 65 bias</td><td>4,225</td></tr>
          <tr><td><strong>Total</strong></td><td>sum</td><td><strong>112,065</strong></td></tr>
        </tbody></table>
        <p>The four heads split the same attention width, so increasing from one to four heads does not itself multiply the QKV parameter count. Most new parameters come from the two feed-forward networks and the second block.</p>
      </> },
      { id: 'run', title: 'Run the complete model', body: <>
        <Code>{`uv run --no-sync python ml/shakespeare_transformer.py --steps 3000`}</Code>
        <p>The run used batch size 32, context length 64, width 64, four heads, two blocks, AdamW learning rate 0.001, weight decay 0.01 and seed 42. Checkpoints were captured at steps 0, 1, 50, 250, 1,000, 2,000 and 3,000.</p>
        <p>Generation invokes the full model on the most recent 64 tokens, selects the last position&apos;s logits, samples one character at temperature 0.9, appends it and repeats.</p>
      </> },
      { id: 'results', title: 'Watch structured dialogue emerge', body: <>
        <table className="lesson-table"><thead><tr><th>Step</th><th>Train loss</th><th>Validation loss</th><th>Visible change</th></tr></thead><tbody>
          <tr><td>0</td><td>—</td><td>4.2800</td><td>Random characters</td></tr>
          <tr><td>50</td><td>2.5454</td><td>2.5389</td><td>Common character transitions</td></tr>
          <tr><td>250</td><td>2.2208</td><td>2.1858</td><td>Speaker formatting and word fragments</td></tr>
          <tr><td>1,000</td><td>1.6798</td><td>1.8625</td><td>Sentence-like dialogue appears</td></tr>
          <tr><td>2,000</td><td>1.5291</td><td>1.7626</td><td>More stable clauses and names</td></tr>
          <tr><td>3,000</td><td>1.5007</td><td><strong>1.7267</strong></td><td>Best held-out model so far</td></tr>
        </tbody></table>
        <blockquote className="model-sample">Of the to winds do is an this place; that I know\nAnd poppn this upon the conce, asrove mucklous\nWhom have much that Comfles…</blockquote>
        <p>The output is still grammatically unstable and semantically unreliable, but clauses are markedly longer and punctuation is more coherent. Recorded training, checkpoint evaluation and generation took 18.81 seconds on the Apple GPU.</p>
      </> },
      { id: 'comparison', title: 'Separate the contribution of the whole block', body: <>
        <table className="lesson-table"><thead><tr><th>Architecture</th><th>Context</th><th>Parameters</th><th>Final validation loss</th></tr></thead><tbody><tr><td>Bigram</td><td>1</td><td>4,225</td><td>≈2.489</td></tr><tr><td>Fixed-context MLP</td><td>8</td><td>43,361</td><td>1.9456</td></tr><tr><td>Single-head attention</td><td>64</td><td>28,993</td><td>2.1889</td></tr><tr><td><strong>Tiny transformer</strong></td><td>64</td><td>112,065</td><td><strong>1.7267</strong></td></tr></tbody></table>
        <p>The transformer beats the attention-only model with the same context length. However, several variables changed together: head count, feed-forward layers, depth, parameter count, learning rate and training steps. We can conclude that the complete configured transformer is better—not that any one added component caused the entire gain.</p>
      </> },
      { id: 'limits', title: 'Know what we have—and have not—built', body: <>
        <p>At 112,065 parameters, this is a tiny educational language model, not a general-purpose LLM. It knows only character statistics from one Shakespeare corpus. It cannot answer questions, follow instructions, retrieve facts or reliably maintain story meaning.</p>
        <p>The next evaluation lesson will freeze the comparison procedure, calculate repeated validation estimates, inspect generalisation gaps, compare samples under identical seeds and document failure patterns before we move to TinyStories.</p>
      </> },
    ],
  },
  {
    slug: 'evaluation',
    title: 'Evaluate every checkpoint with one frozen protocol',
    summary: 'Stop changing weights, apply the same held-out measurements and prompts to every checkpoint, and distinguish improvement from sampling noise and overfitting.',
    outcome: 'The final checkpoint achieved validation loss 1.7205 ± 0.0104 and perplexity 5.59; its 0.2011 generalisation gap shows that further training should not be judged by training loss alone.',
    evidence: 'Completed · evaluation shakespeare-evaluation-001 · 7 checkpoints · 2.32 seconds · 2026-08-21 16:17 UTC',
    sections: [
      { id: 'purpose', title: 'Freeze the rules before reading the result', body: <>
        <p>An <Term id="evaluation-protocol">evaluation protocol</Term> specifies the data, batches, metrics, prompts and random seeds applied to every candidate. Freezing those rules prevents us from quietly choosing easier examples for a preferred model.</p>
        <p>Training answers “did the optimiser reduce error on examples it sampled?” Evaluation asks “does the saved model make better predictions on held-out text, and does the improvement remain visible under repeated measurement?” No gradients or optimiser updates occur in this lesson.</p>
        <aside className="lesson-caveat"><strong>Development evaluation, not a final test</strong><p>The existing corpus has a 90% training and 10% validation split but no untouched test set. We have already used validation results to guide architecture choices, so this evidence supports development decisions rather than an unbiased final performance claim. The final lesson will repeat complete training across seeds, but that replication cannot retroactively create an untouched test set.</p></aside>
      </> },
      { id: 'protocol', title: 'Use identical batches and seeds', body: <>
        <p>The script loads each of the seven <Term id="checkpoint">checkpoints</Term> and evaluates it on five repeatable estimates. Each repeat contains 20 batches × 32 windows × 64 positions = 40,960 next-character predictions. The five repeats therefore cover 204,800 sampled predictions per data split and checkpoint.</p>
        <Code>{`repeated_loss(
    model,
    validation_data,
    base_seed=200_042,  # identical for every checkpoint
    repeats=5,
    batches=20,
    batch_size=32,
    context_size=64,
)`}</Code>
        <p>The same seed schedule selects the same window starts for every checkpoint. We report the mean and population standard deviation across the five repeats. This is not five separately trained models: it measures batch-sampling variability for one saved model.</p>
      </> },
      { id: 'run', title: 'Run the read-only evaluator', body: <>
        <Code>{`uv run --no-sync python ml/shakespeare_evaluate.py`}</Code>
        <p>The evaluator uses Python 3.12.13 and MLX 0.32.0 on <code>Device(gpu, 0)</code>. It reconstructs the exact 112,065-parameter architecture from <code>experiments/shakespeare-transformer-001/config.json</code>, then calls MLX <code>load_weights</code> for each Safetensors file. See the <Source href="https://ml-explore.github.io/mlx/build/html/python/nn/module.html">official MLX module documentation</Source>.</p>
        <p>Learner results are written to <code>public/data/local/shakespeare-evaluation.json</code>; the frozen protocol and timing are recorded under <code>work/experiments/shakespeare-evaluation-001/config.json</code>. The committed reference copies preserve this course&apos;s 2.319-second evaluation separately.</p>
      </> },
      { id: 'metrics', title: 'Read loss, perplexity and the generalisation gap together', body: <>
        <p><Term id="perplexity">Perplexity</Term> is <code>exp(cross-entropy loss)</code>. It expresses average predictive uncertainty on a multiplicative scale: 5.59 is substantially less uncertain than 72.17, but it does not mean the model has exactly 5.59 equally likely characters at every position. The relationship between cross-entropy and perplexity is described in the <Source href="https://web.stanford.edu/~jurafsky/slp3/3.pdf">Stanford Speech and Language Processing chapter</Source>.</p>
        <p>The <Term id="generalisation-gap">generalisation gap</Term> here is validation loss minus training loss. A positive, growing gap means the model predicts its training distribution better than the held-out tail of the corpus.</p>
        <table className="lesson-table"><thead><tr><th>Step</th><th>Train loss</th><th>Validation loss ± std</th><th>Gap</th><th>Perplexity</th></tr></thead><tbody>
          <tr><td>0</td><td>4.2814</td><td>4.2790 ± 0.0010</td><td>−0.0024</td><td>72.17</td></tr>
          <tr><td>1</td><td>3.6573</td><td>3.6770 ± 0.0059</td><td>0.0198</td><td>39.53</td></tr>
          <tr><td>50</td><td>2.5299</td><td>2.5367 ± 0.0113</td><td>0.0068</td><td>12.64</td></tr>
          <tr><td>250</td><td>2.1326</td><td>2.1732 ± 0.0083</td><td>0.0406</td><td>8.79</td></tr>
          <tr><td>1,000</td><td>1.7009</td><td>1.8633 ± 0.0109</td><td>0.1624</td><td>6.44</td></tr>
          <tr><td>2,000</td><td>1.5737</td><td>1.7639 ± 0.0103</td><td>0.1902</td><td>5.84</td></tr>
          <tr><td><strong>3,000</strong></td><td><strong>1.5194</strong></td><td><strong>1.7205 ± 0.0104</strong></td><td><strong>0.2011</strong></td><td><strong>5.59</strong></td></tr>
        </tbody></table>
      </> },
      { id: 'prompt-test', title: 'Hold generation settings constant too', body: <>
        <p>Loss measures every target character but cannot show whether output feels coherent. We therefore pass the same three <Term id="prompt">prompts</Term> through every checkpoint: <code>To be, or not to be</code>, <code>My lord, the night is</code>, and <code>ROMEO:\n</code>. Each receives 120 new characters at temperature 0.8 with a fixed sampling seed.</p>
        <table className="lesson-table"><thead><tr><th>Checkpoint</th><th>Continuation after “To be, or not to be”</th></tr></thead><tbody><tr><td>Random · step 0</td><td><code>SHSYItAA$RRYAzDA&amp;D?D3RTzOKayv,…</code></td></tr><tr><td>Step 50</td><td><code>terithe s whopre moe tas, Thane, theres…</code></td></tr><tr><td>Step 1,000</td><td><code>that so sound Which hear aid news all shall…</code></td></tr><tr><td>Step 3,000</td><td><code>that sees whose his heaven! Be Out love some…</code></td></tr></tbody></table>
        <p>The later output has recognisable words, clauses and line breaks, but it still lacks reliable meaning. Prompt examples are qualitative evidence, not a substitute for held-out loss, and one attractive sample must not be treated as typical behaviour.</p>
      </> },
      { id: 'decision', title: 'Turn the evidence into the next experiment', body: <>
        <p>Validation loss improves at every recorded checkpoint, so step 3,000 remains the best saved model. Yet the gap grows from roughly zero to 0.2011. The next training experiment should therefore save more frequent late checkpoints and test regularisation or a learning-rate schedule rather than merely chasing lower training loss.</p>
        <p>The immediate next lesson does not change the model at all. It packages these checkpoints behind a local inference service so you can probe them with your own text and see how checkpoint, temperature and seed affect generation.</p>
      </> },
    ],
  },
  {
    slug: 'prompt-playground',
    title: 'Prompt saved checkpoints from the wiki',
    summary: 'Connect the browser interface to a loopback-only Python inference service, load any saved checkpoint, encode learner-written text and generate a continuation one character at a time.',
    outcome: 'The wiki can complete an arbitrary valid Shakespeare prompt with random, minimally trained, baseline-final or improved-recipe weights and compare them under identical settings.',
    evidence: 'Completed · local API on 127.0.0.1:8001 · baseline and warmup-cosine runs · Apple GPU generation',
    sections: [
      { id: 'two-processes', title: 'Separate the interface from model inference', body: <>
        <p>The wiki runs in a JavaScript development server, while MLX and the checkpoints live in Python. A small <Term id="inference-service">inference service</Term> connects them: the browser sends a JSON request through an <Term id="api">API</Term>, Python invokes the model, and JSON returns the continuation.</p>
        <Code>{`Browser UI :3000
    │ POST /generate { prompt, checkpoint, temperature, ... }
    ▼
Python inference service :8001
    │ encode → load weights → autoregressive sampling
    ▼
MLX model on Apple GPU`}</Code>
        <p>This boundary keeps the educational model code readable and allows checkpoints to remain local. The service binds only to <code>127.0.0.1</code>, the loopback address of this Mac; it is not a hosted public model endpoint.</p>
      </> },
      { id: 'start', title: 'Start the two local processes', body: <>
        <Code>{`# Terminal 1 — wiki (already running in this lab)
npm run dev

# Terminal 2 — checkpoint inference
uv run --no-sync python ml/shakespeare_inference_server.py`}</Code>
        <p>The browser checks <code>GET http://127.0.0.1:8001/health</code> when the page loads. A green “Ready” badge confirms that Python found the baseline and improved-recipe checkpoint files and MLX reports the Apple GPU. If Python stops, the lesson remains readable and the playground shows the exact restart command.</p>
        <p>The server is built with Python&apos;s standard <code>ThreadingHTTPServer</code>. Python explicitly warns that <Source href="https://docs.python.org/3/library/http.server.html">http.server is not recommended for production</Source>; it is appropriate here only because this is a local teaching service with bounded inputs.</p>
      </> },
      { id: 'load', title: 'Reconstruct and cache a selected checkpoint', body: <>
        <Code>{`model = TinyTransformerLanguageModel(
    vocabulary_size=65,
    context_size=64,
    model_size=64,
    head_count=4,
    block_count=2,
)
model.load_weights("checkpoint-3000.safetensors")
mx.eval(model.parameters())`}</Code>
        <p>A checkpoint contains learned tensors, not the Python architecture. The service reconstructs the same class and dimensions recorded by the training configuration before loading weights. Models are loaded lazily on first selection and cached in memory, so later requests avoid reading the same 440 KB file again.</p>
        <p>Selecting baseline step 0 invokes genuine random initial weights saved before training. Baseline step 1 shows the minimally trained state, while “Improved recipe · step 3,000” loads the later warmup-and-cosine result. “Compare every checkpoint” runs the seven baseline stages plus the improved final model with all other controls unchanged.</p>
      </> },
      { id: 'encode-prompt', title: 'Encode the prompt and respect the 64-character context', body: <>
        <p>The <Term id="prompt">prompt</Term> is encoded with the same 65-character vocabulary used during training. A character outside that vocabulary produces an explicit error rather than being silently replaced. This simple tokenizer therefore accepts Shakespeare&apos;s letters, spaces and known punctuation but not arbitrary emoji or unseen Unicode characters.</p>
        <Code>{`generated = [char_to_id[ch] for ch in prompt]
context = generated[-model.context_size:]  # at most 64 characters
logits = model(mx.array([context]))[0, -1]`}</Code>
        <p>The page may retain a longer prompt for display, but only its final 64 characters can influence the first generated character. After each new character is appended, the oldest character falls out of the window. The result card reports how many prompt characters were initially visible.</p>
      </> },
      { id: 'sampling', title: 'Generate one character at a time', body: <>
        <Code>{`for _ in range(characters):
    logits = model(context)[0, -1] / temperature
    next_id = mx.random.categorical(logits)
    generated.append(next_id)
    context = generated[-64:]`}</Code>
        <p>This is <Term id="autoregressive">autoregressive</Term> inference. Each sampled character becomes input for the next model call. <Term id="temperature">Temperature</Term> rescales logits: lower values favour the model&apos;s highest-scoring options; higher values increase variety and errors. The seed makes categorical sampling repeatable for the same model, prompt and settings.</p>
        <p>Output length controls work, not context: asking for 400 new characters causes 400 forward passes, while every pass still sees at most 64 characters.</p>
      </> },
      { id: 'invoke', title: 'See the exact request behind the button', body: <>
        <Code>{`curl -X POST http://127.0.0.1:8001/generate \
  -H 'Content-Type: application/json' \
  --data '{
    "prompt": "To be, or not to be",
    "run": "baseline",
    "checkpoint": 3000,
    "temperature": 0.8,
    "characters": 80,
    "seed": 42
  }'`}</Code>
        <p>The measured response took 0.086 seconds and continued: <code> blood thing whose same; and stay speak.\n\nABTRUTUS:…</code>. Repeating the request with the same values produces the same sampled continuation; changing the seed explores another valid path through the probability distribution.</p>
      </> },
      { id: 'controls', title: 'Use the playground as an experiment, not a slot machine', body: <>
        <ol><li>Enter one prompt and keep it unchanged.</li><li>Set temperature 0.8, seed 42 and a modest output length.</li><li>Compare every checkpoint to isolate the effect of training progress.</li><li>Then hold the final checkpoint fixed and change one control at a time.</li><li>Record failures as well as unusually good samples.</li></ol>
        <p>The service validates checkpoint names, temperature 0.1–2.0, output length 1–500 and prompt length up to 2,000 characters. Requests are serialised around MLX generation to avoid concurrent mutation of its shared random generator.</p>
      </> },
      { id: 'limits', title: 'Understand what prompting does not change', body: <>
        <p>A prompt conditions existing weights; it does not teach new facts or update the model. This character model has no instruction training, so “Write a sonnet about Mars” is merely another character prefix rather than a command it understands.</p>
        <p>The next lesson returns to training. We will preserve this playground and frozen evaluation as acceptance tests while changing one training technique at a time. That prevents a visually pleasing cherry-picked completion from overriding worse held-out evidence.</p>
      </> },
    ],
  },
  {
    slug: 'training-improvements',
    title: 'Improve training without changing the transformer',
    summary: 'Hold architecture, data order and evaluation fixed while testing a warmup-plus-cosine learning-rate schedule and global gradient clipping as separate interventions.',
    outcome: 'Warmup plus cosine decay produced the new best validation loss, 1.7051 ± 0.0082, while clipping at norm 1.0 affected only two updates and finished worse than the baseline.',
    evidence: 'Completed · 2 controlled 3,000-step runs · experiment shakespeare-training-improvements-001 · 30.03 seconds total',
    sections: [
      { id: 'question', title: 'Improve the recipe before enlarging the model', body: <>
        <p>The transformer architecture stays at 112,065 parameters. This lesson asks whether a better optimisation recipe can improve held-out predictions without adding capacity. That is cheaper and easier to interpret than changing model size and training behaviour together.</p>
        <p>We use a <Term id="controlled-experiment">controlled experiment</Term>: corpus and 90/10 split, seed 42, batch size 32, 64-character context, model initialisation, AdamW weight decay 0.01, training batches and 3,000 updates remain fixed. One variant changes only the <Term id="learning-rate-schedule">learning-rate schedule</Term>; the other changes only gradient handling.</p>
      </> },
      { id: 'variants', title: 'Define the three comparable recipes', body: <>
        <table className="lesson-table"><thead><tr><th>Recipe</th><th>Learning rate</th><th>Gradient clipping</th><th>What it tests</th></tr></thead><tbody><tr><td>Constant baseline</td><td>0.001 throughout</td><td>None</td><td>Previously trained reference</td></tr><tr><td>Warmup + cosine</td><td>0.0001 → 0.001 over 100 steps, then → 0.0001</td><td>None</td><td>Learning-rate schedule only</td></tr><tr><td>Gradient clipping</td><td>0.001 throughout</td><td>Global norm 1.0</td><td>Clipping only</td></tr></tbody></table>
        <p>We deliberately did not add dropout, longer training and scheduling simultaneously. If that bundle won, we would not know which change helped. Those remain candidates only if the evidence points to a problem they can address.</p>
      </> },
      { id: 'schedule', title: 'Warm up, then make smaller late updates', body: <>
        <p><Term id="warmup">Warmup</Term> raises the learning rate gradually for the first 100 updates. <Term id="cosine-decay">Cosine decay</Term> then lowers it smoothly from 0.001 to 0.0001. The intention is conservative initial movement followed by increasingly fine adjustments near the end.</p>
        <Code>{`learning_rate = optim.join_schedules([
    optim.linear_schedule(0.0001, 0.001, steps=100),
    optim.cosine_decay(
        0.001,
        decay_steps=2900,
        end=0.0001,
    ),
], boundaries=[100])

optimiser = optim.AdamW(
    learning_rate=learning_rate,
    weight_decay=0.01,
)`}</Code>
        <p>MLX accepts a schedule callable directly as the optimiser&apos;s learning rate; see the <Source href="https://ml-explore.github.io/mlx/build/html/python/optimizers.html">official MLX optimiser documentation</Source>. Cosine annealing was popularised for neural-network optimisation in <Source href="https://arxiv.org/abs/1608.03983">SGDR: Stochastic Gradient Descent with Warm Restarts</Source>; this experiment uses one decay curve without restarts.</p>
      </> },
      { id: 'clipping', title: 'Limit unusually large gradient updates', body: <>
        <p>A <Term id="gradient-norm">gradient norm</Term> compresses the magnitude of all parameter gradients into one value. <Term id="gradient-clipping">Gradient clipping</Term> proportionally rescales the complete gradient tree only when that value exceeds 1.0.</p>
        <Code>{`gradients, total_norm = optim.clip_grad_norm(
    gradients,
    max_norm=1.0,
)
optimiser.update(model, gradients)`}</Code>
        <p>Clipping can stabilise training when rare exploding gradients cause destructive updates, a technique studied in <Source href="https://arxiv.org/abs/1211.5063">On the difficulty of training recurrent neural networks</Source>. It is not automatically beneficial: if gradients are already well behaved, clipping changes almost nothing or can suppress useful movement.</p>
      </> },
      { id: 'run', title: 'Run both variants and preserve checkpoints', body: <>
        <Code>{`uv run --no-sync python ml/shakespeare_training_improvements.py`}</Code>
        <p>The Python 3.12.13 and MLX 0.32.0 script trained both variants on the Apple GPU. Each run saved checkpoints at steps 0, 250, 1,000, 2,000 and 3,000 plus a JSON configuration. My Lab reads <code>public/data/local/shakespeare-training-improvements.json</code>; the reference dashboard reads its committed counterpart.</p>
        <p>The measured experiment took 30.03 seconds overall. Peak Metal allocation was 170.8 MB for both new runs. Per-run times are not compared as a speed benchmark because the original baseline captured more generated samples during training than the two new variants.</p>
      </> },
      { id: 'checkpoints', title: 'Watch the learning-rate schedule change the path', body: <>
        <table className="lesson-table"><thead><tr><th>Step</th><th>Warmup/cosine LR</th><th>Warmup/cosine validation</th><th>Clipped validation</th><th>Captured UTC</th></tr></thead><tbody>
          <tr><td>0</td><td>0.0001000</td><td>4.2800</td><td>4.2800</td><td>17:27:37 / 17:27:49</td></tr>
          <tr><td>250</td><td>0.0009942</td><td>2.2304</td><td>2.1755</td><td>17:27:38 / 17:27:50</td></tr>
          <tr><td>1,000</td><td>0.0008029</td><td>1.8516</td><td>1.8478</td><td>17:27:41 / 17:27:54</td></tr>
          <tr><td>2,000</td><td>0.0003396</td><td>1.7348</td><td>1.7569</td><td>17:27:45 / 17:28:00</td></tr>
          <tr><td>3,000</td><td>0.0001000</td><td><strong>1.7058</strong></td><td>1.7275</td><td>17:27:49 / 17:28:06</td></tr>
        </tbody></table>
        <p>Early in training, warmup is deliberately slower. By step 2,000 its smaller learning rate has overtaken the constant-rate clipped run. The checkpoint loss uses 30 sampled validation batches; final selection still uses the larger frozen protocol below.</p>
      </> },
      { id: 'frozen-results', title: 'Apply exactly the same final evaluation', body: <>
        <table className="lesson-table"><thead><tr><th>Recipe</th><th>Train loss</th><th>Validation loss ± std</th><th>Gap</th><th>Perplexity</th><th>Δ validation</th></tr></thead><tbody><tr><td>Constant baseline</td><td>1.5194</td><td>1.7205 ± 0.0104</td><td>0.2011</td><td>5.59</td><td>reference</td></tr><tr><td><strong>Warmup + cosine</strong></td><td><strong>1.5124</strong></td><td><strong>1.7051 ± 0.0083</strong></td><td><strong>0.1927</strong></td><td><strong>5.50</strong></td><td><strong>−0.0154</strong></td></tr><tr><td>Gradient clipping</td><td>1.5187</td><td>1.7288 ± 0.0101</td><td>0.2101</td><td>5.63</td><td>+0.0083</td></tr></tbody></table>
        <p>Observed result: warmup plus cosine is the new best saved recipe. It lowers mean validation loss by 0.0154 and slightly reduces the generalisation gap. Gradient clipping exceeded its threshold on only 2 of 3,000 updates; its maximum pre-clipping norm was 3.2710. This run provides no evidence that clipping helps this stable configuration.</p>
      </> },
      { id: 'generation', title: 'Compare prompted output without cherry-picking settings', body: <>
        <p>The frozen prompt, temperature 0.8 and sampling seed produce this continuation from the improved recipe:</p>
        <blockquote className="model-sample">To be, or not to be that seeing of this
To same grace the shalthings and been such
One and semberdant which unsight from the advilo&apos;s eame…</blockquote>
        <p>The text remains nonsensical despite improved local form. The prompt playground now exposes “Improved recipe · step 3,000” alongside every baseline checkpoint, so you can probe both recipes without retraining.</p>
      </> },
      { id: 'limits', title: 'Make a cautious decision from a single training seed', body: <>
        <p>The five evaluation repeats vary sampled batches, not model training. Both new recipes were trained only once with seed 42. The 0.0154 advantage is promising, but multiple full training seeds are required before claiming that the schedule reliably wins rather than benefiting from run-to-run numerical variation.</p>
        <p>Validation improved at every saved warmup/cosine checkpoint, so sparse <Term id="checkpoint">checkpoint</Term> evidence does not justify early stopping before step 3,000. The next scaling lesson will carry the schedule forward as the provisional recipe, compare capacity and context as separate factors, and preserve this baseline for every result.</p>
      </> },
    ],
  },
  {
    slug: 'scaling-experiment', title: 'Scale width, depth and context separately',
    summary: 'Carry forward the warmup-and-cosine recipe, change one architectural dimension at a time, and measure quality, parameters, runtime and Metal memory.',
    outcome: 'Doubling width to 128 features produced the strongest model: validation loss 1.5862 and perplexity 4.89 with 420,673 parameters and 295 MB peak Metal allocation.',
    evidence: 'Completed · 4 × 3,000-step runs · shakespeare-scaling-001 · 106.21 seconds',
    sections: [
      { id: 'design', title: 'Ask three different scaling questions', body: <><p>More <Term id="capacity">capacity</Term> can come from wider token representations, more sequential blocks, or more visible context. We compare these separately against the 64-wide, two-block, 64-context reference. Seed 42, data order, batch size 32, 3,000 updates, weight decay and warmup/cosine recipe remain fixed.</p><table className="lesson-table"><thead><tr><th>Candidate</th><th>Width</th><th>Blocks</th><th>Context</th><th>Changed factor</th></tr></thead><tbody><tr><td>Reference</td><td>64</td><td>2</td><td>64</td><td>None</td></tr><tr><td>Wider</td><td>128</td><td>2</td><td>64</td><td>Width only</td></tr><tr><td>Deeper</td><td>64</td><td>4</td><td>64</td><td>Depth only</td></tr><tr><td>Longer context</td><td>64</td><td>2</td><td>128</td><td>Context only</td></tr></tbody></table></> },
      { id: 'cost', title: 'Predict how each change affects cost', body: <><p>Doubling width makes most dense matrices roughly four times larger because both matrix axes grow. Doubling depth duplicates transformer blocks. Doubling context adds few weights, but attention work grows approximately with sequence length squared: a 128 × 128 attention-score grid contains four times as many entries as a 64 × 64 grid.</p><p>This is why parameter count alone cannot predict memory or runtime. The experiment records peak Metal allocation and elapsed time as well as loss.</p></> },
      { id: 'run', title: 'Train all four candidates', body: <><Code>{`uv run --no-sync python ml/shakespeare_scaling.py`}</Code><p>Python 3.12.13 and MLX 0.32.0 trained on the Apple GPU. Every run saved random, 250, 1,000, 2,000 and 3,000-step Safetensors checkpoints, timestamps, configuration and generated evidence. The full comparison took 106.21 seconds.</p></> },
      { id: 'results', title: 'Compare quality and resource use', body: <><table className="lesson-table"><thead><tr><th>Candidate</th><th>Parameters</th><th>Validation loss</th><th>Perplexity</th><th>Time</th><th>Peak Metal</th></tr></thead><tbody><tr><td>Reference</td><td>112,065</td><td>1.7051</td><td>5.50</td><td>15.4 s</td><td>170 MB</td></tr><tr><td><strong>Wider</strong></td><td><strong>420,673</strong></td><td><strong>1.5862</strong></td><td><strong>4.89</strong></td><td>23.4 s</td><td>295 MB</td></tr><tr><td>Deeper</td><td>211,521</td><td>1.6504</td><td>5.21</td><td>29.0 s</td><td>198 MB</td></tr><tr><td>Longer context</td><td>116,161</td><td>1.6827</td><td>5.38</td><td>36.2 s</td><td>406 MB</td></tr></tbody></table><p>All three changes improved loss. Width won on absolute quality; depth delivered a smaller but cheaper parameter increase. Longer context added few weights yet cost the most memory and runtime because attention processed longer sequences.</p></> },
      { id: 'sample', title: 'Inspect the wider model under the frozen prompt', body: <><blockquote className="model-sample">To be, or not to be that says who<br/>We may base your with whose thine, and yet you.<br/><br/>VOLUMNIA:<br/>Stay as a grace that safe! I find him…</blockquote><p>The output has stronger dialogue formatting and longer grammatical fragments, but invented words and incoherent meaning remain. The numerical result—not this one sample—is the selection basis.</p></> },
      { id: 'decision', title: 'Select width, then demand replication', body: <><p>The 128-wide candidate improves validation loss by 0.1190 over the same-recipe reference, while remaining comfortably within the Mac&apos;s memory. We select it for final confirmation.</p><p>This comparison still uses one training seed and a validation split already used for development. The final lesson repeats the selected architecture with seeds 42, 43 and 44. It cannot retroactively create an untouched test set, so the final claim remains explicitly in-domain and development-evaluated.</p></> },
    ],
  },
  {
    slug: 'final-evaluation', title: 'Confirm and close the Shakespeare project',
    summary: 'Repeat the selected 420,673-parameter architecture across three training seeds, choose a reproducible checkpoint, and audit what the project proves and what it does not.',
    outcome: 'Across seeds 42–44, the final architecture averaged validation loss 1.5846 ± 0.0040; seed 43 was selected at 1.5791 and is now the default prompt model.',
    evidence: 'Complete · shakespeare-final-001 · 3 training seeds · final checkpoint seed 43',
    sections: [
      { id: 'final-spec', title: 'Freeze the final specification', body: <><table className="lesson-table"><tbody><tr><th>Architecture</th><td>Decoder-only transformer; width 128, 2 blocks, 4 heads, context 64</td></tr><tr><th>Parameters</th><td>420,673</td></tr><tr><th>Training</th><td>3,000 updates; batch 32; AdamW; weight decay 0.01</td></tr><tr><th>Schedule</th><td>100-step warmup 0.0001 → 0.001; cosine decay to 0.0001</td></tr><tr><th>Data</th><td>1,003,854 training and 111,540 validation characters; vocabulary 65</td></tr><tr><th>Hardware</th><td>Apple GPU through MLX on the 32 GB MacBook Air</td></tr></tbody></table><p>Nothing except the <Term id="seed">training seed</Term> changes between confirmation runs.</p></> },
      { id: 'repeat', title: 'Repeat complete training, not just evaluation batches', body: <><Code>{`uv run --no-sync python ml/shakespeare_final.py`}</Code><p>Seed 42 reuses the identical scaling checkpoint. Seeds 43 and 44 start from independently initialised random weights and receive their own deterministic batch sequences. This measures <Term id="training-seed-variance">training-seed variance</Term>, unlike the five evaluation repeats that only vary sampled evaluation windows.</p></> },
      { id: 'three-seeds', title: 'Read the replicated result', body: <><table className="lesson-table"><thead><tr><th>Seed</th><th>Validation loss ± batch std</th><th>Perplexity</th><th>Training time</th></tr></thead><tbody><tr><td>42</td><td>1.5862 ± 0.0091</td><td>4.89</td><td>23.45 s</td></tr><tr><td><strong>43</strong></td><td><strong>1.5791 ± 0.0109</strong></td><td><strong>4.85</strong></td><td>21.52 s</td></tr><tr><td>44</td><td>1.5885 ± 0.0128</td><td>4.90</td><td>21.93 s</td></tr></tbody></table><p>The mean across independently trained models is <strong>1.5846</strong>, with between-seed population standard deviation <strong>0.0040</strong>. All three beat the 64-wide model at 1.7051, so the width result is not dependent on seed 42 alone.</p></> },
      { id: 'selection', title: 'Select seed 43 without overstating it', body: <><p>Seed 43 has the lowest observed validation loss and becomes the default playground model. Its frozen-prompt continuation begins:</p><blockquote className="model-sample">To be, or not to be that says with<br/>Clifford and grace loves<br/>so his pulpisy&apos;d years.<br/><br/>BALTHASAR:<br/>How banished that safe!…</blockquote><p>The tiny loss difference between seeds is not an architectural discovery. Seed 43 is a practical checkpoint choice; the three-seed mean is the more honest architecture summary.</p></> },
      { id: 'journey', title: 'Measure the full learning journey', body: <><table className="lesson-table"><thead><tr><th>Stage</th><th>Parameters</th><th>Context</th><th>Validation loss</th></tr></thead><tbody><tr><td>Random transformer</td><td>112,065</td><td>64</td><td>4.2790</td></tr><tr><td>Bigram</td><td>4,225</td><td>1</td><td>≈2.489</td></tr><tr><td>Fixed-context MLP</td><td>43,361</td><td>8</td><td>1.9456</td></tr><tr><td>First transformer</td><td>112,065</td><td>64</td><td>1.7205</td></tr><tr><td>Improved recipe</td><td>112,065</td><td>64</td><td>1.7051</td></tr><tr><td><strong>Final wider model</strong></td><td><strong>420,673</strong></td><td>64</td><td><strong>1.5846 mean</strong></td></tr></tbody></table><p>The main gains came from learning context-sensitive structure and then adding width; optimisation refinement helped modestly.</p></> },
      { id: 'limits', title: 'State the boundary of the achievement', body: <><p>This is a successful small character language-model experiment, not a general LLM. It imitates local Shakespeare-like statistics from one corpus. It does not understand instructions, guarantee grammatical text, reason about plots, or possess factual knowledge.</p><p>There is no untouched final test set: the 10% validation tail influenced development decisions throughout the project. The figures describe repeated in-domain validation performance, not unbiased generalisation to new authors or unseen Shakespeare editions.</p></> },
      { id: 'complete', title: 'What you can now reconstruct yourself', body: <><p>You have seen the entire loop: source and tokenize data, initialise random weights, construct context models and attention, assemble a transformer, train with gradients and AdamW, save checkpoints, evaluate held-out loss, diagnose overfitting, tune a schedule, scale architecture, measure memory, replicate seeds and serve local inference to a UI.</p><p>The Shakespeare project is complete. TinyStories will replace characters with subword tokens and scale toward coherent simple English while preserving these experimental habits.</p></> },
    ],
  },
];

export function getShakespeareLesson(slug: string) {
  return shakespeareLessons.find(lesson => lesson.slug === slug);
}
