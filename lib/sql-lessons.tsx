import type { ReactNode } from 'react';
import BeginnerTerm from '@/app/components/BeginnerTerm';
import SqlDatasetAuditPanel from '@/app/components/SqlDatasetAuditPanel';
import SqlExperimentPanel from '@/app/components/SqlExperimentPanel';
import type { RichLesson } from './shakespeare-lessons';

function Term({ id, children }: { id: string; children: ReactNode }) {
  return <BeginnerTerm id={id}>{children}</BeginnerTerm>;
}

function Source({ href, children }: { href: string; children: ReactNode }) {
  return <a className="inline-source" href={href} target="_blank" rel="noreferrer">{children} ↗</a>;
}

function Code({ children }: { children: string }) {
  return <pre className="lesson-code"><code>{children}</code></pre>;
}

export const sqlLessons: RichLesson[] = [
  {
    slug: 'task-and-dataset-audit',
    title: 'Define the task before adapting a model',
    summary: 'Freeze what the model may see, what it must emit, how correctness will be measured, and what WikiSQL can and cannot support before choosing pretrained weights.',
    outcome: 'Executable guide ready; no reviewed Reference result has been promoted.',
    evidence: 'My Lab command available · reference evidence remains Not run yet · no model training',
    sections: [
    {
      id: 'departure', title: 'Change the learning question after pretraining', body: <>
        <p>Projects 1 and 2 began with random weights and taught one corpus through next-token prediction. English → SQL will instead begin with a pretrained model and later add a small task-specific adapter. Lesson 1 does neither: it defines the target transformation and the evidence needed to judge it.</p>
        <p>The task is <Term id="semantic-parsing">semantic parsing</Term>: turn a natural-language question plus a table <Term id="schema">schema</Term> into a structured program. The program is useful only if its execution returns the intended answer.</p>
        <aside className="lesson-caveat"><strong>No model has earned a place yet</strong><p>A fluent pretrained model may still invent columns, produce unsafe operations, or answer from memorised world knowledge. Dataset and evaluation boundaries come before model selection.</p></aside>
      </>,
    },
    {
      id: 'contract', title: 'Freeze the permitted input and output', body: <>
        <table className="lesson-table"><tbody>
          <tr><th>Model input</th><td>One natural-language question plus the column names and types for exactly one table</td></tr>
          <tr><th>Hidden from the model</th><td>Reference query, expected answer, test examples and—under the original WikiSQL protocol—table cell contents</td></tr>
          <tr><th>Model output</th><td>A restricted single-table <code>SELECT</code> logical form with one selected column, optional aggregation and zero or more conditions</td></tr>
          <tr><th>Primary success measure</th><td><Term id="execution-accuracy">Execution accuracy</Term> on the held-out split</td></tr>
          <tr><th>Safety boundary</th><td>Never run generated SQL against a writable or production database</td></tr>
        </tbody></table>
        <p>This contract follows the <Source href="https://github.com/salesforce/WikiSQL">official WikiSQL protocol</Source>: inference receives the question and table schema, not table contents. Later prompt formatting and fine-tuning must preserve that boundary.</p>
      </>,
    },
    {
      id: 'provenance', title: 'Separate dataset provenance from licence certainty', body: <>
        <p><Source href="https://arxiv.org/abs/1709.00103">Seq2SQL</Source> introduced WikiSQL as 80,654 annotated question/query examples based on Wikipedia tables. The official archive is version 1.1; its changelog says examples with condition-value mismatches were removed.</p>
        <p>The upstream repository carries a BSD-3-Clause licence for its code, but it does not explicitly say that the bundled dataset is licensed under the same terms. An <Source href="https://github.com/salesforce/WikiSQL/issues/91">upstream licensing question</Source> remains unanswered, and the repository was archived in 2025. This course may download the source for local study, but redistribution and public-release claims remain blocked pending legal review.</p>
      </>,
    },
    {
      id: 'download', title: 'Acquire one verified upstream artifact', body: <>
        <Code>{`uv run --no-sync python scripts/download_wikisql.py`}</Code>
        <p>The downloader retrieves the official 25 MB <code>data.tar.bz2</code> archive, requires SHA-256 <code>755c…0881</code>, and copies only the expected train, development and test JSONL/table/SQLite files. It does not use unrestricted archive extraction.</p>
        <p>Raw files and their manifest remain ignored under <code>data/raw/wikisql-1.1/</code>. The manifest records each file hash, source URL, retrieval time and the unresolved dataset-licence status.</p>
      </>,
    },
    {
      id: 'structure', title: 'Audit every structured example before measuring a model', body: <>
        <p>Each example must contain a non-empty question, a known table identifier and a <Term id="logical-form">logical form</Term>. Column indices must fit the matching schema; aggregation and condition operators must belong to the declared WikiSQL grammar.</p>
        <p>The audit counts split sizes, table IDs, question lengths, aggregation operators, condition operators and multi-condition queries. It also measures exact normalized question overlap across splits separately from table overlap. Identical wording on different tables is not direct answer leakage, but it warns that question templates recur.</p>
      </>,
    },
    {
      id: 'correctness', title: 'Use execution accuracy without discarding exact match', body: <>
        <p><Term id="logical-form-exact-match">Logical-form exact match</Term> asks whether the generated structure is identical to the reference. It is strict and useful for diagnosis, but two different SQL strings can represent the same operation—for example, reversing two conditions joined by <code>AND</code>.</p>
        <p>Execution accuracy runs the candidate and reference against the same frozen table and compares returned rows. It therefore becomes the primary metric, while exact match helps explain structural differences. Agreement on one table state is still not proof that two programs are universally equivalent.</p>
      </>,
    },
    {
      id: 'safe-execution', title: 'Treat every query as untrusted program text', body: <>
        <p>The Lesson 1 smoke test never executes free-form model output. It renders only the dataset&apos;s structured references through a tiny allowlist: <code>SELECT</code>, six aggregation choices, three comparison operators and <code>AND</code>. Values travel as SQLite parameters rather than being joined into SQL text.</p>
        <p><Term id="sqlite">SQLite</Term> databases open in read-only mode and enable <code>query_only</code>. These controls make the benchmark audit safer; they do not turn arbitrary generated SQL into something suitable for a production database.</p>
      </>,
    },
    {
      id: 'run', title: 'Run the audit and keep the evidence local', body: <>
        <Code>{`uv run --no-sync python ml/sql_task_dataset_audit.py`}</Code>
        <p>The program verifies every source hash and structured example, then executes a seeded sample of 100 reference queries from each split. It writes dashboard JSON to <code>public/data/local/sql-task-dataset-audit.json</code> and configuration to <code>work/experiments/sql-task-dataset-audit-001/config.json</code>.</p>
        <p>Neither output is committed automatically. Reference mode stays blank until a maintainer separately reviews and explicitly promotes a run.</p>
        <SqlDatasetAuditPanel />
      </>,
    },
    {
      id: 'limits-next', title: 'Carry the contract into base-model selection', body: <>
        <p>This lesson can establish that the files are traceable, examples fit the declared grammar, split boundaries are measurable and the restricted evaluator works. It cannot establish model accuracy, LoRA suitability, safety on arbitrary databases or permission to redistribute the dataset.</p>
        <p>Lesson 2 will use this frozen contract to compare candidate pretrained models by licence, local memory, tokenizer and context conventions, latency and zero-training execution accuracy. The untouched test split must remain closed during that selection.</p>
      </>,
    },
    ],
  },
  {
    slug: 'base-model-selection',
    title: 'Choose the smallest defensible pretrained base',
    summary: 'Compare pinned, permissively licensed MLX models under one development-only protocol before any task training.',
    outcome: 'Executable guide ready; learner evidence selects a local base model without opening test.',
    evidence: 'My Lab command available · two pinned candidates · reference evidence remains Not run yet',
    sections: [
      { id: 'question', title: 'Turn model choice into a controlled experiment', body: <><p>A larger or more famous model is not automatically the right base. This lesson asks which candidate fits the laptop, follows the restricted output contract and produces the strongest zero-training development result.</p><p>The candidates are pinned MLX conversions of <Source href="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct">Qwen2.5 0.5B Instruct</Source> and <Source href="https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct">SmolLM2 360M Instruct</Source>. Both upstream model cards record Apache-2.0 licences.</p></> },
      { id: 'boundary', title: 'Keep the final exam sealed', body: <><p>A seeded sample comes only from <code>dev</code>, the development split used for choices. The test file is not loaded. The same questions, compact schema representation, greedy decoding and 96-token limit apply to both models.</p><aside className="lesson-caveat"><strong>Development is for decisions</strong><p>Choosing a model after viewing test accuracy would tune the course to its final exam. That makes the reported test result less trustworthy.</p></aside></> },
      { id: 'criteria', title: 'Predeclare eligibility and selection', body: <><p>A candidate must be ungated, revision-pinned, permissively licensed, load through MLX and stay below 8 GiB peak MLX allocation. Among eligible models, development execution accuracy wins; ties use exact match, valid SQL, lower peak memory and then model identifier.</p><p><Term id="quantization">Quantization</Term> stores weights with fewer bits. It reduces local memory and download cost, but it can slightly change model behaviour and does not reduce the upstream parameter count.</p></> },
      { id: 'safe-output', title: 'Measure programs without executing model text', body: <><p>The model sees column IDs, names and types but no table rows or reference answer. Generated text must parse into the allowlisted WikiSQL logical form. Only that checked structure is rebuilt as parameterized SQL and run against read-only SQLite.</p><p>Invalid grammar, unknown columns and unsupported operations become explicit failures rather than reaching the database.</p></> },
      { id: 'run', title: 'Run the two-model screen', body: <><Code>{`uv run python ml/sql_base_model_selection.py`}</Code><p>The command downloads roughly 600 MB of pinned model repositories to the Hugging Face cache, invokes each model in a separate process and writes only learner-owned evidence to <code>public/data/local/sql-base-model-selection.json</code> and <code>work/experiments/sql-base-model-selection-001/</code>.</p><SqlExperimentPanel filename="sql-base-model-selection.json" lesson="base-model-selection" /></> },
      { id: 'interpret', title: 'Read zero accuracy as evidence, not failure of the lesson', body: <><p>A pretrained chat model can produce fluent explanations while failing a rigid program interface. Valid-SQL rate diagnoses formatting; exact match diagnoses structure; execution tests the returned answer. Memory and latency explain practical cost but do not replace correctness.</p><p>The selected model is a starting point for adaptation, not a claim that it already solves WikiSQL.</p></> },
      { id: 'limits-next', title: 'Freeze the winner and change only the prompt', body: <><p>A small development sample cannot rank models universally, and two candidates do not cover the full open-model landscape. Model-card licences also do not resolve WikiSQL dataset redistribution.</p><p>Lesson 3 holds the selected revision fixed and compares compact, SQL-DDL-like and JSON schema serializations on a different seeded development sample.</p></> },
    ],
  },
  {
    slug: 'prompt-formatting',
    title: 'Treat the prompt as a versioned interface',
    summary: 'Hold the base model fixed while comparing three schema serializations that preserve the same information boundary.',
    outcome: 'Executable guide ready; learner evidence freezes one prompt format without test leakage.',
    evidence: 'My Lab command available · model held fixed · reference evidence remains Not run yet',
    sections: [
      { id: 'inherit', title: 'Inherit the selected model without reopening the choice', body: <><p>Lesson 3 reads Lesson 2&apos;s selected model ID and exact revision. It does not add another candidate or reinterpret the earlier score. This isolates prompt representation as the changed variable.</p><p>The model-specific chat template still wraps every prompt because instruction-tuned models use different control tokens. The semantic content remains matched.</p></> },
      { id: 'three-formats', title: 'Compare three views of the same schema', body: <><table className="lesson-table"><tbody><tr><th>Compact</th><td><code>col0: Player [text] | col1: No. [text]</code></td></tr><tr><th>DDL</th><td>A <code>CREATE TABLE data</code> declaration with original names in comments</td></tr><tr><th>JSON</th><td>Objects containing table, column ID, name and type</td></tr></tbody></table><p>No format contains table rows, a reference query or expected result.</p></> },
      { id: 'control', title: 'Change serialization, not the task', body: <><p>Question, column identifiers, names, types, output grammar, chat template, greedy decoding and token limit stay fixed. A fresh seeded development sample avoids selecting the format on exactly the same 24 examples used for the model screen.</p><p>Execution accuracy selects the format; exact match, validity and prompt-token count break ties in that order.</p></> },
      { id: 'leakage', title: 'Reject answer-shaped convenience', body: <><p>Including cell values can help a model copy condition text, but it changes the official WikiSQL input contract. Including worked examples selected from the evaluation rows can leak targets more directly. This lesson uses zero demonstrations.</p><aside className="lesson-caveat"><strong>Consistent does not mean harmless</strong><p>A perfectly regular prompt can still leak the answer. Every field must be justified by the task contract.</p></aside></> },
      { id: 'run', title: 'Run the matched prompt comparison', body: <><Code>{`uv run python ml/sql_prompt_formatting.py`}</Code><p>The command reuses the selected cached weights, evaluates all three formats and writes local evidence to <code>public/data/local/sql-prompt-formatting.json</code> plus the ignored experiment directory.</p><SqlExperimentPanel filename="sql-prompt-formatting.json" lesson="prompt-formatting" /></> },
      { id: 'interpret', title: 'Separate token cost from task evidence', body: <><p>A shorter prompt reduces prefill work, but the shortest format is selected only after correctness ties. A higher valid-SQL count can be useful even when execution remains low because the adapter can only learn reliably from an unambiguous target interface.</p><p>The result applies to this model revision and task. Another tokenizer may split the same serialization differently.</p></> },
      { id: 'limits-next', title: 'Freeze the prompt before generating training records', body: <><p>Prompt selection remains development evidence. It says nothing about the untouched test split and does not prove that one wording transfers to joins or unseen database dialects.</p><p>Lesson 4 serializes training and validation examples exactly this way, masks prompt tokens from the loss and trains only LoRA parameters in the final eight transformer layers.</p></> },
    ],
  },
  {
    slug: 'lora-fine-tuning',
    title: 'Adapt frozen weights through a low-rank path',
    summary: 'Compare the base, a zero-effect random LoRA checkpoint and a bounded trained adapter while test remains closed.',
    outcome: 'Executable guide ready; learner evidence preserves before and after adapter checkpoints.',
    evidence: 'My Lab command available · 512 training examples · 100 updates · reference evidence remains Not run yet',
    sections: [
      { id: 'why-lora', title: 'Change a small path instead of every base weight', body: <><p><Source href="https://arxiv.org/abs/2106.09685">Low-Rank Adaptation</Source>, or LoRA, freezes each pretrained weight matrix and learns two much smaller matrices whose product adds a task-specific update. The adapter depends on the base; it is not a complete standalone model.</p><p><Source href="https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md">MLX-LM&apos;s LoRA guide</Source> supports this route directly on Apple silicon.</p></> },
      { id: 'before-state', title: 'Save the adapter before it can learn', body: <><p>MLX-LM initializes the first low-rank matrix randomly and the second to zero. Their initial product is therefore zero, so the random adapter should reproduce the base model exactly. The lesson saves and invokes this checkpoint instead of merely assuming the property.</p><p>Any output difference at this point would reveal a configuration or loading error.</p></> },
      { id: 'data', title: 'Build bounded records from training and development only', body: <><p>A seeded 512-example subset comes from <code>train</code>; 64 validation examples come from <code>dev</code>. Each record contains a system instruction, the frozen Lesson 3 prompt and canonical SQL completion. File hashes make the exact subsets auditable.</p><p><Term id="loss-mask">Loss masking</Term> excludes prompt tokens from the training objective, so optimizer updates are driven by the SQL completion rather than teaching the model to repeat the question.</p></> },
      { id: 'recipe', title: 'Freeze one laptop-sized training recipe', body: <><table className="lesson-table"><tbody><tr><th>Base</th><td>Selected 4-bit model, unchanged</td></tr><tr><th>Adapter</th><td>Rank 8, scale 20, final 8 layers</td></tr><tr><th>Batch and budget</th><td>Batch 2, 100 optimizer updates</td></tr><tr><th>Optimizer</th><td>Adam at <code>1e-5</code></td></tr><tr><th>Sequence cap</th><td>512 tokens with prompt masking</td></tr></tbody></table></> },
      { id: 'run', title: 'Create both adapter checkpoints and evaluate them', body: <><Code>{`uv run python ml/sql_lora_fine_tuning.py`}</Code><p>The command writes prepared records, logs, a zero-update adapter and a trained adapter under <code>work/experiments/sql-lora-fine-tuning-001/</code>. Dashboard evidence goes only to <code>public/data/local/sql-lora-fine-tuning.json</code>.</p><SqlExperimentPanel filename="sql-lora-fine-tuning.json" lesson="lora-fine-tuning" /></> },
      { id: 'interpret', title: 'Require behavioral evidence beside loss', body: <><p>Training and validation loss show how well the adapter predicts recorded completions. Development valid-SQL, exact-match and execution counts show whether that lower token loss changes the actual program behavior.</p><p>A tiny adapter file is evidence of parameter efficiency, not automatically of accuracy, safety or lower end-to-end memory; the base weights remain loaded.</p></> },
      { id: 'limits-next', title: 'Stop selecting before opening test', body: <><p>The short run may underfit, while repeating recipes until development improves would create its own selection bias. The course freezes this one teaching recipe and preserves disappointing results rather than manufacturing a win.</p><p>Lesson 5 verifies the adapter checksum, opens a seeded test sample once and compares only the already frozen base and trained adapter.</p></> },
    ],
  },
  {
    slug: 'execution-evaluation',
    title: 'Open the test split once and judge returned answers',
    summary: 'Evaluate the frozen base and adapter cohort with restricted parsing, read-only execution and an explicit failure taxonomy.',
    outcome: 'Executable final lesson ready; learner test evidence closes the English-to-SQL project.',
    evidence: 'My Lab command available · no post-test selection · reference evidence remains Not run yet',
    sections: [
      { id: 'freeze', title: 'Lock every decision before the final run', body: <><p>The model revision, adapter checksum, prompt style, greedy decoder, token limit, evaluator, sample size and seed are written before test examples are loaded. Only the base and trained adapter enter the cohort.</p><p>After observing test results, the command marks <code>noFurtherSelection</code>. A new recipe would require a new development cycle and a separately declared final evaluation.</p></> },
      { id: 'metric', title: 'Make execution primary and exact match diagnostic', body: <><p>Generated output first becomes a checked logical form. Execution accuracy compares its returned rows with the reference query&apos;s rows. Logical-form exact match remains useful for structure, while valid-SQL rate shows whether the model followed the interface at all.</p><p>The dashboard reports counts and a 95% <Term id="wilson-interval">Wilson interval</Term> for each execution proportion. The interval shows how imprecise a 100-example sample is under repeated sampling; it does not remove benchmark bias or turn this sample into a full-test result. Rows are compared without relying on result order because WikiSQL queries do not include <code>ORDER BY</code>.</p></> },
      { id: 'sandbox', title: 'Keep generated programs away from writable data', body: <><p>Free-form output never reaches SQLite. The parser permits one selected column, six aggregation states, three condition operators and <code>AND</code>. The evaluator rebuilds SQL with quoted identifiers and parameterized values, then opens the benchmark database read-only with <code>query_only</code>.</p><aside className="lesson-caveat"><strong>A benchmark sandbox is not production safety</strong><p>It does not cover arbitrary SQL, denial-of-service behavior, permissions, private data or multi-user database systems.</p></aside></> },
      { id: 'errors', title: 'Group failures by the first visible structural cause', body: <><p>Failures are counted as invalid grammar, wrong selected column, wrong aggregation, wrong condition count, wrong condition details or a residual execution mismatch. The taxonomy makes a score actionable without pretending that one automatic label captures model intent.</p><p>Representative records preserve the question, reference SQL, generated text, parsed form and outcome for review.</p></> },
      { id: 'run', title: 'Run the one-time frozen evaluation', body: <><Code>{`uv run python ml/sql_execution_evaluation.py`}</Code><p>The default evaluates a declared 100-example test sample. This is small enough for a laptop course and must be reported as a sample rather than the full 15,878-example test split.</p><SqlExperimentPanel filename="sql-execution-evaluation.json" lesson="execution-evaluation" /></> },
      { id: 'interpret', title: 'Report the comparison even when adaptation loses', body: <><p>The trained-minus-base deltas show whether the bounded adapter improved executable behavior, exact structure and format adherence. A negative or zero delta remains valid evidence: short supervised adaptation can overfit, optimize token loss without fixing semantics or simply be underpowered.</p><p>Do not replace the selected checkpoint after seeing this result.</p></> },
      { id: 'close', title: 'Close the course with a bounded capability claim', body: <><p>Project 1 exposed learning from random character weights. Project 2 scaled pretraining to subword stories. Project 3 began with pretrained weights, added a compact adapter and judged a generated program by execution.</p><p>The result remains bounded to WikiSQL&apos;s single-table grammar, the pinned local models, the sampled evaluation and the unresolved dataset-redistribution licence. The next honest step is an independently designed project with its own frozen evidence contract.</p></> },
    ],
  },
];

export function getSqlLesson(slug: string) {
  return sqlLessons.find(lesson => lesson.slug === slug);
}
