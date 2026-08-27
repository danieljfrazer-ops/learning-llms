import type { ReactNode } from 'react';
import BeginnerTerm from '@/app/components/BeginnerTerm';
import SqlDatasetAuditPanel from '@/app/components/SqlDatasetAuditPanel';
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
];

export function getSqlLesson(slug: string) {
  return sqlLessons.find(lesson => lesson.slug === slug);
}
