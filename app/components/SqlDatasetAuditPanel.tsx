'use client';

import { useEffect, useState } from 'react';
import { useEvidenceUrl } from './EvidenceMode';

type Split = {
  questions: number;
  tables: number;
  questionWords: { median: number; p95: number; maximum: number };
  questionsWithMultipleConditions: number;
  conditionValuesMentionedInQuestion: { mentioned: number; total: number };
  validationErrors: Record<string, number>;
};

type AuditEvidence = {
  status: string;
  elapsedSeconds: number;
  taskContract: { input: string; output: string; primaryMetric: string; secondaryMetric: string; prohibitedUse: string; modelTraining: string };
  dataset: {
    name: string;
    totalQuestions: number;
    totalTables: number;
    licensing: { repositoryCodeLicense: string; datasetLicenseStatus: string };
    splits: Record<'train' | 'dev' | 'test', Split>;
    pairwiseLeakage: Record<string, { sharedTableIds: number; exactNormalisedQuestions: number }>;
  };
  referenceExecution: {
    safety: string;
    samples: Record<'train' | 'dev' | 'test', { executed: number; succeeded: number; failures: unknown[] }>;
    equivalenceExample: { question: string; logicalFormsDiffer: boolean; resultsMatch: boolean; resultRowCount: number; comparison: string };
  };
  limitations: string[];
};

export default function SqlDatasetAuditPanel() {
  const evidenceUrl = useEvidenceUrl('sql-task-dataset-audit.json');
  const [loaded, setLoaded] = useState<{ url: string; data: AuditEvidence | null }>({ url: '', data: null });
  useEffect(() => {
    fetch(`${evidenceUrl}?t=${Date.now()}`, { cache: 'no-store' })
      .then(response => response.ok ? response.json() : null)
      .then(data => setLoaded({ url: evidenceUrl, data }))
      .catch(() => setLoaded({ url: evidenceUrl, data: null }));
  }, [evidenceUrl]);
  const audit = loaded.url === evidenceUrl ? loaded.data : null;
  const splits = audit ? (['train', 'dev', 'test'] as const) : [];
  const execution = audit ? splits.reduce((sum, split) => sum + audit.referenceExecution.samples[split].succeeded, 0) : null;
  const duplicateQuestions = audit ? Object.values(audit.dataset.pairwiseLeakage).reduce((sum, item) => sum + item.exactNormalisedQuestions, 0) : null;

  return <section className="evaluation-panel sql-audit-panel" aria-label="English to SQL task and dataset audit evidence">
    <div className="evaluation-head"><div><p className="kicker">PROJECT 03 · LESSON 01 EVIDENCE</p><h2>{audit ? 'The task is executable; the dataset licence still needs review' : 'Define correctness before adapting a model'}</h2></div><span>{audit?.status ?? (loaded.url === evidenceUrl ? 'Not run yet' : 'Loading evidence…')}</span></div>
    <div className="evaluation-summary">
      <div><small>QUESTIONS</small><strong>{audit?.dataset.totalQuestions.toLocaleString() ?? '—'}</strong></div>
      <div><small>OBSERVED TABLES</small><strong>{audit?.dataset.totalTables.toLocaleString() ?? '—'}</strong></div>
      <div><small>SAFE EXECUTION SMOKES</small><strong>{execution == null ? '—' : `${execution} / 300`}</strong></div>
      <div><small>CROSS-SPLIT QUESTION MATCHES</small><strong>{duplicateQuestions ?? '—'}</strong></div>
    </div>
    {audit ? <>
      <div className="transition-evidence-grid">
        <article><strong>Input boundary</strong><p>{audit.taskContract.input}.</p></article>
        <article><strong>Correctness hierarchy</strong><p>{audit.taskContract.primaryMetric}; {audit.taskContract.secondaryMetric.toLowerCase()} remains diagnostic.</p></article>
        <article><strong>Licence gate</strong><p>{audit.dataset.licensing.datasetLicenseStatus}.</p></article>
        <article><strong>Equivalent programs</strong><p>Reordering two <code>AND</code> conditions changed the SQL text but returned the same {audit.referenceExecution.equivalenceExample.resultRowCount} row(s).</p></article>
      </div>
      <table className="lesson-table"><thead><tr><th>Split</th><th>Questions</th><th>Tables</th><th>Median words</th><th>95th percentile</th><th>Multi-condition</th><th>Structural errors</th></tr></thead><tbody>
        {splits.map(split => <tr key={split}><td>{split}</td><td>{audit.dataset.splits[split].questions.toLocaleString()}</td><td>{audit.dataset.splits[split].tables.toLocaleString()}</td><td>{audit.dataset.splits[split].questionWords.median}</td><td>{audit.dataset.splits[split].questionWords.p95}</td><td>{audit.dataset.splits[split].questionsWithMultipleConditions.toLocaleString()}</td><td>{Object.values(audit.dataset.splits[split].validationErrors).reduce((sum, count) => sum + count, 0)}</td></tr>)}
      </tbody></table>
      <aside className="lesson-caveat"><strong>Do not cross this boundary</strong><p>{audit.taskContract.prohibitedUse}. The audit uses {audit.referenceExecution.safety.toLowerCase()}.</p></aside>
      <p className="evaluation-foot">No model was selected, invoked, or trained. Audit runtime: {audit.elapsedSeconds.toFixed(2)} seconds.</p>
    </> : <p className="evaluation-foot">Run the downloader and audit in My Lab to create your evidence. Published Reference results are maintained separately.</p>}
  </section>;
}
