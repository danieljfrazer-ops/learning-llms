'use client';

import { useEffect, useState } from 'react';
import { useEvidenceUrl } from './EvidenceMode';

type MetricRow = { checkpoint?: string; style?: string; sampleSize: number; validSql: number; logicalFormExact: number; executionCorrect: number; peakMemoryBytes?: number };
type ModelResult = MetricRow & { candidate: { id: string; license: string; upstreamParametersBillions: number }; eligible: boolean };
type SqlEvidence = {
  status: string;
  lesson: string;
  results?: ModelResult[] | MetricRow[];
  selectedModel?: { id: string };
  selectedStyle?: string;
  checkpoints?: MetricRow[];
  cohort?: MetricRow[];
  comparison?: { executionCorrectDelta: number; exactMatchDelta: number; validSqlDelta: number };
  zeroEffectCheck?: { randomAdapterMatchesBaseOutputs: boolean };
  training?: { adapterBytes: number; lossTrace: { iteration: number; kind: string; loss: number }[] };
  limits: string[];
};

const labels: Record<string, string> = {
  'base-model-selection': 'PROJECT 03 · LESSON 02 EVIDENCE',
  'prompt-formatting': 'PROJECT 03 · LESSON 03 EVIDENCE',
  'lora-fine-tuning': 'PROJECT 03 · LESSON 04 EVIDENCE',
  'execution-evaluation': 'PROJECT 03 · LESSON 05 EVIDENCE',
};

function name(row: MetricRow | ModelResult) {
  if ('candidate' in row) return row.candidate.id.split('/').at(-1) ?? row.candidate.id;
  return row.style ?? row.checkpoint ?? 'candidate';
}

export default function SqlExperimentPanel({ filename, lesson }: { filename: string; lesson: string }) {
  const evidenceUrl = useEvidenceUrl(filename);
  const [loaded, setLoaded] = useState<{ url: string; data: SqlEvidence | null }>({ url: '', data: null });
  useEffect(() => {
    fetch(`${evidenceUrl}?t=${Date.now()}`, { cache: 'no-store' })
      .then(response => response.ok ? response.json() : null)
      .then(data => setLoaded({ url: evidenceUrl, data }))
      .catch(() => setLoaded({ url: evidenceUrl, data: null }));
  }, [evidenceUrl]);
  const evidence = loaded.url === evidenceUrl ? loaded.data : null;
  const rows = evidence?.results ?? evidence?.checkpoints ?? evidence?.cohort ?? [];
  const selected = evidence?.selectedModel?.id.split('/').at(-1) ?? evidence?.selectedStyle ?? (lesson === 'execution-evaluation' ? 'frozen cohort' : '—');

  return <section className="evaluation-panel sql-audit-panel" aria-label={`${lesson} evidence`}>
    <div className="evaluation-head"><div><p className="kicker">{labels[lesson]}</p><h2>{evidence ? 'Measured locally under the frozen protocol' : 'Run this lesson to create learner evidence'}</h2></div><span>{evidence?.status ?? (loaded.url === evidenceUrl ? 'Not run yet' : 'Loading evidence…')}</span></div>
    <div className="evaluation-summary">
      <div><small>CANDIDATES</small><strong>{rows.length || '—'}</strong></div>
      <div><small>SELECTED</small><strong>{selected}</strong></div>
      <div><small>TEST BOUNDARY</small><strong>{lesson === 'execution-evaluation' && evidence ? 'after freeze' : 'closed'}</strong></div>
      <div><small>PRIMARY METRIC</small><strong>execution</strong></div>
    </div>
    {evidence ? <>
      <table className="lesson-table"><thead><tr><th>Candidate</th><th>Valid SQL</th><th>Exact</th><th>Execution</th><th>Sample</th></tr></thead><tbody>
        {rows.map(row => <tr key={name(row)}><td>{name(row)}</td><td>{row.validSql}</td><td>{row.logicalFormExact}</td><td>{row.executionCorrect}</td><td>{row.sampleSize}</td></tr>)}
      </tbody></table>
      {evidence.zeroEffectCheck && <p className="evaluation-foot">Random adapter matched base outputs: <strong>{String(evidence.zeroEffectCheck.randomAdapterMatchesBaseOutputs)}</strong>. Trained adapter size: {evidence.training ? `${(evidence.training.adapterBytes / 1024).toFixed(1)} KiB` : '—'}.</p>}
      {evidence.comparison && <p className="evaluation-foot">Trained minus base on the frozen test sample: execution {evidence.comparison.executionCorrectDelta >= 0 ? '+' : ''}{evidence.comparison.executionCorrectDelta}; exact {evidence.comparison.exactMatchDelta >= 0 ? '+' : ''}{evidence.comparison.exactMatchDelta}; valid SQL {evidence.comparison.validSqlDelta >= 0 ? '+' : ''}{evidence.comparison.validSqlDelta}.</p>}
      <aside className="lesson-caveat"><strong>Interpret inside the boundary</strong><p>{evidence.limits[0]}</p></aside>
    </> : <p className="evaluation-foot">My Lab remains blank until the command finishes. No missing metric is filled with a reference or placeholder value.</p>}
  </section>;
}
