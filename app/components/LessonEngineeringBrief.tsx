import type { LessonEngineeringBrief as Brief } from '@/lib/lesson-engineering-briefs';

export default function LessonEngineeringBrief({ brief }: { brief: Brief }) {
  return <section className="lesson-engineering-brief" aria-label="Lesson engineering brief">
    <div className="evaluation-head"><div><p className="kicker">ENGINEERING BRIEF · BEFORE YOU RUN</p><h2>{brief.question}</h2></div><span>STATE + EVIDENCE</span></div>
    <div className="transition-evidence-grid">
      <article><strong>State entering</strong><p>{brief.enters}</p></article>
      <article><strong>Controlled change</strong><p>{brief.changes}</p></article>
      <article><strong>Decision evidence</strong><p>{brief.evidence}</p></article>
      <article><strong>State handed forward</strong><p>{brief.handoff}</p></article>
    </div>
    <aside className="lesson-caveat"><strong>Claim boundary</strong><p>{brief.boundary}</p></aside>
    <details className="engineering-runbook"><summary>Open the run contract</summary>
      <pre className="lesson-code"><code>{brief.run.command}</code></pre>
      <dl>
        <div><dt>Reads</dt><dd>{brief.run.reads}</dd></div>
        <div><dt>Computes</dt><dd>{brief.run.computes}</dd></div>
        <div><dt>Writes</dt><dd>{brief.run.writes}</dd></div>
        <div><dt>State change</dt><dd>{brief.run.mutates}</dd></div>
        <div><dt>Complete when</dt><dd>{brief.run.complete}</dd></div>
        <div><dt>If it fails</dt><dd>{brief.run.recovery}</dd></div>
        <div><dt>Practical cost</dt><dd>{brief.run.cost}</dd></div>
      </dl>
    </details>
  </section>;
}
