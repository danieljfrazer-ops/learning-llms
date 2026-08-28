import Link from '@/app/components/CourseLink';
import { notFound } from 'next/navigation';
import WikiChrome from '@/app/components/WikiChrome';
import LiveMetrics from '@/app/components/LiveMetrics';
import ModelComparison from '@/app/components/ModelComparison';
import { getProject, projects } from '@/lib/wiki-data';
import { BeginnerProjectGuide } from '@/app/components/BeginnerGuidance';
import { LocalOnly, ReferenceOnly, StageState } from '@/app/components/EvidenceMode';
import { ProjectJourneyVisual } from '@/app/components/LearningVisual';

export function generateStaticParams() { return projects.map(project => ({ slug: project.slug })); }

export default async function ProjectPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params; const project = getProject(slug); if (!project) notFound();
  return <WikiChrome active={project.slug}>
    <article className="article-page">
      <div className="breadcrumbs"><Link href="/">Dashboard</Link><span>/</span><span>Projects</span><span>/</span><strong>{project.shortName}</strong></div>
      <header className="article-hero"><div><p className="kicker">PROJECT {project.number} · {project.method.toUpperCase()}</p><h1>{project.name}</h1><p>{project.objective}</p></div><ReferenceOnly><div className="stage-stamp"><span>REFERENCE COURSE</span><strong>{project.stage}</strong><small>{project.progress}% of planned journey</small><div className="progress-track"><i style={{ width: `${Math.max(project.progress, 2)}%` }} /></div></div></ReferenceOnly><LocalOnly><div className="stage-stamp local-stage-stamp"><span>YOUR LOCAL LAB</span><strong>Ready to reproduce</strong><small>Run lessons in order; your dashboards begin empty.</small><div className="progress-track"><i style={{ width: '2%' }} /></div></div></LocalOnly></header>
      <BeginnerProjectGuide project={project.slug} />
      <ProjectJourneyVisual name={project.shortName} stages={project.stages} />

      {project.slug === 'shakespeare' && <LiveMetrics />}
      {project.slug === 'shakespeare' && <ModelComparison />}

      {project.slug === 'shakespeare' && <ReferenceOnly><section className="experiment-note">
        <div><p className="kicker">PROJECT COMPLETE · THREE-SEED CONFIRMATION</p><h2>A 420,673-parameter model emerged from random weights</h2><p>The selected 128-wide transformer averaged validation loss 1.5846 ± 0.0040 across three independently trained seeds, compared with 4.2790 before learning. Seed 43 is the promptable final checkpoint at loss 1.5791 and perplexity 4.85.</p><p>Review the <Link href="/projects/shakespeare/lessons/final-evaluation">final comparison</Link> or use the <Link href="/projects/shakespeare/lessons/prompt-playground">prompt playground</Link>. The model remains a tiny character imitator—not a conversational LLM—and the lesson documents the absence of an untouched final test set.</p></div>
        <dl><div><dt>Lessons complete</dt><dd>12 / 12</dd></div><div><dt>Mean validation loss</dt><dd>1.5846</dd></div><div><dt>Selected perplexity</dt><dd>4.85</dd></div><div><dt>Final parameters</dt><dd>420,673</dd></div><div><dt>Peak Metal memory</dt><dd>295 MB</dd></div><div><dt>Confirmation seeds</dt><dd>3</dd></div></dl>
      </section></ReferenceOnly>}
      {project.slug === 'shakespeare' && <LocalOnly><section className="experiment-note local-lab-note"><div><p className="kicker">MY LAB · BLANK WORKSPACE</p><h2>Your results remain separate from the course reference</h2><p>Start with lesson 01 and run each command from your clone. Generated JSON appears under <code>public/data/local/</code>; checkpoints and configurations appear under <code>work/experiments/</code>. Both locations are ignored by Git.</p><p>Switch back to Reference results whenever you want to compare with the original 32 GB Apple-silicon run.</p></div><dl><div><dt>Published files changed</dt><dd>0</dd></div><div><dt>First executable model</dt><dd>Lesson 03</dd></div><div><dt>Live refresh</dt><dd>Every 2 s</dd></div><div><dt>Evidence owner</dt><dd>You</dd></div></dl></section></LocalOnly>}

      <section className="lesson-section" id="stages"><div className="lesson-title"><span>01</span><div><p className="kicker">THE ROADMAP</p><h2>Stages and lessons</h2><p className="section-intro">Open any stage for its procedure, commands, theory, inline glossary links, sources, evidence and limitations.</p></div></div><div className="stage-list">{project.stages.map((stage, index) => <Link href={`/projects/${project.slug}/lessons/${stage.slug}`} className={`stage-row stage-${stage.state}`} key={stage.name}><span className="stage-index">{String(index + 1).padStart(2, '0')}</span><div><h3>{stage.name}</h3><p>{stage.lesson}</p></div><span className="stage-action"><strong><StageState reference={stage.state} /></strong><i>Open lesson →</i></span></Link>)}</div></section>

      <section className="lesson-section two-column" id="dataset"><div><div className="lesson-title"><span>02</span><div><p className="kicker">THE MATERIAL</p><h2>Dataset</h2></div></div><h3>{project.dataset.name}</h3><p>{project.dataset.detail}</p><a className="text-link" href={project.dataset.source} target="_blank" rel="noreferrer">{project.dataset.sourceLabel} ↗</a></div><aside className="callout"><strong>Why audit data first?</strong><p>A model learns statistical patterns from exactly what we give it. Provenance, licences, duplicates and train/test leakage affect what conclusions we can honestly draw.</p></aside></section>

      <section className="lesson-section integration-note"><p className="kicker">HOW TO READ THIS WIKI</p><h2>Concepts and references now live inside the lessons.</h2><p>Glossary terms are linked at their first meaningful use, while external sources sit beside the claim, tool or dataset they support. The standalone <Link href="/glossary">A–Z glossary</Link> remains available for later lookup.</p></section>
    </article>
  </WikiChrome>;
}
