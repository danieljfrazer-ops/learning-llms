import Link from 'next/link';
import { notFound } from 'next/navigation';
import WikiChrome from '@/app/components/WikiChrome';
import LiveMetrics from '@/app/components/LiveMetrics';
import ModelComparison from '@/app/components/ModelComparison';
import { getProject, projects } from '@/lib/wiki-data';

export function generateStaticParams() { return projects.map(project => ({ slug: project.slug })); }

export default async function ProjectPage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params; const project = getProject(slug); if (!project) notFound();
  return <WikiChrome active={project.slug}>
    <article className="article-page">
      <div className="breadcrumbs"><Link href="/">Dashboard</Link><span>/</span><span>Projects</span><span>/</span><strong>{project.shortName}</strong></div>
      <header className="article-hero"><div><p className="kicker">PROJECT {project.number} · {project.method.toUpperCase()}</p><h1>{project.name}</h1><p>{project.objective}</p></div><div className="stage-stamp"><span>CURRENT STAGE</span><strong>{project.stage}</strong><small>{project.progress}% of planned journey</small><div className="progress-track"><i style={{ width: `${Math.max(project.progress, 2)}%` }} /></div></div></header>

      {project.slug === 'shakespeare' && <LiveMetrics />}
      {project.slug === 'shakespeare' && <ModelComparison />}

      {project.slug === 'shakespeare' && <section className="experiment-note">
        <div><p className="kicker">FOUR MODELS · OBSERVED RESULTS</p><h2>The complete transformer is now strongest</h2><p>The one-character bigram reached validation loss 2.49, the eight-character feed-forward model 1.9456, and single-head attention 2.1889. Combining four attention heads with nonlinear feed-forward processing, residual paths, normalisation and two blocks reduced validation loss to 1.7267.</p><p>This is evidence for the complete system—not proof that any one addition caused the gain. The transformer also has more parameters and training steps. The <Link href="/projects/shakespeare/lessons/evaluation">evaluation lesson</Link> will separate useful comparisons from variables we have not controlled.</p></div>
        <dl><div><dt>Models trained</dt><dd>4</dd></div><div><dt>Best validation loss</dt><dd>1.7267</dd></div><div><dt>Best architecture</dt><dd>2-block transformer</dd></div><div><dt>Largest context</dt><dd>64 chars</dd></div><div><dt>Compute</dt><dd>MLX · Apple GPU</dd></div><div><dt>Reproducibility seed</dt><dd>42</dd></div></dl>
      </section>}

      <section className="lesson-section" id="stages"><div className="lesson-title"><span>01</span><div><p className="kicker">THE ROADMAP</p><h2>Stages and lessons</h2><p className="section-intro">Open any stage for its procedure, commands, theory, inline glossary links, sources, evidence and limitations.</p></div></div><div className="stage-list">{project.stages.map((stage, index) => <Link href={`/projects/${project.slug}/lessons/${stage.slug}`} className={`stage-row stage-${stage.state}`} key={stage.name}><span className="stage-index">{String(index + 1).padStart(2, '0')}</span><div><h3>{stage.name}</h3><p>{stage.lesson}</p></div><span className="stage-action"><strong>{stage.state}</strong><i>Open lesson →</i></span></Link>)}</div></section>

      <section className="lesson-section two-column" id="dataset"><div><div className="lesson-title"><span>02</span><div><p className="kicker">THE MATERIAL</p><h2>Dataset</h2></div></div><h3>{project.dataset.name}</h3><p>{project.dataset.detail}</p><a className="text-link" href={project.dataset.source} target="_blank" rel="noreferrer">{project.dataset.sourceLabel} ↗</a></div><aside className="callout"><strong>Why audit data first?</strong><p>A model learns statistical patterns from exactly what we give it. Provenance, licences, duplicates and train/test leakage affect what conclusions we can honestly draw.</p></aside></section>

      <section className="lesson-section integration-note"><p className="kicker">HOW TO READ THIS WIKI</p><h2>Concepts and references now live inside the lessons.</h2><p>Glossary terms are linked at their first meaningful use, while external sources sit beside the claim, tool or dataset they support. The standalone <Link href="/glossary">A–Z glossary</Link> remains available for later lookup.</p></section>
    </article>
  </WikiChrome>;
}
