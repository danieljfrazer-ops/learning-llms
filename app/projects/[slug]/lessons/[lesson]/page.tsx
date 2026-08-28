import Link from 'next/link';
import { notFound } from 'next/navigation';
import WikiChrome from '@/app/components/WikiChrome';
import ModelComparison from '@/app/components/ModelComparison';
import PromptPlayground from '@/app/components/PromptPlayground';
import EvaluationPanel from '@/app/components/EvaluationPanel';
import TrainingImprovementsPanel from '@/app/components/TrainingImprovementsPanel';
import ScalingPanel from '@/app/components/ScalingPanel';
import FinalModelPanel from '@/app/components/FinalModelPanel';
import TinyStoriesPromptPlayground from '@/app/components/TinyStoriesPromptPlayground';
import TinyStoriesFinalPanel from '@/app/components/TinyStoriesFinalPanel';
import { BeginnerLessonIntro, BeginnerSectionNote, PlannedBeginnerGuide } from '@/app/components/BeginnerGuidance';
import { LocalEvidencePlaceholder, LocalOnly, ReferenceOnly, StageState } from '@/app/components/EvidenceMode';
import { getProject, projects } from '@/lib/wiki-data';
import { getShakespeareLesson } from '@/lib/shakespeare-lessons';
import { getTinyStoriesLesson } from '@/lib/tinystories-lessons';
import { getSqlLesson } from '@/lib/sql-lessons';
import { LessonVisual } from '@/app/components/LearningVisual';
import LessonEngineeringBrief from '@/app/components/LessonEngineeringBrief';
import { lessonEngineeringBriefs } from '@/lib/lesson-engineering-briefs';

const referenceEvidenceSections: Record<string, string[]> = {
  'random-baseline': ['sample'],
  'bigram-training': ['checkpoints', 'interpret'],
  'context-windows': ['results'],
  'self-attention': ['results', 'interpret'],
  'tiny-transformer': ['results', 'comparison'],
  evaluation: ['metrics', 'prompt-test', 'decision'],
  'training-improvements': ['checkpoints', 'frozen-results', 'generation', 'limits'],
  'scaling-experiment': ['results', 'sample', 'decision'],
  'final-evaluation': ['three-seeds', 'selection', 'journey'],
  'final-story-model': ['measured-results', 'behaviour', 'selection', 'cross-project'],
};

export function generateStaticParams() {
  return projects.flatMap(project => project.stages.map(stage => ({ slug: project.slug, lesson: stage.slug })));
}

export default async function LessonPage({ params }: { params: Promise<{ slug: string; lesson: string }> }) {
  const { slug, lesson: lessonSlug } = await params;
  const project = getProject(slug);
  const stageIndex = project?.stages.findIndex(stage => stage.slug === lessonSlug) ?? -1;
  const stage = project?.stages[stageIndex];
  if (!project || !stage) notFound();
  const richLesson = project.slug === 'shakespeare'
    ? getShakespeareLesson(lessonSlug)
    : project.slug === 'tinystories' ? getTinyStoriesLesson(lessonSlug)
    : project.slug === 'sql' ? getSqlLesson(lessonSlug) : undefined;
  const previous = project.stages[stageIndex - 1];
  const next = project.stages[stageIndex + 1];
  const onward = project.slug === 'shakespeare' ? { href: '/projects/tinystories', label: 'Next project · TinyStories' } : project.slug === 'tinystories' ? { href: '/projects/sql', label: 'Next project · English → SQL' } : { href: '/continue', label: 'Continue independently' };

  return <WikiChrome active={project.slug}>
    <article className="article-page lesson-page">
      <div className="breadcrumbs"><Link href="/">Dashboard</Link><span>/</span><Link href={`/projects/${project.slug}`}>{project.shortName}</Link><span>/</span><strong>{stage.name}</strong></div>
      <header className="lesson-hero">
        <div><p className="kicker">{project.name.toUpperCase()} · LESSON {String(stageIndex + 1).padStart(2, '0')}</p><h1>{richLesson?.title ?? stage.name}</h1><p>{richLesson?.summary ?? stage.lesson}</p></div>
        <aside><ReferenceOnly><span className={`lesson-status lesson-status-${stage.state}`}>{stage.state}</span><small>REFERENCE OUTCOME</small><strong>{richLesson?.outcome ?? 'This lesson is planned. Its detailed procedure and measured evidence will be added when work begins.'}</strong>{richLesson && <p>{richLesson.evidence}</p>}</ReferenceOnly><LocalOnly><span className="lesson-status lesson-status-planned">your lab</span><small>REPRODUCTION GOAL</small><strong>{stage.lesson}</strong><p>Your measured outcome appears in the local dashboard when you run this stage.</p></LocalOnly></aside>
      </header>

      <nav className="lesson-switcher" aria-label="Project lessons">
        {project.stages.map((item, index) => <Link key={item.slug} className={item.slug === lessonSlug ? 'current' : ''} href={`/projects/${project.slug}/lessons/${item.slug}`}><span>{String(index + 1).padStart(2, '0')}</span><strong>{item.name}</strong><small><StageState reference={item.state} /></small></Link>)}
      </nav>

      {richLesson && <BeginnerLessonIntro lessonSlug={lessonSlug} />}
      <LessonVisual lessonSlug={lessonSlug} />
      {lessonEngineeringBriefs[lessonSlug] && <LessonEngineeringBrief brief={lessonEngineeringBriefs[lessonSlug]} />}

      {project.slug === 'shakespeare' && ['context-windows', 'self-attention', 'tiny-transformer'].includes(lessonSlug) && <ModelComparison />}
      {project.slug === 'shakespeare' && lessonSlug === 'evaluation' && <EvaluationPanel />}
      {project.slug === 'shakespeare' && lessonSlug === 'prompt-playground' && <PromptPlayground />}
      {project.slug === 'shakespeare' && lessonSlug === 'training-improvements' && <TrainingImprovementsPanel />}
      {project.slug === 'shakespeare' && lessonSlug === 'scaling-experiment' && <ScalingPanel />}
      {project.slug === 'shakespeare' && lessonSlug === 'final-evaluation' && <FinalModelPanel />}
      {project.slug === 'tinystories' && lessonSlug === 'tinystories-playground' && <TinyStoriesPromptPlayground />}
      {project.slug === 'tinystories' && lessonSlug === 'final-story-model' && <TinyStoriesFinalPanel />}

      {richLesson ? <div className="lesson-layout">
        <aside className="lesson-toc"><p>IN THIS LESSON</p>{richLesson.sections.map(section => <a key={section.id} href={`#${section.id}`}>{section.title}</a>)}</aside>
        <div className="lesson-body">{richLesson.sections.map((section, index) => {
          const isReferenceEvidence = referenceEvidenceSections[lessonSlug]?.includes(section.id);
          return <section id={section.id} key={section.id}><div className="section-number">{String(index + 1).padStart(2, '0')}</div><h2>{section.title}</h2>{isReferenceEvidence ? <><ReferenceOnly><BeginnerSectionNote lessonSlug={lessonSlug} sectionId={section.id} />{section.body}</ReferenceOnly><LocalEvidencePlaceholder title={section.title} /></> : <><BeginnerSectionNote lessonSlug={lessonSlug} sectionId={section.id} />{section.body}</>}</section>;
        })}</div>
      </div> : <section className="planned-lesson">
        <p className="kicker">PLANNED LESSON</p><h2>What this stage will cover</h2><p>{stage.lesson}</p>
        <PlannedBeginnerGuide lessonSlug={lessonSlug} />
        <div className="planned-grid"><article><strong>Before we begin</strong><p>We will record the exact dataset state, model configuration and baseline that this stage inherits.</p></article><article><strong>During the work</strong><p>Commands, code decisions and newly introduced concepts will be explained inline as they occur.</p></article><article><strong>Evidence required</strong><p>No result will be claimed without measured output, a saved configuration and a comparison with the preceding stage.</p></article></div>
      </section>}

      <footer className="lesson-pagination">
        {previous ? <Link href={`/projects/${project.slug}/lessons/${previous.slug}`}><small>← PREVIOUS LESSON</small><strong>{previous.name}</strong></Link> : <span />}
        {next ? <Link className="next" href={`/projects/${project.slug}/lessons/${next.slug}`}><small>NEXT LESSON →</small><strong>{next.name}</strong></Link> : <Link className="next" href={onward.href}><small>CONTINUE →</small><strong>{onward.label}</strong></Link>}
      </footer>
    </article>
  </WikiChrome>;
}
