import Link from '@/app/components/CourseLink';
import WikiChrome from '@/app/components/WikiChrome';
import { glossary } from '@/lib/wiki-data';
import BeginnerGlossaryNote from '@/app/components/BeginnerGlossaryNote';
import { LearningVisual } from '@/app/components/LearningVisual';

export default function Glossary() {
  return <WikiChrome active="glossary"><article className="article-page"><div className="breadcrumbs"><Link href="/">Dashboard</Link><span>/</span><strong>Glossary</strong></div><header className="article-hero single"><div><p className="kicker">REFERENCE · A–Z</p><h1>Technical terms, decoded</h1><p>The exact vocabulary used by practitioners, paired with a plain-language explanation. Terms are added when they first appear in a lesson. Beginner Mode also shows a concrete analogy for the most important ideas.</p></div></header><LearningVisual compact title="Four kinds of words you will meet" description="Use this map to place a new term before reading its precise definition below." kind="compare" steps={[{ label: 'Data', detail: 'token · vocabulary · batch · split', tone: 'input' }, { label: 'Model', detail: 'weight · embedding · attention · layer', tone: 'process' }, { label: 'Training', detail: 'loss · gradient · optimiser · checkpoint', tone: 'warning' }, { label: 'Evidence', detail: 'validation · perplexity · protocol', tone: 'evidence' }]} /><div className="glossary-list">{glossary.map(([term, definition]) => <article id={term.toLowerCase().replaceAll(' ', '-')} key={term}><h2>{term}</h2><p>{definition}</p><BeginnerGlossaryNote term={term} /></article>)}</div></article></WikiChrome>;
}
