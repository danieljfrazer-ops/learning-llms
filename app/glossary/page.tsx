import Link from 'next/link';
import WikiChrome from '@/app/components/WikiChrome';
import { glossary } from '@/lib/wiki-data';

export default function Glossary() {
  return <WikiChrome active="glossary"><article className="article-page"><div className="breadcrumbs"><Link href="/">Dashboard</Link><span>/</span><strong>Glossary</strong></div><header className="article-hero single"><div><p className="kicker">REFERENCE · A–Z</p><h1>Technical terms, decoded</h1><p>The exact vocabulary used by practitioners, paired with a plain-language explanation. Terms are added when they first appear in a lesson.</p></div></header><div className="glossary-list">{glossary.map(([term, definition]) => <article id={term.toLowerCase().replaceAll(' ', '-')} key={term}><h2>{term}</h2><p>{definition}</p></article>)}</div></article></WikiChrome>;
}
