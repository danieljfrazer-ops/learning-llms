import Link from 'next/link';
import { BeginnerModeToggle } from './BeginnerMode';

export default function WikiChrome({ active, children }: { active: string; children: React.ReactNode }) {
  const link = (href: string, label: string, icon: string, key: string) => <Link className={`nav-link ${active === key ? 'active' : ''}`} href={href}><span>{icon}</span>{label}</Link>;
  return <div className="site-shell">
    <aside className="sidebar">
      <Link className="brand" href="/"><span className="brand-mark">L</span><span><strong>Learning</strong><small>Language Models</small></span></Link>
      <nav aria-label="Wiki navigation">
        <p className="nav-label">COURSE</p>{link('/', 'Dashboard', '⌂', 'home')}{link('/concepts/training-lifecycle', 'Learning path', '◎', 'path')}
        <p className="nav-label">PROJECTS</p>{link('/projects/shakespeare', 'Shakespeare', '01', 'shakespeare')}{link('/projects/tinystories', 'TinyStories', '02', 'tinystories')}{link('/projects/sql', 'English → SQL', '03', 'sql')}
        <p className="nav-label">REFERENCE</p>{link('/glossary', 'Glossary', 'A–Z', 'glossary')}
      </nav>
      <div className="machine-card"><span className="live-dot" /> LOCAL LAB<strong>MacBook Air · 32 GB</strong><small>Apple M5 · MLX</small></div>
    </aside>
    <main className="main-content"><header className="topbar"><div><span className="eyebrow">LEARNING LANGUAGE MODELS</span><span className="sync-state">● Living documentation</span></div><div className="topbar-tools"><BeginnerModeToggle /><Link className="search-link" href="/glossary">Technical glossary <kbd>A–Z</kbd></Link></div></header>{children}</main>
  </div>;
}
