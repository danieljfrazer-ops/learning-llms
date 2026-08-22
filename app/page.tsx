import Link from 'next/link';
import { projects as projectRecords } from '@/lib/wiki-data';
import { BeginnerModeToggle, BeginnerOnly } from '@/app/components/BeginnerMode';

const projects = projectRecords.map(project => ({ ...project, href: `/projects/${project.slug}` }));

export default function Home() {
  return (
    <div className="site-shell">
      <aside className="sidebar">
        <Link className="brand" href="/"><span className="brand-mark">L</span><span><strong>Learning</strong><small>Language Models</small></span></Link>
        <nav aria-label="Wiki navigation">
          <p className="nav-label">COURSE</p>
          <Link className="nav-link active" href="/"><span>⌂</span> Dashboard</Link>
          <Link className="nav-link" href="/concepts/training-lifecycle"><span>◎</span> Learning path</Link>
          <p className="nav-label">PROJECTS</p>
          <Link className="nav-link" href="/projects/shakespeare"><span>01</span> Shakespeare</Link>
          <Link className="nav-link" href="/projects/tinystories"><span>02</span> TinyStories</Link>
          <Link className="nav-link" href="/projects/sql"><span>03</span> English → SQL</Link>
          <p className="nav-label">REFERENCE</p>
          <Link className="nav-link" href="/glossary"><span>A–Z</span> Glossary</Link>
          <a className="nav-link" href="#toolkit"><span>⌘</span> Toolkit</a>
        </nav>
        <div className="machine-card"><span className="live-dot" /> LOCAL LAB<strong>MacBook Air · 32 GB</strong><small>Apple silicon · MLX</small></div>
      </aside>

      <main className="main-content">
        <header className="topbar"><div><span className="eyebrow">LIVING WIKI</span><span className="sync-state">● Documentation synced</span></div><div className="topbar-tools"><BeginnerModeToggle /><a className="search-link" href="#projects">Jump to projects <kbd>↓</kbd></a></div></header>
        <section className="hero">
          <div>
            <p className="kicker">BUILDING LANGUAGE MODELS FROM FIRST PRINCIPLES</p>
            <h1>Learn by watching<br /><em>language emerge.</em></h1>
            <p className="hero-copy">A hands-on record of every dataset, tensor, training run and mistake—as we teach small models to predict what comes next.</p>
            <BeginnerOnly className="home-beginner"><strong>New to all of this?</strong><span>An LLM is fundamentally an autocomplete system trained on enormous amounts of text. Here we build tiny versions so every moving part stays visible. Switch Beginner Mode off whenever the extra coaching is no longer useful.</span></BeginnerOnly>
            <div className="hero-actions"><Link className="button primary" href="/projects/shakespeare">Continue project <span>→</span></Link><Link className="button secondary" href="/concepts/training-lifecycle">See the learning path</Link></div>
          </div>
          <div className="terminal-card" aria-label="Current experiment status">
            <div className="terminal-title"><span>● ● ●</span><code>shakespeare · progression</code></div>
            <pre><span className="muted">$ course.status()</span>{'\n'}{'{'}{'\n'}  <span className="key">shakespeare_project</span>: <span className="value">&quot;complete&quot;</span>,{'\n'}  <span className="key">final_mean_validation_loss</span>: <span className="value">1.5846</span>,{'\n'}  <span className="key">next</span>: <span className="value">&quot;TinyStories&quot;</span>{'\n'}{'}'}</pre>
            <div className="terminal-footer"><span className="live-dot" /> Final 420,673-parameter model ready to prompt</div>
          </div>
        </section>

        <section className="section" id="projects">
          <div className="section-heading"><div><p className="kicker">THE CURRICULUM</p><h2>Three models, one mental map</h2></div><p>We begin with random noise, scale into language, then learn how fine-tuning differs from training from scratch.</p></div>
          <div className="project-grid">
            {projects.map((project) => <Link href={project.href} className="project-card" key={project.name}><div className="project-card-top"><span className="project-number">{project.number}</span><span className={`status status-${project.progress ? 'active' : 'planned'}`}>{project.status}</span></div><h3>{project.name}</h3><p>{project.description}</p><div className="progress-label"><span>{project.stage}</span><span>{project.progress}%</span></div><div className="progress-track"><span style={{ width: `${Math.max(project.progress, 2)}%` }} /></div></Link>)}
          </div>
        </section>

        <section className="section toolkit" id="toolkit"><p className="kicker">OUR TOOLKIT</p><div className="tool-row"><strong>MLX</strong><span>Apple-silicon-native tensor operations and automatic differentiation</span><strong>Python</strong><span>Data preparation, experiments and reproducible training scripts</span><strong>Living dashboard</strong><span>Loss curves and generated samples from each saved checkpoint</span></div></section>
      </main>
    </div>
  );
}
