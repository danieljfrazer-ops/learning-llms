import Link from '@/app/components/CourseLink';
import { projects as projectRecords } from '@/lib/wiki-data';
import { BeginnerModeToggle, BeginnerOnly } from '@/app/components/BeginnerMode';
import { EvidenceMachineCard, EvidenceModeToggle, LocalOnly, ReferenceOnly } from '@/app/components/EvidenceMode';
import { CourseMapVisual } from '@/app/components/LearningVisual';

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
          <p className="nav-label">INDEPENDENT</p>
          <Link className="nav-link" href="/continue"><span>04+</span> Continue yourself</Link>
          <p className="nav-label">REFERENCE</p>
          <Link className="nav-link" href="/glossary"><span>A–Z</span> Glossary</Link>
          <a className="nav-link" href="#toolkit"><span>⌘</span> Toolkit</a>
        </nav>
        <EvidenceMachineCard />
      </aside>

      <main className="main-content">
        <header className="topbar"><div><span className="eyebrow">LIVING WIKI</span><span className="sync-state">● Documentation synced</span></div><div className="topbar-tools"><EvidenceModeToggle /><BeginnerModeToggle /><a className="search-link" href="#projects">Jump to projects <kbd>↓</kbd></a></div></header>
        <section className="hero">
          <div>
            <p className="kicker">BUILDING LANGUAGE MODELS FROM FIRST PRINCIPLES</p>
            <h1>Learn by watching<br /><em>language emerge.</em></h1>
            <p className="hero-copy">A hands-on record of every dataset, tensor, training run and mistake—as we teach small models to predict what comes next.</p>
            <BeginnerOnly className="home-beginner"><strong>New to all of this?</strong><span>A language model repeatedly calculates probabilities for the next piece of text. During training, known text supplies the correct next pieces and numerical error signals gradually alter millions—or here, thousands—of adjustable weights.</span><dl><div><dt>Why start tiny?</dt><dd>The mechanism is the same family of prediction, loss, gradient, and sampling ideas used at larger scale, but runs finish quickly enough for us to inspect every stage.</dd></div><div><dt>What this is not</dt><dd>We are not downloading a hidden Shakespeare expert or programming grammar rules. We begin with random weights and measure which patterns training actually creates.</dd></div></dl></BeginnerOnly>
            <div className="hero-actions"><Link className="button primary" href="/projects/sql/lessons/task-and-dataset-audit">Begin English → SQL <span>→</span></Link><Link className="button secondary" href="/concepts/training-lifecycle">See the learning path</Link></div>
          </div>
          <div className="terminal-card" aria-label="Current experiment status">
            <div className="terminal-title"><span>● ● ●</span><code>course · progression</code></div>
            <ReferenceOnly><pre><span className="muted">$ course.status()</span>{'\n'}{'{'}{'\n'}  <span className="key">shakespeare</span>: <span className="value">&quot;complete&quot;</span>,{'\n'}  <span className="key">tinystories</span>: <span className="value">&quot;complete&quot;</span>,{'\n'}  <span className="key">english_to_sql</span>: <span className="value">&quot;complete&quot;</span>,{'\n'}  <span className="key">next</span>: <span className="value">&quot;independent project&quot;</span>{'\n'}{'}'}</pre><div className="terminal-footer"><span className="live-dot" /> Reviewed evidence published for all three core projects</div></ReferenceOnly>
            <LocalOnly><pre><span className="muted">$ my_lab.status()</span>{'\n'}{'{'}{'\n'}  <span className="key">evidence</span>: <span className="value">&quot;your local result files&quot;</span>,{'\n'}  <span className="key">writes_to</span>: <span className="value">&quot;ignored local workspace&quot;</span>,{'\n'}  <span className="key">continue</span>: <span className="value">&quot;open a project lesson&quot;</span>{'\n'}{'}'}</pre><div className="terminal-footer"><span className="live-dot" /> Dashboards show “Not run yet” until each script completes</div></LocalOnly>
          </div>
        </section>

        <section className="section" id="projects">
          <div className="section-heading"><div><p className="kicker">THE CURRICULUM</p><h2>Three models, one mental map</h2></div><p>We begin with random noise, scale into language, then learn how fine-tuning differs from training from scratch.</p></div>
          <CourseMapVisual />
          <div className="project-grid">
            {projects.map((project) => <Link href={project.href} className="project-card" key={project.name}><div className="project-card-top"><span className="project-number">{project.number}</span><span className={`status status-${project.progress ? 'active' : 'planned'}`}>{project.status}</span></div><h3>{project.name}</h3><p>{project.description}</p><div className="progress-label"><span>{project.stage}</span><span>{project.progress}%</span></div><div className="progress-track"><span style={{ width: `${Math.max(project.progress, 2)}%` }} /></div></Link>)}
          </div>
        </section>

        <section className="section">
          <div className="section-heading"><div><p className="kicker">PUBLIC EXPERIENCE</p><h2>Read here; reproduce in your own lab</h2></div><p>The hosted wiki shows every completed core lesson and reviewed result. Training, checkpoints, datasets, and learner prompts remain on the visitor&apos;s own machine.</p></div>
          <div className="project-grid">
            <article className="project-card"><div className="project-card-top"><span className="project-number">WEB</span><span className="status status-active">READ</span></div><h3>Complete reference course</h3><p>Twenty-nine core lessons, Beginner Mode, diagrams, glossary, measured dashboards, limitations, and sources—without blank results masquerading as evidence.</p></article>
            <article className="project-card"><div className="project-card-top"><span className="project-number">LAB</span><span className="status status-active">RUN</span></div><h3>Clone and reproduce</h3><p>Hardware-aware setup, verified downloads, readable experiments, ignored local evidence, and loopback-only Shakespeare and TinyStories playgrounds.</p></article>
            <article className="project-card"><div className="project-card-top"><span className="project-number">AI</span><span className="status status-planned">GATED</span></div><h3>Honest browser models</h3><p>Public Shakespeare prompting will ship only after the genuine checkpoints pass MLX/browser parity. TinyStories and SQL show their current local-only boundaries explicitly.</p></article>
          </div>
        </section>

        <section className="section toolkit" id="toolkit"><p className="kicker">OUR TOOLKIT</p><div className="tool-row"><strong>MLX</strong><span>Apple-silicon-native tensor operations and automatic differentiation</span><strong>Python</strong><span>Data preparation, experiments and reproducible training scripts</span><strong>Living dashboard</strong><span>Loss curves and generated samples from each saved checkpoint</span></div></section>
      </main>
    </div>
  );
}
