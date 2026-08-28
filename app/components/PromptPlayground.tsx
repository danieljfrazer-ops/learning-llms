'use client';

import { useEffect, useState, useSyncExternalStore } from 'react';

type Health = { status: string; runs: { id: string; checkpoints: number[]; contextSize: number }[]; device: string };
type Completion = { prompt: string; continuation: string; run: string; checkpoint: number; temperature: number; seed: number; generatedCharacters: number; contextCharactersUsed: number; elapsedSeconds: number };

const endpoint = 'http://127.0.0.1:8001';
const presets = ['To be, or not to be', 'My lord, the night is', 'ROMEO:\n'];

function checkpointLabel(run: string, step: number) {
  if (run === 'final') return `Final 420K model · seed 43 · step ${step.toLocaleString()}`;
  if (run === 'warmup-cosine') return `Improved recipe · step ${step.toLocaleString()}`;
  if (step === 0) return 'Random weights · step 0';
  if (step === 1) return 'Minimally trained · step 1';
  return `Checkpoint · step ${step.toLocaleString()}`;
}

export default function PromptPlayground() {
  const isLocalSite = useSyncExternalStore(
    () => () => undefined,
    () => ['localhost', '127.0.0.1'].includes(window.location.hostname),
    () => null,
  );
  const [health, setHealth] = useState<Health | null>(null);
  const [prompt, setPrompt] = useState(presets[0]);
  const [selection, setSelection] = useState('final:3000');
  const [temperature, setTemperature] = useState(0.8);
  const [characters, setCharacters] = useState(180);
  const [seed, setSeed] = useState(42);
  const [results, setResults] = useState<Completion[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!isLocalSite) return;
    fetch(`${endpoint}/health`)
      .then(response => response.ok ? response.json() : Promise.reject(new Error('Inference service did not respond')))
      .then((nextHealth: Health) => {
        setHealth(nextHealth);
        const firstRun = nextHealth.runs[0];
        const firstCheckpoint = firstRun?.checkpoints[0];
        if (firstRun && firstCheckpoint !== undefined) setSelection(`${firstRun.id}:${firstCheckpoint}`);
      })
      .catch(() => setError('The local inference service is offline. Start it with the command shown below.'));
  }, [isLocalSite]);

  const hasCheckpoints = Boolean(health?.runs.some(run => run.checkpoints.length));

  async function requestCompletion(run: string, step: number) {
    const response = await fetch(`${endpoint}/generate`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, run, checkpoint: step, temperature, characters, seed }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error ?? 'Generation failed');
    return body as Completion;
  }

  async function generate(compareAll = false) {
    setBusy(true); setError('');
    try {
      const [selectedRun, selectedStep] = selection.split(':');
      const comparisons: [string, number][] = compareAll
        ? (health?.runs.flatMap(run => run.checkpoints.map(step => [run.id, step] as [string, number])) ?? [])
        : [[selectedRun, Number(selectedStep)]];
      const completions = [];
      for (const [run, step] of comparisons) completions.push(await requestCompletion(run, step));
      setResults(completions);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : 'Generation failed');
    } finally { setBusy(false); }
  }

  return <section className="playground" id="prompt-playground">
    <div className="playground-heading"><div><p className="kicker">CHECKPOINT PLAYGROUND</p><h2>Prompt the model yourself</h2><p>In a local clone, every completion comes from a checkpoint on your machine. The public release will enable this control only after the same course checkpoints pass browser/MLX parity.</p></div><span className={`service-state ${hasCheckpoints ? 'ready' : 'offline'}`}>{hasCheckpoints ? `● Ready · ${health?.device}` : isLocalSite === false ? '○ Browser export pending' : health ? '○ No local checkpoints' : '○ Service offline'}</span></div>
    <div className="playground-grid">
      <div className="playground-controls">
        <label>Beginning of the passage<textarea value={prompt} maxLength={2000} rows={5} onChange={event => setPrompt(event.target.value)} /></label>
        <div className="preset-row">{presets.map(preset => <button type="button" key={preset} onClick={() => setPrompt(preset)}>{preset.replace('\n', ' ↵')}</button>)}</div>
        <label>Model checkpoint<select value={hasCheckpoints ? selection : ''} disabled={!hasCheckpoints} onChange={event => setSelection(event.target.value)}>{hasCheckpoints ? health?.runs.flatMap(run => run.checkpoints.map(step => <option value={`${run.id}:${step}`} key={`${run.id}:${step}`}>{checkpointLabel(run.id, step)}</option>)) : <option value="">Train a checkpoint first</option>}</select></label>
        <div className="control-pair"><label>Temperature <strong>{temperature.toFixed(1)}</strong><input type="range" min="0.1" max="1.5" step="0.1" value={temperature} onChange={event => setTemperature(Number(event.target.value))} /></label><label>New characters <strong>{characters}</strong><input type="range" min="40" max="400" step="20" value={characters} onChange={event => setCharacters(Number(event.target.value))} /></label></div>
        <label>Random seed<input type="number" value={seed} onChange={event => setSeed(Number(event.target.value))} /></label>
        <div className="playground-actions"><button type="button" className="run-button" disabled={busy || !prompt || !hasCheckpoints} onClick={() => generate(false)}>{busy ? 'Generating…' : 'Complete with selected checkpoint'}</button><button type="button" className="compare-button" disabled={busy || !prompt || !hasCheckpoints} onClick={() => generate(true)}>Compare every checkpoint</button></div>
        {health && !hasCheckpoints && <aside className="playground-error"><strong>This is the intended blank-canvas state.</strong><span>Complete the Tiny transformer lesson to create your first local checkpoints, then restart this service.</span><code>uv run --no-sync python ml/shakespeare_transformer.py</code></aside>}
        {isLocalSite === false && <aside className="playground-error"><strong>The public browser model is not packaged yet.</strong><span>Read the reviewed checkpoint samples here, or clone the repository to prompt your own local checkpoints. Your prompt is not sent to a hosted model.</span></aside>}
        {isLocalSite !== false && error && <aside className="playground-error"><strong>{error}</strong><code>uv run --no-sync python ml/shakespeare_inference_server.py</code></aside>}
      </div>
      <div className="completion-list" aria-live="polite">{results.length ? results.map(result => <article className="completion-card" key={`${result.run}:${result.checkpoint}`}><div><strong>{checkpointLabel(result.run, result.checkpoint)}</strong><span>{result.elapsedSeconds.toFixed(3)} s · {result.contextCharactersUsed}/64 prompt characters visible</span></div><pre><mark>{result.prompt}</mark>{result.continuation}</pre></article>) : <div className="completion-empty"><strong>Your continuation will appear here.</strong><p>Try the improved checkpoint first, then compare it with random weights using the same prompt.</p></div>}</div>
    </div>
  </section>;
}
