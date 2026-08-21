'use client';

import { useEffect, useState } from 'react';

type Health = { status: string; checkpoints: number[]; contextSize: number; device: string };
type Completion = { prompt: string; continuation: string; checkpoint: number; temperature: number; seed: number; generatedCharacters: number; contextCharactersUsed: number; elapsedSeconds: number };

const endpoint = 'http://127.0.0.1:8001';
const fallbackCheckpoints = [0, 1, 50, 250, 1000, 2000, 3000];
const presets = ['To be, or not to be', 'My lord, the night is', 'ROMEO:\n'];

function checkpointLabel(step: number) {
  if (step === 0) return 'Random weights · step 0';
  if (step === 1) return 'Minimally trained · step 1';
  return `Checkpoint · step ${step.toLocaleString()}`;
}

export default function PromptPlayground() {
  const [health, setHealth] = useState<Health | null>(null);
  const [prompt, setPrompt] = useState(presets[0]);
  const [checkpoint, setCheckpoint] = useState(3000);
  const [temperature, setTemperature] = useState(0.8);
  const [characters, setCharacters] = useState(180);
  const [seed, setSeed] = useState(42);
  const [results, setResults] = useState<Completion[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch(`${endpoint}/health`)
      .then(response => response.ok ? response.json() : Promise.reject(new Error('Inference service did not respond')))
      .then(setHealth)
      .catch(() => setError('The local inference service is offline. Start it with the command shown below.'));
  }, []);

  async function requestCompletion(step: number) {
    const response = await fetch(`${endpoint}/generate`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, checkpoint: step, temperature, characters, seed }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error ?? 'Generation failed');
    return body as Completion;
  }

  async function generate(compareAll = false) {
    setBusy(true); setError('');
    try {
      const steps = compareAll ? (health?.checkpoints ?? fallbackCheckpoints) : [checkpoint];
      const completions = [];
      for (const step of steps) completions.push(await requestCompletion(step));
      setResults(completions);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : 'Generation failed');
    } finally { setBusy(false); }
  }

  return <section className="playground" id="prompt-playground">
    <div className="playground-heading"><div><p className="kicker">LIVE LOCAL INFERENCE</p><h2>Prompt the model yourself</h2><p>Every completion comes from a saved model checkpoint on your Mac. Comparing all checkpoints keeps the prompt, temperature and seed fixed so training is the variable you can see.</p></div><span className={`service-state ${health ? 'ready' : 'offline'}`}>{health ? `● Ready · ${health.device}` : '○ Service offline'}</span></div>
    <div className="playground-grid">
      <div className="playground-controls">
        <label>Beginning of the passage<textarea value={prompt} maxLength={2000} rows={5} onChange={event => setPrompt(event.target.value)} /></label>
        <div className="preset-row">{presets.map(preset => <button type="button" key={preset} onClick={() => setPrompt(preset)}>{preset.replace('\n', ' ↵')}</button>)}</div>
        <label>Checkpoint<select value={checkpoint} onChange={event => setCheckpoint(Number(event.target.value))}>{(health?.checkpoints ?? fallbackCheckpoints).map(step => <option value={step} key={step}>{checkpointLabel(step)}</option>)}</select></label>
        <div className="control-pair"><label>Temperature <strong>{temperature.toFixed(1)}</strong><input type="range" min="0.1" max="1.5" step="0.1" value={temperature} onChange={event => setTemperature(Number(event.target.value))} /></label><label>New characters <strong>{characters}</strong><input type="range" min="40" max="400" step="20" value={characters} onChange={event => setCharacters(Number(event.target.value))} /></label></div>
        <label>Random seed<input type="number" value={seed} onChange={event => setSeed(Number(event.target.value))} /></label>
        <div className="playground-actions"><button type="button" className="run-button" disabled={busy || !prompt} onClick={() => generate(false)}>{busy ? 'Generating…' : 'Complete with selected checkpoint'}</button><button type="button" className="compare-button" disabled={busy || !prompt} onClick={() => generate(true)}>Compare every checkpoint</button></div>
        {error && <aside className="playground-error"><strong>{error}</strong><code>.venv/bin/python ml/shakespeare_inference_server.py</code></aside>}
      </div>
      <div className="completion-list" aria-live="polite">{results.length ? results.map(result => <article className="completion-card" key={result.checkpoint}><div><strong>{checkpointLabel(result.checkpoint)}</strong><span>{result.elapsedSeconds.toFixed(3)} s · {result.contextCharactersUsed}/64 prompt characters visible</span></div><pre><mark>{result.prompt}</mark>{result.continuation}</pre></article>) : <div className="completion-empty"><strong>Your continuation will appear here.</strong><p>Try the final checkpoint first, then compare it with random weights using the same prompt.</p></div>}</div>
    </div>
  </section>;
}
