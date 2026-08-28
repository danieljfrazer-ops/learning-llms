'use client';

import { useEffect, useState, useSyncExternalStore } from 'react';

type Checkpoint = { id: string; label: string; history: string; checkpointSha256: string };
type Health = { status: string; error: string | null; checkpoints: Checkpoint[]; architecture: { contextSize?: number; parameterCount?: number }; device: string };
type Completion = { prompt: string; continuation: string; checkpoint: string; checkpointLabel: string; seed: number; temperature: number; maximumTokens: number; generatedTokens: number; endedWithEos: boolean; promptTokens: number; contextTokensUsed: number; contextSize: number; elapsedSeconds: number; weightsUpdated: false };

const endpoint = 'http://127.0.0.1:8002';
const presets = [
  'Once upon a time, a small fox found a silver key. The fox',
  'Mia saw a cold kitten waiting in the rain. She decided to',
  'The little dragon was afraid of its own fire, so its friend said',
];

export default function TinyStoriesPromptPlayground() {
  const isLocalSite = useSyncExternalStore(
    () => () => undefined,
    () => ['localhost', '127.0.0.1'].includes(window.location.hostname),
    () => null,
  );
  const [health, setHealth] = useState<Health | null>(null);
  const [prompt, setPrompt] = useState(presets[0]);
  const [checkpoint, setCheckpoint] = useState('final-selected');
  const [temperature, setTemperature] = useState(0.8);
  const [maximumTokens, setMaximumTokens] = useState(64);
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
        if (nextHealth.status !== 'ready') setError(nextHealth.error ?? 'The service found no evaluated checkpoints.');
        setCheckpoint(current => nextHealth.checkpoints.some(item => item.id === current) ? current : (nextHealth.checkpoints[0]?.id ?? current));
      })
      .catch(() => setError('The local inference service is offline. Start it with the command shown below.'));
  }, [isLocalSite]);

  const ready = health?.status === 'ready' && health.checkpoints.length > 0;

  async function requestCompletion(checkpointId: string) {
    const response = await fetch(`${endpoint}/generate`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, checkpoint: checkpointId, temperature, maximumTokens, seed }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error ?? 'Generation failed');
    return body as Completion;
  }

  async function generate(compareAll = false) {
    setBusy(true); setError('');
    try {
      const checkpointIds = compareAll ? (health?.checkpoints.map(item => item.id) ?? []) : [checkpoint];
      const completions: Completion[] = [];
      for (const checkpointId of checkpointIds) completions.push(await requestCompletion(checkpointId));
      setResults(completions);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : 'Generation failed');
    } finally { setBusy(false); }
  }

  return <section className="playground" id="tinystories-live-playground">
    <div className="playground-heading"><div><p className="kicker">CHECKPOINT PLAYGROUND</p><h2>Continue your own story opening</h2><p>A local clone can send the same prompt and sampling controls to random and trained checkpoints. The larger browser runtime remains follow-up work; the hosted wiki does not send your prompt to an external model.</p></div><span className={`service-state ${ready ? 'ready' : 'offline'}`}>{ready ? `● Ready · ${health.device}` : isLocalSite === false ? '○ Local clone required' : health ? '○ Checkpoints unavailable' : '○ Service offline'}</span></div>
    <div className="playground-grid">
      <div className="playground-controls">
        <label>Story opening<textarea value={prompt} maxLength={2000} rows={6} onChange={event => setPrompt(event.target.value)} /></label>
        <div className="preset-row">{presets.map((preset, index) => <button type="button" key={preset} onClick={() => setPrompt(preset)}>Example {index + 1}</button>)}</div>
        <label>Evaluated checkpoint<select value={ready ? checkpoint : ''} disabled={!ready} onChange={event => setCheckpoint(event.target.value)}>{ready ? health.checkpoints.map(item => <option value={item.id} key={item.id}>{item.label}</option>) : <option value="">Complete Lesson 10 first</option>}</select></label>
        <div className="control-pair"><label>Temperature <strong>{temperature.toFixed(1)}</strong><input aria-label="Temperature" type="range" min="0.1" max="1.5" step="0.1" value={temperature} onChange={event => setTemperature(Number(event.target.value))} /></label><label>Maximum new tokens <strong>{maximumTokens}</strong><input aria-label="Maximum new tokens" type="range" min="16" max="160" step="8" value={maximumTokens} onChange={event => setMaximumTokens(Number(event.target.value))} /></label></div>
        <label>Sampling seed<input type="number" min="0" max="2147483647" value={seed} onChange={event => setSeed(Number(event.target.value))} /></label>
        <div className="playground-actions"><button type="button" className="run-button" disabled={busy || !prompt.trim() || !ready} onClick={() => generate(false)}>{busy ? 'Generating…' : 'Complete with selected checkpoint'}</button><button type="button" className="compare-button" disabled={busy || !prompt.trim() || !ready} onClick={() => generate(true)}>Compare all available checkpoints</button></div>
        {isLocalSite === false && <aside className="playground-error"><strong>TinyStories generation is local-only in the initial public release.</strong><span>Its 5.82M-parameter runtime and roughly 22.2 MiB checkpoints need a separately tested browser port. The reviewed fixed outputs remain visible above.</span></aside>}
        {isLocalSite !== false && error && <aside className="playground-error"><strong>{error}</strong><code>uv run --no-sync python ml/tinystories_inference_server.py</code></aside>}
      </div>
      <div className="completion-list" aria-live="polite">{results.length ? results.map(result => <article className="completion-card" key={result.checkpoint}><div><strong>{result.checkpointLabel}</strong><span>{result.elapsedSeconds.toFixed(3)} s · {result.generatedTokens}/{result.maximumTokens} tokens{result.endedWithEos ? ' · EOS' : ' · token limit'}</span></div><pre><mark>{result.prompt}</mark>{result.continuation}</pre><p className="completion-metadata">Prompt: {result.promptTokens} tokens · initially visible: {result.contextTokensUsed}/{result.contextSize} · temperature {result.temperature.toFixed(1)} · seed {result.seed} · weights updated: no</p></article>) : <div className="completion-empty"><strong>Your continuation will appear here.</strong><p>Start with the final selected checkpoint when available, then compare every listed checkpoint while the prompt, temperature, length, and seed stay fixed.</p></div>}</div>
    </div>
  </section>;
}
