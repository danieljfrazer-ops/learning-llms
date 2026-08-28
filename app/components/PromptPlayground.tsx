'use client';

import { useEffect, useRef, useState, useSyncExternalStore } from 'react';

type Health = { status: string; runs: { id: string; checkpoints: number[]; contextSize: number }[]; device: string };
type BrowserModel = { id: string; label: string; step: number; contextSize: number; parameterCount: number };
type BrowserManifest = { runtimeVersion: string; models: BrowserModel[]; limitations: string[] };
type Completion = { prompt: string; continuation: string; run: string; checkpoint: number; temperature: number; seed: number; generatedCharacters: number; contextCharactersUsed: number; elapsedSeconds: number };
type PendingRequest = { resolve: (completion: Completion) => void; reject: (problem: Error) => void };

const endpoint = 'http://127.0.0.1:8001';
const presets = ['To be, or not to be', 'My lord, the night is', 'ROMEO:\n'];

function checkpointLabel(run: string, step: number) {
  if (run === 'random') return 'Random weights · step 0';
  if (run === 'minimal') return 'Minimally trained · step 1';
  if (run === 'baseline') return 'Baseline 112K transformer · step 3,000';
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
  const [browserManifest, setBrowserManifest] = useState<BrowserManifest | null>(null);
  const [prompt, setPrompt] = useState(presets[0]);
  const [selection, setSelection] = useState('final');
  const [temperature, setTemperature] = useState(0.8);
  const [characters, setCharacters] = useState(180);
  const [seed, setSeed] = useState(42);
  const [results, setResults] = useState<Completion[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const workerRef = useRef<Worker | null>(null);
  const nextRequestId = useRef(1);
  const pendingRequests = useRef(new Map<number, PendingRequest>());

  useEffect(() => {
    if (isLocalSite === null) return;
    if (isLocalSite) {
      fetch(`${endpoint}/health`)
        .then(response => response.ok ? response.json() : Promise.reject(new Error('Inference service did not respond')))
        .then((nextHealth: Health) => {
          setHealth(nextHealth);
          const preferred = nextHealth.runs.find(run => run.id === 'final') ?? nextHealth.runs[0];
          const preferredStep = preferred?.checkpoints.at(-1);
          if (preferred && preferredStep !== undefined) setSelection(`${preferred.id}:${preferredStep}`);
        })
        .catch(() => setError('The local inference service is offline. Start it with the command shown below.'));
      return;
    }

    // Keep the browser-only ONNX runtime outside the server module graph. The
    // prebuild step emits this worker and its WASM runtime as static assets.
    const worker = new Worker('/browser/shakespeare-worker.js', { type: 'module' });
    const requests = pendingRequests.current;
    workerRef.current = worker;
    worker.onmessage = event => {
      const message = event.data as { type: string; requestId: number; manifest?: BrowserManifest; result?: Completion; error?: string };
      if (message.type === 'ready' && message.manifest) {
        setBrowserManifest(message.manifest);
        setSelection(message.manifest.models.some(model => model.id === 'final') ? 'final' : message.manifest.models[0]?.id ?? '');
        return;
      }
      const pending = requests.get(message.requestId);
      if (!pending) {
        if (message.type === 'error') setError(message.error ?? 'Browser inference failed');
        return;
      }
      requests.delete(message.requestId);
      if (message.type === 'result' && message.result) pending.resolve(message.result);
      else pending.reject(new Error(message.error ?? 'Browser inference failed'));
    };
    worker.onerror = () => setError('The on-device model runtime could not start in this browser. Reference samples remain available below.');
    worker.postMessage({ type: 'load', requestId: 0 });
    return () => {
      worker.terminate();
      workerRef.current = null;
      for (const pending of requests.values()) pending.reject(new Error('Browser model worker stopped'));
      requests.clear();
    };
  }, [isLocalSite]);

  const localHasCheckpoints = Boolean(health?.runs.some(run => run.checkpoints.length));
  const browserHasCheckpoints = Boolean(browserManifest?.models.length);
  const hasCheckpoints = isLocalSite ? localHasCheckpoints : browserHasCheckpoints;

  async function requestCompletion(run: string, step: number) {
    if (isLocalSite) {
      const response = await fetch(`${endpoint}/generate`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, run, checkpoint: step, temperature, characters, seed }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.error ?? 'Generation failed');
      return body as Completion;
    }
    const worker = workerRef.current;
    if (!worker) throw new Error('The browser model worker is not ready');
    const requestId = nextRequestId.current++;
    const completion = new Promise<Completion>((resolve, reject) => pendingRequests.current.set(requestId, { resolve, reject }));
    worker.postMessage({ type: 'generate', requestId, modelId: run, prompt, temperature, characters, seed });
    return completion;
  }

  async function generate(compareAll = false) {
    setBusy(true); setError('');
    try {
      const comparisons: [string, number][] = isLocalSite
        ? (compareAll
            ? (health?.runs.flatMap(run => run.checkpoints.map(step => [run.id, step] as [string, number])) ?? [])
            : [[selection.split(':')[0], Number(selection.split(':')[1])]])
        : (compareAll
            ? (browserManifest?.models.map(model => [model.id, model.step] as [string, number]) ?? [])
            : [[selection, browserManifest?.models.find(model => model.id === selection)?.step ?? 0]]);
      const completions = [];
      for (const [run, step] of comparisons) completions.push(await requestCompletion(run, step));
      setResults(completions);
    } catch (problem) {
      setError(problem instanceof Error ? problem.message : 'Generation failed');
    } finally { setBusy(false); }
  }

  const stateLabel = hasCheckpoints
    ? isLocalSite ? `● Ready · ${health?.device}` : '● Ready · on-device WebAssembly'
    : isLocalSite === false ? '○ Loading browser model' : health ? '○ No local checkpoints' : '○ Service offline';

  return <section className="playground" id="prompt-playground">
    <div className="playground-heading"><div><p className="kicker">CHECKPOINT PLAYGROUND</p><h2>Prompt the model yourself</h2><p>{isLocalSite === false ? 'The public wiki runs the reviewed course checkpoints inside a Web Worker using WebAssembly. Prompts and generated text stay in this browser.' : 'In a local clone, every completion comes from a checkpoint on your machine through the loopback-only MLX service.'}</p></div><span className={`service-state ${hasCheckpoints ? 'ready' : 'offline'}`}>{stateLabel}</span></div>
    <div className="playground-grid">
      <div className="playground-controls">
        <label>Beginning of the passage<textarea value={prompt} maxLength={2000} rows={5} onChange={event => setPrompt(event.target.value)} /></label>
        <div className="preset-row">{presets.map(preset => <button type="button" key={preset} onClick={() => setPrompt(preset)}>{preset.replace('\n', ' ↵')}</button>)}</div>
        <label>Model checkpoint<select value={hasCheckpoints ? selection : ''} disabled={!hasCheckpoints} onChange={event => setSelection(event.target.value)}>{hasCheckpoints ? (isLocalSite ? health?.runs.flatMap(run => run.checkpoints.map(step => <option value={`${run.id}:${step}`} key={`${run.id}:${step}`}>{checkpointLabel(run.id, step)}</option>)) : browserManifest?.models.map(model => <option value={model.id} key={model.id}>{model.label}</option>)) : <option value="">Loading checkpoints…</option>}</select></label>
        <div className="control-pair"><label>Temperature <strong>{temperature.toFixed(1)}</strong><input type="range" min="0.1" max="1.5" step="0.1" value={temperature} onChange={event => setTemperature(Number(event.target.value))} /></label><label>New characters <strong>{characters}</strong><input type="range" min="40" max="400" step="20" value={characters} onChange={event => setCharacters(Number(event.target.value))} /></label></div>
        <label>Random seed<input type="number" value={seed} onChange={event => setSeed(Number(event.target.value))} /></label>
        <div className="playground-actions"><button type="button" className="run-button" disabled={busy || !prompt || !hasCheckpoints} onClick={() => generate(false)}>{busy ? 'Generating on this device…' : 'Complete with selected checkpoint'}</button><button type="button" className="compare-button" disabled={busy || !prompt || !hasCheckpoints} onClick={() => generate(true)}>Compare every checkpoint</button></div>
        {health && !localHasCheckpoints && <aside className="playground-error"><strong>This is the intended blank-canvas state.</strong><span>Complete the Tiny transformer lesson to create your first local checkpoints, then restart this service.</span><code>uv run --no-sync python ml/shakespeare_transformer.py</code></aside>}
        {isLocalSite === false && browserHasCheckpoints && <aside className="playground-note"><strong>Your prompt stays on this device.</strong><span>The static ONNX files run in a browser worker. Nothing is sent to a public inference API, saved to the course, or used for training.</span></aside>}
        {error && <aside className="playground-error"><strong>{error}</strong>{isLocalSite && <code>uv run --no-sync python ml/shakespeare_inference_server.py</code>}</aside>}
      </div>
      <div className="completion-list" aria-live="polite">{results.length ? results.map(result => <article className="completion-card" key={`${result.run}:${result.checkpoint}`}><div><strong>{checkpointLabel(result.run, result.checkpoint)}</strong><span>{result.elapsedSeconds.toFixed(3)} s · {result.contextCharactersUsed}/64 prompt characters visible</span></div><pre><mark>{result.prompt}</mark>{result.continuation}</pre></article>) : <div className="completion-empty"><strong>Your continuation will appear here.</strong><p>Try the final checkpoint first, then compare it with random weights using the same prompt.</p></div>}</div>
    </div>
  </section>;
}
