/// <reference lib="webworker" />

import * as ort from 'onnxruntime-web/wasm';

type BrowserModel = {
  id: string;
  label: string;
  file: string;
  step: number;
  sourceRunId: string;
  contextSize: number;
  parameterCount: number;
};

type Manifest = {
  runtimeVersion: string;
  vocabulary: string[];
  models: BrowserModel[];
  limitations: string[];
};

type GenerateRequest = {
  type: 'generate';
  requestId: number;
  modelId: string;
  prompt: string;
  temperature: number;
  characters: number;
  seed: number;
};

type LoadRequest = { type: 'load'; requestId: number };
type WorkerRequest = LoadRequest | GenerateRequest;

const workerScope: DedicatedWorkerGlobalScope = self as unknown as DedicatedWorkerGlobalScope;
const sessions = new Map<string, Promise<ort.InferenceSession>>();
let manifestPromise: Promise<Manifest> | null = null;

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = '/browser/';

function manifest(): Promise<Manifest> {
  manifestPromise ??= fetch('/models/shakespeare/manifest.json', { cache: 'force-cache' }).then(async response => {
    if (!response.ok) throw new Error(`Model manifest returned HTTP ${response.status}`);
    return response.json() as Promise<Manifest>;
  });
  return manifestPromise;
}

async function sessionFor(model: BrowserModel): Promise<ort.InferenceSession> {
  let pending = sessions.get(model.id);
  if (!pending) {
    pending = fetch(`/models/shakespeare/${model.file}`, { cache: 'force-cache' })
      .then(async response => {
        if (!response.ok) throw new Error(`Checkpoint returned HTTP ${response.status}`);
        return response.arrayBuffer();
      })
      .then(bytes => ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] }));
    sessions.set(model.id, pending);
  }
  return pending;
}

function encode(prompt: string, vocabulary: string[]): number[] {
  const characterIds = new Map(vocabulary.map((character, id) => [character, id]));
  const unknown = [...new Set([...prompt].filter(character => !characterIds.has(character)))];
  if (unknown.length) throw new Error(`Prompt contains characters outside the 65-character vocabulary: ${JSON.stringify(unknown)}`);
  return [...prompt].map(character => characterIds.get(character)!);
}

function mulberry32(seed: number) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function sample(logits: Float32Array, temperature: number, random: () => number): number {
  if (temperature === 0) {
    let best = 0;
    for (let index = 1; index < logits.length; index += 1) if (logits[index] > logits[best]) best = index;
    return best;
  }
  let maximum = -Infinity;
  for (const logit of logits) maximum = Math.max(maximum, logit / temperature);
  const probabilities = new Float64Array(logits.length);
  let total = 0;
  for (let index = 0; index < logits.length; index += 1) {
    probabilities[index] = Math.exp(logits[index] / temperature - maximum);
    total += probabilities[index];
  }
  let threshold = random() * total;
  for (let index = 0; index < probabilities.length; index += 1) {
    threshold -= probabilities[index];
    if (threshold <= 0) return index;
  }
  return probabilities.length - 1;
}

async function generate(request: GenerateRequest) {
  const modelManifest = await manifest();
  const model = modelManifest.models.find(candidate => candidate.id === request.modelId);
  if (!model) throw new Error(`Unknown browser checkpoint ${JSON.stringify(request.modelId)}`);
  if (!request.prompt) throw new Error('Enter at least one prompt character');
  if (request.prompt.length > 2_000) throw new Error('Prompt must contain at most 2,000 characters');
  if (request.temperature !== 0 && (request.temperature < 0.1 || request.temperature > 2)) {
    throw new Error('Temperature must be between 0.1 and 2.0');
  }
  if (request.characters < 1 || request.characters > 500) throw new Error('Generated length must be between 1 and 500 characters');

  const modelSession = await sessionFor(model);
  const promptIds = encode(request.prompt, modelManifest.vocabulary);
  const generated = [...promptIds];
  const continuation: number[] = [];
  const random = mulberry32(request.seed);
  const startedAt = performance.now();
  for (let index = 0; index < request.characters; index += 1) {
    const context = generated.slice(-model.contextSize);
    const input = new ort.Tensor('int64', BigInt64Array.from(context, value => BigInt(value)), [1, context.length]);
    const result = await modelSession.run({ token_ids: input });
    const allLogits = result.logits.data as Float32Array;
    const lastLogits = allLogits.subarray(allLogits.length - modelManifest.vocabulary.length);
    const nextId = sample(lastLogits, request.temperature, random);
    generated.push(nextId);
    continuation.push(nextId);
  }
  return {
    prompt: request.prompt,
    continuation: continuation.map(tokenId => modelManifest.vocabulary[tokenId]).join(''),
    run: model.id,
    checkpoint: model.step,
    temperature: request.temperature,
    seed: request.seed,
    generatedCharacters: continuation.length,
    contextCharactersUsed: Math.min(request.prompt.length, model.contextSize),
    elapsedSeconds: (performance.now() - startedAt) / 1000,
  };
}

workerScope.addEventListener('message', async event => {
  const request = event.data as WorkerRequest;
  try {
    if (request.type === 'load') {
      const modelManifest = await manifest();
      workerScope.postMessage({ type: 'ready', requestId: request.requestId, manifest: modelManifest });
      return;
    }
    workerScope.postMessage({ type: 'result', requestId: request.requestId, result: await generate(request) });
  } catch (problem) {
    workerScope.postMessage({
      type: 'error',
      requestId: request.requestId,
      error: problem instanceof Error ? problem.message : 'Browser inference failed',
    });
  }
});

export {};
