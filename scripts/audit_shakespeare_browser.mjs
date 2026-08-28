#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import * as ort from 'onnxruntime-web/wasm';

const root = resolve(import.meta.dirname, '..');
const modelDirectory = resolve(root, 'public/models/shakespeare');
const manifest = JSON.parse(await readFile(resolve(modelDirectory, 'manifest.json'), 'utf8'));
const fixture = JSON.parse(await readFile(resolve(modelDirectory, manifest.parityFixture.file), 'utf8'));
const requiredModels = ['random', 'minimal', 'baseline', 'final'];
const maximumAssetBytes = 25 * 1024 * 1024;

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

function fail(message) {
  throw new Error(`Shakespeare browser parity audit failed: ${message}`);
}

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function argmax(values) {
  let best = 0;
  for (let index = 1; index < values.length; index += 1) if (values[index] > values[best]) best = index;
  return best;
}

if (manifest.runtimeVersion !== '1.29.0') fail(`manifest runtime ${manifest.runtimeVersion} does not match the pinned browser runtime`);
if (manifest.vocabulary.length !== 65 || new Set(manifest.vocabulary).size !== 65) fail('vocabulary is not the frozen set of 65 unique characters');
if (manifest.unknownCharacterPolicy !== 'reject') fail('unknown-character policy must be explicit and rejecting');
if (!manifest.promptsStayOnDevice) fail('manifest does not state that prompts remain on-device');
if (requiredModels.some(modelId => !manifest.models.some(model => model.id === modelId))) fail('one or more required checkpoint stages are missing');
const fixtureBytes = await readFile(resolve(modelDirectory, manifest.parityFixture.file));
if (sha256(fixtureBytes) !== manifest.parityFixture.sha256) fail('parity fixture checksum does not match its manifest');

let comparisonCount = 0;
let largestDifference = 0;
let documentedNearTies = 0;
for (const model of manifest.models) {
  const modelBytes = await readFile(resolve(modelDirectory, model.file));
  if (modelBytes.byteLength > maximumAssetBytes) fail(`${model.file} exceeds Cloudflare's 25 MiB asset limit`);
  if (sha256(modelBytes) !== model.sha256) fail(`${model.file} checksum does not match its manifest`);
  if (!model.parameters.length || model.parameters.some(parameter => !parameter.sourceSha256 || !parameter.exportedSha256)) {
    fail(`${model.id} lacks layer-by-layer source/export checksums`);
  }
  const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ['wasm'] });
  const modelFixtures = fixture.models[model.id];
  if (!Array.isArray(modelFixtures) || !modelFixtures.length) fail(`${model.id} has no MLX fixture`);
  for (const test of modelFixtures) {
    const encoded = [...test.prompt].map(character => manifest.vocabulary.indexOf(character));
    if (encoded.some(tokenId => tokenId < 0) || encoded.join(',') !== test.tokenIds.join(',')) {
      fail(`${model.id} tokenizer IDs differ for ${JSON.stringify(test.prompt)}`);
    }
    const result = await session.run({
      token_ids: new ort.Tensor('int64', BigInt64Array.from(encoded, tokenId => BigInt(tokenId)), [1, encoded.length]),
    });
    const logits = result.logits.data;
    const lastLogits = logits.subarray(logits.length - manifest.vocabulary.length);
    if (lastLogits.length !== test.lastLogits.length) fail(`${model.id} logit shape differs for ${JSON.stringify(test.prompt)}`);
    let localDifference = 0;
    for (let index = 0; index < lastLogits.length; index += 1) {
      localDifference = Math.max(localDifference, Math.abs(lastLogits[index] - test.lastLogits[index]));
    }
    largestDifference = Math.max(largestDifference, localDifference);
    if (localDifference > fixture.logitAbsoluteTolerance) {
      fail(`${model.id} logits differ by ${localDifference} for ${JSON.stringify(test.prompt)}`);
    }

    const generated = [...encoded];
    const greedyIds = [];
    for (let index = 0; index < fixture.greedyCharacters; index += 1) {
      const context = generated.slice(-model.contextSize);
      const stepResult = await session.run({
        token_ids: new ort.Tensor('int64', BigInt64Array.from(context, tokenId => BigInt(tokenId)), [1, context.length]),
      });
      const stepLogits = stepResult.logits.data;
      const nextId = argmax(stepLogits.subarray(stepLogits.length - manifest.vocabulary.length));
      generated.push(nextId);
      greedyIds.push(nextId);
    }
    const firstDifference = greedyIds.findIndex((tokenId, index) => tokenId !== test.greedyTokenIds[index]);
    if (firstDifference >= 0) {
      const mlxMargin = test.greedyTopTwoMargins[firstDifference];
      if (mlxMargin > fixture.greedyNearTieTolerance) {
        fail(`${model.id} greedy continuation differs outside a numerical near-tie for ${JSON.stringify(test.prompt)}`);
      }
      documentedNearTies += 1;
    }
    comparisonCount += 1;
  }
}

console.log(
  `Shakespeare browser parity audit passed: ${manifest.models.length} checkpoints, ${comparisonCount} MLX/WebAssembly comparisons, maximum absolute logit difference ${largestDifference.toExponential(3)}, ${documentedNearTies} documented numerical near-tie.`,
);
