import { readdir, stat } from 'node:fs/promises';
import path from 'node:path';

const serverRoot = path.resolve('dist/server');
const staticRoot = path.resolve('dist/client/browser');
const workerLimit = 3 * 1024 * 1024;
const staticAssetLimit = 25 * 1024 * 1024;

async function filesBelow(root) {
  const entries = await readdir(root, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const target = path.join(root, entry.name);
    if (entry.isDirectory()) files.push(...await filesBelow(target));
    else files.push(target);
  }
  return files;
}

const serverFiles = await filesBelow(serverRoot);
const serverBytes = (await Promise.all(serverFiles.map(file => stat(file)))).reduce((sum, file) => sum + file.size, 0);
const serverWasm = serverFiles.filter(file => file.endsWith('.wasm'));
if (serverWasm.length) throw new Error(`Browser WebAssembly leaked into the Cloudflare Worker module graph: ${serverWasm.join(', ')}`);
if (serverBytes > workerLimit) throw new Error(`Uncompressed Worker modules total ${serverBytes} bytes, above the conservative ${workerLimit}-byte free-plan gate`);

for (const file of await filesBelow(staticRoot)) {
  const details = await stat(file);
  if (details.size > staticAssetLimit) throw new Error(`Static browser asset exceeds Cloudflare's 25 MiB limit: ${file}`);
}

console.log(`Cloudflare bundle audit passed: ${serverBytes} server bytes, browser runtime isolated as static assets.`);
