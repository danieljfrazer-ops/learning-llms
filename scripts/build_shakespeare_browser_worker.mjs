import { copyFile, mkdir } from 'node:fs/promises';
import { build } from 'esbuild';

const outputDirectory = new URL('../public/browser/', import.meta.url);
await mkdir(outputDirectory, { recursive: true });

await build({
  entryPoints: [new URL('../app/workers/shakespeare-browser.worker.ts', import.meta.url).pathname],
  outfile: new URL('shakespeare-worker.js', outputDirectory).pathname,
  bundle: true,
  format: 'esm',
  platform: 'browser',
  target: ['es2022'],
  minify: true,
  sourcemap: false,
  legalComments: 'none',
});

await copyFile(
  new URL('../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm', import.meta.url),
  new URL('ort-wasm-simd-threaded.wasm', outputDirectory),
);
await copyFile(
  new URL('../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs', import.meta.url),
  new URL('ort-wasm-simd-threaded.mjs', outputDirectory),
);

console.log('Built static Shakespeare browser worker and WebAssembly runtime modules.');
