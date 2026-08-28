#!/usr/bin/env node

import { access, readdir, rm } from 'node:fs/promises';
import { constants } from 'node:fs';
import { resolve } from 'node:path';

const repositoryRoot = resolve(import.meta.dirname, '..');
const clientRoot = resolve(repositoryRoot, 'dist', 'client');
const localEvidence = resolve(clientRoot, 'data', 'local');

if (!localEvidence.startsWith(`${clientRoot}/`)) {
  throw new Error('Refusing to scrub a path outside dist/client');
}

try {
  await access(localEvidence, constants.F_OK);
  await rm(localEvidence, { recursive: true, force: false });
  console.log('Removed learner-local evidence copies from the deploy artifact.');
} catch (error) {
  if (error?.code !== 'ENOENT') throw error;
  console.log('Deploy artifact contained no learner-local evidence.');
}

const remaining = (await readdir(clientRoot, { recursive: true }))
  .filter(path => path.startsWith('data/local/'));
if (remaining.length) {
  throw new Error(`Private build scrub failed: ${remaining.join(', ')}`);
}
