'use client';

import Link from 'next/link';
import type { ReactNode } from 'react';
import { glossary } from '@/lib/wiki-data';
import { useBeginnerMode } from './BeginnerMode';

const definitions = Object.fromEntries(glossary.map(([term, definition]) => [term.toLowerCase().replaceAll(' ', '-'), definition]));

export default function BeginnerTerm({ id, children }: { id: string; children: ReactNode }) {
  const { enabled } = useBeginnerMode();
  return <span className={`beginner-term ${enabled ? 'expanded' : ''}`}><Link className="inline-term" href={`/glossary#${id}`}>{children}</Link>{enabled && <span className="beginner-term-definition"><b>Plain English:</b> {definitions[id] ?? 'Open the glossary for a full definition.'}</span>}</span>;
}
