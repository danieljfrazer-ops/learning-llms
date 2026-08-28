'use client';

import Link from './CourseLink';
import type { ReactNode } from 'react';
import { glossary } from '@/lib/wiki-data';
import { useBeginnerMode } from './BeginnerMode';
import { beginnerGlossaryGuidance } from './BeginnerGlossaryNote';

const entries = Object.fromEntries(glossary.map(([term, definition]) => [term.toLowerCase().replaceAll(' ', '-'), { term, definition }]));

export default function BeginnerTerm({ id, children }: { id: string; children: ReactNode }) {
  const { enabled } = useBeginnerMode();
  const entry = entries[id];
  const guide = entry ? beginnerGlossaryGuidance[entry.term] : undefined;
  return <span className={`beginner-term ${enabled ? 'expanded' : ''}`}><Link className="inline-term" href={`/glossary#${id}`}>{children}</Link>{enabled && <span className="beginner-term-definition"><span><b>Plain English</b>{entry?.definition ?? 'Open the glossary for a full definition.'}</span>{guide && <span><b>How to use the idea here</b>{guide.use}</span>}</span>}</span>;
}
