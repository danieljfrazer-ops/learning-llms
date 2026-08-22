'use client';

import { BeginnerOnly } from './BeginnerMode';
import { beginnerGuidance } from '@/lib/beginner-guidance';

export function BeginnerLessonIntro({ lessonSlug }: { lessonSlug: string }) {
  const guide = beginnerGuidance[lessonSlug];
  if (!guide) return null;
  return <BeginnerOnly className="beginner-lesson-intro"><p className="kicker">BEGINNER MODE · MENTAL MODEL</p><h2>Picture it like this</h2><p>{guide.picture}</p><aside><strong>What to understand by the end</strong><span>{guide.goal}</span></aside></BeginnerOnly>;
}

export function BeginnerSectionNote({ lessonSlug, sectionId }: { lessonSlug: string; sectionId: string }) {
  const note = beginnerGuidance[lessonSlug]?.notes[sectionId];
  if (!note) return null;
  return <BeginnerOnly className="beginner-section-note"><strong>Beginner’s lens</strong><p>{note}</p></BeginnerOnly>;
}

export function BeginnerProjectGuide({ project }: { project: string }) {
  const copy: Record<string, { title: string; body: string }> = {
    shakespeare: { title: 'What are we really building?', body: 'An autocomplete engine for Shakespeare characters. It studies which character tends to follow a visible history, then repeatedly predicts one more. The project grows that history and the machinery used to interpret it.' },
    tinystories: { title: 'How will this differ?', body: 'Instead of learning letter by letter, this model will use reusable word pieces—closer to assembling sentences from Lego bricks than individual grains of sand.' },
    sql: { title: 'Why is this a different kind of project?', body: 'We will begin with a model that already knows language, then teach a narrower translation skill: turning a question into a database instruction whose answer can be checked by running it.' },
  };
  const item = copy[project];
  return item ? <BeginnerOnly className="beginner-project-guide"><strong>{item.title}</strong><p>{item.body}</p></BeginnerOnly> : null;
}
