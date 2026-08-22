'use client';

import { BeginnerOnly } from './BeginnerMode';
import { beginnerGuidance } from '@/lib/beginner-guidance';
import { plannedBeginnerGuidance } from '@/lib/planned-beginner-guidance';

export function BeginnerLessonIntro({ lessonSlug }: { lessonSlug: string }) {
  const guide = beginnerGuidance[lessonSlug];
  if (!guide) return null;
  return <BeginnerOnly className="beginner-lesson-intro"><p className="kicker">BEGINNER MODE · START HERE</p><h2>Background before the detail</h2><p>{guide.background}</p><div className="beginner-intro-grid"><section><strong>Picture it like this</strong><p>{guide.picture}</p></section><section><strong>A common misconception</strong><p>{guide.misconception}</p></section></div><aside><strong>What to understand by the end</strong><span>{guide.goal}</span></aside></BeginnerOnly>;
}

export function BeginnerSectionNote({ lessonSlug, sectionId }: { lessonSlug: string; sectionId: string }) {
  const note = beginnerGuidance[lessonSlug]?.notes[sectionId];
  if (!note) return null;
  return <BeginnerOnly className="beginner-section-note"><p className="beginner-note-label">BEGINNER’S DEEPER EXPLANATION</p><dl><div><dt>Why this matters</dt><dd>{note.why}</dd></div><div><dt>What is actually happening</dt><dd>{note.mechanism}</dd></div><div><dt>A useful comparison</dt><dd>{note.analogy}</dd></div>{note.boundary && <div><dt>Where the comparison stops</dt><dd>{note.boundary}</dd></div>}</dl></BeginnerOnly>;
}

export function BeginnerProjectGuide({ project }: { project: string }) {
  const copy: Record<string, { title: string; body: string; mechanism: string; boundary: string }> = {
    shakespeare: { title: 'What are we really building?', body: 'A small autocomplete system trained from random numbers. It never receives grammar rules or a list of quotations; instead it repeatedly sees earlier characters and is corrected on the character that actually followed.', mechanism: 'The first model uses one earlier character. Later models receive longer histories and gain better ways to route and transform that information. Generating a passage means predicting one character, appending it, and repeating.', boundary: 'Shakespeare-like output shows learned textual patterns, not comprehension of plots, speakers, or meaning.' },
    tinystories: { title: 'How will this differ?', body: 'TinyStories moves from individual characters to reusable word fragments and from imitation of one authorial corpus toward simple English story structure.', mechanism: 'A tokenizer learns or applies a vocabulary of common fragments. The transformer predicts one fragment at a time, allowing the same number of prediction steps to cover much more text than character tokens.', boundary: 'Simple coherent stories are a harder capability test, but fluency still does not establish factual knowledge or human understanding.' },
    sql: { title: 'Why is this a different kind of project?', body: 'Instead of creating every weight from randomness, we begin with a pretrained model whose weights already encode broad text patterns, then adapt a small part of it for a precise translation task.', mechanism: 'The input combines a natural-language question with a table description. The model emits SQL, and we run that SQL against a sandboxed database to check whether the returned answer is correct.', boundary: 'Producing executable SQL for familiar schemas does not mean the model understands the business meaning of the database or can safely query arbitrary systems.' },
  };
  const item = copy[project];
  return item ? <BeginnerOnly className="beginner-project-guide"><strong>{item.title}</strong><p>{item.body}</p><dl><div><dt>How the learning works</dt><dd>{item.mechanism}</dd></div><div><dt>Important boundary</dt><dd>{item.boundary}</dd></div></dl></BeginnerOnly> : null;
}

export function PlannedBeginnerGuide({ lessonSlug }: { lessonSlug: string }) {
  const item = plannedBeginnerGuidance[lessonSlug];
  if (!item) return null;
  return <BeginnerOnly className="beginner-section-note planned-beginner-guide"><p className="beginner-note-label">BEGINNER’S PREVIEW</p><dl><div><dt>Why this stage exists</dt><dd>{item.purpose}</dd></div><div><dt>What we will actually do</dt><dd>{item.mechanism}</dd></div><div><dt>Picture it</dt><dd>{item.picture}</dd></div><div><dt>Important boundary</dt><dd>{item.boundary}</dd></div></dl></BeginnerOnly>;
}
