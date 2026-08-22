'use client';
import { BeginnerOnly } from './BeginnerMode';

const analogies: Record<string, string> = {
  Token: 'Like choosing whether a reading exercise uses letters, syllables, or whole words as its movable pieces.',
  Weight: 'One adjustable dial among thousands; training discovers useful dial positions.',
  Logit: 'A raw vote before the votes are converted into percentages.',
  Softmax: 'A pie maker that turns arbitrary scores into slices adding up to 100%.',
  Loss: 'The model’s error score for a set of answers; lower is better.',
  Gradient: 'A small arrow attached to each dial showing which way would reduce the error.',
  AdamW: 'The coach that decides how far to turn each dial using current and recent gradient arrows.',
  Checkpoint: 'A save-game slot containing the model’s learned dial positions.',
  'Context window': 'The size of the note card the model may look back at while guessing the next token.',
  Embedding: 'A learned description card made of numbers rather than words.',
  Tensor: 'A spreadsheet that may have more than two dimensions.',
  'Self-attention': 'A meeting in which each position decides which earlier speakers are relevant.',
  'Causal mask': 'A screen covering future answers during a next-token exam.',
  'Attention head': 'One specialist reader looking for one learned kind of relationship.',
  'Feed-forward network': 'A private workbench that processes each position after attention gathers information.',
  'Residual connection': 'A bypass lane that preserves the original message while adding new processing.',
  Perplexity: 'A translation of loss into average uncertainty; lower means fewer plausible next-token choices remain.',
  Seed: 'A repeatable starting ticket for pseudo-random choices.',
  Inference: 'Using the trained student to answer questions without giving further lessons.',
  Temperature: 'A creativity dial controlling how adventurous probability sampling becomes.',
};

export default function BeginnerGlossaryNote({ term }: { term: string }) {
  const analogy = analogies[term];
  return analogy ? <BeginnerOnly className="glossary-analogy"><strong>Picture it:</strong> {analogy}</BeginnerOnly> : null;
}
