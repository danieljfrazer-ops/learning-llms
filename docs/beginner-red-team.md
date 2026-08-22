# Beginner clarity red-team

Date: 2026-08-22

## Verdict before remediation

The wiki was technically detailed but still assumed too much prior knowledge. A motivated beginner could reproduce commands while failing to build a durable mental model of why the machinery exists.

## Systemic failure modes found

1. **Definitions were navigational rather than instructional.** Linked terms opened the glossary, forcing the learner to leave the sentence where the idea first mattered.
2. **Mechanics often preceded purpose.** Tensor shapes, commands and parameter arithmetic appeared before a concrete picture of what information was moving or why.
3. **The learning path compressed the hard concepts.** “Tokenise,” “backpropagate,” “checkpoint” and “held-out” were presented as steps without everyday-language bridges.
4. **Analogies were sporadic.** Excellent local explanations existed, but there was no guaranteed mental model for every stage or section.
5. **Planned projects exposed jargon early.** TinyStories and SQL roadmap pages named subword tokenisation, LoRA and execution evaluation without an optional beginner translation.
6. **No persistence contract existed.** Future lessons could satisfy the old content checklist without updating beginner guidance.

## Remediation implemented

- Persistent Beginner Mode toggle in the global top bar.
- Lesson-level mental model and learning goal for all 12 Shakespeare lessons.
- A distinct “Beginner’s lens” for every substantive Shakespeare lesson section.
- In-place plain-English expansion of glossary-linked terms.
- Concrete analogies for core glossary concepts.
- Beginner previews on all planned TinyStories and English-to-SQL lessons.
- Plain-language expansion of every step on the global learning path.
- Updated maintenance skill and executable completeness audit.

## Analogy safety rule

Human metaphors describe information flow, not consciousness. Lessons must not imply that attention literally chooses, models understand, checkpoints remember experiences, or optimisers act with intent. Standard technical prose remains authoritative; Beginner Mode is an additional bridge.
