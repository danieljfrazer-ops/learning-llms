# Beginner explanation quality red-team

Date: 2026-08-22

## Verdict before this rewrite

The first Beginner Mode release solved discoverability and coverage, but not teaching depth. Its cards were usually accurate and concise, yet many simply translated or shortened the paragraph immediately below. A first-time learner gained labels and friendly imagery without consistently gaining a causal model of the computation.

## Failure modes found

1. **Summary masquerading as explanation.** Cards often repeated the section’s conclusion without explaining the problem that made the component necessary.
2. **Metaphor without mapping.** Images such as students, meetings, kitchens, and workbenches were memorable, but sometimes left unclear which real values or operations corresponded to which familiar objects.
3. **Missing operational trace.** Cards named tokens, gradients, attention, or checkpoints without consistently stating what entered, what changed, and what came out.
4. **Insufficient prerequisite bridge.** Lesson introductions assumed learners already understood the relationship between architecture, weights, training, inference, and evaluation.
5. **Anthropomorphic leakage.** “Looks for,” “chooses,” “learns,” and “remembers” could be read literally unless the numerical mechanism and boundary appeared nearby.
6. **Shallow non-lesson pages.** The learning lifecycle, project overviews, glossary aids, and planned-stage previews received briefer treatment than completed Shakespeare lessons.
7. **Coverage audit without a depth contract.** The automated check proved that a card existed, not that it contained purpose, mechanism, analogy, and limitations.

## Remediation

- Replaced every one-line section lens with a four-part teaching card: why it matters, what actually happens, a useful comparison, and a boundary where needed.
- Added prerequisite background and a misconception correction to all 12 completed lesson introductions.
- Mapped model operations explicitly to metaphors and described where human or physical comparisons become false.
- Reworked the training lifecycle into the same why/mechanism/comparison structure.
- Expanded project overviews with learning mechanism and capability boundaries.
- Added substantive structured previews for all 10 planned TinyStories and SQL stages.
- Expanded the 20 most central glossary analogies into intuition, project use, and caution.
- Added a reusable beginner-writing standard and strengthened the maintenance audit to require the new structured content.

## Ongoing quality gate

A card fails even if technically correct when hiding it removes only a shorter restatement of adjacent prose. Future reviews must apply the subtraction, causal-gap, operational-trace, unfamiliar-noun, analogy-map, boundary, and course-connection tests documented in the LearningLLMs skill.
