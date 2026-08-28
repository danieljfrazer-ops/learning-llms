# Tiny Shakespeare lesson quality red-team

Date: 2026-08-28

## Audience and method

This review treated a motivated software or AI engineer with no prior machine-learning training as the target learner. Every Shakespeare lesson was read in course order and challenged on five dimensions:

1. Does it answer a causal question rather than merely introduce machinery?
2. Can the learner distinguish incoming state, controlled change, evidence, and outgoing state?
3. Can the learner run it without guessing what is read, written, mutated, considered complete, or safe to retry?
4. Are comparisons honest about confounded architecture changes, evaluation noise, development-set reuse, and training-seed variation?
5. Are provenance, licence, hardware, process, checkpoint, and capability boundaries visible before they matter?

The review also applied the Beginner Mode subtraction and operational-trace tests, checked each conceptual visual, traced every command to its current script, and compared the lesson claims with the saved-state behavior of those scripts.

## Verdict

Shakespeare remains the strongest conceptual introduction in the course. Its progression from a random 65×65 table through context, attention, a transformer, evaluation, optimisation, scaling, and replication is unusually legible. Beginner Mode already explains why each mechanism exists and carefully avoids treating attention or generation as human understanding.

The missing layer was an engineer-readable state contract. Important facts—fresh initialisation versus checkpoint reuse, weight-only versus resumable state, output locations, completion signals, restart behavior, and practical cost—were spread across prose and source. All twelve lessons now receive the same engineering brief used by TinyStories and SQL.

Two content inconsistencies were also corrected:

- The dataset lesson now uses the repository's checksum-verifying atomic downloader instead of teaching an unverified direct `curl` command. It distinguishes the upstream repository licence from the separately unresolved redistribution status of the compiled corpus and derived checkpoints.
- The evaluation lesson no longer promises that the final lesson will create a fresh test strategy. The final run repeats complete training across seeds, but it continues to use validation evidence that shaped earlier choices and therefore cannot become an untouched test retrospectively.

## Lesson disposition

| Lesson | Red-team focus | Disposition |
|---|---|---|
| 01 · Lab setup | Make environment state and backend assumptions reproducible | Added install/read/write/completion/recovery contract and zero-model-state boundary |
| 02 · Data and tokenisation | Align the lesson with the actual verified downloader and expose data rights | Replaced direct download, added hash/atomic-write behavior and conservative licence boundary |
| 03 · Random baseline | Preserve the causal before-training checkpoint despite sharing Lesson 4's command | Added shared-command state boundary and exact step-zero completion signal |
| 04 · Bigram training | Distinguish a weight-only checkpoint cohort from resumable training | Added restart requirement, evidence outputs, and noisy-late-difference boundary |
| 05 · Context windows | Avoid attributing the full improvement to context alone | Made fresh initialisation and simultaneous capacity change explicit |
| 06 · Self-attention | Prevent a teaching comparison being mistaken for a clean ablation | Made changed context/capacity/architecture and fresh-history boundaries explicit |
| 07 · Tiny transformer | Preserve full architecture and checkpoint lineage for downstream lessons | Added input/output, completion, non-resumability, and capability contract |
| 08 · Evaluation | Separate batch-sampling variation from training variation and final testing | Corrected the future-test promise and added evaluator-only mutation/retry contract |
| 09 · Prompt playground | Keep exploration separate from selection or training | Added process, loopback, cache, metadata, and cherry-picking boundaries |
| 10 · Training improvements | Clarify that candidates are fresh matched trainings, not baseline continuations | Added three-variant lineage, prerequisite, restart, and one-seed boundary |
| 11 · Scaling experiment | Clarify fresh histories and local budget-limited selection | Added four-run quality–memory–time contract and no-scaling-law boundary |
| 12 · Final evaluation | Separate architecture replication from practical checkpoint selection | Added seed-42 reuse, seed-43/44 fresh training, small-sample, and no-test boundary |

## Residual boundaries

- No untouched Shakespeare test split exists. All reported validation values are development evidence after repeated validation-guided choices.
- Training scripts save weight-only checkpoints. Interrupted ordinary runs restart; they do not provide exact optimiser/data-order continuation like TinyStories Lesson 7.
- Architecture comparisons intentionally change multiple coupled properties at several teaching transitions. They reveal useful progression but are not pure scientific ablations.
- Three final seeds are a useful stability check, not a precise estimate of the full training-result distribution.
- Tiny Shakespeare is a narrow, compiled subset of Shakespeare-like dialogue. Provenance and a hash do not establish completeness, representativeness, or redistribution permission.
- Prompted text and automated loss do not establish understanding, factual reliability, literary coherence, or production fitness.

## Ongoing quality gate

`python3 scripts/audit_lesson_engineering_briefs.py` now covers all 29 executable core-course lessons. It catches missing or stale briefs; human review must still test whether each brief correctly names incoming state, the controlled change, written artifacts, completion, retry behavior, cost, and justified claim.
