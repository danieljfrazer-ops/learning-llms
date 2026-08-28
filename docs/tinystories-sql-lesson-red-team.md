# TinyStories and English → SQL lesson quality red-team

Date: 2026-08-28

## Audience and review method

This review treated a motivated software or AI engineer with no prior machine-learning training as the target learner. It inspected every TinyStories and English → SQL lesson against four questions:

1. Can the learner explain why the lesson exists and what causal question it answers?
2. Can the learner identify the exact state entering the lesson, the one controlled change, and the state handed forward?
3. Can the learner run the work without guessing what is read, computed, written, mutated, considered complete, or safe to retry?
4. Can the learner state what the evidence supports and what remains unmeasured?

The review also checked the Beginner Mode contract, glossary links, visual coverage, dataset and checkpoint provenance, reference/local evidence separation, and the transition between inherited-history and fresh-start experiments.

## Findings and remediation

The conceptual teaching was already strong: the lessons usually introduced purpose before mechanism, exposed random or minimally trained behavior, connected metrics to visible outputs, and stated important limitations. The main weakness was operational fragmentation. Commands, prerequisite artifacts, mutation boundaries, completion signals, and recovery rules existed in prose or scripts but were not available as one predictable pre-run contract.

Every TinyStories and SQL lesson now begins with an engineering brief containing:

- the causal question;
- incoming state and controlled change;
- decision evidence and outgoing state;
- the claim boundary; and
- a collapsible run contract covering reads, computation, writes, mutation, completion, recovery, and practical cost.

This is deliberately additive. Beginner Mode still teaches the mental model, the visual still shows the information flow, and the main lesson remains the authoritative explanation. The brief gives engineers a compact state-and-artifact map before they act.

The SQL final evaluation also now reports a 95% Wilson interval beside each execution count and surfaces the most frequent failure group. This makes the sampling uncertainty of the 100-example test cohort visible without implying that an interval repairs dataset bias or establishes production safety.

## TinyStories disposition

| Lesson | Red-team focus | Disposition |
|---|---|---|
| 01 · Transition to subwords | Distinguish plumbing success from learned capability | Added explicit zero-update, artifact, invariant, and recovery contract |
| 02 · Dataset audit | Make trust, provenance, and split decisions actionable | Added audit completion and upstream-failure boundary |
| 03 · Tokenizer experiment | Separate tokenizer selection from model quality | Added controlled-change and exact handoff contract |
| 04 · Sequence batching | Expose boundary masks and tensor-state requirements | Added read/compute/write and invariant-based completion contract |
| 05 · Random GPT baseline | Define the official causal checkpoint zero | Added strict reload and configuration-drift recovery contract |
| 06 · First pretraining | Prevent weight-only checkpoints being mistaken for resumable state | Added explicit restart requirement and checkpoint limitation |
| 07 · Checkpoints dashboard | Explain what exact continuation requires | Added model, optimiser, trainer, RNG/data-order, manifest, and tolerance contract |
| 08 · Training recipe | Preserve inherited history across matched branches | Added matched-batch, isolated-branch, and no-universal-winner boundary |
| 09 · Scaling budget | Prevent Lesson 8 history being spliced into fresh architectures | Added fresh-start lineage and gated-selection contract |
| 10 · Story evaluation | Separate frozen behavioral evidence from training | Added checkpoint cohort, proxy limits, and evaluator-only mutation contract |
| 11 · Playground | Separate exploration from unbiased evaluation | Added loopback, hash, reproducibility, and process-lifecycle contract |
| 12 · Final story model | State seed replication and test-set limits precisely | Added three-path lineage and made the absence of an untouched test split explicit |

## English → SQL disposition

| Lesson | Red-team focus | Disposition |
|---|---|---|
| 01 · Task and dataset audit | Define the allowed program and licence boundary before model work | Added source, evaluator, mutation, and blocked-redistribution contract |
| 02 · Base-model selection | Keep test closed and separate mechanical failures from poor behavior | Added candidate eligibility, cache effects, and selection contract |
| 03 · Prompt formatting | Prevent prompt choice from changing task semantics | Added matched-input and development-only handoff contract |
| 04 · LoRA fine-tuning | Make frozen base versus trainable adapter state legible | Added zero-effect parity, adapter identity, and prerequisite mismatch contract |
| 05 · Execution evaluation | Avoid overclaiming from a small final sample | Added interval estimates, leading failure groups, immutable post-test decision rule, and rerun boundary |

## Residual boundaries

- TinyStories has no untouched test split. Its final measurements are development evidence after repeated validation-guided choices.
- English → SQL reports a seeded 100-example sample, not the complete WikiSQL test split, and the unresolved redistribution licence remains a publication blocker.
- Automated surface metrics, failure groups, and intervals support diagnosis but do not measure human usefulness, production security, or broad language understanding.
- Run contracts describe expected state; the scripts and evidence manifests remain the executable source of truth.
- English → SQL reference evidence was promoted only after a separate maintainer review. Learner-local results are still never copied into Reference mode automatically.

## Ongoing quality gate

For each new executable lesson, ask a reviewer to hide the implementation and reconstruct the state transition using only the lesson. If they cannot name the incoming artifacts, controlled change, output artifacts, completion signal, retry boundary, cost class, and justified claim, the lesson is not complete even when its command succeeds. Run `python3 scripts/audit_lesson_engineering_briefs.py` to catch missing or stale briefs before review.
