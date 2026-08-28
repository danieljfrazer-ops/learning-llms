# Future public-release checklist

Do not publish solely because the code builds. Complete this checklist deliberately.

For the current completed work and remaining blockers, see the [public-release readiness report](release-readiness-report.md). Run `python3 scripts/audit_public_repo.py --release` immediately before freezing a candidate; its owner-decision and browser-artifact failures are intentional until those gates are closed.

## Legal and provenance

- [x] Add Apache-2.0 for code and CC BY 4.0 for original lesson content, with a clear `NOTICE` and data-specific exceptions.
- [x] Review dependencies, dataset provenance, derived weights, samples, and artwork; record the conclusions and unresolved jurisdiction caveat in [licensing and data](licensing-and-data.md).
- [x] Keep downloaded datasets out of Git and link to original sources and terms.
- [x] Review generated Shakespeare samples and document that they are illustrative model output, not redistributed corpus text.

## Repository hygiene

- [x] Run Gitleaks across full Git history, not only the working tree.
- [x] Audit file sizes and exclude accidental checkpoints, caches, datasets, environments, learner evidence, and machine-specific paths.
- [x] Keep the four compact browser checkpoints as ordinary versioned static assets and publish their checksums.
- [x] Name the final GitHub repository in README; add the public-site URL after Cloudflare resolves the hostname.
- [x] Add issue, lesson-feedback, security-reporting, and pull-request templates.

## Reproduction

- [x] Test a clean clone on the documented Apple profile.
- [ ] Test at least one Linux profile before describing it as verified.
- [x] Run the system report, safe fixtures, local inference smoke test, lint, audits, and production build. Long training runs remain opt-in learner exercises.
- [x] Confirm My Lab begins empty and never modifies Reference results.
- [x] Confirm reference dashboards work without downloading training checkpoints.
- [x] Record expected disk use and approximate time ranges by hardware class.

## Teaching quality

- [x] Red-team Shakespeare, TinyStories, and English-to-SQL lessons for a non-technical learner.
- [x] Run the Beginner Mode audit and the human subtraction/causal/analogy tests.
- [x] Verify external links, dataset descriptions, and hardware claims against primary sources.
- [x] Confirm all displayed metrics and samples trace to committed reference JSON and configuration.
- [x] Complete semantic, focus-visible, keyboard, mobile-responsive, and reduced-motion checks on the release candidate.

## Browser playground release gate

Complete the detailed [GitHub and Cloudflare publication plan](publishing-github-and-cloudflare.md) before either public launch.

- [x] Export selected Shakespeare checkpoints for browser execution.
- [x] Provide WebAssembly execution without a public inference API. WebGPU is deliberately deferred: it is an optional enhancement, not a release requirement or compatibility baseline.
- [x] Prove tokenizer, parameter, logits, greedy-generation, and checkpoint-selection parity against MLX.
- [x] Publish model manifests, checksums, run identifiers, numerical tolerances, download sizes, and browser limitations.
- [x] Confirm public prompts remain on the visitor's device.
- [x] Confirm an unsupported or failed browser receives a useful explanation while published reference evidence remains readable.

## Publication

- [ ] Freeze one release-candidate commit for both GitHub and Cloudflare.
- [ ] Create the public repository only after the above review.
- [ ] Protect the default branch and require build/lint/audit checks.
- [ ] Create a tagged first release with a clear scope and known limitations.
- [ ] Build and preview the Cloudflare wiki from that exact tagged commit.
- [ ] Verify that lesson content and model files are static-first and no public inference server is required.
- [ ] Smoke-test browser inference, Reference/My Lab separation, accessibility, metadata, and rollback before changing site access.
- [ ] Obtain explicit final approval before making the Cloudflare deployment public.
- [ ] Link the public wiki to the tagged source and the repository to the public wiki.
- [ ] Publish no learner-local files, prompts, credentials, or machine-specific paths.
