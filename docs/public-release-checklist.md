# Future public-release checklist

Do not publish solely because the code builds. Complete this checklist deliberately.

## Legal and provenance

- [ ] Choose and add an explicit repository licence.
- [ ] Confirm that dependencies, copied snippets, datasets, samples, and artwork are compatible with that licence.
- [ ] Keep downloaded datasets out of Git and link to original sources and terms.
- [ ] Review whether generated Shakespeare samples create any additional attribution requirement.

## Repository hygiene

- [ ] Run a secret scan across full Git history, not only the working tree.
- [ ] Audit file sizes and remove accidental checkpoints, caches, datasets, environments, and personal paths.
- [ ] Decide whether selected checkpoints belong in release assets or Git LFS; publish checksums if so.
- [ ] Replace preparation-status wording in README with the final repository URL and release status.
- [ ] Add issue and pull-request templates only if the project will accept community contributions.

## Reproduction

- [ ] Test a clean clone on the documented Apple profile.
- [ ] Test at least one Linux profile before describing it as verified.
- [ ] Run the system report, dataset downloader, every supported script, inference service, lint, and production build.
- [ ] Confirm My Lab begins empty and never modifies Reference results.
- [ ] Confirm reference dashboards work without downloading checkpoints.
- [ ] Record expected disk use and approximate time ranges by hardware class.

## Teaching quality

- [ ] Red-team every lesson and planned project for unexplained prerequisites.
- [ ] Run the Beginner Mode audit and the human subtraction/causal/analogy tests.
- [ ] Verify all external links, dataset descriptions, and hardware claims against primary sources.
- [ ] Confirm all displayed metrics and samples trace to committed reference JSON and configuration.
- [ ] Complete accessibility, keyboard, responsive, and reduced-motion checks.

## Publication

- [ ] Create the public repository only after the above review.
- [ ] Protect the default branch and require build/lint/audit checks.
- [ ] Create a tagged first release with a clear scope and known limitations.
- [ ] Publish no learner-local files, prompts, credentials, or machine-specific paths.
