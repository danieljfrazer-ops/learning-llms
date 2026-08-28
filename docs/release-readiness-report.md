# Public-release readiness report

Status date: 28 August 2026. This report describes the prepared working tree; it is not permission to publish.

## Completed preparation

- All 29 core lessons have been red-teamed for beginner completeness, conceptual visuals, evidence boundaries, engineering expectations, and bounded claims.
- Twenty-four reviewed dashboard JSON files now cover all evidence-bearing Shakespeare, TinyStories, and English → SQL stages. Machine-specific paths were rewritten without changing measured values; SQL benchmark rows were withheld while redistribution rights remain unresolved.
- Reproducibility metadata for the newly promoted TinyStories and SQL runs is committed without their learner checkpoints, optimizer state, raw datasets, or adapters.
- README onboarding covers Apple silicon, Linux MLX profiles, unsupported backends, reduced-compute adaptations, data download, inference, and evidence ownership.
- The public visitor experience states exactly what is readable, runnable locally, browser-capable, and still gated.
- Repository-native lesson audits and a least-privilege GitHub Actions workflow make validation runnable by a stranger without a private Codex skill installation.
- Dependabot, a private vulnerability-reporting policy, ignored secret/local paths, a high-confidence full-history credential audit, Cloudflare response headers, a 25 MiB asset check, and absolute-path checks establish the initial security baseline.
- The Cloudflare runbook now matches the actual Workers + static-assets build instead of describing a Pages-only deployment.
- The production build removes any ignored `public/data/local/` copies from `dist`, preventing a maintainer's My Lab evidence or prompts from entering a deploy artifact.
- Internal course links use durable document navigation because rendered QA found errors in Vinext beta's client prefetch/router path; this trades speculative transitions for reliable deep links and back/forward navigation.

## Deliberate release blockers

1. **Owner choices:** public display name, GitHub username/repository slug, source/course licence, whether public issues and pull requests are welcome, and `workers.dev` versus a purchased custom domain.
2. **Browser Shakespeare:** export selected checkpoints, publish manifests/checksums, implement WebAssembly-compatible inference off the main thread, and prove tokenizer/logit/greedy parity with MLX.
3. **Legal review:** decide the licence scope for original code and course prose; keep WikiSQL itself unredistributed; confirm whether derived Shakespeare/TinyStories weights may be distributed.
4. **Independent clean-clone QA:** test the Apple path and at least one Linux path before changing “available” to “verified”; record timings and disk use.
5. **Final security and UX QA:** supplement the repository's high-confidence history audit with an independent maintained secret scanner, then run an external-link check, accessibility/keyboard/responsive/reduced-motion pass, production preview smoke test, and rollback rehearsal.

`python3 scripts/audit_public_repo.py` enforces preparation invariants. `python3 scripts/audit_public_repo.py --release` intentionally fails until the licence and browser-model manifest exist.

## Publication boundary

No remote, GitHub repository, tag, Cloudflare application, deployment, DNS record, paid plan, or public access setting is created by this preparation. Those actions use one reviewed commit and require a separate final instruction from the owner.
