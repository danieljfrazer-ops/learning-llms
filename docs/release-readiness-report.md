# Public-release readiness report

Status date: 28 August 2026. The GitHub repository is public; the Cloudflare production site still requires the final named-access approval recorded below.

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
- A clean clone on 32 GB Apple silicon completed dependency setup, the verified Tiny Shakespeare download, a real 400-step bigram run, all course audits, lint, and a production build. Linux remains unverified rather than being implied by that result.
- All 52 unique external links resolved during the release-candidate check. Desktop, mobile, local, and public-like hosted states were rendered and smoke-tested, including the SQL evidence view and disabled hosted playground controls.
- The Next, React, Vinext, Vite, Cloudflare, and Wrangler dependency set was upgraded together; a clean `npm ci` and `npm audit --audit-level=low` now report zero known vulnerabilities. ESLint remains on its compatible 9.x line because the React lint plugin in the current Next release fails under ESLint 10.
- Four reviewed Shakespeare checkpoints were exported to checksum-pinned ONNX files. A Web Worker runs them with ONNX Runtime Web/WebAssembly without a public inference API; 12 MLX/browser comparisons passed with a 0.015 logit tolerance and one documented near-tied minimally trained greedy choice.
- Public-like rendered QA generated from the selected final checkpoint in 0.217 seconds, compared all four checkpoints, rejected an out-of-vocabulary emoji clearly, emitted no browser warnings/errors, and reflowed at 390 px without horizontal scrolling.
- The owner selected Apache-2.0 for software/model files, CC BY 4.0 for original educational content, the `danieljfrazer-ops/learning-llms` identity, public issues/pull requests, private commit-email rewriting, and the free `workers.dev` hostname.
- GitHub is published at <https://github.com/danieljfrazer-ops/learning-llms>; the resolved Cloudflare hostname is <https://learning-llms.daniel-j-frazer.workers.dev>.
- Cloudflare Workers is confirmed on the Free plan dashboard at 0 / 100,000 requests for the day and $0.00 billable usage before this release.
- Gitleaks 8.30.1 independently scanned all 31 pre-release commits. Its 21 initial generic-key matches were verified as published `tokenizerSha256` integrity fields; a narrow line-level allowlist documents that false-positive class, after which the full-history scan passed.

## Deliberate release blockers

1. **Remaining platform claim:** Linux must remain labelled framework-supported but unverified until a clean-clone test records its timings and disk use; this does not block an Apple-verified release.
2. **Final operational QA:** rehearse rollback against the exact tagged Cloudflare candidate before public access.

`python3 scripts/audit_public_repo.py --release` enforces repository/licence/model-manifest invariants. `npm run audit:browser` independently executes the published ONNX files through WebAssembly and compares them with frozen MLX fixtures.

## Publication boundary

No remote, GitHub repository, tag, Cloudflare application, deployment, DNS record, paid plan, or public access setting is created by this preparation. Those actions use one reviewed commit and require a separate final instruction from the owner.
