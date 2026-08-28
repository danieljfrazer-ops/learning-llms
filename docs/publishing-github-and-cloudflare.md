# Publishing GitHub and the Cloudflare wiki

This is the coordinated release record for the first public version of the three core projects. Daniel Frazer approved the GitHub and Cloudflare publication on 28 August 2026. The exact resolved `workers.dev` hostname still receives a final named approval before production access is enabled.

## Intended public experience

The same release should provide two complementary ways to learn:

1. **Public GitHub repository** — people can clone the complete course, reproduce the experiments, and populate My Lab with their own evidence.
2. **Public Cloudflare wiki** — people can read all 29 completed core lessons and reviewed reference results without installing anything, then run selected Shakespeare checkpoints inside their browser after the parity gate passes.

The public playground must use the model trained by this course. It must not quietly substitute an unrelated hosted language model.

## Release gate: browser-executable Shakespeare model

Complete this before either public launch.

### Select and package the evidence

- [x] Freeze the reviewed Shakespeare architecture, vocabulary, configuration, and selected run identifiers.
- [x] Select random, minimally trained, baseline transformer, and final checkpoints.
- [x] Store the compact browser weights as versioned repository and Cloudflare static assets.
- [x] Publish a checksum, parameter count, training step, seed, and source run identifier for every browser artifact.
- [x] Review the repository licence, upstream corpus status, and derived-weight distribution decision.

### Build browser inference

- [x] Export each selected MLX checkpoint to ONNX.
- [x] Implement the course's exact character vocabulary, encoding, context truncation, logits processing, temperature, and checkpoint behavior.
- [x] Run inference in a Web Worker so generation does not freeze navigation.
- [x] Use WebAssembly as the compatibility baseline. WebGPU remains a future performance enhancement and is not required for this small model.
- [x] Lazy-load model files only when the visitor opens or starts the playground.
- [x] Keep prompts on the visitor's device and state this beside the playground.
- [x] Provide honest loading, unsupported-browser, corrupt-model, memory, validation, and generation-failure states.

[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) documents browser inference with WebAssembly and WebGPU. Its [browser compatibility table](https://onnxruntime.ai/docs/get-started/with-javascript/web.html) should be checked again at release time because browser support changes.

### Prove that the port is the same model

Do not approve the browser playground merely because its output looks Shakespeare-like.

- [x] Compare MLX and browser token IDs for fixed input strings, including newlines and unknown-input handling.
- [x] Compare checkpoint conversion manifests layer by layer.
- [x] Compare logits from fixed token sequences within the published `0.015` maximum-absolute-difference tolerance.
- [x] Require identical greedy continuations except for one documented near-tie in the minimally trained checkpoint; final and baseline continuations match.
- [x] Keep deterministic browser sampling while documenting that its JavaScript random-number generator is not MLX's generator.
- [x] Check random, minimally trained, baseline, and final checkpoints.
- [x] Test checkpoint switching and compare-all behavior without stale weights or labels.
- [x] Record download sizes and observed browser generation behavior; speeds remain device-dependent rather than promised benchmarks.

### Preserve local and public behaviour

The playground should choose its engine from the environment:

| Context | Engine | Evidence meaning |
|---|---|---|
| Local clone | Existing loopback Python/MLX service | The learner's checkpoints |
| Public Cloudflare wiki | Browser model using published static weights | Reviewed reference checkpoints |
| Browser cannot execute model | No generation; show published samples and a clear explanation | Reference evidence remains readable |

Reference results and My Lab remain separate. On the public wiki, My Lab should explain that training and learner evidence require a local clone; it must not imply that browser prompting retrains the model.

## Prepare one release candidate

After all three core projects and browser inference are complete:

- [ ] Freeze one candidate commit containing lessons, reference JSON, model manifests, browser artifacts, reproduction instructions, and release notes.
- [ ] Run every repository, lesson, Python, browser-parity, accessibility, and production-build check against that exact commit.
- [ ] Test a clean clone independently from the working development checkout.
- [ ] Build the Cloudflare artifact from the same commit that will receive the Git tag.
- [ ] Confirm the site is static-first: lesson pages, reference JSON, model files, runtime files, and fonts should be static assets wherever possible.
- [ ] Confirm there is no public inference API, secret, learner prompt logging, analytics identifier, or unreviewed external request.
- [ ] Prepare a rollback target before changing any public access.

Cloudflare currently documents static-asset requests as free and unlimited, while dynamic Functions or Workers consume plan allowances. Recheck the live [static-assets billing documentation](https://developers.cloudflare.com/workers/static-assets/billing-and-limitations/) and [platform limits](https://developers.cloudflare.com/workers/platform/limits/) immediately before release rather than assuming today's free tier will remain unchanged.

### Match the Cloudflare product to this build

This repository uses the Cloudflare Vite plugin. `npm run build` produces a Worker entry point plus a static asset directory and generated `dist/server/wrangler.json`; it is not a Pages-only static export. Use **Workers Builds** Git integration and name the Cloudflare Worker `learning-llms` to match the generated configuration. Run `npm run preview` to exercise the production artifact locally before any upload.

The free-plan budget at the August 2026 review is:

- static asset requests: free and unlimited;
- Worker requests: 100,000 per day on Workers Free, with requests rejected after the limit rather than silently creating paid usage;
- individual static assets: at most 25 MiB; and
- no database, object storage, analytics, public inference API, or paid Worker feature required by the initial design.

A `workers.dev` hostname has no domain-registration cost. A custom portfolio domain is optional and its registration/renewal is outside Cloudflare Workers' free hosting allowance. Before release, verify that the account is still on Workers Free and has not opted into the paid plan.

## Coordinated publication sequence

These actions require explicit owner approval at the time of release.

1. Create the public GitHub repository and push only the reviewed release candidate.
2. Enable branch protection and the required lint, build, audit, and browser-parity checks.
3. Create the first tagged GitHub release, including scope, hardware evidence, limitations, artifact checksums, and browser support.
4. Create or connect a `learning-llms` Worker through Workers Builds Git integration and make `main` the production branch.
5. Deploy the exact tagged commit to a non-public or preview URL first.
6. Smoke-test representative lessons, internal links, Reference/My Lab behaviour, every published checkpoint, and the WebAssembly compatibility path. WebGPU is not part of this release.
7. Obtain a final explicit approval that names the resolved public site access.
8. Make the Cloudflare wiki public and verify the production URL, security headers, metadata, accessibility, and rollback path.
9. Add reciprocal links: GitHub links to the hosted wiki; the wiki links to the tagged source and reproduction instructions.

Cloudflare's [Workers Builds Git integration guide](https://developers.cloudflare.com/workers/ci-cd/builds/git-integration/) explains automatic production and preview deployments. Keep production credentials in Cloudflare's integration rather than adding an account token to the public repository or a contributor-triggerable workflow.

## Post-release monitoring

- [ ] Verify that a new visitor can read lessons without executing server code.
- [ ] Monitor build failures, static-asset sizes, browser-model load failures, and free-tier policy changes.
- [ ] Treat public prompts as private: do not add prompt collection later without a separate privacy decision and visible consent.
- [ ] Re-run parity tests whenever training code, model architecture, vocabulary, export code, inference code, or checkpoints change.
- [ ] Publish corrected model artifacts under a new release rather than silently replacing files whose checksums were documented.

## Definition of ready

The combined release is ready only when a stranger can either clone the repository and reproduce the course, or open the public wiki and run the genuine selected Shakespeare checkpoints in their browser, with the origin and limitations of every result made clear.
