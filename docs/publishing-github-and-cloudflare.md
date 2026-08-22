# Publishing GitHub and the Cloudflare wiki

This is the coordinated release plan for the end of the three core projects. It is preparation guidance, not permission to publish. Creating the GitHub repository, changing site access, connecting Cloudflare, or deploying publicly always requires an explicit final request from the repository owner.

## Intended public experience

The same release should provide two complementary ways to learn:

1. **Public GitHub repository** — people can clone the complete course, reproduce the experiments, and populate My Lab with their own evidence.
2. **Public Cloudflare wiki** — people can read the completed lessons and reference results without installing anything, then run selected Shakespeare checkpoints inside their browser.

The public playground must use the model trained by this course. It must not quietly substitute an unrelated hosted language model.

## Release gate: browser-executable Shakespeare model

Complete this before either public launch.

### Select and package the evidence

- [ ] Freeze the reviewed Shakespeare architecture, vocabulary, configuration, and selected run identifiers.
- [ ] Select a small, educational checkpoint set: random weights, minimally trained, baseline transformer, and final model.
- [ ] Decide whether browser weights live in the repository, GitHub release assets, or Cloudflare static assets.
- [ ] Publish a checksum, parameter count, training step, seed, and source run identifier for every browser artifact.
- [ ] Confirm that the repository licence and dataset terms permit distributing the selected weights.

### Build browser inference

- [ ] Export each selected MLX checkpoint to a browser-readable model, preferably ONNX or ORT format.
- [ ] Implement the exact character vocabulary, encoding, context truncation, logits processing, temperature, and sampling rules used by the local model.
- [ ] Run inference off the main page thread so generation does not freeze navigation.
- [ ] Use WebAssembly as the compatibility baseline; add WebGPU as an optional acceleration path rather than the only path.
- [ ] Lazy-load model files only when the visitor opens or starts the playground.
- [ ] Keep prompts on the visitor's device and state this beside the playground.
- [ ] Provide honest loading, unsupported-browser, corrupt-model, out-of-memory, and generation-failure states.

[ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) documents browser inference with WebAssembly and WebGPU. Its [browser compatibility table](https://onnxruntime.ai/docs/get-started/with-javascript/web.html) should be checked again at release time because browser support changes.

### Prove that the port is the same model

Do not approve the browser playground merely because its output looks Shakespeare-like.

- [ ] Compare MLX and browser token IDs for fixed input strings, including newlines and unknown-input handling.
- [ ] Compare checkpoint parameters or conversion manifests layer by layer.
- [ ] Compare logits from fixed token sequences within a documented numerical tolerance.
- [ ] Require identical greedy continuations for fixed prompts wherever the numerical backend permits it.
- [ ] Test seeded sampling separately and document any random-number-generator difference that prevents exact text equality.
- [ ] Check random, minimally trained, baseline, and final checkpoints—not only the most attractive model.
- [ ] Test checkpoint switching without stale weights or cached labels.
- [ ] Record model download size, initialisation time, generation speed, browser, device, and execution backend.

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

## Coordinated publication sequence

These actions require explicit owner approval at the time of release.

1. Create the public GitHub repository and push only the reviewed release candidate.
2. Enable branch protection and the required lint, build, audit, and browser-parity checks.
3. Create the first tagged GitHub release, including scope, hardware evidence, limitations, artifact checksums, and browser support.
4. Connect the repository to Cloudflare using the chosen Git integration and make `main` the production branch.
5. Deploy the exact tagged commit to a non-public or preview URL first.
6. Smoke-test representative lessons, internal links, Reference/My Lab behaviour, every published checkpoint, WebAssembly fallback, and at least one supported WebGPU browser.
7. Obtain a final explicit approval that names the resolved public site access.
8. Make the Cloudflare wiki public and verify the production URL, security headers, metadata, accessibility, and rollback path.
9. Add reciprocal links: GitHub links to the hosted wiki; the wiki links to the tagged source and reproduction instructions.

Cloudflare's [Git integration guide](https://developers.cloudflare.com/pages/get-started/git-integration/) explains automatic production and preview deployments. Choose Git integration deliberately: Cloudflare currently warns that a Pages project created through Git integration cannot later be converted to Direct Upload, or vice versa, without creating a different project.

## Post-release monitoring

- [ ] Verify that a new visitor can read lessons without executing server code.
- [ ] Monitor build failures, static-asset sizes, browser-model load failures, and free-tier policy changes.
- [ ] Treat public prompts as private: do not add prompt collection later without a separate privacy decision and visible consent.
- [ ] Re-run parity tests whenever training code, model architecture, vocabulary, export code, inference code, or checkpoints change.
- [ ] Publish corrected model artifacts under a new release rather than silently replacing files whose checksums were documented.

## Definition of ready

The combined release is ready only when a stranger can either clone the repository and reproduce the course, or open the public wiki and run the genuine selected Shakespeare checkpoints in their browser, with the origin and limitations of every result made clear.
