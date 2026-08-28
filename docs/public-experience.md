# What the public release will contain

LearningLLMs has two deliberately different public surfaces.

## Public GitHub repository

A visitor can clone the complete course code, install a hardware-appropriate dependency profile, download source datasets from their original hosts, and execute lessons in order. The clone includes:

- all 29 core lesson explanations, Beginner Mode guidance, diagrams, glossary entries, commands, limitations, and further reading;
- reviewed Reference result JSON and reproducibility metadata for Shakespeare, TinyStories, and English → SQL;
- readable MLX experiment, evaluation, and loopback-only inference code;
- four checksum-pinned Shakespeare ONNX checkpoints plus the WebAssembly browser worker and MLX parity fixture;
- empty, ignored locations for the learner's own data, checkpoints, prompts, and dashboard evidence; and
- automated content, evidence, Python, lint, and production-build checks.

The clone does **not** include downloaded datasets, the pretrained SQL base model, SQL adapters, TinyStories checkpoints, caches, credentials, or learner-local evidence. Published SQL dashboards contain aggregate measurements and failure counts, not redistributed WikiSQL question/query rows while the dataset licence remains unresolved. Those omissions protect licensing boundaries, repository size, privacy, and the distinction between published and learner evidence.

## Public Cloudflare wiki

A visitor can read every completed core lesson and every reviewed Reference dashboard without installing Python or downloading a dataset. Optional independent-project material is labelled as an extension, not presented as completed evidence.

The **Reference results / My lab** control still has a useful meaning online:

- Reference results show the maintainer's reviewed run.
- My lab explains the reproduction goal and remains empty until the visitor uses a local clone. A browser visit does not train a model or save evidence into this repository.

## Prompt-playground capability

| Project | Local clone | Public website at the next release | Why |
|---|---|---|---|
| Shakespeare | Prompt random, minimally trained, baseline, improved, and final local checkpoints through loopback port `8001` | Prompt genuine random, one-update, baseline and selected-final course checkpoints on-device through WebAssembly | Four ONNX files total about 2.9 MiB; the 13 MiB runtime and model execute in a Web Worker, and prompts are never sent to a public inference API. |
| TinyStories | Compare the evaluated random and trained local checkpoints through loopback port `8002` | Read reference outputs; browser generation remains a documented follow-up | Each checkpoint is about 22.2 MiB, close to Cloudflare's 25 MiB per-file limit, and the 5.82M-parameter model needs a tested browser runtime and fallback rather than a slow or fragile port. |
| English → SQL | Run the pinned quantized base model and LoRA adapter locally, then evaluate restricted logical forms in read-only SQLite | Read all reference comparisons; no public generation or SQL execution in the initial release | The base model download is hundreds of megabytes, redistribution rights must be checked, and safely reproducing the Python parser plus SQLite evaluation in-browser is a separate engineering and security project. |

The public site must never silently replace a course-trained checkpoint with a hosted general-purpose model. If a browser cannot run a published model, it will show the measured reference samples and a clear limitation instead of a broken or misleading control.

## Hardware expectations

The wiki itself needs only a current Node.js runtime. Training uses MLX and is best supported on Apple silicon. The reference machine has 32 GiB unified memory. Shakespeare is the broadest practical entry point; later TinyStories and SQL runs require more memory, disk, downloads, and time. Linux MLX profiles are provided, but a release must not call them verified until clean-clone testing is recorded on matching hardware. AMD GPU, Intel Mac, and native Windows users need a backend port; WSL2 may support an appropriate Linux route, but it is not equivalent to the tested Apple reference.

Reduced batch size, fewer updates, fewer seeds, or a smaller architecture can make a lesson practical on weaker hardware. Such a run remains valuable evidence, but it must be labelled as an adaptation and not compared as if its protocol were identical.

## Privacy, security, and cost boundary

The intended site is static-first, has no account system, database, public inference API, prompt collection, analytics identifier, advertising, or payment flow. Security headers deny framing and unnecessary browser capabilities while allowing the narrowly scoped WebAssembly runtime. Browser generation runs off the main thread and retains prompts in memory on the visitor&apos;s device. Local inference binds to the learner's loopback interface only. Generated SQL never receives production credentials and is restricted before read-only benchmark execution.

The default publication path can operate at £0 using a public GitHub repository, standard GitHub-hosted Actions, Cloudflare's free plan, and a `pages.dev` or `workers.dev` hostname. A custom domain is optional and normally costs money to register. Cloudflare and GitHub limits and pricing can change; the owner must check the live plan at release time. Staying on a free plan avoids usage billing, but exceeding a free quota can make builds or requests fail rather than guaranteeing unlimited service.
