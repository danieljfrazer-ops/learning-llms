# LearningLLMs

Build small language models locally, watch them change from random noise into structured text, and understand every important step along the way.

LearningLLMs is both:

- an executable machine-learning laboratory for modest personal hardware; and
- a living beginner-friendly wiki that connects commands, code, theory, checkpoints, metrics, generated samples, limitations, and further reading.

The course starts with a 4,225-weight character model rather than a downloaded black box. It then adds training, context, self-attention, a complete transformer, controlled evaluation, local prompting, optimisation experiments, scaling, and multi-seed confirmation. Later projects move to subword pretraining and parameter-efficient fine-tuning.

> Repository status: all three core projects and their reviewed reference dashboards are complete. The Apple-silicon clean-clone path is verified; public release is still blocked by the owner&apos;s licence/identity decisions, Linux clean-clone testing, and the browser-executable Shakespeare parity gate. No GitHub remote or public Cloudflare deployment has been created yet.

## What you will learn

- How text becomes tokens, tensors, logits, probabilities, and generated text
- The difference between architecture, random weights, training, and inference
- How loss, gradients, and an optimiser change model weights
- Why context windows, attention heads, feed-forward layers, residual paths, and normalisation exist
- How to save checkpoints and compare untrained, intermediate, and trained models
- How to design controlled experiments and avoid cherry-picking attractive samples
- How pretraining differs from adapting a pretrained model with LoRA
- How to state hardware, data, evaluation, and generalisation limitations honestly

No previous machine-learning knowledge is assumed. **Beginner Mode** adds deeper explanations, mapped metaphors, misconception warnings, and inline definitions throughout the wiki.

## Course map

| Project | Purpose | Starting point | Status in the reference course |
|---|---|---|---|
| 01 · Tiny Shakespeare | Expose the complete language-model loop at character scale | Random weights | 12 lessons complete |
| 02 · TinyStories | Train a small GPT-like model on simple English using subword tokens | Random weights → resumable pretraining | 12 lessons and reviewed reference evidence complete |
| 03 · English → SQL | Contrast pretraining with task adaptation and executable evaluation | Pretrained small model | 5 lessons and reviewed reference evidence complete |
| 04 · Sentiment laboratory | Independently build a classifier with robustness and calibration checks | Fully scaffolded extension | Optional |
| 05 · Dialogue summarisation | Ask your coding agent to scaffold a careful sequence-to-sequence project | Agent brief only | Optional |
| 06 · Your own question | Choose a domain, task, data, baseline, and evidence standard | Themes only | Open exploration |

## Two evidence modes

The top bar contains a **Reference results / My lab** switch. On the hosted wiki, Reference results are immediately readable; My lab explains what a learner will create after cloning rather than pretending a public page can train on their machine.

- **Reference results** displays the committed measurements from the original 32 GB Apple-silicon course run. It lets readers inspect completed lessons immediately.
- **My lab** begins with empty dashboards and reproduction goals. Training scripts write your JSON results to `public/data/local/` and checkpoints to `work/experiments/`. Both are ignored by Git, so your work cannot overwrite the published evidence accidentally.

Switch between the modes whenever you want to compare your machine and model with the reference run. See [How evidence modes work](docs/evidence-modes.md).

## Quick start · Apple silicon

Prerequisites: macOS 14+, an Apple-silicon Mac, [Node.js 22.13+](https://nodejs.org/), Git, and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```sh
# From the repository root
npm ci
uv sync --extra apple
uv run --no-sync python scripts/download_tiny_shakespeare.py
npm run dev
```

Open <http://localhost:3000>, switch to **My lab**, and begin with Shakespeare lesson 01.

Run the first trainable baseline with:

```sh
uv run --no-sync python ml/shakespeare_bigram.py
```

The open dashboard checks for local results every two seconds. The first captured sample comes from random weights, before any optimiser update.

## Other hardware

The wiki itself runs anywhere Node.js does. The model scripts use MLX 0.32.0.

| Machine | Current repository path | Practical expectation |
|---|---|---|
| Apple silicon, 8 GB | `uv sync --extra apple` | Shakespeare should fit; reduce batch size for later projects if required |
| Apple silicon, 16–32 GB | `uv sync --extra apple` | Recommended course path; 32 GB is the reference machine |
| Linux + supported NVIDIA GPU | `uv sync --extra linux-cuda12` or `linux-cuda13` | Same MLX model code; validate driver, CUDA, and GPU architecture first |
| Linux CPU-only | `uv sync --extra linux-cpu` | Early Shakespeare lessons are practical; full scaling runs will be slower |
| Linux + AMD GPU | PyTorch/ROCm adaptation required | Wiki works; current MLX scripts are not a ROCm implementation |
| Native Windows | Backend adaptation required | Use WSL2/Linux where compatible, or port the numerical model code to PyTorch/DirectML |
| Intel Mac | Backend adaptation required | MLX’s packaged macOS path requires Apple silicon |

For the complete local learning experience, treat **8 GB memory as the minimum starting point and 16 GB as the practical minimum for later projects; 32 GB is recommended**. The wiki alone needs far less. Keep several gigabytes of disk space free for package environments, datasets, and checkpoints; CUDA environments require substantially more. Exact feasibility depends on backend, batch size, context, and model size.

Run `python3 scripts/system_report.py` for a local recommendation. Read [Hardware and backend guide](docs/hardware.md) before attempting a non-Apple setup. Hardware support changes over time; that guide links to official MLX, PyTorch, ROCm, and DirectML documentation rather than freezing installation commands indefinitely.

## Reproduce the Shakespeare course

After setup and dataset download, follow the wiki lessons or run the scripts in this order:

```sh
uv run --no-sync python ml/shakespeare_bigram.py
uv run --no-sync python ml/shakespeare_context.py
uv run --no-sync python ml/shakespeare_attention.py
uv run --no-sync python ml/shakespeare_transformer.py
uv run --no-sync python ml/shakespeare_evaluate.py
uv run --no-sync python ml/shakespeare_training_improvements.py
uv run --no-sync python ml/shakespeare_scaling.py
uv run --no-sync python ml/shakespeare_final.py
```

Start local checkpoint-backed prompting separately:

```sh
uv run --no-sync python ml/shakespeare_inference_server.py
```

The later scaling and three-seed confirmation runs intentionally cost more time. Do not treat matching the reference numbers exactly as success: hardware, low-level numerical execution, and random paths can produce small differences. Reproduction means preserving the method, measuring your result, and explaining meaningful divergence.

For a blank-canvas walkthrough, expected artifacts, troubleshooting, and reduced-compute options, read [Reproduce the course](docs/reproduce-course.md).

## Begin TinyStories

Project 2 adds Hugging Face Tokenizers 0.23.1 but continues to use the same platform-specific MLX profile. Its first lesson downloads only a deterministic 1,200-story sample and invokes a provisional random model; it performs no pretraining:

```sh
uv run --no-sync python scripts/download_tinystories_sample.py
uv run --no-sync python ml/tinystories_transition.py
uv run --no-sync python ml/tinystories_dataset_audit.py
uv run --no-sync python ml/tinystories_tokenizer_experiment.py
uv run --no-sync python ml/tinystories_sequence_batching.py
uv run --no-sync python ml/tinystories_random_baseline.py
uv run --no-sync python ml/tinystories_first_pretraining.py
uv run --no-sync python ml/tinystories_checkpoint_resume.py
uv run --no-sync python ml/tinystories_training_recipe.py
uv run --no-sync python ml/tinystories_scaling_budget.py
uv run --no-sync python ml/tinystories_story_evaluation.py
uv run --no-sync python ml/tinystories_final.py
```

Start the TinyStories prompt service separately after Lesson 10:

```sh
uv run --no-sync python ml/tinystories_inference_server.py
```

Open all twelve TinyStories lessons in order. The dataset audit verifies hashes and freezes the bounded development sample; the tokenizer experiment compares 512–4,096-piece byte-level BPE vocabularies and freezes one local tokenizer; sequence batching creates story-isolated 32 × 128 tensors; and the random-baseline command saves and reloads the official 5.82M-parameter checkpoint zero. Those preparation commands perform no optimiser update. Lesson 6 performs the first 500 AdamW updates. Lesson 7 adds complete model, optimiser, trainer, and manifest state, then tests a fresh-process reload. Lesson 8 branches that exact state into three matched learning-rate policies. Lesson 9 screens width, depth, context, and data coverage before one clean longer run. Lesson 10 freezes the resulting weights and compares validation, scenario, repetition, diversity, and training-overlap evidence without updating the model. Lesson 11 exposes evaluated checkpoints through a loopback-only inference service and controlled wiki playground. Lesson 12 repeats the clean 700-update recipe across training seeds 42–44, selects one practical checkpoint, and compares Projects 1 and 2 without ranking incompatible token losses. Reference results contain only separately reviewed course runs; My Lab reads your ignored local evidence, tokenizer, and checkpoint files.

## Begin English → SQL

Project 3 begins by defining the task and auditing WikiSQL before choosing or adapting a pretrained model:

```sh
uv run --no-sync python scripts/download_wikisql.py
uv run --no-sync python ml/sql_task_dataset_audit.py
```

The downloader verifies the official WikiSQL 1.1 archive and writes raw files only under `data/raw/`. The audit performs no model training: it validates the structured examples, measures split overlap, and smoke-tests a restricted read-only evaluator. Dataset redistribution remains blocked pending review because the upstream repository does not explicitly resolve the dataset licence.

Continue through the development-only model and prompt selections, bounded LoRA adaptation, and one-time sampled test evaluation:

```sh
uv run python ml/sql_base_model_selection.py
uv run python ml/sql_prompt_formatting.py
uv run python ml/sql_lora_fine_tuning.py
uv run python ml/sql_execution_evaluation.py
```

The first three commands never load the test split. Generated text is not executed directly: it must parse into the restricted WikiSQL logical form before the evaluator rebuilds parameterized SQL for a read-only SQLite database. All measurements and adapter files remain learner-local unless a maintainer separately reviews and explicitly promotes reference evidence.

## Repository map

```text
app/                    wiki routes and interactive dashboards
lib/                    structured lessons, project records, and teaching guidance
ml/                     readable MLX data, model, training, evaluation, and inference code
scripts/                setup, dataset verification, system reporting, and maintainer tools
public/data/reference/  committed reference JSON displayed by Reference results
public/data/local/      your ignored live dashboard JSON, created by training
experiments/            committed reference configurations; large weights are not committed
work/experiments/       your ignored configurations and model checkpoints
data/raw/               downloaded datasets, ignored by Git
docs/                   hardware, reproduction, evidence, agent, and extension guides
```

## Working with a coding agent

Give your agent the repository and ask it to read `AGENTS.md` before acting. That file protects reference evidence, defines where learner runs belong, and lists the checks required after lesson or infrastructure changes.

A useful opening request is:

> Help me reproduce the Shakespeare course from lesson 01. Read AGENTS.md and the lesson before acting. Explain each command, keep all generated evidence in My lab, and stop after each lesson to compare my result with the reference evidence.

See [Agent-assisted learning](docs/agent-workflow.md) for prompts that preserve learning rather than turning the project into an unexplained one-command build.

## Public website and playgrounds

The planned Cloudflare site is a read-first portfolio and learning experience: all 29 completed lessons, Beginner Mode, diagrams, glossary entries, and reviewed Reference dashboards are visible without installing anything. Blank learner evidence is never presented as a completed result.

Local clones can already prompt Shakespeare and TinyStories checkpoints through loopback-only services. The first public release remains gated on a genuine in-browser Shakespeare port with MLX parity; TinyStories browser inference is a documented follow-up because its checkpoints and runtime are much larger. English → SQL remains local-only initially because its pretrained base is hundreds of megabytes, redistribution needs review, and safe browser execution would require a separately tested parser and SQLite sandbox. No hosted general-purpose model will be substituted for a course checkpoint.

See [What the public release will contain](docs/public-experience.md) for the exact visitor experience, omissions, privacy boundary, hardware adaptations, and zero-cost assumptions.

## Continue independently

After the three core projects, open the wiki’s **Continue yourself** page or read [Independent project extensions](docs/independent-projects.md). The three suggestions deliberately remove scaffolding:

1. a complete eight-lesson sentiment-classification plan;
2. a dialogue-summarisation brief for your agent to scaffold; and
3. open themes from which you define the question and evidence yourself.

## Reproducibility and repository policy

- Reference JSON is committed; learner JSON and checkpoints are ignored.
- Dataset files are downloaded from their original sources and are not committed.
- Generated samples and metrics must come from real runs, never placeholders.
- Every result must identify its configuration, dataset split, seed, hardware/backend, and limitations.
- Checkpoints may be large and are intentionally excluded from ordinary Git history. A future public release should use documented release assets or Git LFS only if downloadable weights are genuinely needed.
- Do not run `scripts/promote_reference_results.py --confirm` unless deliberately publishing a reviewed new reference run.

## Documentation

- [Hardware and backend guide](docs/hardware.md)
- [Reproduce from a blank clone](docs/reproduce-course.md)
- [Reference results versus My lab](docs/evidence-modes.md)
- [Agent-assisted learning workflow](docs/agent-workflow.md)
- [Tiny Shakespeare lesson quality red-team](docs/shakespeare-lesson-red-team.md)
- [TinyStories and English → SQL lesson quality red-team](docs/tinystories-sql-lesson-red-team.md)
- [Independent project extensions](docs/independent-projects.md)
- [Public visitor experience and capability matrix](docs/public-experience.md)
- [Current public-release readiness report](docs/release-readiness-report.md)
- [Publishing GitHub and the Cloudflare wiki](docs/publishing-github-and-cloudflare.md)
- [Future public-release checklist](docs/public-release-checklist.md)

## Before public release

This repository still needs an explicit licence choice, an independent maintained secret-scanner pass, Linux clean-clone testing, dataset-terms review, accessibility QA, and a browser-executable Shakespeare export proven equivalent to the MLX model. The Apple-silicon clean-clone path and the repository's full-history credential audit have passed. Run `python3 scripts/audit_public_repo.py` during preparation and `python3 scripts/audit_public_repo.py --release` for the owner-decision and browser-artifact gates. GitHub and the public Cloudflare wiki will be released from one reviewed candidate under the [coordinated publication plan](docs/publishing-github-and-cloudflare.md). No remote repository, public site, or deployment is created by the current setup work.
