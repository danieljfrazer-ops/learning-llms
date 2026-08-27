# Reproduce the course from a blank clone

This guide describes the learner path. It intentionally keeps generated data and checkpoints separate from the committed reference evidence.

## 1. Inspect the machine

```sh
python3 scripts/system_report.py
```

Read [the hardware guide](hardware.md), choose a supported dependency profile, and record the report with your experiment notes.

Minimum tooling:

- Git
- Node.js 22.13 or newer
- npm
- uv
- enough free disk space for dependencies, data, and local checkpoints

The wiki can run without the Python model environment, but training cannot.

## 2. Install the wiki

```sh
npm ci
npm run dev
```

Open <http://localhost:3000>. Select **My lab** so dashboards and outcomes represent your work rather than the reference run. Beginner Mode is independent and may be enabled at the same time.

## 3. Install the model backend

Choose one documented profile:

```sh
uv sync --extra apple
uv sync --extra linux-cpu
uv sync --extra linux-cuda12
uv sync --extra linux-cuda13
```

Do not combine CUDA 12 and CUDA 13 profiles. On non-Apple systems, verify requirements against the current MLX documentation before choosing.

## 4. Download and verify data

```sh
uv run --no-sync python scripts/download_tiny_shakespeare.py
```

The downloader verifies SHA-256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` before replacing the destination. The dataset remains ignored under `data/raw/`.

## 5. Execute lessons in order

Read each wiki lesson before running its command. The intended rhythm is:

1. predict what should happen;
2. inspect the command and source path;
3. run the stage;
4. watch the My Lab dashboard;
5. record observed metrics and samples;
6. compare with Reference results;
7. explain differences before continuing.

| Stage | Command | Primary local artifact |
|---|---|---|
| Bigram | `uv run --no-sync python ml/shakespeare_bigram.py` | `public/data/local/shakespeare-metrics.json` |
| Fixed context | `uv run --no-sync python ml/shakespeare_context.py` | `shakespeare-context-metrics.json` |
| Self-attention | `uv run --no-sync python ml/shakespeare_attention.py` | `shakespeare-attention-metrics.json` |
| Transformer | `uv run --no-sync python ml/shakespeare_transformer.py` | `shakespeare-transformer-metrics.json` |
| Frozen evaluation | `uv run --no-sync python ml/shakespeare_evaluate.py` | `shakespeare-evaluation.json` |
| Training recipes | `uv run --no-sync python ml/shakespeare_training_improvements.py` | `shakespeare-training-improvements.json` |
| Scaling | `uv run --no-sync python ml/shakespeare_scaling.py` | `shakespeare-scaling.json` |
| Final seeds | `uv run --no-sync python ml/shakespeare_final.py` | `shakespeare-final.json` |

Each training stage also writes configurations and Safetensors checkpoints below `work/experiments/`.

## 6. Prompt checkpoints

Keep the wiki running, then start the local inference service in another terminal:

```sh
uv run --no-sync python ml/shakespeare_inference_server.py
```

Open the Prompt playground. The service binds only to `127.0.0.1:8001`, reconstructs the matching architecture, and loads local checkpoint tensors on demand. Prompting performs inference; it does not update weights.

## 7. Decide whether reproduction succeeded

Do not require identical floating-point values. Check instead that:

- random checkpoints produce unstructured samples;
- training loss and validation loss fall substantially;
- longer-context models improve beyond the bigram under their recorded recipes;
- the complete transformer beats the incomplete attention-only design in this course;
- local result files match the expected schema;
- checkpoints reload and generate text;
- controlled comparisons retain their stated constants; and
- your report distinguishes model quality, runtime, and memory.

Investigate large differences in dataset hash, vocabulary size, parameter count, configuration, device, dependency version, and random seed before blaming hardware.

## 8. Start Project 2 without pretraining

TinyStories Lesson 1 adds one cross-platform dependency, Hugging Face Tokenizers, through the existing lock file. Re-run the selected `uv sync --extra ...` command after pulling that change.

Download the deterministic bounded sample and invoke the provisional random model:

```sh
uv run --no-sync python scripts/download_tinystories_sample.py
uv run --no-sync python ml/tinystories_transition.py
uv run --no-sync python ml/tinystories_dataset_audit.py
uv run --no-sync python ml/tinystories_tokenizer_experiment.py
uv run --no-sync python ml/tinystories_sequence_batching.py
uv run --no-sync python ml/tinystories_random_baseline.py
```

The downloader retains 1,000 rows from the official training split and 200 from the validation split, with source row numbers, selected page offsets and SHA-256 hashes in `data/raw/tinystories-sample/manifest.json`. If matching files already exist, it verifies and reuses them.

The transition experiment trains a provisional 2,048-piece byte-level BPE tokenizer, creates a 938,496-parameter random transformer, performs one fixed validation forward pass and saves a step-zero checkpoint. It performs zero optimiser updates. Expected local artifacts are:

- `public/data/local/tinystories-transition.json`;
- `work/experiments/tinystories-transition-001/config.json`;
- `work/experiments/tinystories-transition-001/provisional-tokenizer.json`; and
- `work/experiments/tinystories-transition-001/checkpoint-0000.safetensors`.

The fetched dataset can change upstream. Compare your manifest and resulting measurements with Reference results rather than assuming future hashes must match forever. A hash mismatch is a reason to inspect provenance, not permission to copy the reference metric into My Lab.

The audit command performs no neural-network training. It verifies the manifest hashes and JSON rows, measures exact and normalised duplicates, compares every validation story with every training story using word five-gram Jaccard similarity, counts repeated openings and declared content terms, and checks the provisional tokenizer for unknown IDs and NFKC round-trip mismatches. It creates:

- `public/data/local/tinystories-dataset-audit.json`; and
- `work/experiments/tinystories-dataset-audit-001/config.json`.

Passing these checks freezes the named `tinystories-development-v1` split for tokenizer, batching, baseline and short-recipe comparisons. It does not certify the complete source dataset, approve this small slice for final scaling, or turn a keyword screen into a safety assessment.

The tokenizer experiment requires that learner-local audit. It trains 512, 1,024, 2,048 and 4,096-piece byte-level BPE candidates on the training split only; validation stories measure compression and reversibility afterward. It saves:

- `public/data/local/tinystories-tokenizer.json`;
- `work/experiments/tinystories-tokenizer-001/candidate-*.json`;
- `work/experiments/tinystories-tokenizer-001/tokenizer.json`; and
- `work/experiments/tinystories-tokenizer-001/config.json`.

The hard gates require stable special-token IDs, zero unknown IDs and zero NFKC round-trip mismatches. The documented selection policy weighs held-out sequence length against the embedding/output parameters induced by vocabulary size. It is a development decision, not a downstream model-quality measurement, and must be revisited if the training corpus or normalisation changes.

The sequence-batching experiment requires the learner-local tokenizer result and verifies both the selected tokenizer checksum and dataset hashes before proceeding. It wraps each story independently with beginning/end markers, slices context-128 windows with a one-token overlap, right-pads final windows, and masks padded targets. It saves:

- `public/data/local/tinystories-batching.json`;
- `work/experiments/tinystories-batching-001/config.json`; and
- `work/experiments/tinystories-batching-001/batch-preview.json`.

The audit requires every within-story adjacent token pair to appear exactly once, zero cross-story targets, exact story reconstruction, aligned `32 × 128` input/target/loss-mask tensors, and masked padding. It also reports a deliberately naïve concatenation comparison to show why high tensor utilisation is not sufficient evidence of correct batching. This command materialises arrays but performs no model forward pass, loss calculation, gradient, optimiser update, or checkpoint write.

The random-baseline experiment requires that learner-local batching result. It verifies the frozen tokenizer and validation-data checksums, then materialises the selected width-256, eight-head, six-block decoder with 5,816,320 seeded random parameters. It evaluates every validation window once with padding excluded, generates from three fixed prompts, and saves:

- `public/data/local/tinystories-random-baseline.json`;
- `work/experiments/tinystories-random-baseline-001/config.json`; and
- `work/experiments/tinystories-random-baseline-001/checkpoint-0000.safetensors`.

The script loads that checkpoint strictly into a fresh matching architecture and requires a fixed logit slice to remain identical. Loss, probability, prompt, forward-throughput, and MLX allocator measurements are all before-state evidence: the experiment constructs no optimiser and performs zero weight updates. Its peak allocation excludes total process and system memory, while forward throughput excludes backward, gradient, and optimiser work. Lower-memory adaptations may choose the documented width-192 candidate, but must record that architecture change rather than compare it as if it were the official baseline.

The first-pretraining experiment requires the learner-local batching and random-baseline results. It verifies the frozen data, tokenizer, and checkpoint-zero hashes; loads the exact measured starting weights; and then adds masked automatic differentiation plus AdamW with learning rate `0.0003` and weight decay `0.01`. Run:

```sh
uv run --no-sync python ml/tinystories_first_pretraining.py
```

The seeded data order shuffles all 2,292 training windows, includes the final short batch, and reshuffles after a complete pass. Full held-out evaluation, fixed-prompt generation, and weight-only checkpoints are captured at steps 0, 1, 10, 50, 100, 250, and 500. It creates:

- `public/data/local/tinystories-first-pretraining.json`;
- `work/experiments/tinystories-first-pretraining-001/config.json`; and
- `work/experiments/tinystories-first-pretraining-001/checkpoint-*.safetensors`.

Expect several hundred megabytes of checkpoint files and a larger MLX allocation than the forward-only baseline because gradients and AdamW state must coexist with weights and activations. Do not compare wall time or peak memory as though they were hardware-independent. These Lesson 6 files preserve weights for evaluation and inference, but not AdamW moments or the shuffle cursor; exact interrupted-run continuation is intentionally deferred to the resumable-checkpoint lesson.

Lesson 7 starts a declared continuation segment from the Lesson 6 step-500 weights. It cannot reconstruct the optimiser history that Lesson 6 never saved, so it initializes fresh AdamW and data-order state once, records that boundary, and tests a later save at step 550:

```sh
uv run --no-sync python ml/tinystories_checkpoint_resume.py
```

The control route saves at step 550 and continues in memory to step 600. A second Python process loads that exact step-550 directory and independently continues to 600. Each loadable directory contains `model.safetensors`, `optimizer.safetensors`, `trainer-state.json`, and a checksum-bearing `manifest.json` written last. Live evidence is written to `public/data/local/tinystories-checkpoints-dashboard.json`; protocol, traces, and complete saves live under `work/experiments/tinystories-checkpoint-resume-001/`.

Model and optimiser tensors must reload with zero difference before continuation. Batch hashes, target counts, cursor, and next-batch identity must match exactly. Separate accelerator processes may not be bitwise deterministic, so the protocol records numerical tolerances before the comparison and reports every observed floating-point difference. Do not call a run exact when it merely falls inside tolerance, and do not treat resume validation as evidence that model quality improved.

Lesson 8 uses the complete resumed step-600 state rather than resetting weights, AdamW history, or data order. It freezes three 100-update candidates before execution and changes only learning-rate policy:

```sh
uv run --no-sync python ml/tinystories_training_recipe.py
```

The control keeps `3e-4`, the lower constant candidate uses `1.5e-4`, and cosine decay moves from `3e-4` to `3e-5`. Every fresh worker receives the same next 100 batch identities, complete validation set, prompts and generation seeds. The predeclared rule selects an alternative only when its complete validation loss improves on the control by at least `0.001`; otherwise the control remains selected.

Live and final evidence is written to `public/data/local/tinystories-training-recipe.json`. The frozen protocol, traces, results, selection record and complete step-700 candidate checkpoints live under `work/experiments/tinystories-training-recipe-001/`. Expect roughly 200 MiB for three model-plus-AdamW states. The chosen checkpoint is a local continuation artifact; training does not promote it or any measurement to Reference results.

Lesson 9 cannot load that selected checkpoint into wider, deeper, or longer-context candidates because their tensor shapes differ. It therefore runs a fresh-start screen with a common model seed, batch seed, tokenizer, validation set, optimiser and 100-update budget:

```sh
uv run --no-sync python ml/tinystories_scaling_budget.py
```

The five probes are the 256-wide/six-block/context-128 control, width 320, eight blocks, context 256, and the control trained on a deterministic half of the training stories. The data probe diagnoses coverage and is not eligible for architecture selection. Feasible architectures must stay below 4 GiB peak MLX allocation and 250 ms mean update time on the documented reference machine; an alternative must improve complete validation loss by at least `0.01` or the control remains selected.

After applying that frozen rule, the script rebuilds only the selected architecture and trains it for 700 updates: constant `3e-4` through update 600 and cosine decay to `3e-5` over the final 100. Live evidence is `public/data/local/tinystories-scaling-budget.json`; protocol, probe weights, traces, selection, and the complete selected checkpoint are under `work/experiments/tinystories-scaling-budget-001/`. The recorded run used about 195 MiB. Treat its hardware timings as local measurements and its 1,000-story quality screen as development evidence, not full-dataset scaling proof.

Lesson 10 freezes all weights and evaluates three checkpoints—the random before-state, Lesson 8's inherited-history step 700, and Lesson 9's clean-history step 700—under one protocol:

```sh
uv run --no-sync python ml/tinystories_story_evaluation.py
```

The command grades every held-out target, then generates five declared scenarios with three fixed seeds, temperature `0.8`, and a 64-token cap. It separately records scenario-keyword and contradiction-list proxies, repeated four-grams, distinct bigrams, cross-seed three-gram similarity, sentence closure, exact 5–12-word training spans, and maximum five-gram Jaccard overlap. These are deterministic surface measurements rather than semantic judges; inspect their fixed outputs and limitations beside the aggregate values.

The evaluator writes no checkpoint and performs no optimiser update. Its protocol and full result live under `work/experiments/tinystories-story-evaluation-001/`; the polling dashboard reads `public/data/local/tinystories-story-evaluation.json`. Rerunning requires an empty run directory so an older frozen protocol cannot be silently mixed with new evidence.

Lesson 11 reuses exactly that evaluated cohort. Keep the wiki running and start a second process:

```sh
uv run --no-sync python ml/tinystories_inference_server.py
```

The service reads and verifies the Lesson 10 local evidence, frozen tokenizer, and all three checkpoint hashes, then binds only to `127.0.0.1:8002`. It writes no files and never constructs an optimiser. Open the TinyStories prompt-playground lesson, enter an opening, and keep checkpoint, temperature, seed, and maximum new-token count visible. “Compare all available checkpoints” holds those sampling inputs fixed and invokes the random, inherited-history, and clean-history weights sequentially; after Lesson 12 it also includes the checksum-verified final selected seed. The page reports prompt tokens initially visible inside context 128, generated-token count, EOS versus token-limit stopping, elapsed time, and the explicit fact that weights were not updated.

If the service reports missing or mismatched files, rerun the prerequisite lesson that owns the artifact rather than copying Reference evidence. This is a local development server built with Python's `http.server`; do not expose port 8002 publicly. The eventual public wiki requires the separately planned browser-executable export and parity checks, not this Python endpoint.

Lesson 12 freezes the clean Lesson 9 recipe and repeats complete training across seeds 42, 43, and 44:

```sh
uv run --no-sync python ml/tinystories_final.py
```

The command verifies and reuses seed 42's exact Lesson 9/Lesson 10 checkpoint, then launches two separate MLX workers. Seeds 43 and 44 each create fresh random model weights, fresh AdamW state, and independent batch orders before performing all 700 updates. Every checkpoint receives complete held-out evaluation and the unchanged Lesson 10 five-prompt/three-sampling-seed behaviour protocol. The command writes live and final evidence to `public/data/local/tinystories-final.json`; protocol, traces, configurations, and the two new weight-only checkpoints live under `work/experiments/tinystories-final-001/`.

The selection rule keeps the lowest complete validation loss as a practical inference checkpoint, with the lower seed breaking an exact tie. Report the across-seed mean, population standard deviation, and range as the recipe result rather than quoting only that selected minimum. The cross-project table uses committed Shakespeare reference evidence and labels that provenance explicitly; absolute character-token and subword-token losses are not ranked. On lower-memory or slower hardware, run one fresh seed first as a pipeline check, but record that reduced run as incomplete confirmation rather than copying or inventing the missing seeds.

After Lesson 12, restart `ml/tinystories_inference_server.py`. Its checksum-verified checkpoint list adds the selected final seed when `tinystories-final.json` is complete.

## 9. Begin English → SQL with a task and dataset audit

Project 3 does not begin by downloading a pretrained model. First freeze what the model may see, the restricted program it must produce, and how correctness will be measured:

```sh
uv run --no-sync python scripts/download_wikisql.py
uv run --no-sync python ml/sql_task_dataset_audit.py
```

The downloader retrieves the official WikiSQL 1.1 archive, verifies SHA-256 `755c…0881`, and safely writes only the nine expected split, table, and SQLite files under `data/raw/wikisql-1.1/`. These raw files and their manifest are ignored. The repository code is BSD-3-Clause, but the upstream project does not explicitly resolve the dataset licence; treat redistribution as blocked pending review.

The audit checks all structured examples against their table schemas, measures exact normalized question and table-ID overlap between train, development, and test, and executes a seeded 100-query sample per split. Execution uses read-only SQLite connections, query-only mode, whitelisted reference operators, quoted identifiers, and parameterized values. It never executes free-form model output and performs no model training. It writes:

- `public/data/local/sql-task-dataset-audit.json`; and
- `work/experiments/sql-task-dataset-audit-001/config.json`.

Keep the test split closed during base-model selection. A successful audit proves the declared checks and evaluator work; it does not measure model accuracy, establish SQL safety beyond the benchmark, or grant dataset redistribution rights.

## 10. Keep your work private or publish it deliberately

My Lab artifacts are ignored. To preserve your journey, create a separate branch or repository policy for selected JSON/configuration files; do not commit raw datasets or every checkpoint by default. Review dataset terms and remove private prompts before sharing.

Reference promotion is reserved for maintainers and requires an explicit confirmation flag. Learners should normally leave `public/data/reference/` unchanged.
