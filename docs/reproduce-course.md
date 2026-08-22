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
```

The downloader retains 1,000 rows from the official training split and 200 from the validation split, with source row numbers, selected page offsets and SHA-256 hashes in `data/raw/tinystories-sample/manifest.json`. If matching files already exist, it verifies and reuses them.

The transition experiment trains a provisional 2,048-piece byte-level BPE tokenizer, creates a 938,496-parameter random transformer, performs one fixed validation forward pass and saves a step-zero checkpoint. It performs zero optimiser updates. Expected local artifacts are:

- `public/data/local/tinystories-transition.json`;
- `work/experiments/tinystories-transition-001/config.json`;
- `work/experiments/tinystories-transition-001/provisional-tokenizer.json`; and
- `work/experiments/tinystories-transition-001/checkpoint-0000.safetensors`.

The fetched dataset can change upstream. Compare your manifest and resulting measurements with Reference results rather than assuming future hashes must match forever. A hash mismatch is a reason to inspect provenance, not permission to copy the reference metric into My Lab.

## 9. Keep your work private or publish it deliberately

My Lab artifacts are ignored. To preserve your journey, create a separate branch or repository policy for selected JSON/configuration files; do not commit raw datasets or every checkpoint by default. Review dataset terms and remove private prompts before sharing.

Reference promotion is reserved for maintainers and requires an explicit confirmation flag. Learners should normally leave `public/data/reference/` unchanged.
