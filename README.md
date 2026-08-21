# LearningLLMs

An executable course in building small language models on a 32 GB Apple-silicon Mac. The local wiki explains each experiment and displays checkpoint samples and metrics as they are produced.

## Project shape

- `ml/` contains deliberately readable training code.
- `data/raw/` contains downloaded source data and is ignored by Git.
- `experiments/` contains run configurations and model checkpoints.
- `app/`, `lib/`, and `public/data/` power the living wiki and experiment dashboard.
- The discoverable Codex skill at `~/.codex/skills/learning-llm-wiki` defines how lessons and evidence are recorded.

## Local development

Use Node 22 or newer for the wiki and Python 3.12 with MLX for the experiments.

The Shakespeare progression currently contains four runnable models:

- `ml/shakespeare_bigram.py` — one-character lookup baseline
- `ml/shakespeare_context.py` — eight-character feed-forward model
- `ml/shakespeare_attention.py` — single-head causal attention over 64 characters
- `ml/shakespeare_transformer.py` — two-block, four-head decoder-only transformer
- `ml/shakespeare_training_improvements.py` — controlled learning-rate and gradient-clipping comparison

Each script writes live dashboard metrics under `public/data/` and reproducibility metadata under `experiments/`.

## Evaluation and prompting

Run the frozen checkpoint evaluation with:

```sh
.venv/bin/python ml/shakespeare_evaluate.py
```

The interactive prompt lesson needs the local inference service alongside the wiki:

```sh
.venv/bin/python ml/shakespeare_inference_server.py
```

It binds to `127.0.0.1:8001`, loads saved transformer checkpoints on demand, and never updates their weights.
