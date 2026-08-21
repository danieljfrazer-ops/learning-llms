# LearningLLMs

An executable course in building small language models on a 32 GB Apple-silicon Mac. The local wiki explains each experiment and displays checkpoint samples and metrics as they are produced.

## Project shape

- `ml/` contains deliberately readable training code.
- `data/raw/` contains downloaded source data and is ignored by Git.
- `experiments/` contains run configurations and model checkpoints.
- `app/`, `lib/`, and `public/data/` power the living wiki and experiment dashboard.
- The discoverable Codex skill at `~/.codex/skills/learning-llm-wiki` defines how lessons and evidence are recorded.

## Local development

Use Node 22 or newer for the wiki and Python 3.12 with MLX for the experiments. The first experiment is `ml/shakespeare_bigram.py`.
