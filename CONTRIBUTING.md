# Contributing

LearningLLMs values explanations and evidence as much as code.

Before proposing a change, read `AGENTS.md` and the relevant documentation under `docs/`.

## Contributions should

- keep reference and learner-local evidence separate;
- include reproducible commands and configuration;
- identify dataset provenance and terms;
- preserve the untrained baseline when applicable;
- add or update lessons, inline sources, glossary entries, and Beginner Mode guidance when introducing concepts;
- state limitations and avoid claims unsupported by measured output; and
- avoid committing datasets, local prompts, caches, environments, or large checkpoints.

## Validation

```sh
python3 ~/.codex/skills/learning-llm-wiki/scripts/audit_beginner_guidance.py
python3 -m compileall -q ml scripts
npm run lint
npm run build
git diff --check
```

Non-Apple backend contributions should record the operating system, CPU/GPU, memory, driver/runtime, framework package, and numerical differences. A backend port should compare data batches, parameter counts, logits or loss on a fixed small fixture, checkpoint reload, and result schema—not only final generated prose.

Reference-result promotion and public releases are maintainer actions and are not part of an ordinary contribution.
