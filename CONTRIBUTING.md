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

Browser-model contributions must preserve checkpoint provenance and compare tokenisation, fixed-input logits, greedy generation, and checkpoint selection against the MLX implementation. WebGPU-only support is insufficient; retain the documented WebAssembly compatibility path and never send public playground prompts to an inference service without an explicit privacy and architecture change.

## Validation

```sh
python3 scripts/audit_beginner_guidance.py
python3 scripts/audit_lesson_visuals.py
python3 scripts/audit_comparison_evidence.py
python3 scripts/audit_lesson_engineering_briefs.py
python3 scripts/audit_public_repo.py
python3 scripts/audit_git_history.py
python3 -m compileall -q ml scripts
npm run lint
npm run build
git diff --check
```

Non-Apple backend contributions should record the operating system, CPU/GPU, memory, driver/runtime, framework package, and numerical differences. A backend port should compare data batches, parameter counts, logits or loss on a fixed small fixture, checkpoint reload, and result schema—not only final generated prose.

Reference-result promotion and public releases are maintainer actions and are not part of an ordinary contribution. GitHub and Cloudflare publication follow the coordinated gate in `docs/publishing-github-and-cloudflare.md`.
