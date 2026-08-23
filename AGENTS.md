# LearningLLMs agent instructions

This repository is an executable course. Preserve both reproducibility and the learner’s understanding.

## Before acting

1. Read `README.md`, the current lesson, `docs/evidence-modes.md`, and `docs/hardware.md` when setup or execution is involved.
2. Inspect the machine non-destructively with `python3 scripts/system_report.py` before choosing a model backend.
3. Explain assumptions when hardware or installed software differs from the documented reference.

## Evidence boundaries

- Treat `public/data/reference/` and committed files under `experiments/` as published course evidence. Do not overwrite them during ordinary learning work.
- Write learner dashboard JSON only to `public/data/local/` and learner checkpoints/configuration only to `work/experiments/`.
- Keep downloaded datasets under `data/raw/`.
- Never invent, copy, or backfill a learner metric. Missing evidence must remain “Not run yet.”
- Run `scripts/promote_reference_results.py --confirm` only when the user explicitly requests a reviewed reference update. Promotion is separate from training and from public release.

## Teaching behaviour

- Work one lesson at a time unless the user asks otherwise.
- Before a command, explain what it reads, the principal computation, which state may change, and what files it writes.
- After a run, identify observed evidence, compare it cautiously with the reference, explain divergence, and recap the new concept.
- Show random or minimally trained behaviour before final behaviour whenever the model supports checkpoints.
- Keep exact technical terms, define them at first meaningful use, and update the glossary and Beginner Mode guidance in context.

## Content maintenance

Use the discoverable `learning-llm-wiki` skill for lesson, experiment, checkpoint, evaluation, tool, or result changes. Its beginner-writing standard and content contract are mandatory for new lessons.

When adding a project or lesson:

- preserve dataset provenance, licence, splits, configuration, hardware/backend, commands, outputs, limitations, and next experiment;
- add reference/local evidence handling from the start;
- provide a blank state rather than fabricated placeholder results;
- keep authoritative sources beside the claim or tool they support; and
- distinguish observed results from expectations.

## Required checks

After relevant changes, run:

```sh
python3 ~/.codex/skills/learning-llm-wiki/scripts/audit_beginner_guidance.py
python3 ~/.codex/skills/learning-llm-wiki/scripts/audit_lesson_visuals.py
python3 scripts/audit_comparison_evidence.py
python3 -m compileall -q ml scripts
npm run lint
npm run build
git diff --check
```

Also run the smallest safe experiment or fixture that validates changed Python behaviour. Do not launch long scaling or multi-seed training merely as a generic check.

## Publication boundary

Do not create a GitHub repository, add a remote, publish a package, deploy the wiki, promote reference results, or choose a licence unless the user explicitly asks for that separate action. Use `docs/public-release-checklist.md` and `docs/publishing-github-and-cloudflare.md` when publication is requested.

Treat GitHub and Cloudflare as one coordinated release candidate. Before either public launch, require the browser-executable Shakespeare model, MLX/browser parity evidence, static-first hosting, private or preview smoke testing, documented rollback, and a final approval that explicitly names the public site access. Never substitute a hosted general-purpose model for the course's trained checkpoint without clearly defining it as a different experiment.
