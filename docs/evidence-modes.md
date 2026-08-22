# Reference results and My lab

The wiki must serve two audiences at once:

1. a reader who wants to inspect the completed course and its real measurements; and
2. a learner who wants an initially blank laboratory populated only by their own runs.

The evidence-mode toggle separates those experiences without duplicating the lesson site.

## Reference results

Reference mode reads committed JSON from `public/data/reference/`. These files preserve the original course run and make dashboards useful immediately after cloning. Hand-authored lesson result sections and completion outcomes are also visible in this mode.

Reference configurations live under `experiments/`. Safetensors checkpoints are excluded from normal Git history; the dashboard JSON contains representative samples and metrics without requiring large binary downloads.

## My lab

My Lab reads `public/data/local/`. A fresh clone has no files there, so dashboards show “Not run yet.” Training scripts create or atomically update the directory as work progresses.

Model checkpoints and local run configurations go to `work/experiments/`. Raw downloaded data goes to `data/raw/`. All three learner-owned locations are ignored by Git.

In My Lab mode:

- course lessons and procedures remain available;
- reference outcome labels become reproduction goals;
- roadmap labels say “guide ready” rather than implying that the learner completed them;
- hand-authored reference-result sections become blank evidence placeholders; and
- live dashboards populate from learner JSON when scripts run.

The mode choice is stored only in browser local storage. It changes presentation and fetch paths; it never moves, deletes, or rewrites experiment files.

## Data flow

```text
training script
  ├─> public/data/local/<experiment>.json ──> My lab dashboard
  └─> work/experiments/<run-id>/           ──> checkpoints + configuration

reviewed maintainer promotion
  └─> public/data/reference/<experiment>.json ──> Reference dashboard
```

The local inference server prefers checkpoints in `work/experiments/`. During this repository’s migration period it can fall back to the original local `experiments/` directories when ignored checkpoint files already exist there.

## Publishing new reference evidence

Promotion is a maintainer action, not a learner step. First review the local JSON, configurations, samples, data provenance, hardware metadata, and lesson conclusions. Then run:

```sh
python3 scripts/promote_reference_results.py --confirm
```

The explicit flag exists because promotion overwrites committed reference JSON. Inspect the Git diff and rerun all audits before committing. Never promote automatically after training.

## Future improvements

- Add a small manifest so local completion status can be calculated per lesson rather than displaying “guide ready.”
- Add schema validation for every result JSON before dashboards consume it.
- If public checkpoint downloads are valuable, publish selected weights as versioned release assets with checksums instead of placing large binaries in Git history.
