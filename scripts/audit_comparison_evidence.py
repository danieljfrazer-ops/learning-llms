#!/usr/bin/env python3
"""Fail when a comparison dashboard hides or hard-codes its intended contrast.

This is a presentation audit, not a statistical-quality test. It checks that
available experiment evidence contains discriminating fixed examples and that
the UI derives selections from evidence rather than card position.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "public/data/local"
REFERENCE = ROOT / "public/data/reference"


class AuditFailure(RuntimeError):
    pass


checks = 0
skips = 0


def load_if_present(path: Path) -> dict | None:
    global skips
    if not path.is_file():
        skips += 1
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    global checks
    checks += 1
    if not condition:
        raise AuditFailure(message)


def require_distinct(values: list[str], label: str) -> None:
    require(len(values) >= 2, f"{label}: comparison needs at least two values")
    require(len(set(values)) >= 2, f"{label}: every displayed example is identical")


def audit_tokenizers() -> None:
    evidence = load_if_present(LOCAL / "tinystories-tokenizer.json")
    if not evidence:
        return
    candidates = evidence["candidates"]
    prompt_count = min(len(candidate["promptPieces"]) for candidate in candidates)
    signatures = [
        {tuple(candidate["promptPieces"][index]["tokens"]) for candidate in candidates}
        for index in range(prompt_count)
    ]
    require(any(len(items) > 1 for items in signatures), "Tokenizer prompts never expose a segmentation difference")
    require(max(map(len, signatures)) == 3, "Expected the strongest frozen tokenizer prompt to expose three segmentations")


def audit_shakespeare() -> None:
    architecture_files = [
        "shakespeare-metrics.json",
        "shakespeare-context-metrics.json",
        "shakespeare-attention-metrics.json",
        "shakespeare-transformer-metrics.json",
    ]
    runs = [load_if_present(REFERENCE / filename) for filename in architecture_files]
    present = [run for run in runs if run]
    if present:
        require_distinct([run["checkpoints"][-1]["sample"] for run in present], "Shakespeare architecture samples")
        require_distinct([str(run["checkpoints"][-1]["validationLoss"]) for run in present], "Shakespeare architecture losses")

    recipes = load_if_present(REFERENCE / "shakespeare-training-improvements.json")
    if recipes:
        require_distinct([item["evaluation"]["continuation"] for item in recipes["variants"]], "Shakespeare recipe samples")
        require_distinct([str(item["evaluation"]["validationLossMean"]) for item in recipes["variants"]], "Shakespeare recipe losses")

    scaling = load_if_present(REFERENCE / "shakespeare-scaling.json")
    if scaling:
        require(scaling["selectedRunId"] in {item["runId"] for item in scaling["variants"]}, "Shakespeare scaling selection is absent from variants")
        require_distinct([item["evaluation"]["continuation"] for item in scaling["variants"]], "Shakespeare scaling samples")

    final = load_if_present(REFERENCE / "shakespeare-final.json")
    if final:
        require(final["selectedRunId"] in {item["runId"] for item in final["seeds"]}, "Shakespeare final selection is absent from seeds")
        require_distinct([item["continuation"] for item in final["seeds"]], "Shakespeare seed samples")


def audit_tinystories_training() -> None:
    pretraining = load_if_present(LOCAL / "tinystories-first-pretraining.json")
    if pretraining:
        checkpoints = pretraining["checkpoints"]
        require_distinct([str(item["validation"]["crossEntropyLoss"]) for item in checkpoints], "TinyStories checkpoint losses")
        for prompt_index in range(min(len(item["samples"]) for item in checkpoints)):
            require_distinct([item["samples"][prompt_index]["continuation"] for item in checkpoints], f"TinyStories checkpoint prompt {prompt_index + 1}")

    recipes = load_if_present(LOCAL / "tinystories-training-recipe.json")
    if recipes:
        require(recipes["selection"]["selectedCandidateId"] in {item["candidate"]["id"] for item in recipes["variants"]}, "TinyStories recipe selection is absent from variants")
        require_distinct([item["samples"][0]["continuation"] for item in recipes["variants"]], "TinyStories recipe samples")

    evaluation = load_if_present(LOCAL / "tinystories-story-evaluation.json")
    if evaluation:
        for scenario in evaluation["protocol"]["generation"]["scenarios"]:
            samples = [
                next(item for item in result["fixedExamples"] if item["scenarioId"] == scenario["id"])["continuation"]
                for result in evaluation["results"]
            ]
            require_distinct(samples, f"TinyStories story scenario {scenario['id']}")

    final = load_if_present(LOCAL / "tinystories-final.json")
    if final:
        require(final["selectedSeed"] in {item["seed"] for item in final["seeds"]}, "TinyStories final selection is absent from seeds")
        require_distinct([item["samples"][0]["continuation"] for item in final["seeds"]], "TinyStories final seed samples")


def audit_dashboard_source() -> None:
    forbidden = {
        "app/components/TinyStoriesTokenizerPanel.tsx": "promptPieces[0]",
        "app/components/TrainingImprovementsPanel.tsx": "index === 1 ? 'recipe-winner'",
        "app/components/FinalModelPanel.tsx": "d?'43'",
        "app/components/ScalingPanel.tsx": "x.factor==='Width'",
    }
    for filename, pattern in forbidden.items():
        source = (ROOT / filename).read_text(encoding="utf-8")
        require(pattern not in source, f"{filename} still hard-codes comparison evidence with {pattern!r}")


def main() -> None:
    audit_tokenizers()
    audit_shakespeare()
    audit_tinystories_training()
    audit_dashboard_source()
    print(f"Comparison evidence audit passed: {checks} checks; {skips} unavailable evidence files skipped.")


if __name__ == "__main__":
    main()
