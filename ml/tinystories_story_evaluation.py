"""Evaluate TinyStories checkpoints with frozen loss and behaviour lenses.

This script performs inference only. It compares the random baseline, the
Lesson 8 inherited-history checkpoint, and the Lesson 9 clean-history
checkpoint on complete validation loss plus fixed prompt scenarios, repetition,
cross-seed diversity, sentence closure, and overlap with training text.
Keyword checks are deliberately labelled proxies rather than semantic judges.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import statistics
import time
import unicodedata
from collections import defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

from paths import RAW_DATA_DIR, local_result, local_run
from shakespeare_transformer import TinyTransformerLanguageModel, atomic_json_write
from tinystories_checkpoint_resume import file_sha256
from tinystories_random_baseline import evaluate_validation, random_completion
from tinystories_sequence_batching import build_examples, sha256
from tinystories_transition import load_jsonl


DEFAULT_DATA_DIR = RAW_DATA_DIR / "tinystories-sample"
DEFAULT_TOKENIZER = local_run("tinystories-tokenizer-001") / "tokenizer.json"
DEFAULT_BASELINE_EVIDENCE = local_result("tinystories-random-baseline.json")
DEFAULT_RECIPE_EVIDENCE = local_result("tinystories-training-recipe.json")
DEFAULT_SCALING_EVIDENCE = local_result("tinystories-scaling-budget.json")
DEFAULT_RANDOM_CHECKPOINT = local_run("tinystories-random-baseline-001") / "checkpoint-0000.safetensors"
DEFAULT_METRICS = local_result("tinystories-story-evaluation.json")
DEFAULT_RUN_DIR = local_run("tinystories-story-evaluation-001")
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
GENERATION_SEEDS = (3101, 3102, 3103)
TEMPERATURE = 0.8
MAXIMUM_TOKENS = 64

SCENARIOS = (
    {
        "id": "care-for-kitten",
        "prompt": "Nia found a cold kitten in the rain. To help the kitten, Nia",
        "expectedAny": ("help", "home", "warm", "blanket", "food", "took", "brought", "gave", "carried"),
        "contradictionTerms": ("dragon", "spaceship", "robot"),
        "intent": "Continue an explicitly stated caring action without replacing the situation.",
    },
    {
        "id": "repair-harm",
        "prompt": "Ben broke his sister's red toy by accident. He decided to",
        "expectedAny": ("fix", "tell", "sorry", "apologize", "apologise", "help", "replace", "repair"),
        "contradictionTerms": ("kitten", "dragon", "spaceship"),
        "intent": "Continue a cause-and-response setup with a plausible repair action.",
    },
    {
        "id": "boat-danger",
        "prompt": "The small boat had a hole and water rushed in. Mia and Tom",
        "expectedAny": ("water", "hole", "boat", "shore", "swim", "help", "save", "out"),
        "contradictionTerms": ("spaceship", "dragon", "school"),
        "intent": "Remain connected to the immediate boat problem and its participants.",
    },
    {
        "id": "seed-growth",
        "prompt": "Lily planted a tiny seed and watered it every day. After many days, the seed",
        "expectedAny": ("grew", "grow", "plant", "sprout", "flower", "tree", "green"),
        "contradictionTerms": ("boat", "spaceship", "dragon"),
        "intent": "Continue a simple delayed cause with a related growth outcome.",
    },
    {
        "id": "fear-and-friend",
        "prompt": "A little dragon was afraid of its own fire. Its friend said",
        "expectedAny": ("afraid", "brave", "help", "try", "fire", "friend", "can", "don't"),
        "contradictionTerms": ("spaceship", "computer", "school bus"),
        "intent": "Continue dialogue that remains connected to fear, fire, or reassurance.",
    },
)


def words(text: str) -> list[str]:
    return WORD_RE.findall(unicodedata.normalize("NFKC", text).lower())


def ngrams(tokens: list[str], width: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[index:index + width]) for index in range(max(0, len(tokens) - width + 1))}


def repeated_ngram_fraction(tokens: list[str], width: int) -> float:
    sequence = [tuple(tokens[index:index + width]) for index in range(max(0, len(tokens) - width + 1))]
    return 0.0 if not sequence else 1 - len(set(sequence)) / len(sequence)


def distinct_ngram_ratio(tokens: list[str], width: int) -> float:
    sequence = [tuple(tokens[index:index + width]) for index in range(max(0, len(tokens) - width + 1))]
    return 0.0 if not sequence else len(set(sequence)) / len(sequence)


def pairwise_jaccard(sets: list[set]) -> float:
    values = []
    for left, right in combinations(sets, 2):
        union = left | right
        values.append(len(left & right) / len(union) if union else 1.0)
    return statistics.mean(values) if values else 1.0


def training_overlap_index(train_rows: list[dict]) -> dict:
    story_fivegrams: list[set] = []
    inverted: dict[tuple[str, ...], set[int]] = defaultdict(set)
    exact_by_width = {width: set() for width in range(5, 13)}
    for story_index, row in enumerate(train_rows):
        tokens = words(row["text"])
        fivegrams = ngrams(tokens, 5)
        story_fivegrams.append(fivegrams)
        for gram in fivegrams:
            inverted[gram].add(story_index)
        for width in exact_by_width:
            exact_by_width[width].update(ngrams(tokens, width))
    return {"storyFivegrams": story_fivegrams, "inverted": inverted, "exactByWidth": exact_by_width}


def overlap_measure(tokens: list[str], index: dict) -> dict:
    generated = ngrams(tokens, 5)
    intersections: dict[int, int] = defaultdict(int)
    for gram in generated:
        for story_index in index["inverted"].get(gram, ()):
            intersections[story_index] += 1
    best_jaccard = 0.0
    best_story_index = None
    for story_index, intersection in intersections.items():
        union = len(generated) + len(index["storyFivegrams"][story_index]) - intersection
        score = intersection / union if union else 1.0
        if score > best_jaccard:
            best_jaccard = score
            best_story_index = story_index
    longest = 0
    for width in range(min(12, len(tokens)), 4, -1):
        if ngrams(tokens, width) & index["exactByWidth"][width]:
            longest = width
            break
    return {
        "maximumTrainingStoryFiveGramJaccard": best_jaccard,
        "closestTrainingStoryIndex": best_story_index,
        "longestExactTrainingSpanWordsUpTo12": longest,
        "hasExactEightWordTrainingSpan": longest >= 8,
    }


def checked_manifest(directory: Path, expected_step: int = 700) -> Path:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("absoluteStep") != expected_step:
        raise RuntimeError(f"Incomplete or unexpected checkpoint manifest: {manifest_path}")
    for filename, expected in manifest["files"].items():
        path = directory / filename
        if not path.is_file() or file_sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Checkpoint checksum failed: {path}")
    return directory / "model.safetensors"


def build_model(config: dict, checkpoint: Path) -> TinyTransformerLanguageModel:
    model = TinyTransformerLanguageModel(
        config["vocabularySize"], config["contextSize"], config["modelSize"],
        config["attentionHeads"], config["transformerBlocks"],
    )
    model.load_weights(str(checkpoint), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model


def evaluate_candidate(candidate: dict, config: dict, tokenizer: Tokenizer, validation_examples: list[dict], overlap_index: dict) -> dict:
    model = build_model(config, Path(candidate["checkpoint"]))
    validation = evaluate_validation(model, validation_examples, config["batchSize"])
    samples = []
    scenario_summaries = []
    for scenario in SCENARIOS:
        scenario_samples = []
        for seed in GENERATION_SEEDS:
            generated = random_completion(
                model, tokenizer, scenario["prompt"], seed=seed,
                maximum_tokens=MAXIMUM_TOKENS, temperature=TEMPERATURE,
            )
            continuation_words = words(generated["continuation"])
            expected_hits = sorted(set(continuation_words) & set(scenario["expectedAny"]))
            normalised_continuation = " ".join(continuation_words)
            contradiction_hits = sorted(term for term in scenario["contradictionTerms"] if term in normalised_continuation)
            overlap = overlap_measure(continuation_words, overlap_index)
            measured = {
                **generated,
                "scenarioId": scenario["id"],
                "expectedKeywordHits": expected_hits,
                "adherenceProxyPass": bool(expected_hits),
                "contradictionHits": contradiction_hits,
                "contradictionProxyPass": not contradiction_hits,
                "endedWithSentencePunctuation": bool(generated["continuation"].rstrip().endswith((".", "!", "?"))),
                "distinctBigramRatio": distinct_ngram_ratio(continuation_words, 2),
                "repeatedFourGramFraction": repeated_ngram_fraction(continuation_words, 4),
                **overlap,
            }
            scenario_samples.append(measured)
            samples.append(measured)
        scenario_summaries.append({
            "scenario": scenario,
            "samples": scenario_samples,
            "adherenceProxyPassRate": statistics.mean(item["adherenceProxyPass"] for item in scenario_samples),
            "contradictionProxyPassRate": statistics.mean(item["contradictionProxyPass"] for item in scenario_samples),
            "meanPairwiseThreeGramJaccard": pairwise_jaccard([ngrams(words(item["continuation"]), 3) for item in scenario_samples]),
            "uniqueContinuations": len({item["continuation"] for item in scenario_samples}),
        })
    result = {
        "candidate": candidate,
        "validation": validation,
        "aggregate": {
            "samples": len(samples),
            "adherenceProxyPassRate": statistics.mean(item["adherenceProxyPass"] for item in samples),
            "contradictionProxyPassRate": statistics.mean(item["contradictionProxyPass"] for item in samples),
            "sentenceClosureRate": statistics.mean(item["endedWithSentencePunctuation"] or item["endedWithEos"] for item in samples),
            "meanDistinctBigramRatio": statistics.mean(item["distinctBigramRatio"] for item in samples),
            "meanRepeatedFourGramFraction": statistics.mean(item["repeatedFourGramFraction"] for item in samples),
            "meanPairwiseThreeGramJaccardAcrossSeeds": statistics.mean(item["meanPairwiseThreeGramJaccard"] for item in scenario_summaries),
            "exactEightWordTrainingSpanRate": statistics.mean(item["hasExactEightWordTrainingSpan"] for item in samples),
            "maximumObservedTrainingFiveGramJaccard": max(item["maximumTrainingStoryFiveGramJaccard"] for item in samples),
            "maximumObservedExactTrainingSpanWordsUpTo12": max(item["longestExactTrainingSpanWordsUpTo12"] for item in samples),
        },
        "scenarios": scenario_summaries,
        "fixedExamples": [item for item in samples if item["seed"] == GENERATION_SEEDS[0]],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--baseline-evidence", type=Path, default=DEFAULT_BASELINE_EVIDENCE)
    parser.add_argument("--recipe-evidence", type=Path, default=DEFAULT_RECIPE_EVIDENCE)
    parser.add_argument("--scaling-evidence", type=Path, default=DEFAULT_SCALING_EVIDENCE)
    parser.add_argument("--random-checkpoint", type=Path, default=DEFAULT_RANDOM_CHECKPOINT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    required = [args.tokenizer, args.baseline_evidence, args.recipe_evidence, args.scaling_evidence, args.random_checkpoint, args.data_dir / "train.jsonl", args.data_dir / "validation.jsonl"]
    for path in required:
        if not path.is_file():
            raise SystemExit(f"Missing prerequisite: {path}")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise SystemExit(f"Run directory already contains files: {args.run_dir}; move it aside before rerunning")
    baseline = json.loads(args.baseline_evidence.read_text(encoding="utf-8"))
    recipe = json.loads(args.recipe_evidence.read_text(encoding="utf-8"))
    scaling = json.loads(args.scaling_evidence.read_text(encoding="utf-8"))
    if any(item.get("status") != "Complete" for item in (baseline, recipe, scaling)):
        raise SystemExit("Lessons 5, 8, and 9 must have complete learner-local evidence")
    config = baseline["configuration"]
    data_hashes = {split: sha256(args.data_dir / f"{split}.jsonl") for split in ("train", "validation")}
    if data_hashes != config["dataHashes"] or sha256(args.tokenizer) != config["tokenizerSha256"]:
        raise SystemExit("Dataset or tokenizer differs from the frozen baseline contract")
    if file_sha256(args.random_checkpoint) != baseline["checkpoint"]["sha256"]:
        raise SystemExit("Random checkpoint differs from the Lesson 5 evidence")
    recipe_dir = Path(recipe["selection"]["selectedCheckpointDirectory"])
    scaling_dir = Path(scaling["selection"]["selectedCheckpointDirectory"])
    candidates = [
        {"id": "random", "label": "Random checkpoint · 0 updates", "history": "Lesson 5 before-state", "checkpoint": str(args.random_checkpoint.resolve()), "checkpointSha256": file_sha256(args.random_checkpoint)},
        {"id": "inherited-700", "label": "Inherited history · step 700", "history": "Lessons 6→7→8 with a declared optimiser reset at step 500", "checkpoint": str(checked_manifest(recipe_dir).resolve()), "checkpointSha256": file_sha256(recipe_dir / "model.safetensors")},
        {"id": "clean-700", "label": "Clean history · 700 updates", "history": "Lesson 9 fresh start with one clean 600-constant + 100-cosine path", "checkpoint": str(checked_manifest(scaling_dir).resolve()), "checkpointSha256": file_sha256(scaling_dir / "model.safetensors")},
    ]
    train_rows = load_jsonl(args.data_dir / "train.jsonl")
    validation_rows = load_jsonl(args.data_dir / "validation.jsonl")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    validation_examples, _ = build_examples(tokenizer, validation_rows, config["contextSize"])
    protocol = {
        "evaluationId": args.run_dir.name,
        "datasetId": config["datasetId"], "dataHashes": data_hashes,
        "tokenizerSha256": config["tokenizerSha256"], "architecture": {
            key: config[key] for key in ("vocabularySize", "contextSize", "modelSize", "attentionHeads", "transformerBlocks", "parameterCount", "batchSize")
        },
        "weightsUpdated": False,
        "completeValidation": True,
        "generation": {"scenarios": list(SCENARIOS), "seeds": list(GENERATION_SEEDS), "temperature": TEMPERATURE, "maximumTokens": MAXIMUM_TOKENS},
        "proxyDefinitions": {
            "adherence": "At least one predeclared scenario-related keyword occurs in the continuation.",
            "contradiction": "No predeclared obviously unrelated keyword occurs in the continuation.",
            "repetition": "Fraction of generated word four-grams that duplicate an earlier four-gram in the same continuation.",
            "diversity": "Mean pairwise word three-gram Jaccard similarity across the three seeds for each prompt; lower means less surface overlap.",
            "overlap": "Exact generated spans of 5–12 words plus maximum word-five-gram Jaccard against any training story.",
        },
        "interpretationBoundary": "Keyword and surface-form proxies do not understand plot, reference, grammar, causality, truth, safety, or semantic consistency. Fixed examples must be inspected beside them.",
        "candidates": candidates,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_write(args.run_dir / "protocol.json", protocol)
    payload = {
        "status": "Evaluating", "evaluationId": args.run_dir.name,
        "updatedAt": datetime.now(UTC).isoformat(), "protocol": protocol,
        "progress": {"phase": "Building training-overlap index", "candidate": None, "completedCandidates": 0, "plannedCandidates": len(candidates)},
        "results": [],
        "limitations": [
            "The corpus is a bounded 1,000/200-story development sample, not the full TinyStories distribution.",
            "Only three sampling seeds per scenario are observed; sampling variance remains substantial.",
            "Keyword checks are narrow deterministic proxies, not semantic ratings or proof of consistency.",
            "Validation was already used during model development, so it is not an untouched final test set.",
            "The two trained checkpoints have different optimisation histories and are compared descriptively rather than as a controlled winner-selection test.",
            "Training overlap detects exact surface spans and word-shingle similarity; paraphrased memorisation can escape it and template language can trigger it.",
        ],
        "nextExperiment": "Expose the evaluated checkpoints in a controlled local prompt playground with visible temperature, seed, and length settings.",
    }
    atomic_json_write(args.metrics, payload)
    started = time.perf_counter()
    overlap_index = training_overlap_index(train_rows)
    for candidate in candidates:
        payload["progress"] = {"phase": "Evaluating checkpoint", "candidate": candidate["id"], "completedCandidates": len(payload["results"]), "plannedCandidates": len(candidates)}
        atomic_json_write(args.metrics, payload)
        result = evaluate_candidate(candidate, config, tokenizer, validation_examples, overlap_index)
        payload["results"].append(result)
        payload["progress"]["completedCandidates"] = len(payload["results"])
        payload["updatedAt"] = datetime.now(UTC).isoformat()
        atomic_json_write(args.metrics, payload)
        print(f"{candidate['id']:13s} validation={result['validation']['crossEntropyLoss']:.4f} adherence-proxy={result['aggregate']['adherenceProxyPassRate']:.1%} repeated-4gram={result['aggregate']['meanRepeatedFourGramFraction']:.1%}", flush=True)
    payload.update({
        "status": "Complete", "updatedAt": datetime.now(UTC).isoformat(),
        "progress": {"phase": "Evaluation complete", "candidate": None, "completedCandidates": len(candidates), "plannedCandidates": len(candidates)},
        "elapsedSeconds": time.perf_counter() - started,
        "environment": {"python": platform.python_version(), "mlx": getattr(mx, "__version__", "unknown"), "device": str(mx.default_device()), "machine": platform.machine(), "platform": platform.platform()},
        "conclusionBoundary": "Training is expected to improve these bounded measurements, but no single proxy establishes coherent story understanding. Do not use this evaluation to retroactively choose between the two differently trained step-700 histories.",
    })
    atomic_json_write(args.metrics, payload)
    atomic_json_write(args.run_dir / "result.json", payload)
    print(json.dumps({"status": payload["status"], "elapsedSeconds": payload["elapsedSeconds"], "results": [{"id": item["candidate"]["id"], "validation": item["validation"]["crossEntropyLoss"], **item["aggregate"]} for item in payload["results"]]}, indent=2))


if __name__ == "__main__":
    main()
