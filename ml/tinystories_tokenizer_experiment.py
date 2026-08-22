"""Compare and freeze a TinyStories subword tokenizer.

This lesson trains tokenizer merge rules, not neural-model weights. It requires
the learner-local Lesson 2 audit, trains several byte-level BPE candidates on
the frozen training split only, evaluates them on both splits, and saves the
selected tokenizer plus learner-local evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import statistics
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer

from paths import RAW_DATA_DIR, local_result, local_run
from shakespeare_transformer import atomic_json_write
from tinystories_transition import SPECIAL_TOKENS, build_tokenizer, load_jsonl


DEFAULT_DATA_DIR = RAW_DATA_DIR / "tinystories-sample"
DEFAULT_AUDIT = local_result("tinystories-dataset-audit.json")
DEFAULT_METRICS = local_result("tinystories-tokenizer.json")
DEFAULT_RUN_DIR = local_run("tinystories-tokenizer-001")
DEFAULT_VOCABULARY_SIZES = (512, 1_024, 2_048, 4_096)
REFERENCE_VOCABULARY_SIZE = 2_048
MODEL_WIDTH_PROXY = 256
PROMPTS = (
    "Once upon a time, there was",
    "Lily found a small blue box",
    "The little dragon wanted to help",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[int], proportion: float) -> int:
    ordered = sorted(values)
    return ordered[min(math.ceil(len(ordered) * proportion) - 1, len(ordered) - 1)]


def evaluate_split(tokenizer: Tokenizer, rows: list[dict]) -> dict:
    counts: list[int] = []
    unknown_tokens = 0
    round_trip_mismatches = 0
    unknown_id = tokenizer.token_to_id("<unk>")
    for row in rows:
        encoding = tokenizer.encode(row["text"])
        counts.append(len(encoding.ids))
        unknown_tokens += sum(token_id == unknown_id for token_id in encoding.ids)
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=True)
        if decoded != unicodedata.normalize("NFKC", row["text"]):
            round_trip_mismatches += 1
    characters = sum(len(row["text"]) for row in rows)
    total_tokens = sum(counts)
    return {
        "stories": len(rows),
        "characters": characters,
        "tokens": total_tokens,
        "charactersPerToken": characters / total_tokens,
        "tokensPerStory": {
            "minimum": min(counts),
            "median": statistics.median(counts),
            "mean": statistics.mean(counts),
            "p95": percentile(counts, 0.95),
            "maximum": max(counts),
        },
        "unknownTokens": unknown_tokens,
        "nfkcRoundTripMismatches": round_trip_mismatches,
    }


def vocabulary_parameter_proxy(vocabulary_size: int, model_width: int) -> dict:
    token_embedding = vocabulary_size * model_width
    output_projection = model_width * vocabulary_size + vocabulary_size
    return {
        "assumedModelWidth": model_width,
        "tokenEmbeddingParameters": token_embedding,
        "outputProjectionParameters": output_projection,
        "totalVocabularyDependentParameters": token_embedding + output_projection,
        "scope": "Exact for the course's untied embedding/output design at this width; not a final architecture parameter count",
    }


def candidate_passes(candidate: dict) -> bool:
    expected_ids = list(range(len(SPECIAL_TOKENS)))
    return (
        candidate["specialTokenIds"] == expected_ids
        and candidate["train"]["unknownTokens"] == 0
        and candidate["validation"]["unknownTokens"] == 0
        and candidate["train"]["nfkcRoundTripMismatches"] == 0
        and candidate["validation"]["nfkcRoundTripMismatches"] == 0
    )


def select_candidate(candidates: list[dict]) -> tuple[dict, str]:
    passing = [candidate for candidate in candidates if candidate_passes(candidate)]
    if not passing:
        raise RuntimeError("No tokenizer candidate passed the representation-integrity gates")
    by_size = {candidate["actualVocabularySize"]: candidate for candidate in passing}
    if REFERENCE_VOCABULARY_SIZE not in by_size:
        raise RuntimeError("The comparison must include the provisional 2,048-piece tokenizer")

    reference = by_size[REFERENCE_VOCABULARY_SIZE]
    reference_tokens = reference["validation"]["tokens"]
    smaller = sorted(
        (
            candidate for candidate in passing
            if candidate["actualVocabularySize"] < REFERENCE_VOCABULARY_SIZE
            and candidate["validation"]["tokens"] <= reference_tokens * 1.10
        ),
        key=lambda candidate: candidate["actualVocabularySize"],
    )
    if smaller:
        selected = smaller[0]
        return selected, "Smallest passing vocabulary within a 10% validation-token penalty versus 2,048"

    larger = sorted(
        (
            candidate for candidate in passing
            if candidate["actualVocabularySize"] > REFERENCE_VOCABULARY_SIZE
            and candidate["validation"]["tokens"] <= reference_tokens * 0.80
        ),
        key=lambda candidate: candidate["actualVocabularySize"],
    )
    if larger:
        selected = larger[0]
        return selected, "Smallest passing larger vocabulary delivering at least 20% fewer validation tokens than 2,048"

    return reference, "Retain 2,048: smaller candidates cost more than 10% extra validation tokens and larger candidates save less than 20%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--vocabulary-sizes", type=int, nargs="+", default=DEFAULT_VOCABULARY_SIZES)
    parser.add_argument("--model-width-proxy", type=int, default=MODEL_WIDTH_PROXY)
    args = parser.parse_args()

    vocabulary_sizes = sorted(set(args.vocabulary_sizes))
    if REFERENCE_VOCABULARY_SIZE not in vocabulary_sizes:
        raise SystemExit("Vocabulary candidates must include the provisional size 2048")
    if any(size < 300 for size in vocabulary_sizes):
        raise SystemExit("Byte-level candidates must leave room for the 256-byte alphabet and special tokens")
    if not args.audit.is_file():
        raise SystemExit(f"Missing learner-local Lesson 2 audit: {args.audit}\nRun ml/tinystories_dataset_audit.py first.")

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    decision = audit.get("datasetDecision", {})
    if audit.get("status") != "Complete" or decision.get("datasetId") != "tinystories-development-v1":
        raise SystemExit("Lesson 2 audit does not freeze tinystories-development-v1")

    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    observed_hashes = {
        "train": sha256(args.data_dir / "train.jsonl"),
        "validation": sha256(args.data_dir / "validation.jsonl"),
    }
    if observed_hashes != decision.get("sourceManifestHashes"):
        raise SystemExit("Dataset files no longer match the hashes approved by the Lesson 2 audit")
    if observed_hashes != {"train": manifest["train"]["sha256"], "validation": manifest["validation"]["sha256"]}:
        raise SystemExit("Dataset files no longer match their source manifest")

    train_rows = load_jsonl(args.data_dir / "train.jsonl")
    validation_rows = load_jsonl(args.data_dir / "validation.jsonl")
    train_texts = [row["text"] for row in train_rows]
    args.run_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    candidates: list[dict] = []
    for requested_size in vocabulary_sizes:
        candidate_started = time.perf_counter()
        tokenizer = build_tokenizer(train_texts, requested_size)
        actual_size = tokenizer.get_vocab_size()
        candidate_path = args.run_dir / f"candidate-{actual_size}.json"
        tokenizer.save(str(candidate_path), pretty=True)
        candidate = {
            "requestedVocabularySize": requested_size,
            "actualVocabularySize": actual_size,
            "algorithm": "byte-level BPE with NFKC normalisation and minimum pair frequency 2",
            "trainedOn": "tinystories-development-v1 training split only",
            "specialTokens": SPECIAL_TOKENS,
            "specialTokenIds": [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS],
            "trainingSeconds": time.perf_counter() - candidate_started,
            "artifact": {
                "filename": candidate_path.name,
                "bytes": candidate_path.stat().st_size,
                "sha256": sha256(candidate_path),
            },
            "train": evaluate_split(tokenizer, train_rows),
            "validation": evaluate_split(tokenizer, validation_rows),
            "modelCostProxy": vocabulary_parameter_proxy(actual_size, args.model_width_proxy),
            "promptPieces": [
                {"text": prompt, "tokens": tokenizer.encode(prompt).tokens}
                for prompt in PROMPTS
            ],
        }
        candidate["passesIntegrityGates"] = candidate_passes(candidate)
        candidates.append(candidate)

    selected, rationale = select_candidate(candidates)
    selected_size = selected["actualVocabularySize"]
    selected_candidate_path = args.run_dir / selected["artifact"]["filename"]
    selected_path = args.run_dir / "tokenizer.json"
    shutil.copyfile(selected_candidate_path, selected_path)

    config = {
        "runId": args.run_dir.name,
        "datasetId": decision["datasetId"],
        "sourceManifestHashes": observed_hashes,
        "tokenizersVersion": tokenizers.__version__,
        "candidateVocabularySizes": vocabulary_sizes,
        "specialTokens": SPECIAL_TOKENS,
        "normalisation": "NFKC",
        "preTokenizer": "ByteLevel(add_prefix_space=False)",
        "minimumMergeFrequency": 2,
        "selectionPolicy": {
            "hardGates": "fixed special-token IDs, zero unknown tokens, and zero NFKC round-trip mismatches on both splits",
            "smallerCandidateRule": "select the smallest passing candidate with no more than 10% extra validation tokens versus 2,048",
            "largerCandidateRule": "otherwise select the smallest passing candidate with at least 20% fewer validation tokens versus 2,048",
            "fallback": "retain 2,048",
        },
        "modelWidthProxy": args.model_width_proxy,
        "modelTraining": "none",
        "selectedVocabularySize": selected_size,
        "selectedTokenizerSha256": sha256(selected_path),
    }
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    payload = {
        "status": "Complete",
        "experimentId": args.run_dir.name,
        "updatedAt": datetime.now(UTC).isoformat(),
        "purpose": "Tokenizer comparison and freeze; no neural-model weights or optimiser updates",
        "dataset": {
            "datasetId": decision["datasetId"],
            "trainStories": len(train_rows),
            "validationStories": len(validation_rows),
            "sourceManifestHashes": observed_hashes,
        },
        "configuration": config,
        "candidates": candidates,
        "decision": {
            "selectedVocabularySize": selected_size,
            "selectedTokenizer": "tokenizer.json",
            "selectedTokenizerSha256": sha256(selected_path),
            "rationale": rationale,
            "scope": "Frozen for the next TinyStories batching and development-model lessons; revisit if the training corpus is scaled or normalisation changes",
        },
        "elapsedSeconds": time.perf_counter() - started,
        "environment": {
            "operatingSystem": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "tokenizers": tokenizers.__version__,
        },
        "limitations": [
            "Tokenizer compression and vocabulary-dependent parameter cost do not measure downstream language-model quality.",
            "The selected vocabulary is fitted to a 1,000-story development sample and must be reconsidered for a larger final corpus.",
            "The 10% and 20% selection thresholds express a documented course preference, not a universal optimum.",
            "The 256-wide model-cost proxy is not the final architecture parameter count.",
            "NFKC normalisation can erase distinctions that matter in other languages or tasks.",
        ],
    }
    atomic_json_write(args.metrics, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
