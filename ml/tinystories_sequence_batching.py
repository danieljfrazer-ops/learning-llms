"""Build and audit story-isolated causal batches for TinyStories.

This lesson changes data arrangement, not model weights. It loads the frozen
development tokenizer, adds story-boundary tokens, creates overlapping
input/target windows, masks padding targets, materialises fixed MLX batches,
and writes learner-local evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
from tokenizers import Tokenizer

from paths import RAW_DATA_DIR, local_result, local_run
from shakespeare_transformer import atomic_json_write
from tinystories_transition import load_jsonl


DEFAULT_DATA_DIR = RAW_DATA_DIR / "tinystories-sample"
DEFAULT_TOKENIZER_EVIDENCE = local_result("tinystories-tokenizer.json")
DEFAULT_TOKENIZER = local_run("tinystories-tokenizer-001") / "tokenizer.json"
DEFAULT_METRICS = local_result("tinystories-batching.json")
DEFAULT_RUN_DIR = local_run("tinystories-batching-001")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_story(tokenizer: Tokenizer, text: str) -> list[int]:
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    if bos is None or eos is None:
        raise RuntimeError("Tokenizer is missing beginning/end story tokens")
    return [bos, *tokenizer.encode(text).ids, eos]


def build_examples(tokenizer: Tokenizer, rows: list[dict], context_size: int) -> tuple[list[dict], dict[int, list[int]]]:
    pad = tokenizer.token_to_id("<pad>")
    if pad is None:
        raise RuntimeError("Tokenizer is missing the padding token")
    examples: list[dict] = []
    encoded_by_row: dict[int, list[int]] = {}
    for row in rows:
        full_story = encode_story(tokenizer, row["text"])
        encoded_by_row[row["sourceRow"]] = full_story
        for chunk_index, start in enumerate(range(0, len(full_story) - 1, context_size)):
            visible = full_story[start:start + context_size + 1]
            valid_targets = len(visible) - 1
            padded = visible + [pad] * (context_size + 1 - len(visible))
            examples.append({
                "sourceRow": row["sourceRow"],
                "chunkIndex": chunk_index,
                "storyTokenStart": start,
                "inputIds": padded[:-1],
                "targetIds": padded[1:],
                "lossMask": [1.0] * valid_targets + [0.0] * (context_size - valid_targets),
                "validTargetCount": valid_targets,
            })
    return examples, encoded_by_row


def audit_examples(
    tokenizer: Tokenizer,
    rows: list[dict],
    examples: list[dict],
    encoded_by_row: dict[int, list[int]],
    context_size: int,
) -> dict:
    pad = tokenizer.token_to_id("<pad>")
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    expected_pairs = sum(len(ids) - 1 for ids in encoded_by_row.values())
    observed_pairs = sum(example["validTargetCount"] for example in examples)
    target_counts: Counter[int] = Counter()
    reconstruction_errors = 0
    mask_errors = 0
    shape_errors = 0

    by_source: dict[int, list[dict]] = {}
    for example in examples:
        by_source.setdefault(example["sourceRow"], []).append(example)
        if not (
            len(example["inputIds"]) == context_size
            and len(example["targetIds"]) == context_size
            and len(example["lossMask"]) == context_size
        ):
            shape_errors += 1
        valid = example["validTargetCount"]
        target_counts.update(example["targetIds"][:valid])
        if not all(value == 1.0 for value in example["lossMask"][:valid]) or any(example["lossMask"][valid:]):
            mask_errors += 1
        if any(token_id != pad for token_id in example["targetIds"][valid:]):
            mask_errors += 1

    for source_row, story_examples in by_source.items():
        story_examples.sort(key=lambda example: example["storyTokenStart"])
        reconstructed = [story_examples[0]["inputIds"][0]]
        for example in story_examples:
            reconstructed.extend(example["targetIds"][:example["validTargetCount"]])
        if reconstructed != encoded_by_row[source_row]:
            reconstruction_errors += 1

    capacity = len(examples) * context_size
    return {
        "stories": len(rows),
        "examples": len(examples),
        "contextSize": context_size,
        "validTargetTokens": observed_pairs,
        "expectedAdjacentStoryPairs": expected_pairs,
        "paddingTargetPositions": capacity - observed_pairs,
        "targetUtilisation": observed_pairs / capacity,
        "examplesPerStoryMean": len(examples) / len(rows),
        "maximumChunksForOneStory": max(Counter(example["sourceRow"] for example in examples).values()),
        "eosTargets": target_counts[eos],
        "bosTargets": target_counts[bos],
        "crossStoryTargetPairs": 0,
        "shapeErrors": shape_errors,
        "maskErrors": mask_errors,
        "storyReconstructionErrors": reconstruction_errors,
        "pairCoverageMatches": observed_pairs == expected_pairs,
    }


def naïve_concatenation_comparison(encoded_by_row: dict[int, list[int]], context_size: int) -> dict:
    story_sequences = list(encoded_by_row.values())
    stream_tokens = sum(len(ids) for ids in story_sequences)
    target_pairs = stream_tokens - 1
    capacity = math.ceil(target_pairs / context_size) * context_size
    return {
        "method": "Concatenate complete encoded stories, then slice one continuous stream",
        "targetPairs": target_pairs,
        "crossStoryTargetPairs": max(0, len(story_sequences) - 1),
        "windows": math.ceil(target_pairs / context_size),
        "targetUtilisation": target_pairs / capacity,
        "boundaryRisk": "A later story can attend to unrelated earlier-story tokens, and EOS→BOS boundary pairs become optimisation targets",
    }


def materialise_batch(examples: list[dict], indices: list[int]) -> tuple[mx.array, mx.array, mx.array]:
    selected = [examples[index] for index in indices]
    inputs = mx.array([example["inputIds"] for example in selected], dtype=mx.int32)
    targets = mx.array([example["targetIds"] for example in selected], dtype=mx.int32)
    loss_mask = mx.array([example["lossMask"] for example in selected], dtype=mx.float32)
    mx.eval(inputs, targets, loss_mask)
    return inputs, targets, loss_mask


def batch_evidence(examples: list[dict], indices: list[int], tensors: tuple[mx.array, mx.array, mx.array]) -> dict:
    inputs, targets, loss_mask = tensors
    return {
        "exampleIndices": indices,
        "sourceRows": [examples[index]["sourceRow"] for index in indices],
        "shapes": {
            "inputs": list(inputs.shape),
            "targets": list(targets.shape),
            "lossMask": list(loss_mask.shape),
        },
        "dtypes": {
            "inputs": str(inputs.dtype),
            "targets": str(targets.dtype),
            "lossMask": str(loss_mask.dtype),
        },
        "validTargets": sum(examples[index]["validTargetCount"] for index in indices),
        "paddingTargets": int(loss_mask.size) - sum(examples[index]["validTargetCount"] for index in indices),
        "tensorStorageBytes": int(inputs.nbytes + targets.nbytes + loss_mask.nbytes),
    }


def piece(tokenizer: Tokenizer, token_id: int) -> str:
    return tokenizer.id_to_token(token_id) or "<?>"


def padding_preview(tokenizer: Tokenizer, examples: list[dict]) -> dict:
    example = next(item for item in examples if item["validTargetCount"] < len(item["lossMask"]))
    valid = example["validTargetCount"]
    start = max(0, valid - 6)
    stop = min(len(example["lossMask"]), valid + 4)
    positions = []
    for position in range(start, stop):
        positions.append({
            "position": position,
            "inputId": example["inputIds"][position],
            "inputPiece": piece(tokenizer, example["inputIds"][position]),
            "targetId": example["targetIds"][position],
            "targetPiece": piece(tokenizer, example["targetIds"][position]),
            "countsForLoss": bool(example["lossMask"][position]),
        })
    return {
        "sourceRow": example["sourceRow"],
        "chunkIndex": example["chunkIndex"],
        "storyTokenStart": example["storyTokenStart"],
        "validTargetCount": valid,
        "positions": positions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--tokenizer-evidence", type=Path, default=DEFAULT_TOKENIZER_EVIDENCE)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.context_size < 2 or args.batch_size < 1:
        raise SystemExit("context size must be at least 2 and batch size must be positive")
    if not args.tokenizer_evidence.is_file() or not args.tokenizer.is_file():
        raise SystemExit("Missing learner-local Lesson 3 evidence or selected tokenizer; run ml/tinystories_tokenizer_experiment.py first")

    tokenizer_evidence = json.loads(args.tokenizer_evidence.read_text(encoding="utf-8"))
    selected = tokenizer_evidence.get("decision", {})
    if tokenizer_evidence.get("status") != "Complete":
        raise SystemExit("Lesson 3 tokenizer experiment is not complete")
    if sha256(args.tokenizer) != selected.get("selectedTokenizerSha256"):
        raise SystemExit("Selected tokenizer checksum does not match Lesson 3 evidence")

    dataset = tokenizer_evidence["dataset"]
    observed_hashes = {
        "train": sha256(args.data_dir / "train.jsonl"),
        "validation": sha256(args.data_dir / "validation.jsonl"),
    }
    if observed_hashes != dataset.get("sourceManifestHashes"):
        raise SystemExit("Dataset files do not match the hashes inherited from Lesson 3")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    train_rows = load_jsonl(args.data_dir / "train.jsonl")
    validation_rows = load_jsonl(args.data_dir / "validation.jsonl")
    started = time.perf_counter()
    train_examples, train_encoded = build_examples(tokenizer, train_rows, args.context_size)
    validation_examples, validation_encoded = build_examples(tokenizer, validation_rows, args.context_size)
    train_audit = audit_examples(tokenizer, train_rows, train_examples, train_encoded, args.context_size)
    validation_audit = audit_examples(tokenizer, validation_rows, validation_examples, validation_encoded, args.context_size)

    for name, audit in (("train", train_audit), ("validation", validation_audit)):
        if not audit["pairCoverageMatches"] or audit["shapeErrors"] or audit["maskErrors"] or audit["storyReconstructionErrors"]:
            raise RuntimeError(f"{name} batching invariants failed: {audit}")

    rng = random.Random(args.seed)
    train_indices = list(range(len(train_examples)))
    rng.shuffle(train_indices)
    train_indices = train_indices[:args.batch_size]
    validation_indices = list(range(min(args.batch_size, len(validation_examples))))

    mx.reset_peak_memory()
    train_tensors = materialise_batch(train_examples, train_indices)
    validation_tensors = materialise_batch(validation_examples, validation_indices)
    peak_memory = mx.get_peak_memory()
    causal_blocked = args.context_size * (args.context_size - 1) // 2
    causal_allowed = args.context_size * (args.context_size + 1) // 2

    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "runId": args.run_dir.name,
        "datasetId": dataset["datasetId"],
        "sourceManifestHashes": observed_hashes,
        "tokenizerVocabularySize": tokenizer.get_vocab_size(),
        "tokenizerSha256": sha256(args.tokenizer),
        "contextSize": args.context_size,
        "batchSize": args.batch_size,
        "seed": args.seed,
        "boundaryPolicy": "Each story receives BOS/EOS and is windowed independently",
        "windowPolicy": "context+1 token chunks with stride=context, preserving every adjacent pair exactly once",
        "paddingPolicy": "Right-pad final story chunk; multiply per-token loss by lossMask before averaging over valid targets",
        "attentionPolicy": "Standard lower-triangular causal mask; no cross-story positions share a window",
        "modelTraining": "none",
    }
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    preview = {
        "trainBatch": batch_evidence(train_examples, train_indices, train_tensors),
        "validationBatch": batch_evidence(validation_examples, validation_indices, validation_tensors),
        "paddingBoundary": padding_preview(tokenizer, validation_examples),
        "causalMaskPreview": [
            [{"keyPosition": column, "allowed": column <= row} for column in range(8)]
            for row in range(8)
        ],
    }
    (args.run_dir / "batch-preview.json").write_text(json.dumps(preview, indent=2) + "\n", encoding="utf-8")

    payload = {
        "status": "Complete",
        "experimentId": args.run_dir.name,
        "updatedAt": datetime.now(UTC).isoformat(),
        "purpose": "Freeze story-isolated causal sequence batching; no model forward pass or weight updates",
        "dataset": {
            "datasetId": dataset["datasetId"],
            "trainStories": len(train_rows),
            "validationStories": len(validation_rows),
            "sourceManifestHashes": observed_hashes,
        },
        "configuration": config,
        "isolatedWindows": {
            "train": train_audit,
            "validation": validation_audit,
        },
        "naiveConcatenation": {
            "train": naïve_concatenation_comparison(train_encoded, args.context_size),
            "validation": naïve_concatenation_comparison(validation_encoded, args.context_size),
        },
        "fixedBatches": {
            "train": preview["trainBatch"],
            "validation": preview["validationBatch"],
        },
        "paddingBoundaryPreview": preview["paddingBoundary"],
        "causalMask": {
            "shapePerExample": [args.context_size, args.context_size],
            "allowedCurrentOrEarlierRelationships": causal_allowed,
            "blockedFutureRelationships": causal_blocked,
            "preview": preview["causalMaskPreview"],
        },
        "decision": {
            "selectedMethod": "story-isolated-overlap-one-padded-windows-v1",
            "reason": "Preserves every within-story adjacent target pair exactly once, blocks cross-story context, and masks all padding targets",
            "tradeoff": "Uses more padded target positions than naïve concatenation; clarity and boundary correctness are prioritised for the development model",
            "scope": "Frozen for the TinyStories random baseline and initial development training; revisit segment-aware packing during scaling",
        },
        "elapsedSeconds": time.perf_counter() - started,
        "environment": {
            "operatingSystem": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "mlxDevice": str(mx.default_device()),
            "peakMlxAllocationBytes": peak_memory,
        },
        "limitations": [
            "The efficiency comparison counts target positions; it does not benchmark model throughput or activation memory.",
            "Right padding wastes compute because the current model still calculates logits at masked positions.",
            "Story-isolated windows discard attention context before each 128-token chunk boundary, although overlapping one token preserves every adjacent prediction target.",
            "The simple lower-triangular attention mask is safe here only because separate stories never share a window.",
            "Segment-aware packing may recover efficiency later but requires additional attention-mask and position-reset tests.",
        ],
    }
    atomic_json_write(args.metrics, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
