"""Freeze and evaluate the official random TinyStories GPT baseline.

This experiment constructs the architecture that later pretraining will use,
but performs zero optimiser updates. It evaluates every frozen validation
window with the Lesson 4 loss mask, records fixed-prompt generations, saves
checkpoint zero, and proves that the checkpoint reloads without changing
logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from tokenizers import Tokenizer

from paths import RAW_DATA_DIR, local_result, local_run
from shakespeare_transformer import TinyTransformerLanguageModel, atomic_json_write
from tinystories_sequence_batching import build_examples, materialise_batch, sha256
from tinystories_transition import load_jsonl


DEFAULT_DATA_DIR = RAW_DATA_DIR / "tinystories-sample"
DEFAULT_TOKENIZER = local_run("tinystories-tokenizer-001") / "tokenizer.json"
DEFAULT_BATCHING_EVIDENCE = local_result("tinystories-batching.json")
DEFAULT_METRICS = local_result("tinystories-random-baseline.json")
DEFAULT_RUN_DIR = local_run("tinystories-random-baseline-001")
DEFAULT_PROMPTS = [
    "Once upon a time, there was",
    "Lily found a little blue box",
    "The small dragon wanted to",
]


def architecture_parameter_count(
    vocabulary_size: int,
    context_size: int,
    model_size: int,
    block_count: int,
) -> int:
    """Exact count for this course's untied GPT implementation."""
    token_embedding = vocabulary_size * model_size
    position_embedding = context_size * model_size
    attention_per_block = 4 * model_size * model_size
    feed_forward_per_block = 8 * model_size * model_size + 5 * model_size
    normalisation_per_block = 4 * model_size
    final_normalisation = 2 * model_size
    output_projection = model_size * vocabulary_size + vocabulary_size
    return (
        token_embedding
        + position_embedding
        + block_count * (attention_per_block + feed_forward_per_block + normalisation_per_block)
        + final_normalisation
        + output_projection
    )


def architecture_breakdown(
    vocabulary_size: int,
    context_size: int,
    model_size: int,
    block_count: int,
) -> dict[str, int]:
    components = {
        "tokenEmbedding": vocabulary_size * model_size,
        "positionEmbedding": context_size * model_size,
        "attention": block_count * 4 * model_size * model_size,
        "feedForward": block_count * (8 * model_size * model_size + 5 * model_size),
        "blockNormalisation": block_count * 4 * model_size,
        "finalNormalisation": 2 * model_size,
        "outputProjection": model_size * vocabulary_size + vocabulary_size,
    }
    components["total"] = sum(components.values())
    return components


def candidate_table(vocabulary_size: int, context_size: int, block_count: int) -> list[dict]:
    return [
        {
            "label": "Smaller development candidate",
            "modelSize": 192,
            "attentionHeads": 6,
            "transformerBlocks": block_count,
            "parameterCount": architecture_parameter_count(vocabulary_size, context_size, 192, block_count),
            "decision": "Below the declared 5M lower bound",
        },
        {
            "label": "Selected baseline",
            "modelSize": 256,
            "attentionHeads": 8,
            "transformerBlocks": block_count,
            "parameterCount": architecture_parameter_count(vocabulary_size, context_size, 256, block_count),
            "decision": "Smallest candidate inside the declared 5–15M range",
        },
        {
            "label": "Larger development candidate",
            "modelSize": 320,
            "attentionHeads": 8,
            "transformerBlocks": block_count,
            "parameterCount": architecture_parameter_count(vocabulary_size, context_size, 320, block_count),
            "decision": "Deferred until scaling evidence justifies extra compute",
        },
    ]


def masked_loss(
    model: TinyTransformerLanguageModel,
    inputs: mx.array,
    targets: mx.array,
    loss_mask: mx.array,
) -> tuple[float, int]:
    per_position = nn.losses.cross_entropy(model(inputs), targets, reduction="none")
    loss_sum = mx.sum(per_position * loss_mask)
    valid_targets = int(mx.sum(loss_mask).item())
    return float(loss_sum.item()), valid_targets


def evaluate_validation(
    model: TinyTransformerLanguageModel,
    examples: list[dict],
    batch_size: int,
) -> dict:
    losses: list[float] = []
    valid_counts: list[int] = []
    batch_seconds: list[float] = []
    for start in range(0, len(examples), batch_size):
        indices = list(range(start, min(start + batch_size, len(examples))))
        inputs, targets, loss_mask = materialise_batch(examples, indices)
        began = time.perf_counter()
        loss_sum, valid_targets = masked_loss(model, inputs, targets, loss_mask)
        mx.eval(model.parameters())
        batch_seconds.append(time.perf_counter() - began)
        losses.append(loss_sum)
        valid_counts.append(valid_targets)
    elapsed = sum(batch_seconds)
    valid_targets = sum(valid_counts)
    average_loss = sum(losses) / valid_targets
    return {
        "windows": len(examples),
        "batches": len(batch_seconds),
        "validTargets": valid_targets,
        "maskedPaddingTargets": len(examples) * len(examples[0]["lossMask"]) - valid_targets,
        "crossEntropyLoss": average_loss,
        "perplexity": math.exp(average_loss),
        "elapsedSeconds": elapsed,
        "validTargetsPerSecond": valid_targets / elapsed,
        "medianBatchMilliseconds": statistics.median(batch_seconds) * 1_000,
        "batchSeconds": batch_seconds,
    }


def top_predictions(
    model: TinyTransformerLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    count: int = 5,
) -> list[dict]:
    bos = tokenizer.token_to_id("<bos>")
    if bos is None:
        raise RuntimeError("Tokenizer is missing BOS")
    prompt_ids = [bos, *tokenizer.encode(prompt).ids]
    logits = model(mx.array([prompt_ids[-model.context_size :]], dtype=mx.int32))[0, -1]
    probabilities = mx.softmax(logits).tolist()
    ranked = sorted(range(len(probabilities)), key=probabilities.__getitem__, reverse=True)[:count]
    return [
        {
            "tokenId": token_id,
            "tokenPiece": tokenizer.id_to_token(token_id) or "<?>",
            "probability": probabilities[token_id],
        }
        for token_id in ranked
    ]


def random_completion(
    model: TinyTransformerLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    seed: int,
    maximum_tokens: int,
    temperature: float,
) -> dict:
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    if bos is None or eos is None:
        raise RuntimeError("Tokenizer is missing BOS/EOS")
    generated = [bos, *tokenizer.encode(prompt).ids]
    continuation: list[int] = []
    mx.random.seed(seed)
    for _ in range(maximum_tokens):
        context = generated[-model.context_size :]
        logits = model(mx.array([context], dtype=mx.int32))[0, -1] / temperature
        token_id = int(mx.random.categorical(logits).item())
        generated.append(token_id)
        continuation.append(token_id)
        if token_id == eos:
            break
    return {
        "prompt": prompt,
        "seed": seed,
        "temperature": temperature,
        "maximumTokens": maximum_tokens,
        "generatedTokens": len(continuation),
        "endedWithEos": bool(continuation and continuation[-1] == eos),
        "continuation": tokenizer.decode(continuation, skip_special_tokens=True),
        "tokenPieces": [tokenizer.id_to_token(token_id) or "<?>" for token_id in continuation],
    }


def maximum_difference(left: list, right: list) -> float:
    if isinstance(left, list):
        return max((maximum_difference(a, b) for a, b in zip(left, right, strict=True)), default=0.0)
    return abs(float(left) - float(right))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--batching-evidence", type=Path, default=DEFAULT_BATCHING_EVIDENCE)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--model-size", type=int, default=256)
    parser.add_argument("--head-count", type=int, default=8)
    parser.add_argument("--block-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    if not args.batching_evidence.is_file() or not args.tokenizer.is_file():
        raise SystemExit("Missing learner-local Lesson 4 evidence or tokenizer; complete sequence batching first")
    batching = json.loads(args.batching_evidence.read_text(encoding="utf-8"))
    if batching.get("status") != "Complete":
        raise SystemExit("Lesson 4 batching evidence is not complete")
    batch_config = batching["configuration"]
    if sha256(args.tokenizer) != batch_config["tokenizerSha256"]:
        raise SystemExit("Tokenizer checksum differs from the frozen batching contract")
    if batching["decision"]["selectedMethod"] != "story-isolated-overlap-one-padded-windows-v1":
        raise SystemExit("Unexpected batching method; review the Lesson 4 contract before continuing")
    if args.model_size % args.head_count:
        raise SystemExit("model size must be divisible by the number of attention heads")

    context_size = batch_config["contextSize"]
    batch_size = batch_config["batchSize"]
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vocabulary_size = tokenizer.get_vocab_size()
    validation_path = args.data_dir / "validation.jsonl"
    if sha256(validation_path) != batch_config["sourceManifestHashes"]["validation"]:
        raise SystemExit("Validation data checksum differs from the frozen batching contract")
    validation_rows = load_jsonl(validation_path)
    validation_examples, _ = build_examples(tokenizer, validation_rows, context_size)

    started = time.perf_counter()
    mx.reset_peak_memory()
    mx.random.seed(args.seed)
    model = TinyTransformerLanguageModel(
        vocabulary_size,
        context_size,
        args.model_size,
        args.head_count,
        args.block_count,
    )
    model.eval()
    mx.eval(model.parameters())
    flattened = nn.utils.tree_flatten(model.parameters())
    parameter_count = sum(parameter.size for _, parameter in flattened)
    expected_count = architecture_parameter_count(
        vocabulary_size, context_size, args.model_size, args.block_count
    )
    if parameter_count != expected_count:
        raise RuntimeError(f"Parameter count mismatch: model={parameter_count}, formula={expected_count}")

    # Compile/warm the selected shape before measuring the complete validation pass.
    warm_inputs, warm_targets, warm_mask = materialise_batch(
        validation_examples, list(range(min(batch_size, len(validation_examples))))
    )
    masked_loss(model, warm_inputs, warm_targets, warm_mask)
    mx.eval(model.parameters())
    validation = evaluate_validation(model, validation_examples, batch_size)
    peak_evaluation_memory = mx.get_peak_memory()

    samples = [
        random_completion(
            model,
            tokenizer,
            prompt,
            seed=args.seed + 1_000 + index,
            maximum_tokens=args.generation_tokens,
            temperature=args.temperature,
        )
        for index, prompt in enumerate(DEFAULT_PROMPTS)
    ]
    first_prompt_predictions = top_predictions(model, tokenizer, DEFAULT_PROMPTS[0])

    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.run_dir / "checkpoint-0000.safetensors"
    model.save_weights(str(checkpoint_path))
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    parity_inputs = warm_inputs[:1]
    original_slice = model(parity_inputs)[0, :2, :8].tolist()
    reloaded = TinyTransformerLanguageModel(
        vocabulary_size,
        context_size,
        args.model_size,
        args.head_count,
        args.block_count,
    )
    reloaded.load_weights(str(checkpoint_path), strict=True)
    reloaded.eval()
    mx.eval(reloaded.parameters())
    reloaded_slice = reloaded(parity_inputs)[0, :2, :8].tolist()
    reload_difference = maximum_difference(original_slice, reloaded_slice)
    if reload_difference != 0.0:
        raise RuntimeError(f"Checkpoint reload changed fixed logits: maximum difference {reload_difference}")

    breakdown = architecture_breakdown(
        vocabulary_size, context_size, args.model_size, args.block_count
    )
    candidates = candidate_table(vocabulary_size, context_size, args.block_count)
    selected_candidate = next(candidate for candidate in candidates if candidate["label"] == "Selected baseline")
    if args.model_size == 256 and args.head_count == 8 and parameter_count != selected_candidate["parameterCount"]:
        raise RuntimeError("Selected candidate table does not match the materialised model")

    config = {
        "runId": args.run_dir.name,
        "datasetId": batch_config["datasetId"],
        "dataHashes": batch_config["sourceManifestHashes"],
        "tokenizerSha256": sha256(args.tokenizer),
        "vocabularySize": vocabulary_size,
        "contextSize": context_size,
        "batchSize": batch_size,
        "modelSize": args.model_size,
        "attentionHeads": args.head_count,
        "headSize": args.model_size // args.head_count,
        "transformerBlocks": args.block_count,
        "feedForwardMultiplier": 4,
        "parameterCount": parameter_count,
        "parameterDtype": str(flattened[0][1].dtype),
        "seed": args.seed,
        "optimiserUpdates": 0,
        "generation": {
            "prompts": DEFAULT_PROMPTS,
            "seedRule": "model seed + 1000 + prompt index",
            "maximumTokens": args.generation_tokens,
            "temperature": args.temperature,
        },
    }
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    payload = {
        "status": "Complete",
        "experimentId": args.run_dir.name,
        "updatedAt": datetime.now(UTC).isoformat(),
        "purpose": "Official random TinyStories GPT baseline before any optimiser update",
        "configuration": config,
        "architectureDecision": {
            "declaredTargetParameters": [5_000_000, 15_000_000],
            "candidates": candidates,
            "selectedReason": "Choose the smallest candidate inside the declared range; defer added width until a controlled scaling lesson",
            "parameterBreakdown": breakdown,
        },
        "validation": {
            **validation,
            "uniformGuessLoss": math.log(vocabulary_size),
            "lossAboveUniformGuess": validation["crossEntropyLoss"] - math.log(vocabulary_size),
            "protocol": "Every frozen validation window exactly once, ordered by source row and chunk; padding excluded by Lesson 4 loss mask",
        },
        "randomSamples": samples,
        "firstPromptTopPredictions": first_prompt_predictions,
        "checkpoint": {
            "step": 0,
            "optimiserUpdates": 0,
            "file": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_sha256,
            "strictReloadMaximumLogitDifference": reload_difference,
        },
        "memory": {
            "parameterBytes": parameter_count * 4,
            "peakMlxAllocationDuringEvaluationBytes": peak_evaluation_memory,
            "scope": "MLX allocator only during random model materialisation, warmup, and full validation; not total process or unified-memory use",
        },
        "elapsedSeconds": time.perf_counter() - started,
        "environment": {
            "operatingSystem": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "mlx": mx.__version__,
            "tokenizers": __import__("tokenizers").__version__,
            "mlxDevice": str(mx.default_device()),
        },
        "limitations": [
            "Zero optimiser updates means the model has no learned TinyStories next-token patterns.",
            "Random cross-entropy is a plumbing control, not a model-quality target or training forecast.",
            "Three fixed-prompt generations expose random behaviour but cannot characterise every possible sample.",
            "The 5.82M architecture is chosen from a declared laptop-oriented size range, not from trained quality comparisons.",
            "The 1,200-story development slice is suitable for short experiments but is not yet approved as the final scaling corpus.",
            "MLX peak allocation is not total system memory, and warmed throughput does not predict training speed because backward and optimiser work are absent.",
        ],
        "nextExperiment": "Run the first pretraining updates from this exact checkpoint and preserve early samples so learned change remains visible.",
    }
    atomic_json_write(args.metrics, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
