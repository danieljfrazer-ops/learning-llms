"""Audit a bounded TinyStories sample and invoke a provisional random GPT.

This is a pipeline smoke test, not pretraining. It trains only a small BPE
tokenizer, constructs random model weights, measures untrained loss, generates
one deliberately meaningless continuation, and saves learner-local evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from paths import RAW_DATA_DIR, local_result, local_run
from shakespeare_transformer import TinyTransformerLanguageModel, atomic_json_write

DEFAULT_DATA_DIR = RAW_DATA_DIR / "tinystories-sample"
DEFAULT_METRICS = local_result("tinystories-transition.json")
DEFAULT_RUN_DIR = local_run("tinystories-transition-001")
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[int], proportion: float) -> int:
    ordered = sorted(values)
    return ordered[min(math.ceil(len(ordered) * proportion) - 1, len(ordered) - 1)]


def describe(rows: list[dict]) -> dict:
    texts = [row["text"] for row in rows]
    characters = [len(text) for text in texts]
    words = [len(text.split()) for text in texts]
    return {
        "stories": len(texts),
        "characters": sum(characters),
        "words": sum(words),
        "storyCharacters": {
            "minimum": min(characters),
            "median": statistics.median(characters),
            "mean": statistics.mean(characters),
            "p95": percentile(characters, 0.95),
            "maximum": max(characters),
        },
        "exactDuplicateStories": len(texts) - len(set(texts)),
    }


def build_tokenizer(texts: list[str], vocabulary_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocabulary_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
    return tokenizer


def encoded_story(tokenizer: Tokenizer, text: str) -> list[int]:
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    if bos is None or eos is None:
        raise RuntimeError("Tokenizer did not preserve required boundary tokens")
    return [bos, *tokenizer.encode(text).ids, eos]


def fixed_validation_batch(tokenizer: Tokenizer, rows: list[dict], batch_size: int, context_size: int) -> tuple[mx.array, mx.array]:
    token_ids: list[int] = []
    for row in rows:
        token_ids.extend(encoded_story(tokenizer, row["text"]))
        if len(token_ids) >= batch_size * (context_size + 1):
            break
    required = batch_size * (context_size + 1)
    if len(token_ids) < required:
        raise RuntimeError("Validation sample is too small for the requested fixed batch")
    windows = mx.array(token_ids[:required], dtype=mx.int32).reshape(batch_size, context_size + 1)
    return windows[:, :-1], windows[:, 1:]


def random_completion(
    model: TinyTransformerLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    seed: int,
    tokens: int,
    temperature: float,
) -> tuple[str, list[str]]:
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    if bos is None or eos is None:
        raise RuntimeError("Tokenizer is missing boundary tokens")
    generated = [bos, *tokenizer.encode(prompt).ids]
    continuation: list[int] = []
    mx.random.seed(seed)
    for _ in range(tokens):
        context = generated[-model.context_size :]
        logits = model(mx.array([context], dtype=mx.int32))[0, -1] / temperature
        next_id = int(mx.random.categorical(logits).item())
        generated.append(next_id)
        continuation.append(next_id)
        if next_id == eos:
            break
    pieces = [tokenizer.id_to_token(token_id) or "<?>" for token_id in continuation]
    return tokenizer.decode(continuation, skip_special_tokens=True), pieces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--vocabulary-size", type=int, default=2_048)
    parser.add_argument("--context-size", type=int, default=128)
    parser.add_argument("--model-size", type=int, default=128)
    parser.add_argument("--head-count", type=int, default=4)
    parser.add_argument("--block-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    train_rows = load_jsonl(args.data_dir / "train.jsonl")
    validation_rows = load_jsonl(args.data_dir / "validation.jsonl")
    train_texts = [row["text"] for row in train_rows]

    started = time.perf_counter()
    tokenizer_started = time.perf_counter()
    tokenizer = build_tokenizer(train_texts, args.vocabulary_size)
    tokenizer_seconds = time.perf_counter() - tokenizer_started
    actual_vocabulary_size = tokenizer.get_vocab_size()
    train_token_counts = [len(encoded_story(tokenizer, text)) for text in train_texts]
    validation_token_counts = [len(encoded_story(tokenizer, row["text"])) for row in validation_rows]

    mx.reset_peak_memory()
    mx.random.seed(args.seed)
    model = TinyTransformerLanguageModel(
        actual_vocabulary_size,
        args.context_size,
        args.model_size,
        args.head_count,
        args.block_count,
    )
    mx.eval(model.parameters())
    parameter_count = sum(parameter.size for _, parameter in nn.utils.tree_flatten(model.parameters()))
    inputs, targets = fixed_validation_batch(tokenizer, validation_rows, batch_size=8, context_size=args.context_size)
    logits = model(inputs)
    random_loss = float(nn.losses.cross_entropy(logits, targets, reduction="mean").item())
    prompt = "Once upon a time, there was"
    completion, sampled_pieces = random_completion(
        model, tokenizer, prompt, seed=args.seed + 1, tokens=80, temperature=0.9
    )
    mx.eval(logits)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.run_dir / "provisional-tokenizer.json"), pretty=True)
    model.save_weights(str(args.run_dir / "checkpoint-0000.safetensors"))
    config = {
        "runId": args.run_dir.name,
        "stage": "pipeline smoke test; no optimiser updates",
        "seed": args.seed,
        "vocabularySize": actual_vocabulary_size,
        "contextSize": args.context_size,
        "modelSize": args.model_size,
        "attentionHeads": args.head_count,
        "transformerBlocks": args.block_count,
        "parameterCount": parameter_count,
        "tokenizer": "provisional byte-level BPE trained on the bounded training sample",
        "tokenizersVersion": __import__("tokenizers").__version__,
        "mlxDevice": str(mx.default_device()),
    }
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    train_characters = sum(len(text) for text in train_texts)
    validation_characters = sum(len(row["text"]) for row in validation_rows)
    payload = {
        "status": "Complete",
        "experimentId": args.run_dir.name,
        "updatedAt": datetime.now(UTC).isoformat(),
        "purpose": "Transition audit and random-weight pipeline smoke test; no pretraining",
        "sourceManifest": manifest,
        "dataset": {
            "train": describe(train_rows),
            "validation": describe(validation_rows),
            "fullSourceRows": {
                "train": manifest["train"]["sourceRowsTotal"],
                "validation": manifest["validation"]["sourceRowsTotal"],
            },
        },
        "provisionalTokenizer": {
            "algorithm": "byte-level BPE",
            "requestedVocabularySize": args.vocabulary_size,
            "actualVocabularySize": actual_vocabulary_size,
            "specialTokens": SPECIAL_TOKENS,
            "trainingSeconds": tokenizer_seconds,
            "trainTokens": sum(train_token_counts),
            "validationTokens": sum(validation_token_counts),
            "trainCharactersPerToken": train_characters / sum(train_token_counts),
            "validationCharactersPerToken": validation_characters / sum(validation_token_counts),
            "promptTokens": tokenizer.encode(prompt).tokens,
        },
        "randomModel": {
            **config,
            "optimiserUpdates": 0,
            "fixedValidationBatchLoss": random_loss,
            "uniformGuessLoss": math.log(actual_vocabulary_size),
            "prompt": prompt,
            "continuation": completion,
            "sampledTokenPieces": sampled_pieces,
            "peakAcceleratorMemoryBytes": mx.get_peak_memory(),
        },
        "elapsedSeconds": time.perf_counter() - started,
        "environment": {
            "operatingSystem": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
        },
        "limitations": [
            "The bounded sample is a feasibility slice, not a full audit of 2.12 million training stories.",
            "The provisional 2,048-token vocabulary has not been compared with alternatives.",
            "The model received zero optimiser updates, so its continuation is intentionally meaningless.",
            "A single fixed random-loss batch is a smoke-test measurement, not a stable evaluation estimate.",
        ],
    }
    atomic_json_write(args.metrics, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
