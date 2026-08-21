"""Train the smallest useful next-character language model with MLX.

The model has one table: each input character indexes a row of scores for the
next character. It is intentionally simpler than a transformer so the complete
training loop and the effect of learning are easy to inspect.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "raw" / "tiny-shakespeare.txt"
DEFAULT_METRICS = ROOT / "public" / "data" / "shakespeare-metrics.json"
DEFAULT_RUN_DIR = ROOT / "experiments" / "shakespeare-bigram-001"


class BigramLanguageModel(nn.Module):
    """Predict the next character using only the current character."""

    def __init__(self, vocabulary_size: int):
        super().__init__()
        self.next_token_scores = nn.Embedding(vocabulary_size, vocabulary_size)

    def __call__(self, token_ids: mx.array) -> mx.array:
        return self.next_token_scores(token_ids)


def encode(text: str, char_to_id: dict[str, int]) -> list[int]:
    return [char_to_id[character] for character in text]


def decode(token_ids: list[int], id_to_char: list[str]) -> str:
    return "".join(id_to_char[token_id] for token_id in token_ids)


def get_batch(data: mx.array, batch_size: int, block_size: int) -> tuple[mx.array, mx.array]:
    starts = mx.random.randint(0, data.size - block_size - 1, shape=(batch_size,))
    positions = starts[:, None] + mx.arange(block_size)[None, :]
    return data[positions], data[positions + 1]


def loss_fn(model: BigramLanguageModel, inputs: mx.array, targets: mx.array) -> mx.array:
    logits = model(inputs)
    return nn.losses.cross_entropy(logits, targets, reduction="mean")


def estimate_loss(model: BigramLanguageModel, data: mx.array, *, batches: int, batch_size: int, block_size: int) -> float:
    losses = []
    for _ in range(batches):
        inputs, targets = get_batch(data, batch_size, block_size)
        losses.append(loss_fn(model, inputs, targets))
    return float(mx.mean(mx.stack(losses)).item())


def generate(model: BigramLanguageModel, start_id: int, *, characters: int, temperature: float) -> list[int]:
    generated = [start_id]
    current = mx.array([start_id])
    for _ in range(characters):
        logits = model(current)[-1] / temperature
        next_id = int(mx.random.categorical(logits).item())
        generated.append(next_id)
        current = mx.array([next_id])
    return generated


def atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    args = parser.parse_args()

    text = args.data.read_text(encoding="utf-8")
    vocabulary = sorted(set(text))
    char_to_id = {character: index for index, character in enumerate(vocabulary)}
    split_index = int(len(text) * 0.9)
    train_data = mx.array(encode(text[:split_index], char_to_id), dtype=mx.int32)
    validation_data = mx.array(encode(text[split_index:], char_to_id), dtype=mx.int32)

    mx.random.seed(args.seed)
    model = BigramLanguageModel(len(vocabulary))
    mx.eval(model.parameters())
    optimiser = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.0)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: list[dict] = []
    requested_steps = sorted({0, 1, 25, 100, args.steps})
    payload = {
        "status": "Training",
        "runId": args.run_dir.name,
        "model": "Character bigram",
        "parameters": sum(parameter.size for _, parameter in nn.utils.tree_flatten(model.parameters())),
        "updatedAt": datetime.now(UTC).isoformat(),
        "checkpoints": checkpoints,
    }
    atomic_json_write(args.metrics, payload)

    def capture(step: int, train_loss: float | None = None) -> None:
        validation_loss = estimate_loss(model, validation_data, batches=20, batch_size=args.batch_size, block_size=args.block_size) if step else None
        mx.random.seed(args.seed + 10_000 + step)
        sample_ids = generate(model, char_to_id["\n"], characters=280, temperature=0.9)
        checkpoint = {
            "label": "Random weights" if step == 0 else ("One update" if step == 1 else f"After {step} updates"),
            "step": step,
            "trainLoss": train_loss,
            "validationLoss": validation_loss,
            "sample": decode(sample_ids, vocabulary),
        }
        checkpoints.append(checkpoint)
        payload["updatedAt"] = datetime.now(UTC).isoformat()
        atomic_json_write(args.metrics, payload)
        model.save_weights(str(args.run_dir / f"checkpoint-{step:04d}.safetensors"))
        print(f"step={step:4d} train_loss={train_loss!s:>10} validation_loss={validation_loss!s:>10}")

    capture(0)
    mx.random.seed(args.seed)
    last_loss = None
    for step in range(1, args.steps + 1):
        inputs, targets = get_batch(train_data, args.batch_size, args.block_size)
        loss, gradients = loss_and_grad(model, inputs, targets)
        optimiser.update(model, gradients)
        mx.eval(model.parameters(), optimiser.state, loss)
        last_loss = float(loss.item())
        if step in requested_steps:
            capture(step, last_loss)

    payload["status"] = "Complete"
    payload["updatedAt"] = datetime.now(UTC).isoformat()
    atomic_json_write(args.metrics, payload)
    (args.run_dir / "config.json").write_text(json.dumps({
        "seed": args.seed,
        "steps": args.steps,
        "batchSize": args.batch_size,
        "blockSize": args.block_size,
        "learningRate": args.learning_rate,
        "vocabularySize": len(vocabulary),
        "trainingCharacters": train_data.size,
        "validationCharacters": validation_data.size,
        "mlxDevice": str(mx.default_device()),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
