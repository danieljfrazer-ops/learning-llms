"""Train a fixed-window character language model with MLX.

Unlike the bigram baseline, this model sees the previous eight characters at
once. Their embeddings are concatenated in position order and passed through a
small feed-forward network to predict one next character.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from paths import RAW_DATA_DIR, local_result, local_run


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = RAW_DATA_DIR / "tiny-shakespeare.txt"
DEFAULT_METRICS = local_result("shakespeare-context-metrics.json")
DEFAULT_RUN_DIR = local_run("shakespeare-context-001")


class FixedContextLanguageModel(nn.Module):
    """Predict one next character from an ordered fixed-size context."""

    def __init__(self, vocabulary_size: int, context_size: int, embedding_size: int, hidden_size: int):
        super().__init__()
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.token_embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.hidden = nn.Linear(context_size * embedding_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocabulary_size)

    def __call__(self, token_ids: mx.array) -> mx.array:
        embeddings = self.token_embedding(token_ids)
        flattened = embeddings.reshape(embeddings.shape[0], self.context_size * self.embedding_size)
        return self.output(nn.gelu(self.hidden(flattened)))


def load_data(path: Path) -> tuple[mx.array, mx.array, list[str], dict[str, int]]:
    text = path.read_text(encoding="utf-8")
    vocabulary = sorted(set(text))
    char_to_id = {character: index for index, character in enumerate(vocabulary)}
    encoded = [char_to_id[character] for character in text]
    split_index = int(len(encoded) * 0.9)
    return (
        mx.array(encoded[:split_index], dtype=mx.int32),
        mx.array(encoded[split_index:], dtype=mx.int32),
        vocabulary,
        char_to_id,
    )


def get_batch(data: mx.array, batch_size: int, context_size: int) -> tuple[mx.array, mx.array]:
    starts = mx.random.randint(0, data.size - context_size - 1, shape=(batch_size,))
    positions = starts[:, None] + mx.arange(context_size)[None, :]
    return data[positions], data[starts + context_size]


def loss_fn(model: FixedContextLanguageModel, inputs: mx.array, targets: mx.array) -> mx.array:
    return nn.losses.cross_entropy(model(inputs), targets, reduction="mean")


def estimate_loss(model: FixedContextLanguageModel, data: mx.array, *, seed: int, batches: int, batch_size: int, context_size: int) -> float:
    mx.random.seed(seed)
    losses = []
    for _ in range(batches):
        inputs, targets = get_batch(data, batch_size, context_size)
        losses.append(loss_fn(model, inputs, targets))
    return float(mx.mean(mx.stack(losses)).item())


def generate(model: FixedContextLanguageModel, newline_id: int, vocabulary: list[str], *, seed: int, characters: int, temperature: float) -> str:
    mx.random.seed(seed)
    context = [newline_id] * model.context_size
    generated: list[int] = []
    for _ in range(characters):
        logits = model(mx.array([context], dtype=mx.int32))[0] / temperature
        next_id = int(mx.random.categorical(logits).item())
        generated.append(next_id)
        context = context[1:] + [next_id]
    return "".join(vocabulary[token_id] for token_id in generated)


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
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--context-size", type=int, default=8)
    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    args = parser.parse_args()

    train_data, validation_data, vocabulary, char_to_id = load_data(args.data)
    mx.random.seed(args.seed)
    model = FixedContextLanguageModel(len(vocabulary), args.context_size, args.embedding_size, args.hidden_size)
    mx.eval(model.parameters())
    optimiser = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    parameter_count = sum(parameter.size for _, parameter in nn.utils.tree_flatten(model.parameters()))

    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: list[dict] = []
    requested_steps = sorted({0, 1, 50, 250, 1_000, args.steps})
    payload = {
        "status": "Training",
        "runId": args.run_dir.name,
        "model": "8-character context MLP",
        "parameters": parameter_count,
        "contextLength": args.context_size,
        "updatedAt": datetime.now(UTC).isoformat(),
        "checkpoints": checkpoints,
    }
    atomic_json_write(args.metrics, payload)

    def capture(step: int, train_loss: float | None = None) -> None:
        validation_loss = estimate_loss(
            model,
            validation_data,
            seed=args.seed + 100_000 + step,
            batches=50,
            batch_size=args.batch_size,
            context_size=args.context_size,
        )
        sample = generate(
            model,
            char_to_id["\n"],
            vocabulary,
            seed=args.seed + 200_000 + step,
            characters=280,
            temperature=0.9,
        )
        checkpoints.append({
            "label": "Random weights" if step == 0 else ("One update" if step == 1 else f"After {step} updates"),
            "step": step,
            "trainLoss": train_loss,
            "validationLoss": validation_loss,
            "sample": sample,
        })
        payload["updatedAt"] = datetime.now(UTC).isoformat()
        atomic_json_write(args.metrics, payload)
        model.save_weights(str(args.run_dir / f"checkpoint-{step:04d}.safetensors"))
        print(f"step={step:4d} train_loss={train_loss!s:>10} validation_loss={validation_loss:.6f}")

    started_at = time.perf_counter()
    capture(0)
    last_loss = None
    for step in range(1, args.steps + 1):
        mx.random.seed(args.seed + step)
        inputs, targets = get_batch(train_data, args.batch_size, args.context_size)
        loss, gradients = loss_and_grad(model, inputs, targets)
        optimiser.update(model, gradients)
        mx.eval(model.parameters(), optimiser.state, loss)
        last_loss = float(loss.item())
        if step in requested_steps:
            capture(step, last_loss)

    elapsed_seconds = time.perf_counter() - started_at
    payload["status"] = "Complete"
    payload["updatedAt"] = datetime.now(UTC).isoformat()
    payload["elapsedSeconds"] = elapsed_seconds
    atomic_json_write(args.metrics, payload)
    (args.run_dir / "config.json").write_text(json.dumps({
        "seed": args.seed,
        "steps": args.steps,
        "batchSize": args.batch_size,
        "contextSize": args.context_size,
        "embeddingSize": args.embedding_size,
        "hiddenSize": args.hidden_size,
        "learningRate": args.learning_rate,
        "weightDecay": 0.01,
        "parameterCount": parameter_count,
        "vocabularySize": len(vocabulary),
        "trainingCharacters": train_data.size,
        "validationCharacters": validation_data.size,
        "elapsedSeconds": elapsed_seconds,
        "mlxDevice": str(mx.default_device()),
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
