"""Train a small complete decoder-only transformer with MLX.

The model uses four-head causal self-attention, a GELU feed-forward sublayer,
pre-layer-normalisation, residual connections, and two stacked transformer
blocks. It predicts the next character at every position in a 64-token window.
"""

from __future__ import annotations

import argparse
import json
import math
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
DEFAULT_METRICS = local_result("shakespeare-transformer-metrics.json")
DEFAULT_RUN_DIR = local_run("shakespeare-transformer-001")


class MultiHeadCausalAttention(nn.Module):
    """Scaled dot-product attention split across independent heads."""

    def __init__(self, model_size: int, head_count: int):
        super().__init__()
        if model_size % head_count != 0:
            raise ValueError("model_size must be divisible by head_count")
        self.head_count = head_count
        self.head_size = model_size // head_count
        self.query_key_value = nn.Linear(model_size, 3 * model_size, bias=False)
        self.projection = nn.Linear(model_size, model_size, bias=False)

    def __call__(self, inputs: mx.array) -> mx.array:
        batch_size, sequence_length, model_size = inputs.shape
        queries, keys, values = mx.split(self.query_key_value(inputs), 3, axis=-1)

        def split_heads(values_to_split: mx.array) -> mx.array:
            return values_to_split.reshape(
                batch_size, sequence_length, self.head_count, self.head_size
            ).transpose(0, 2, 1, 3)

        queries = split_heads(queries)
        keys = split_heads(keys)
        values = split_heads(values)
        scores = (queries @ keys.transpose(0, 1, 3, 2)) / math.sqrt(self.head_size)
        future_mask = mx.triu(mx.full((sequence_length, sequence_length), -1e9), k=1)
        attention_weights = mx.softmax(scores + future_mask, axis=-1)
        attended = attention_weights @ values
        combined = attended.transpose(0, 2, 1, 3).reshape(batch_size, sequence_length, model_size)
        return self.projection(combined)


class FeedForward(nn.Module):
    """Per-position nonlinear transformation with a four-times wider middle."""

    def __init__(self, model_size: int, multiplier: int = 4):
        super().__init__()
        hidden_size = multiplier * model_size
        self.expand = nn.Linear(model_size, hidden_size)
        self.contract = nn.Linear(hidden_size, model_size)

    def __call__(self, inputs: mx.array) -> mx.array:
        return self.contract(nn.gelu(self.expand(inputs)))


class TransformerBlock(nn.Module):
    """Pre-normalised attention and feed-forward residual sublayers."""

    def __init__(self, model_size: int, head_count: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(model_size)
        self.attention = MultiHeadCausalAttention(model_size, head_count)
        self.feed_forward_norm = nn.LayerNorm(model_size)
        self.feed_forward = FeedForward(model_size)

    def __call__(self, inputs: mx.array) -> mx.array:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        return hidden + self.feed_forward(self.feed_forward_norm(hidden))


class TinyTransformerLanguageModel(nn.Module):
    """A complete small decoder-only transformer for next-character prediction."""

    def __init__(self, vocabulary_size: int, context_size: int, model_size: int, head_count: int, block_count: int):
        super().__init__()
        self.context_size = context_size
        self.token_embedding = nn.Embedding(vocabulary_size, model_size)
        self.position_embedding = nn.Embedding(context_size, model_size)
        self.blocks = nn.Sequential(*[
            TransformerBlock(model_size, head_count) for _ in range(block_count)
        ])
        self.final_norm = nn.LayerNorm(model_size)
        self.output = nn.Linear(model_size, vocabulary_size)

    def __call__(self, token_ids: mx.array) -> mx.array:
        positions = mx.arange(token_ids.shape[1])
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        hidden = self.blocks(hidden)
        return self.output(self.final_norm(hidden))


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
    return data[positions], data[positions + 1]


def loss_fn(model: TinyTransformerLanguageModel, inputs: mx.array, targets: mx.array) -> mx.array:
    return nn.losses.cross_entropy(model(inputs), targets, reduction="mean")


def estimate_loss(model: TinyTransformerLanguageModel, data: mx.array, *, seed: int, batches: int, batch_size: int, context_size: int) -> float:
    mx.random.seed(seed)
    losses = []
    for _ in range(batches):
        inputs, targets = get_batch(data, batch_size, context_size)
        losses.append(loss_fn(model, inputs, targets))
    return float(mx.mean(mx.stack(losses)).item())


def generate(model: TinyTransformerLanguageModel, newline_id: int, vocabulary: list[str], *, seed: int, characters: int, temperature: float) -> str:
    mx.random.seed(seed)
    generated = [newline_id]
    for _ in range(characters):
        context = generated[-model.context_size:]
        logits = model(mx.array([context], dtype=mx.int32))[0, -1] / temperature
        next_id = int(mx.random.categorical(logits).item())
        generated.append(next_id)
    return "".join(vocabulary[token_id] for token_id in generated[1:])


def generate_from_prompt(
    model: TinyTransformerLanguageModel,
    prompt: str,
    char_to_id: dict[str, int],
    vocabulary: list[str],
    *,
    seed: int,
    characters: int,
    temperature: float,
) -> str:
    """Return only newly sampled characters, conditioned on ``prompt``."""
    unknown = sorted(set(prompt) - set(char_to_id))
    if unknown:
        raise ValueError(f"Prompt contains characters outside the vocabulary: {unknown!r}")
    generated = [char_to_id[character] for character in prompt]
    if not generated:
        generated = [char_to_id["\n"]]
    mx.random.seed(seed)
    continuation: list[int] = []
    for _ in range(characters):
        context = generated[-model.context_size :]
        logits = model(mx.array([context], dtype=mx.int32))[0, -1] / temperature
        next_id = int(mx.random.categorical(logits).item())
        generated.append(next_id)
        continuation.append(next_id)
    return "".join(vocabulary[token_id] for token_id in continuation)


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
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context-size", type=int, default=64)
    parser.add_argument("--model-size", type=int, default=64)
    parser.add_argument("--head-count", type=int, default=4)
    parser.add_argument("--block-count", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    train_data, validation_data, vocabulary, char_to_id = load_data(args.data)
    mx.random.seed(args.seed)
    model = TinyTransformerLanguageModel(
        len(vocabulary), args.context_size, args.model_size, args.head_count, args.block_count
    )
    mx.eval(model.parameters())
    optimiser = optim.AdamW(learning_rate=args.learning_rate, weight_decay=0.01)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    parameter_count = sum(parameter.size for _, parameter in nn.utils.tree_flatten(model.parameters()))

    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints: list[dict] = []
    requested_steps = sorted({0, 1, 50, 250, 1_000, 2_000, args.steps})
    payload = {
        "status": "Training",
        "runId": args.run_dir.name,
        "model": "Two-block four-head transformer",
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
            seed=args.seed + 500_000 + step,
            batches=30,
            batch_size=args.batch_size,
            context_size=args.context_size,
        )
        sample = generate(
            model,
            char_to_id["\n"],
            vocabulary,
            seed=args.seed + 600_000 + step,
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
        "modelSize": args.model_size,
        "attentionHeads": args.head_count,
        "transformerBlocks": args.block_count,
        "feedForwardMultiplier": 4,
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
