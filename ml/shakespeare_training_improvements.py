"""Run controlled training-recipe experiments for the tiny transformer.

Architecture, data order, seed, batch size and update count stay fixed. One
variant changes only the learning-rate schedule; another changes only gradient
clipping. Final models are scored with the frozen evaluation protocol.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from shakespeare_transformer import (
    DEFAULT_DATA,
    DEFAULT_RUN_DIR,
    TinyTransformerLanguageModel,
    atomic_json_write,
    estimate_loss,
    generate_from_prompt,
    get_batch,
    load_data,
    loss_fn,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "public" / "data" / "shakespeare-training-improvements.json"
EXPERIMENT_ROOT = ROOT / "experiments"
EXPERIMENT_DIR = EXPERIMENT_ROOT / "shakespeare-training-improvements-001"
SEED = 42
STEPS = 3_000
BATCH_SIZE = 32
CONTEXT_SIZE = 64
MODEL_SIZE = 64
HEAD_COUNT = 4
BLOCK_COUNT = 2
WEIGHT_DECAY = 0.01
PROMPT = "To be, or not to be"
CAPTURE_STEPS = (0, 250, 1_000, 2_000, 3_000)


def write_experiment_config(payload: dict) -> None:
    """Persist the comparison protocol and selected result outside dashboard data."""
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    variants = [{
        "runId": variant["runId"],
        "label": variant["label"],
        "validationLossMean": variant["evaluation"]["validationLossMean"],
        "validationLossStd": variant["evaluation"]["validationLossStd"],
        "perplexity": variant["evaluation"]["perplexity"],
    } for variant in payload["variants"]]
    (EXPERIMENT_DIR / "config.json").write_text(json.dumps({
        "experimentId": payload["experimentId"],
        "createdAt": payload["updatedAt"],
        "controlledVariables": payload["controlledVariables"],
        "variants": variants,
        "selectedRunId": min(variants, key=lambda item: item["validationLossMean"])["runId"],
        "elapsedSeconds": payload["elapsedSeconds"],
    }, indent=2) + "\n", encoding="utf-8")


def frozen_loss(model: TinyTransformerLanguageModel, data: mx.array, base_seed: int) -> tuple[float, float]:
    values = [
        estimate_loss(
            model, data, seed=base_seed + repeat, batches=20,
            batch_size=BATCH_SIZE, context_size=CONTEXT_SIZE,
        )
        for repeat in range(5)
    ]
    return statistics.mean(values), statistics.pstdev(values)


def evaluate_model(
    model: TinyTransformerLanguageModel,
    train_data: mx.array,
    validation_data: mx.array,
    vocabulary: list[str],
    char_to_id: dict[str, int],
) -> dict:
    train_mean, train_std = frozen_loss(model, train_data, SEED + 100_000)
    validation_mean, validation_std = frozen_loss(model, validation_data, SEED + 200_000)
    continuation = generate_from_prompt(
        model, PROMPT, char_to_id, vocabulary,
        seed=SEED + 300_000, characters=120, temperature=0.8,
    )
    return {
        "trainLossMean": train_mean,
        "trainLossStd": train_std,
        "validationLossMean": validation_mean,
        "validationLossStd": validation_std,
        "generalisationGap": validation_mean - train_mean,
        "perplexity": math.exp(validation_mean),
        "prompt": PROMPT,
        "continuation": continuation,
    }


def build_model(vocabulary_size: int) -> TinyTransformerLanguageModel:
    mx.random.seed(SEED)
    model = TinyTransformerLanguageModel(
        vocabulary_size, CONTEXT_SIZE, MODEL_SIZE, HEAD_COUNT, BLOCK_COUNT,
    )
    mx.eval(model.parameters())
    return model


def train_variant(
    name: str,
    label: str,
    schedule_name: str,
    clip_norm: float | None,
    train_data: mx.array,
    validation_data: mx.array,
    vocabulary: list[str],
    char_to_id: dict[str, int],
) -> dict:
    mx.reset_peak_memory()
    model = build_model(len(vocabulary))
    if schedule_name == "warmup_cosine":
        learning_rate = optim.join_schedules([
            optim.linear_schedule(0.0001, 0.001, steps=100),
            optim.cosine_decay(0.001, decay_steps=STEPS - 100, end=0.0001),
        ], boundaries=[100])
    else:
        learning_rate = 0.001
    optimiser = optim.AdamW(learning_rate=learning_rate, weight_decay=WEIGHT_DECAY)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    run_dir = EXPERIMENT_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    clipped_steps = 0
    maximum_gradient_norm = 0.0
    started_at = time.perf_counter()

    def capture(step: int, train_loss: float | None, current_lr: float) -> None:
        validation_loss = estimate_loss(
            model, validation_data, seed=SEED + 500_000 + step,
            batches=30, batch_size=BATCH_SIZE, context_size=CONTEXT_SIZE,
        )
        checkpoints.append({
            "step": step,
            "capturedAt": datetime.now(UTC).isoformat(),
            "trainLoss": train_loss,
            "validationLoss": validation_loss,
            "learningRate": current_lr,
        })
        model.save_weights(str(run_dir / f"checkpoint-{step:04d}.safetensors"))
        print(f"{label:22s} step={step:4d} validation={validation_loss:.4f} lr={current_lr:.7f}")

    capture(0, None, 0.0001 if schedule_name == "warmup_cosine" else 0.001)
    last_loss = None
    for step in range(1, STEPS + 1):
        mx.random.seed(SEED + step)
        inputs, targets = get_batch(train_data, BATCH_SIZE, CONTEXT_SIZE)
        loss, gradients = loss_and_grad(model, inputs, targets)
        if clip_norm is not None:
            gradients, total_norm = optim.clip_grad_norm(gradients, clip_norm)
            mx.eval(total_norm)
            norm_value = float(total_norm.item())
            maximum_gradient_norm = max(maximum_gradient_norm, norm_value)
            clipped_steps += int(norm_value > clip_norm)
        optimiser.update(model, gradients)
        mx.eval(model.parameters(), optimiser.state, loss)
        last_loss = float(loss.item())
        if step in CAPTURE_STEPS:
            current_lr = float(optimiser.learning_rate.item())
            capture(step, last_loss, current_lr)

    elapsed = time.perf_counter() - started_at
    final_evaluation = evaluate_model(
        model, train_data, validation_data, vocabulary, char_to_id,
    )
    config = {
        "runId": name,
        "createdAt": datetime.now(UTC).isoformat(),
        "label": label,
        "seed": SEED,
        "steps": STEPS,
        "batchSize": BATCH_SIZE,
        "contextSize": CONTEXT_SIZE,
        "modelSize": MODEL_SIZE,
        "attentionHeads": HEAD_COUNT,
        "transformerBlocks": BLOCK_COUNT,
        "parameterCount": 112_065,
        "vocabularySize": len(vocabulary),
        "trainingCharacters": train_data.size,
        "validationCharacters": validation_data.size,
        "mlxDevice": str(mx.default_device()),
        "peakMetalMemoryBytes": mx.get_peak_memory(),
        "weightDecay": WEIGHT_DECAY,
        "learningRateRecipe": schedule_name,
        "gradientClipNorm": clip_norm,
        "elapsedSeconds": elapsed,
        "clippedSteps": clipped_steps,
        "maximumGradientNorm": maximum_gradient_norm if clip_norm is not None else None,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {**config, "checkpoints": checkpoints, "evaluation": final_evaluation}


def main() -> None:
    train_data, validation_data, vocabulary, char_to_id = load_data(DEFAULT_DATA)
    payload = {
        "status": "Running",
        "experimentId": "shakespeare-training-improvements-001",
        "updatedAt": datetime.now(UTC).isoformat(),
        "controlledVariables": {
            "architectureParameters": 112_065,
            "seed": SEED,
            "steps": STEPS,
            "batchSize": BATCH_SIZE,
            "contextSize": CONTEXT_SIZE,
            "trainingCharacters": train_data.size,
            "validationCharacters": validation_data.size,
            "evaluationRepeats": 5,
            "evaluationBatchesPerRepeat": 20,
        },
        "variants": [],
    }
    atomic_json_write(OUTPUT, payload)
    overall_started = time.perf_counter()

    baseline = build_model(len(vocabulary))
    baseline.load_weights(str(DEFAULT_RUN_DIR / "checkpoint-3000.safetensors"))
    mx.eval(baseline.parameters())
    baseline_evaluation = evaluate_model(
        baseline, train_data, validation_data, vocabulary, char_to_id,
    )
    baseline_config = json.loads((DEFAULT_RUN_DIR / "config.json").read_text(encoding="utf-8"))
    payload["variants"].append({
        "runId": DEFAULT_RUN_DIR.name,
        "label": "Constant LR baseline",
        "learningRateRecipe": "constant_0.001",
        "gradientClipNorm": None,
        "elapsedSeconds": baseline_config["elapsedSeconds"],
        "evaluation": baseline_evaluation,
    })
    atomic_json_write(OUTPUT, payload)

    variants = (
        ("shakespeare-warmup-cosine-001", "Warmup + cosine", "warmup_cosine", None),
        ("shakespeare-gradient-clipping-001", "Gradient clipping", "constant_0.001", 1.0),
    )
    for name, label, schedule_name, clip_norm in variants:
        result = train_variant(
            name, label, schedule_name, clip_norm,
            train_data, validation_data, vocabulary, char_to_id,
        )
        payload["variants"].append(result)
        payload["updatedAt"] = datetime.now(UTC).isoformat()
        atomic_json_write(OUTPUT, payload)

    payload["status"] = "Complete"
    payload["elapsedSeconds"] = time.perf_counter() - overall_started
    payload["updatedAt"] = datetime.now(UTC).isoformat()
    atomic_json_write(OUTPUT, payload)
    write_experiment_config(payload)


if __name__ == "__main__":
    main()
