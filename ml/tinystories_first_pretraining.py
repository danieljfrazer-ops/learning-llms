"""Run the first controlled TinyStories pretraining journey from checkpoint zero.

The architecture, tokenizer, data contract, prompts, sampling seeds, and full
validation protocol come from Lessons 4 and 5. This lesson adds the smallest
complete learning loop: masked cross-entropy, automatic differentiation, and
AdamW updates. It preserves early checkpoints so the transition away from
random behaviour remains inspectable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tokenizers import Tokenizer

from paths import RAW_DATA_DIR, local_result, local_run
from shakespeare_transformer import TinyTransformerLanguageModel, atomic_json_write
from tinystories_random_baseline import DEFAULT_PROMPTS, evaluate_validation, random_completion
from tinystories_sequence_batching import build_examples, materialise_batch, sha256
from tinystories_transition import load_jsonl


DEFAULT_DATA_DIR = RAW_DATA_DIR / "tinystories-sample"
DEFAULT_TOKENIZER = local_run("tinystories-tokenizer-001") / "tokenizer.json"
DEFAULT_BATCHING_EVIDENCE = local_result("tinystories-batching.json")
DEFAULT_BASELINE_EVIDENCE = local_result("tinystories-random-baseline.json")
DEFAULT_BASELINE_CHECKPOINT = local_run("tinystories-random-baseline-001") / "checkpoint-0000.safetensors"
DEFAULT_METRICS = local_result("tinystories-first-pretraining.json")
DEFAULT_RUN_DIR = local_run("tinystories-first-pretraining-001")
DEFAULT_CAPTURE_STEPS = (0, 1, 10, 50, 100, 250, 500)


def masked_training_loss(
    model: TinyTransformerLanguageModel,
    inputs: mx.array,
    targets: mx.array,
    loss_mask: mx.array,
) -> mx.array:
    per_position = nn.losses.cross_entropy(model(inputs), targets, reduction="none")
    return mx.sum(per_position * loss_mask) / mx.sum(loss_mask)


def gradient_l2_norm(gradients: dict) -> float:
    leaves = [gradient for _, gradient in nn.utils.tree_flatten(gradients)]
    squared_sum = sum((mx.sum(gradient * gradient) for gradient in leaves), mx.array(0.0))
    return float(mx.sqrt(squared_sum).item())


class DeterministicBatchOrder:
    """Seeded shuffled passes over every example, including a final short batch."""

    def __init__(self, example_count: int, batch_size: int, seed: int) -> None:
        self.example_count = example_count
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.order: list[int] = []
        self.position = 0
        self.completed_passes = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        self.order = list(range(self.example_count))
        self.rng.shuffle(self.order)
        self.position = 0

    def next_indices(self) -> list[int]:
        if self.position >= self.example_count:
            self.completed_passes += 1
            self._reshuffle()
        stop = min(self.position + self.batch_size, self.example_count)
        indices = self.order[self.position:stop]
        self.position = stop
        return indices


def checkpoint_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--batching-evidence", type=Path, default=DEFAULT_BATCHING_EVIDENCE)
    parser.add_argument("--baseline-evidence", type=Path, default=DEFAULT_BASELINE_EVIDENCE)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    if args.steps < 1 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise SystemExit("steps and learning rate must be positive; weight decay cannot be negative")
    for required in (args.batching_evidence, args.baseline_evidence, args.baseline_checkpoint, args.tokenizer):
        if not required.is_file():
            raise SystemExit(f"Missing prerequisite: {required}")

    batching = json.loads(args.batching_evidence.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_evidence.read_text(encoding="utf-8"))
    if batching.get("status") != "Complete" or baseline.get("status") != "Complete":
        raise SystemExit("Lessons 4 and 5 must both be complete learner-local runs")
    batch_config = batching["configuration"]
    model_config = baseline["configuration"]
    if sha256(args.tokenizer) != model_config["tokenizerSha256"]:
        raise SystemExit("Tokenizer checksum differs from the Lesson 5 control")
    if checkpoint_sha256(args.baseline_checkpoint) != baseline["checkpoint"]["sha256"]:
        raise SystemExit("Checkpoint-zero checksum differs from the Lesson 5 control")
    if batch_config["sourceManifestHashes"] != model_config["dataHashes"]:
        raise SystemExit("Lesson 4 and Lesson 5 data contracts disagree")

    observed_hashes = {
        split: sha256(args.data_dir / f"{split}.jsonl")
        for split in ("train", "validation")
    }
    if observed_hashes != model_config["dataHashes"]:
        raise SystemExit("Dataset files differ from the frozen Lesson 5 hashes")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    train_rows = load_jsonl(args.data_dir / "train.jsonl")
    validation_rows = load_jsonl(args.data_dir / "validation.jsonl")
    train_examples, _ = build_examples(tokenizer, train_rows, model_config["contextSize"])
    validation_examples, _ = build_examples(tokenizer, validation_rows, model_config["contextSize"])
    total_train_targets = sum(example["validTargetCount"] for example in train_examples)

    model = TinyTransformerLanguageModel(
        model_config["vocabularySize"],
        model_config["contextSize"],
        model_config["modelSize"],
        model_config["attentionHeads"],
        model_config["transformerBlocks"],
    )
    model.load_weights(str(args.baseline_checkpoint), strict=True)
    mx.eval(model.parameters())
    optimiser = optim.AdamW(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    loss_and_grad = nn.value_and_grad(model, masked_training_loss)
    batch_order = DeterministicBatchOrder(len(train_examples), model_config["batchSize"], args.seed)

    captures_requested = sorted(set(step for step in DEFAULT_CAPTURE_STEPS if step <= args.steps) | {args.steps})
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "runId": args.run_dir.name,
        "startedFromRunId": baseline["experimentId"],
        "startedFromCheckpoint": str(args.baseline_checkpoint),
        "startedFromCheckpointSha256": baseline["checkpoint"]["sha256"],
        "datasetId": model_config["datasetId"],
        "dataHashes": observed_hashes,
        "tokenizerSha256": model_config["tokenizerSha256"],
        "vocabularySize": model_config["vocabularySize"],
        "contextSize": model_config["contextSize"],
        "batchSize": model_config["batchSize"],
        "modelSize": model_config["modelSize"],
        "attentionHeads": model_config["attentionHeads"],
        "transformerBlocks": model_config["transformerBlocks"],
        "parameterCount": model_config["parameterCount"],
        "seed": args.seed,
        "optimiser": "AdamW",
        "learningRate": args.learning_rate,
        "weightDecay": args.weight_decay,
        "gradientClipping": None,
        "plannedOptimiserUpdates": args.steps,
        "captureSteps": captures_requested,
        "batchOrder": "Seeded shuffle without replacement; include the short final batch; reshuffle after each complete pass",
        "trainingWindows": len(train_examples),
        "trainingValidTargetsPerCompletePass": total_train_targets,
        "validationWindows": len(validation_examples),
        "validationProtocol": baseline["validation"]["protocol"],
        "generation": model_config["generation"],
    }
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    payload = {
        "status": "Running",
        "experimentId": args.run_dir.name,
        "updatedAt": datetime.now(UTC).isoformat(),
        "purpose": "First controlled optimiser updates from the exact Lesson 5 checkpoint-zero model",
        "configuration": config,
        "progress": {"completedUpdates": 0, "plannedUpdates": args.steps, "validTargetsSeen": 0},
        "trace": [],
        "checkpoints": [],
        "limitations": [
            "This bounded 1,000-story development split is deliberately too small to establish a final TinyStories model.",
            "One seed and one fixed recipe show a learning journey, not uncertainty across independent training runs.",
            "The fixed learning rate and lack of gradient clipping are an intentionally simple baseline recipe for later controlled improvement.",
            "Training-batch loss is noisy and is not directly interchangeable with the complete held-out validation loss.",
            "Prompt samples use three fixed prompts and one seed per prompt, so they reveal examples rather than aggregate quality.",
            "MLX peak allocation is not total process memory or total Apple unified-memory use.",
        ],
        "nextExperiment": "Make checkpoints resumable by preserving optimiser and data-order state, then expose richer live monitoring.",
    }
    atomic_json_write(args.metrics, payload)

    mx.reset_peak_memory()
    started = time.perf_counter()
    valid_targets_seen = 0
    latest_train_loss: float | None = None
    latest_gradient_norm: float | None = None

    def capture(step: int) -> None:
        nonlocal payload
        model.eval()
        validation = evaluate_validation(model, validation_examples, model_config["batchSize"])
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
        if step == 0:
            checkpoint_path = args.baseline_checkpoint
        else:
            checkpoint_path = args.run_dir / f"checkpoint-{step:04d}.safetensors"
            model.save_weights(str(checkpoint_path))
        checkpoint = {
            "step": step,
            "capturedAt": datetime.now(UTC).isoformat(),
            "trainBatchLoss": latest_train_loss,
            "gradientL2Norm": latest_gradient_norm,
            "validation": validation,
            "validTargetsSeen": valid_targets_seen,
            "equivalentCompleteTrainingPasses": valid_targets_seen / total_train_targets,
            "checkpointFile": str(checkpoint_path),
            "checkpointBytes": checkpoint_path.stat().st_size,
            "checkpointSha256": checkpoint_sha256(checkpoint_path),
            "samples": samples,
        }
        payload["checkpoints"].append(checkpoint)
        payload["updatedAt"] = datetime.now(UTC).isoformat()
        payload["progress"] = {
            "completedUpdates": step,
            "plannedUpdates": args.steps,
            "validTargetsSeen": valid_targets_seen,
        }
        payload["memory"] = {
            "peakMlxAllocationBytes": mx.get_peak_memory(),
            "scope": "MLX allocator during this process, including training, complete validation, and prompt generation; not total process or unified-memory use",
        }
        atomic_json_write(args.metrics, payload)
        print(f"checkpoint step={step:4d} validation={validation['crossEntropyLoss']:.4f} targets_seen={valid_targets_seen:,}")
        model.train()

    capture(0)
    step_seconds: list[float] = []
    for step in range(1, args.steps + 1):
        indices = batch_order.next_indices()
        inputs, targets, loss_mask = materialise_batch(train_examples, indices)
        valid_in_batch = sum(train_examples[index]["validTargetCount"] for index in indices)
        began = time.perf_counter()
        loss, gradients = loss_and_grad(model, inputs, targets, loss_mask)
        mx.eval(loss, gradients)
        latest_gradient_norm = gradient_l2_norm(gradients)
        optimiser.update(model, gradients)
        mx.eval(model.parameters(), optimiser.state)
        duration = time.perf_counter() - began
        latest_train_loss = float(loss.item())
        valid_targets_seen += valid_in_batch
        step_seconds.append(duration)
        payload["trace"].append({
            "step": step,
            "trainBatchLoss": latest_train_loss,
            "gradientL2Norm": latest_gradient_norm,
            "validTargets": valid_in_batch,
            "seconds": duration,
        })
        if step in captures_requested:
            capture(step)

    elapsed = time.perf_counter() - started
    payload["status"] = "Complete"
    payload["updatedAt"] = datetime.now(UTC).isoformat()
    payload["progress"] = {
        "completedUpdates": args.steps,
        "plannedUpdates": args.steps,
        "validTargetsSeen": valid_targets_seen,
    }
    payload["trainingPerformance"] = {
        "elapsedSecondsIncludingEvaluationAndGeneration": elapsed,
        "updateSeconds": sum(step_seconds),
        "meanUpdateMilliseconds": sum(step_seconds) / len(step_seconds) * 1_000,
        "validTargetsPerSecondDuringUpdates": valid_targets_seen / sum(step_seconds),
        "peakMlxAllocationBytes": mx.get_peak_memory(),
    }
    payload["observedChange"] = {
        "startingValidationLoss": payload["checkpoints"][0]["validation"]["crossEntropyLoss"],
        "finalValidationLoss": payload["checkpoints"][-1]["validation"]["crossEntropyLoss"],
        "validationLossReduction": payload["checkpoints"][0]["validation"]["crossEntropyLoss"] - payload["checkpoints"][-1]["validation"]["crossEntropyLoss"],
        "finalPerplexity": math.exp(payload["checkpoints"][-1]["validation"]["crossEntropyLoss"]),
    }
    payload["environment"] = {
        "operatingSystem": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "mlx": mx.__version__,
        "tokenizers": __import__("tokenizers").__version__,
        "mlxDevice": str(mx.default_device()),
    }
    atomic_json_write(args.metrics, payload)
    print(json.dumps({
        "status": payload["status"],
        "updates": args.steps,
        "startingValidationLoss": payload["observedChange"]["startingValidationLoss"],
        "finalValidationLoss": payload["observedChange"]["finalValidationLoss"],
        "elapsedSeconds": elapsed,
    }, indent=2))


if __name__ == "__main__":
    main()
