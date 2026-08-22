"""Evaluate every saved tiny-transformer checkpoint with one frozen protocol.

This script never updates model weights. It measures training and validation
loss on repeatable batches, converts validation loss to perplexity, and runs
the same prompts through every checkpoint with fixed sampling seeds.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
from paths import available_run, local_result, local_run

from shakespeare_transformer import (
    DEFAULT_DATA,
    DEFAULT_RUN_DIR,
    TinyTransformerLanguageModel,
    atomic_json_write,
    estimate_loss,
    generate_from_prompt,
    load_data,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = local_result("shakespeare-evaluation.json")
DEFAULT_EVALUATION_DIR = local_run("shakespeare-evaluation-001")
CHECKPOINT_STEPS = (0, 1, 50, 250, 1_000, 2_000, 3_000)
PROMPTS = (
    "To be, or not to be",
    "My lord, the night is",
    "ROMEO:\n",
)


def repeated_loss(
    model: TinyTransformerLanguageModel,
    data: mx.array,
    *,
    base_seed: int,
    repeats: int,
    batches: int,
    batch_size: int,
    context_size: int,
) -> tuple[float, float, list[float]]:
    values = [
        estimate_loss(
            model,
            data,
            seed=base_seed + repeat,
            batches=batches,
            batch_size=batch_size,
            context_size=context_size,
        )
        for repeat in range(repeats)
    ]
    return statistics.mean(values), statistics.pstdev(values), values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=available_run("shakespeare-transformer-001"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batches-per-repeat", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--generated-characters", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    train_data, validation_data, vocabulary, char_to_id = load_data(args.data)
    protocol = {
        "repeats": args.repeats,
        "batchesPerRepeat": args.batches_per_repeat,
        "batchSize": args.batch_size,
        "contextSize": config["contextSize"],
        "predictionsPerRepeat": args.batches_per_repeat * args.batch_size * config["contextSize"],
        "temperature": args.temperature,
        "generatedCharacters": args.generated_characters,
        "baseSeed": args.seed,
        "prompts": list(PROMPTS),
    }
    payload = {
        "status": "Evaluating",
        "evaluationId": args.evaluation_dir.name,
        "sourceRunId": args.run_dir.name,
        "updatedAt": datetime.now(UTC).isoformat(),
        "protocol": protocol,
        "checkpoints": [],
    }
    atomic_json_write(args.output, payload)
    started_at = time.perf_counter()

    for step in CHECKPOINT_STEPS:
        model = TinyTransformerLanguageModel(
            len(vocabulary),
            config["contextSize"],
            config["modelSize"],
            config["attentionHeads"],
            config["transformerBlocks"],
        )
        model.load_weights(str(args.run_dir / f"checkpoint-{step:04d}.safetensors"))
        mx.eval(model.parameters())
        train_mean, train_std, train_repeats = repeated_loss(
            model,
            train_data,
            base_seed=args.seed + 100_000,
            repeats=args.repeats,
            batches=args.batches_per_repeat,
            batch_size=args.batch_size,
            context_size=config["contextSize"],
        )
        validation_mean, validation_std, validation_repeats = repeated_loss(
            model,
            validation_data,
            base_seed=args.seed + 200_000,
            repeats=args.repeats,
            batches=args.batches_per_repeat,
            batch_size=args.batch_size,
            context_size=config["contextSize"],
        )
        prompt_results = []
        for prompt_index, prompt in enumerate(PROMPTS):
            continuation = generate_from_prompt(
                model,
                prompt,
                char_to_id,
                vocabulary,
                seed=args.seed + 300_000 + prompt_index,
                characters=args.generated_characters,
                temperature=args.temperature,
            )
            prompt_results.append({"prompt": prompt, "continuation": continuation})
        payload["checkpoints"].append({
            "step": step,
            "trainLossMean": train_mean,
            "trainLossStd": train_std,
            "trainLossRepeats": train_repeats,
            "validationLossMean": validation_mean,
            "validationLossStd": validation_std,
            "validationLossRepeats": validation_repeats,
            "generalisationGap": validation_mean - train_mean,
            "perplexity": math.exp(validation_mean),
            "promptResults": prompt_results,
        })
        payload["updatedAt"] = datetime.now(UTC).isoformat()
        atomic_json_write(args.output, payload)
        print(
            f"step={step:4d} train={train_mean:.4f} "
            f"validation={validation_mean:.4f} ± {validation_std:.4f} "
            f"perplexity={math.exp(validation_mean):.2f}"
        )

    payload["status"] = "Complete"
    payload["elapsedSeconds"] = time.perf_counter() - started_at
    payload["updatedAt"] = datetime.now(UTC).isoformat()
    atomic_json_write(args.output, payload)
    args.evaluation_dir.mkdir(parents=True, exist_ok=True)
    (args.evaluation_dir / "config.json").write_text(
        json.dumps({
            "evaluationId": args.evaluation_dir.name,
            "sourceRunId": args.run_dir.name,
            "createdAt": payload["updatedAt"],
            "elapsedSeconds": payload["elapsedSeconds"],
            **protocol,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
