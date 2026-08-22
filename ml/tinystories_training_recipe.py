"""Compare TinyStories learning-rate policies from one resumable checkpoint.

Every candidate loads the same step-600 model, AdamW state, and data-order
cursor in a fresh process. Architecture, examples, batches, optimiser family,
weight decay, update budget, validation, prompts, and sampling seeds stay
fixed. Only the learning-rate policy changes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from tokenizers import Tokenizer

from paths import RAW_DATA_DIR, local_result, local_run
from shakespeare_transformer import TinyTransformerLanguageModel, atomic_json_write
from tinystories_checkpoint_resume import (
    StatefulBatchOrder,
    atomic_safetensors,
    file_sha256,
    indices_sha256,
)
from tinystories_first_pretraining import gradient_l2_norm, masked_training_loss
from tinystories_random_baseline import DEFAULT_PROMPTS, evaluate_validation, random_completion
from tinystories_sequence_batching import build_examples, materialise_batch, sha256
from tinystories_transition import load_jsonl


DEFAULT_DATA_DIR = RAW_DATA_DIR / "tinystories-sample"
DEFAULT_TOKENIZER = local_run("tinystories-tokenizer-001") / "tokenizer.json"
DEFAULT_SOURCE_EVIDENCE = local_result("tinystories-checkpoints-dashboard.json")
DEFAULT_SOURCE_CHECKPOINT = (
    local_run("tinystories-checkpoint-resume-001") / "resume" / "checkpoint-0600"
)
DEFAULT_METRICS = local_result("tinystories-training-recipe.json")
DEFAULT_RUN_DIR = local_run("tinystories-training-recipe-001")
SOURCE_STEP = 600
FINAL_STEP = 700
UPDATES = FINAL_STEP - SOURCE_STEP
PROGRESS_INTERVAL = 10
MINIMUM_CONTROL_IMPROVEMENT = 1e-3

CANDIDATES = (
    {
        "id": "constant-3e-4",
        "label": "Control · constant 3e-4",
        "policy": "constant",
        "startLearningRate": 3e-4,
        "endLearningRate": 3e-4,
    },
    {
        "id": "constant-1-5e-4",
        "label": "Lower constant · 1.5e-4",
        "policy": "constant",
        "startLearningRate": 1.5e-4,
        "endLearningRate": 1.5e-4,
    },
    {
        "id": "cosine-3e-4-to-3e-5",
        "label": "Cosine decay · 3e-4 → 3e-5",
        "policy": "cosine",
        "startLearningRate": 3e-4,
        "endLearningRate": 3e-5,
    },
)


def learning_rate(candidate: dict, segment_update: int) -> float:
    """Return the predeclared learning rate for update 1..UPDATES."""
    start = candidate["startLearningRate"]
    end = candidate["endLearningRate"]
    if candidate["policy"] == "constant":
        return start
    progress = (segment_update - 1) / max(UPDATES - 1, 1)
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * progress))


def verified_source(directory: Path, expected_step: int) -> tuple[dict, dict]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("The Lesson 7 source checkpoint has no manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("absoluteStep") != expected_step:
        raise RuntimeError("The Lesson 7 source checkpoint is incomplete or at the wrong step")
    for filename, expected in manifest["files"].items():
        path = directory / filename
        if not path.is_file() or file_sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Source checkpoint checksum failed: {filename}")
    trainer = json.loads((directory / "trainer-state.json").read_text(encoding="utf-8"))
    if trainer.get("absoluteStep") != expected_step:
        raise RuntimeError("Source trainer state disagrees with its manifest")
    return manifest, trainer


def load_source_model(config: dict, directory: Path) -> tuple[
    TinyTransformerLanguageModel, optim.AdamW, StatefulBatchOrder, dict
]:
    manifest, trainer = verified_source(directory, SOURCE_STEP)
    model = TinyTransformerLanguageModel(
        config["vocabularySize"],
        config["contextSize"],
        config["modelSize"],
        config["attentionHeads"],
        config["transformerBlocks"],
    )
    model.load_weights(str(directory / "model.safetensors"), strict=True)
    optimiser = optim.AdamW(
        learning_rate=config["sourceLearningRate"],
        weight_decay=config["weightDecay"],
    )
    optimiser.state = tree_unflatten(mx.load(str(directory / "optimizer.safetensors")))
    batch_order = StatefulBatchOrder.from_state_dict(trainer["dataOrder"])
    mx.eval(model.parameters(), optimiser.state)
    model.train()
    return model, optimiser, batch_order, manifest


def verified_data(protocol: dict) -> tuple[Tokenizer, list[dict], list[dict]]:
    config = protocol["configuration"]
    data_dir = Path(protocol["dataDir"])
    tokenizer_path = Path(protocol["tokenizer"])
    observed = {split: sha256(data_dir / f"{split}.jsonl") for split in ("train", "validation")}
    if observed != config["dataHashes"]:
        raise RuntimeError("Dataset hashes changed after the Lesson 8 protocol was frozen")
    if sha256(tokenizer_path) != config["tokenizerSha256"]:
        raise RuntimeError("Tokenizer hash changed after the Lesson 8 protocol was frozen")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    train_rows = load_jsonl(data_dir / "train.jsonl")
    validation_rows = load_jsonl(data_dir / "validation.jsonl")
    train_examples, _ = build_examples(tokenizer, train_rows, config["contextSize"])
    validation_examples, _ = build_examples(tokenizer, validation_rows, config["contextSize"])
    return tokenizer, train_examples, validation_examples


def update_live(metrics_path: Path, candidate: dict, trace: list[dict]) -> None:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["updatedAt"] = datetime.now(UTC).isoformat()
    payload["progress"] = {
        "phase": f"Training {candidate['label']}",
        "candidateId": candidate["id"],
        "candidateUpdate": trace[-1]["segmentUpdate"],
        "candidatePlannedUpdates": UPDATES,
        "latestTrainLoss": trace[-1]["trainBatchLoss"],
        "latestLearningRate": trace[-1]["learningRate"],
    }
    payload.setdefault("liveTraces", {})[candidate["id"]] = trace
    atomic_json_write(metrics_path, payload)


def save_checkpoint(
    directory: Path,
    model: TinyTransformerLanguageModel,
    optimiser: optim.AdamW,
    batch_order: StatefulBatchOrder,
    candidate: dict,
    protocol_path: Path,
    valid_targets_seen: int,
) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    model_path = directory / "model.safetensors"
    optimiser_path = directory / "optimizer.safetensors"
    trainer_path = directory / "trainer-state.json"
    model.save_weights(str(model_path))
    atomic_safetensors(optimiser_path, tree_flatten(optimiser.state, destination={}))
    atomic_json_write(trainer_path, {
        "schemaVersion": 1,
        "absoluteStep": FINAL_STEP,
        "sourceStep": SOURCE_STEP,
        "recipeCandidateId": candidate["id"],
        "validTargetsSeenThisExperiment": valid_targets_seen,
        "dataOrder": batch_order.state_dict(),
    })
    files = {
        name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for name, path in (
            ("model.safetensors", model_path),
            ("optimizer.safetensors", optimiser_path),
            ("trainer-state.json", trainer_path),
        )
    }
    manifest = {
        "schemaVersion": 1,
        "createdAt": datetime.now(UTC).isoformat(),
        "absoluteStep": FINAL_STEP,
        "candidateId": candidate["id"],
        "protocolSha256": file_sha256(protocol_path),
        "files": files,
        "complete": True,
    }
    atomic_json_write(directory / "manifest.json", manifest)
    return manifest


def worker(protocol_path: Path, candidate_id: str) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    config = protocol["configuration"]
    candidate = next(item for item in protocol["candidates"] if item["id"] == candidate_id)
    metrics_path = Path(protocol["metricsPath"])
    run_dir = Path(protocol["runDir"]) / candidate_id
    run_dir.mkdir(parents=True, exist_ok=False)
    tokenizer, train_examples, validation_examples = verified_data(protocol)
    model, optimiser, batch_order, source_manifest = load_source_model(
        config, Path(protocol["sourceCheckpoint"])
    )
    source_next_batch = indices_sha256(StatefulBatchOrder.from_state_dict(
        batch_order.state_dict()
    ).next_indices())

    model.eval()
    starting_validation = evaluate_validation(model, validation_examples, config["batchSize"])
    model.train()
    loss_and_grad = nn.value_and_grad(model, masked_training_loss)
    trace: list[dict] = []
    valid_targets_seen = 0
    update_seconds: list[float] = []
    mx.reset_peak_memory()
    started = time.perf_counter()

    for segment_update in range(1, UPDATES + 1):
        absolute_step = SOURCE_STEP + segment_update
        indices = batch_order.next_indices()
        inputs, targets, loss_mask = materialise_batch(train_examples, indices)
        rate = learning_rate(candidate, segment_update)
        optimiser.learning_rate = rate
        began = time.perf_counter()
        loss, gradients = loss_and_grad(model, inputs, targets, loss_mask)
        mx.eval(loss, gradients)
        gradient_norm = gradient_l2_norm(gradients)
        optimiser.update(model, gradients)
        mx.eval(model.parameters(), optimiser.state)
        elapsed = time.perf_counter() - began
        valid_targets = sum(train_examples[index]["validTargetCount"] for index in indices)
        valid_targets_seen += valid_targets
        update_seconds.append(elapsed)
        trace.append({
            "absoluteStep": absolute_step,
            "segmentUpdate": segment_update,
            "trainBatchLoss": float(loss.item()),
            "gradientL2Norm": gradient_norm,
            "learningRate": rate,
            "validTargets": valid_targets,
            "validTargetsSeen": valid_targets_seen,
            "batchIndicesSha256": indices_sha256(indices),
            "updateSeconds": elapsed,
        })
        if segment_update % PROGRESS_INTERVAL == 0 or segment_update == UPDATES:
            update_live(metrics_path, candidate, trace)
            print(
                f"{candidate_id:24s} update={segment_update:3d} "
                f"loss={float(loss.item()):.5f} lr={rate:.7f}"
            )

    model.eval()
    final_validation = evaluate_validation(model, validation_examples, config["batchSize"])
    samples = [
        random_completion(
            model,
            tokenizer,
            prompt,
            seed=config["generationSeed"] + index,
            maximum_tokens=config["generationTokens"],
            temperature=config["temperature"],
        )
        for index, prompt in enumerate(DEFAULT_PROMPTS)
    ]
    checkpoint = save_checkpoint(
        run_dir / f"checkpoint-{FINAL_STEP:04d}", model, optimiser, batch_order,
        candidate, protocol_path, valid_targets_seen,
    )
    result = {
        "candidate": candidate,
        "sourceManifestSha256": file_sha256(Path(protocol["sourceCheckpoint"]) / "manifest.json"),
        "sourceModelSha256": source_manifest["files"]["model.safetensors"]["sha256"],
        "sourceOptimizerSha256": source_manifest["files"]["optimizer.safetensors"]["sha256"],
        "sourceNextBatchSha256": source_next_batch,
        "startingValidation": starting_validation,
        "finalValidation": final_validation,
        "validationLossChange": final_validation["crossEntropyLoss"] - starting_validation["crossEntropyLoss"],
        "trace": trace,
        "samples": samples,
        "checkpoint": checkpoint,
        "performance": {
            "elapsedSeconds": time.perf_counter() - started,
            "trainingUpdateSeconds": sum(update_seconds),
            "meanUpdateMilliseconds": statistics.mean(update_seconds) * 1_000,
            "validTargetsPerSecond": valid_targets_seen / sum(update_seconds),
            "peakMlxAllocationBytes": mx.get_peak_memory(),
        },
    }
    atomic_json_write(run_dir / "result.json", result)


def run_worker(protocol_path: Path, candidate_id: str) -> None:
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", candidate_id, "--protocol", str(protocol_path)],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--source-evidence", type=Path, default=DEFAULT_SOURCE_EVIDENCE)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--worker", choices=tuple(item["id"] for item in CANDIDATES), help=argparse.SUPPRESS)
    parser.add_argument("--protocol", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if args.protocol is None:
            raise SystemExit("Internal worker requires --protocol")
        worker(args.protocol, args.worker)
        return
    for path in (args.tokenizer, args.source_evidence, args.source_checkpoint):
        if not path.exists():
            raise SystemExit(f"Missing Lesson 8 prerequisite: {path}")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise SystemExit(f"Run directory already contains files: {args.run_dir}; move it aside before rerunning")

    source = json.loads(args.source_evidence.read_text(encoding="utf-8"))
    if source.get("status") != "Complete" or not source.get("parity", {}).get("resumeValidated"):
        raise SystemExit("Lesson 7 must be complete with validated resume evidence")
    source_manifest, trainer = verified_source(args.source_checkpoint, SOURCE_STEP)
    inherited = source["configuration"]
    if source_manifest["files"]["model.safetensors"]["sha256"] != source["checkpointContract"]["resumedFinalCheckpoint"]["files"]["model.safetensors"]["sha256"]:
        raise SystemExit("Lesson 7 dashboard and source model checkpoint disagree")
    if trainer["dataOrder"]["batchSize"] != inherited["batchSize"]:
        raise SystemExit("Lesson 7 trainer state and batch configuration disagree")

    observed_hashes = {split: sha256(args.data_dir / f"{split}.jsonl") for split in ("train", "validation")}
    if observed_hashes != inherited["dataHashes"] or sha256(args.tokenizer) != inherited["tokenizerSha256"]:
        raise SystemExit("Dataset or tokenizer differs from the Lesson 7 contract")

    configuration = {
        "sourceExperimentId": source["experimentId"],
        "sourceAbsoluteStep": SOURCE_STEP,
        "sourceCheckpointManifestSha256": file_sha256(args.source_checkpoint / "manifest.json"),
        "datasetId": inherited["datasetId"],
        "dataHashes": inherited["dataHashes"],
        "tokenizerSha256": inherited["tokenizerSha256"],
        "vocabularySize": inherited["vocabularySize"],
        "contextSize": inherited["contextSize"],
        "batchSize": inherited["batchSize"],
        "modelSize": inherited["modelSize"],
        "attentionHeads": inherited["attentionHeads"],
        "transformerBlocks": inherited["transformerBlocks"],
        "parameterCount": inherited["parameterCount"],
        "optimiser": "AdamW",
        "sourceLearningRate": inherited["learningRate"],
        "weightDecay": inherited["weightDecay"],
        "updatesPerCandidate": UPDATES,
        "controlledVariables": [
            "source model weights", "source AdamW state", "source data-order cursor",
            "architecture", "dataset and tokenizer", "batch sequence", "weight decay",
            "update count", "validation protocol", "prompts and generation seeds",
        ],
        "changedVariable": "learning-rate policy",
        "selectionRule": {
            "metric": "complete held-out validation cross-entropy at step 700",
            "minimumImprovementOverControl": MINIMUM_CONTROL_IMPROVEMENT,
            "fallback": "Retain constant 3e-4 control if no candidate clears the threshold",
        },
        "lesson7CrossProcessValidationDifference": source["parity"]["finalValidationLossDifference"],
        "generationSeed": 2042,
        "generationTokens": 64,
        "temperature": 0.9,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.run_dir / "protocol.json"
    protocol = {
        "experimentId": args.run_dir.name,
        "createdAt": datetime.now(UTC).isoformat(),
        "protocolPath": str(protocol_path.resolve()),
        "runDir": str(args.run_dir.resolve()),
        "metricsPath": str(args.metrics.resolve()),
        "dataDir": str(args.data_dir.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "sourceCheckpoint": str(args.source_checkpoint.resolve()),
        "configuration": configuration,
        "candidates": list(CANDIDATES),
    }
    atomic_json_write(protocol_path, protocol)
    payload = {
        "status": "Running",
        "experimentId": args.run_dir.name,
        "updatedAt": datetime.now(UTC).isoformat(),
        "purpose": "Compare learning-rate policies from one complete step-600 training state",
        "configuration": configuration,
        "candidates": list(CANDIDATES),
        "progress": {"phase": "Protocol frozen", "candidateUpdate": 0, "candidatePlannedUpdates": UPDATES},
        "liveTraces": {},
        "variants": [],
        "limitations": [
            "The comparison uses one short 100-update continuation on one bounded 1,000-story training sample.",
            "All candidates reuse the same validation set for selection; Lesson 10 needs broader behaviour checks rather than treating this loss as final proof.",
            "Separate MLX worker processes can differ slightly in floating-point execution; Lesson 7 measured the relevant validation difference as numerical context.",
            "Only learning-rate policy changes. This lesson does not establish conclusions about batch size, clipping, model scale, or more data.",
            "Prompt continuations are fixed-seed examples and do not select the winning recipe.",
        ],
        "nextExperiment": "Use the selected complete checkpoint when benchmarking model and data scale within the laptop budget.",
    }
    atomic_json_write(args.metrics, payload)
    overall_started = time.perf_counter()

    for candidate in CANDIDATES:
        run_worker(protocol_path, candidate["id"])
        result = json.loads((args.run_dir / candidate["id"] / "result.json").read_text(encoding="utf-8"))
        payload = json.loads(args.metrics.read_text(encoding="utf-8"))
        payload["variants"].append(result)
        payload["updatedAt"] = datetime.now(UTC).isoformat()
        atomic_json_write(args.metrics, payload)

    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    variants = payload["variants"]
    expected_batches = [item["batchIndicesSha256"] for item in variants[0]["trace"]]
    if any([item["batchIndicesSha256"] for item in variant["trace"]] != expected_batches for variant in variants[1:]):
        raise RuntimeError("Candidate batch sequences differ; controlled comparison is invalid")
    starting_losses = [variant["startingValidation"]["crossEntropyLoss"] for variant in variants]
    if max(starting_losses) - min(starting_losses) > 1e-3:
        raise RuntimeError("Candidate source validations differ beyond the declared tolerance")

    control = variants[0]
    lowest = min(variants, key=lambda item: item["finalValidation"]["crossEntropyLoss"])
    improvement = control["finalValidation"]["crossEntropyLoss"] - lowest["finalValidation"]["crossEntropyLoss"]
    selected = lowest if improvement >= MINIMUM_CONTROL_IMPROVEMENT else control
    payload.update({
        "status": "Complete",
        "updatedAt": datetime.now(UTC).isoformat(),
        "progress": {
            "phase": "Comparison complete",
            "candidateUpdate": UPDATES,
            "candidatePlannedUpdates": UPDATES,
            "latestTrainLoss": selected["trace"][-1]["trainBatchLoss"],
            "latestLearningRate": selected["trace"][-1]["learningRate"],
        },
        "comparison": {
            "identicalBatchSequences": True,
            "comparedBatchesPerCandidate": UPDATES,
            "startingValidationLossRange": max(starting_losses) - min(starting_losses),
            "controlValidationLoss": control["finalValidation"]["crossEntropyLoss"],
            "lowestObservedValidationLoss": lowest["finalValidation"]["crossEntropyLoss"],
            "lowestObservedCandidateId": lowest["candidate"]["id"],
            "improvementOverControl": improvement,
            "minimumRequiredImprovement": MINIMUM_CONTROL_IMPROVEMENT,
        },
        "selection": {
            "selectedCandidateId": selected["candidate"]["id"],
            "selectedLabel": selected["candidate"]["label"],
            "reason": (
                "Lowest validation loss cleared the predeclared improvement threshold"
                if selected is lowest and lowest is not control
                else "No alternative cleared the predeclared improvement threshold; retained the control"
            ),
            "selectedCheckpointDirectory": str(
                args.run_dir / selected["candidate"]["id"] / f"checkpoint-{FINAL_STEP:04d}"
            ),
        },
        "performance": {
            "wholeExperimentElapsedSeconds": time.perf_counter() - overall_started,
            "peakCandidateMlxAllocationBytes": max(
                variant["performance"]["peakMlxAllocationBytes"] for variant in variants
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "mlx": getattr(mx, "__version__", "0.32.0"),
            "device": str(mx.default_device()),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
    })
    atomic_json_write(args.metrics, payload)
    atomic_json_write(args.run_dir / "selection.json", {
        "experimentId": payload["experimentId"],
        "selection": payload["selection"],
        "comparison": payload["comparison"],
        "configuration": configuration,
    })
    print(json.dumps({"status": payload["status"], **payload["selection"], **payload["comparison"]}, indent=2))


if __name__ == "__main__":
    main()
