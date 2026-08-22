"""Prove that a TinyStories training checkpoint can resume exactly.

Lesson 6 saved model weights for inspection but not enough state to continue an
update sequence. This experiment begins a declared new segment at step 500,
then compares an uninterrupted continuation with a fresh process that loads the
step-550 save. Serialized state must round-trip exactly; subsequent floating-
point execution must preserve discrete batches and remain within declared
numerical tolerances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
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
from tinystories_first_pretraining import gradient_l2_norm, masked_training_loss
from tinystories_random_baseline import DEFAULT_PROMPTS, evaluate_validation, random_completion
from tinystories_sequence_batching import build_examples, materialise_batch, sha256
from tinystories_transition import load_jsonl


DEFAULT_DATA_DIR = RAW_DATA_DIR / "tinystories-sample"
DEFAULT_TOKENIZER = local_run("tinystories-tokenizer-001") / "tokenizer.json"
DEFAULT_BATCHING_EVIDENCE = local_result("tinystories-batching.json")
DEFAULT_PRETRAINING_EVIDENCE = local_result("tinystories-first-pretraining.json")
DEFAULT_METRICS = local_result("tinystories-checkpoints-dashboard.json")
DEFAULT_RUN_DIR = local_run("tinystories-checkpoint-resume-001")
SEGMENT_START = 500
INTERRUPTION_STEP = 550
SEGMENT_END = 600
PROGRESS_INTERVAL = 10
LOSS_TOLERANCE = 1e-3
GRADIENT_NORM_TOLERANCE = 1e-3
WEIGHT_TOLERANCE = 2e-2
OPTIMIZER_STATE_TOLERANCE = 1e-3
VALIDATION_LOSS_TOLERANCE = 1e-3


def nested_tuple(value):
    if isinstance(value, list):
        return tuple(nested_tuple(item) for item in value)
    return value


class StatefulBatchOrder:
    """A shuffled example dealer whose exact next position can be serialized."""

    def __init__(self, example_count: int, batch_size: int, seed: int) -> None:
        self.example_count = example_count
        self.batch_size = batch_size
        self.seed = seed
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

    def state_dict(self) -> dict:
        return {
            "exampleCount": self.example_count,
            "batchSize": self.batch_size,
            "seed": self.seed,
            "order": self.order,
            "position": self.position,
            "completedPasses": self.completed_passes,
            "pythonRandomState": self.rng.getstate(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "StatefulBatchOrder":
        instance = cls.__new__(cls)
        instance.example_count = int(state["exampleCount"])
        instance.batch_size = int(state["batchSize"])
        instance.seed = int(state["seed"])
        instance.order = [int(index) for index in state["order"]]
        instance.position = int(state["position"])
        instance.completed_passes = int(state["completedPasses"])
        if sorted(instance.order) != list(range(instance.example_count)):
            raise RuntimeError("Saved data order is not a complete example permutation")
        if not 0 <= instance.position <= instance.example_count:
            raise RuntimeError("Saved data-order cursor is outside the example list")
        instance.rng = random.Random()
        instance.rng.setstate(nested_tuple(state["pythonRandomState"]))
        return instance


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indices_sha256(indices: list[int]) -> str:
    raw = ",".join(str(index) for index in indices).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def build_model(config: dict, checkpoint: Path) -> TinyTransformerLanguageModel:
    model = TinyTransformerLanguageModel(
        config["vocabularySize"],
        config["contextSize"],
        config["modelSize"],
        config["attentionHeads"],
        config["transformerBlocks"],
    )
    model.load_weights(str(checkpoint), strict=True)
    model.train()
    mx.eval(model.parameters())
    return model


def build_optimiser(config: dict) -> optim.AdamW:
    return optim.AdamW(
        learning_rate=config["learningRate"],
        weight_decay=config["weightDecay"],
    )


def verified_data(protocol: dict) -> tuple[Tokenizer, list[dict], list[dict]]:
    data_dir = Path(protocol["dataDir"])
    tokenizer_path = Path(protocol["tokenizer"])
    observed = {
        split: sha256(data_dir / f"{split}.jsonl")
        for split in ("train", "validation")
    }
    if observed != protocol["configuration"]["dataHashes"]:
        raise RuntimeError("Dataset hashes changed after the Lesson 7 protocol was written")
    if sha256(tokenizer_path) != protocol["configuration"]["tokenizerSha256"]:
        raise RuntimeError("Tokenizer checksum changed after the Lesson 7 protocol was written")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    train_rows = load_jsonl(data_dir / "train.jsonl")
    validation_rows = load_jsonl(data_dir / "validation.jsonl")
    context = protocol["configuration"]["contextSize"]
    train_examples, _ = build_examples(tokenizer, train_rows, context)
    validation_examples, _ = build_examples(tokenizer, validation_rows, context)
    return tokenizer, train_examples, validation_examples


def update_live(metrics_path: Path, phase: str, trace: list[dict], current_step: int) -> None:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["updatedAt"] = datetime.now(UTC).isoformat()
    payload["progress"] = {
        "phase": phase,
        "absoluteStep": current_step,
        "segmentCompletedUpdates": current_step - SEGMENT_START,
        "segmentPlannedUpdates": SEGMENT_END - SEGMENT_START,
        "latestTrainLoss": trace[-1]["trainBatchLoss"] if trace else None,
        "latestGradientL2Norm": trace[-1]["gradientL2Norm"] if trace else None,
    }
    payload.setdefault("liveTraces", {})[phase] = trace
    atomic_json_write(metrics_path, payload)


def atomic_safetensors(path: Path, arrays: dict[str, mx.array]) -> None:
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.safetensors")
    mx.save_safetensors(str(temporary), arrays)
    os.replace(temporary, path)


def save_complete_checkpoint(
    directory: Path,
    model: TinyTransformerLanguageModel,
    optimiser: optim.AdamW,
    batch_order: StatefulBatchOrder,
    *,
    absolute_step: int,
    valid_targets_seen: int,
    protocol: dict,
) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    model_path = directory / "model.safetensors"
    optimiser_path = directory / "optimizer.safetensors"
    trainer_path = directory / "trainer-state.json"
    model.save_weights(str(model_path))
    optimiser_arrays = tree_flatten(optimiser.state, destination={})
    atomic_safetensors(optimiser_path, optimiser_arrays)
    trainer_state = {
        "schemaVersion": 1,
        "absoluteStep": absolute_step,
        "segmentStartStep": SEGMENT_START,
        "validTargetsSeenThisSegment": valid_targets_seen,
        "dataOrder": batch_order.state_dict(),
    }
    atomic_json_write(trainer_path, trainer_state)
    files = {
        name: {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for name, path in (
            ("model.safetensors", model_path),
            ("optimizer.safetensors", optimiser_path),
            ("trainer-state.json", trainer_path),
        )
    }
    manifest = {
        "schemaVersion": 1,
        "createdAt": datetime.now(UTC).isoformat(),
        "absoluteStep": absolute_step,
        "protocolSha256": file_sha256(Path(protocol["protocolPath"])),
        "files": files,
        "complete": True,
    }
    atomic_json_write(directory / "manifest.json", manifest)
    return manifest


def load_complete_checkpoint(
    directory: Path,
    config: dict,
    protocol: dict,
) -> tuple[TinyTransformerLanguageModel, optim.AdamW, StatefulBatchOrder, dict, dict]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Checkpoint has no completion manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("schemaVersion") != 1:
        raise RuntimeError("Checkpoint manifest is incomplete or unsupported")
    if manifest["protocolSha256"] != file_sha256(Path(protocol["protocolPath"])):
        raise RuntimeError("Checkpoint belongs to a different training protocol")
    for filename, expected in manifest["files"].items():
        path = directory / filename
        if not path.is_file() or file_sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Checkpoint file failed checksum verification: {filename}")

    model = build_model(config, directory / "model.safetensors")
    optimiser = build_optimiser(config)
    optimiser.state = tree_unflatten(mx.load(str(directory / "optimizer.safetensors")))
    mx.eval(optimiser.state)
    trainer = json.loads((directory / "trainer-state.json").read_text(encoding="utf-8"))
    batch_order = StatefulBatchOrder.from_state_dict(trainer["dataOrder"])
    return model, optimiser, batch_order, trainer, manifest


def preview_next_batch_hash(batch_order: StatefulBatchOrder) -> str:
    clone = StatefulBatchOrder.from_state_dict(batch_order.state_dict())
    return indices_sha256(clone.next_indices())


def maximum_tree_difference(left: dict, right: dict) -> float:
    left_flat = dict(tree_flatten(left))
    right_flat = dict(tree_flatten(right))
    if set(left_flat) != set(right_flat):
        raise RuntimeError("State trees contain different tensor names")
    differences = []
    for name in sorted(left_flat):
        if left_flat[name].shape != right_flat[name].shape:
            raise RuntimeError(f"State tree shape differs for {name}")
        differences.append(mx.max(mx.abs(left_flat[name] - right_flat[name])))
    mx.eval(differences)
    return max((float(value.item()) for value in differences), default=0.0)


def train_updates(
    model: TinyTransformerLanguageModel,
    optimiser: optim.AdamW,
    batch_order: StatefulBatchOrder,
    train_examples: list[dict],
    *,
    start_step: int,
    end_step: int,
    valid_targets_seen: int,
    phase: str,
    trace: list[dict],
    metrics_path: Path,
) -> tuple[list[dict], int, dict]:
    loss_and_grad = nn.value_and_grad(model, masked_training_loss)
    update_seconds: list[float] = []
    mx.reset_peak_memory()
    for absolute_step in range(start_step + 1, end_step + 1):
        indices = batch_order.next_indices()
        inputs, targets, loss_mask = materialise_batch(train_examples, indices)
        valid_targets = sum(train_examples[index]["validTargetCount"] for index in indices)
        began = time.perf_counter()
        loss, gradients = loss_and_grad(model, inputs, targets, loss_mask)
        mx.eval(loss, gradients)
        gradient_norm = gradient_l2_norm(gradients)
        optimiser.update(model, gradients)
        mx.eval(model.parameters(), optimiser.state)
        elapsed = time.perf_counter() - began
        valid_targets_seen += valid_targets
        update_seconds.append(elapsed)
        trace.append({
            "absoluteStep": absolute_step,
            "segmentUpdate": absolute_step - SEGMENT_START,
            "trainBatchLoss": float(loss.item()),
            "gradientL2Norm": gradient_norm,
            "validTargets": valid_targets,
            "validTargetsSeenThisSegment": valid_targets_seen,
            "batchIndicesSha256": indices_sha256(indices),
            "updateSeconds": elapsed,
        })
        if absolute_step % PROGRESS_INTERVAL == 0 or absolute_step == end_step:
            update_live(metrics_path, phase, trace, absolute_step)
            print(f"{phase:11s} step={absolute_step} loss={float(loss.item()):.5f} grad={gradient_norm:.4f}")
    performance = {
        "updates": end_step - start_step,
        "updateSeconds": sum(update_seconds),
        "meanUpdateMilliseconds": statistics.mean(update_seconds) * 1_000,
        "validTargetsPerSecond": sum(item["validTargets"] for item in trace[-len(update_seconds):]) / sum(update_seconds),
        "peakMlxAllocationBytes": mx.get_peak_memory(),
    }
    return trace, valid_targets_seen, performance


def worker(protocol_path: Path, phase: str) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    config = protocol["configuration"]
    metrics_path = Path(protocol["metricsPath"])
    run_dir = Path(protocol["runDir"])
    tokenizer, train_examples, validation_examples = verified_data(protocol)
    source_checkpoint = Path(protocol["sourceCheckpoint"])
    trace: list[dict] = []
    valid_targets_seen = 0
    started = time.perf_counter()

    if phase == "control":
        model = build_model(config, source_checkpoint)
        optimiser = build_optimiser(config)
        optimiser.init(model.trainable_parameters())
        mx.eval(optimiser.state)
        batch_order = StatefulBatchOrder(len(train_examples), config["batchSize"], config["dataOrderSeed"])
        end_step = INTERRUPTION_STEP
    elif phase == "resume":
        checkpoint_dir = run_dir / "control" / f"checkpoint-{INTERRUPTION_STEP:04d}"
        model, optimiser, batch_order, trainer, _ = load_complete_checkpoint(checkpoint_dir, config, protocol)
        trace = [
            item for item in json.loads((run_dir / "control" / "trace.json").read_text(encoding="utf-8"))
            if item["absoluteStep"] <= INTERRUPTION_STEP
        ]
        valid_targets_seen = int(trainer["validTargetsSeenThisSegment"])
        if int(trainer["absoluteStep"]) != INTERRUPTION_STEP:
            raise RuntimeError("Resume checkpoint step does not match the protocol")
        end_step = SEGMENT_END
    else:
        raise RuntimeError(f"Unknown worker phase: {phase}")

    start_step = SEGMENT_START if phase == "control" else INTERRUPTION_STEP
    trace, valid_targets_seen, performance = train_updates(
        model,
        optimiser,
        batch_order,
        train_examples,
        start_step=start_step,
        end_step=end_step,
        valid_targets_seen=valid_targets_seen,
        phase=phase,
        trace=trace,
        metrics_path=metrics_path,
    )

    branch_dir = run_dir / phase
    branch_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_write(branch_dir / "trace.json", trace)
    checkpoint_dir = branch_dir / f"checkpoint-{end_step:04d}"
    manifest = save_complete_checkpoint(
        checkpoint_dir,
        model,
        optimiser,
        batch_order,
        absolute_step=end_step,
        valid_targets_seen=valid_targets_seen,
        protocol=protocol,
    )

    round_trip = None
    if phase == "control" and end_step == INTERRUPTION_STEP:
        restored_model, restored_optimiser, restored_order, restored_trainer, _ = load_complete_checkpoint(
            checkpoint_dir, config, protocol
        )
        round_trip = {
            "maximumModelTensorDifference": maximum_tree_difference(model.parameters(), restored_model.parameters()),
            "maximumOptimizerTensorDifference": maximum_tree_difference(optimiser.state, restored_optimiser.state),
            "dataOrderStateExact": json.loads(json.dumps(batch_order.state_dict())) == json.loads(json.dumps(restored_order.state_dict())),
            "trainerStepExact": restored_trainer["absoluteStep"] == end_step,
            "nextBatchIndicesMatch": preview_next_batch_hash(batch_order) == preview_next_batch_hash(restored_order),
        }

        # Continue the original in-memory control after saving; the lower route
        # will reload the same checkpoint in a fresh process.
        trace, valid_targets_seen, continuation_performance = train_updates(
            model,
            optimiser,
            batch_order,
            train_examples,
            start_step=INTERRUPTION_STEP,
            end_step=SEGMENT_END,
            valid_targets_seen=valid_targets_seen,
            phase=phase,
            trace=trace,
            metrics_path=metrics_path,
        )
        atomic_json_write(branch_dir / "trace.json", trace)
        final_checkpoint_dir = branch_dir / f"checkpoint-{SEGMENT_END:04d}"
        final_manifest = save_complete_checkpoint(
            final_checkpoint_dir,
            model,
            optimiser,
            batch_order,
            absolute_step=SEGMENT_END,
            valid_targets_seen=valid_targets_seen,
            protocol=protocol,
        )
        performance = {
            "updates": performance["updates"] + continuation_performance["updates"],
            "updateSeconds": performance["updateSeconds"] + continuation_performance["updateSeconds"],
            "meanUpdateMilliseconds": (
                performance["updateSeconds"] + continuation_performance["updateSeconds"]
            ) / (SEGMENT_END - SEGMENT_START) * 1_000,
            "validTargetsPerSecond": valid_targets_seen / (
                performance["updateSeconds"] + continuation_performance["updateSeconds"]
            ),
            "peakMlxAllocationBytes": max(
                performance["peakMlxAllocationBytes"], continuation_performance["peakMlxAllocationBytes"]
            ),
        }
        end_step = SEGMENT_END
        checkpoint_dir = final_checkpoint_dir
        manifest = final_manifest

    result = {
        "phase": phase,
        "absoluteStep": end_step,
        "elapsedSeconds": time.perf_counter() - started,
        "performance": performance,
        "validTargetsSeenThisSegment": valid_targets_seen,
        "nextBatchIndicesSha256": preview_next_batch_hash(batch_order),
        "checkpointDirectory": str(checkpoint_dir),
        "manifest": manifest,
        "checkpointRoundTripAtStep550": round_trip,
    }
    if phase in ("control", "resume"):
        model.eval()
        result["validation"] = evaluate_validation(model, validation_examples, config["batchSize"])
    if phase == "resume":
        result["samples"] = [
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
    atomic_json_write(branch_dir / f"{phase}-result.json", result)


def maximum_safetensor_difference(left_path: Path, right_path: Path) -> float:
    left = mx.load(str(left_path))
    right = mx.load(str(right_path))
    if set(left) != set(right):
        raise RuntimeError("Safetensor names differ between control and resumed checkpoints")
    differences = []
    for name in sorted(left):
        if left[name].shape != right[name].shape:
            raise RuntimeError(f"Safetensor shape differs for {name}")
        differences.append(mx.max(mx.abs(left[name] - right[name])))
    mx.eval(differences)
    return max((float(value.item()) for value in differences), default=0.0)


def trace_parity(control: list[dict], resumed: list[dict]) -> dict:
    if len(control) != len(resumed):
        raise RuntimeError("Control and resumed traces have different lengths")
    maximum_loss_difference = 0.0
    maximum_gradient_difference = 0.0
    mismatched_batches = 0
    mismatched_targets = 0
    first_mismatch = None
    for left, right in zip(control, resumed, strict=True):
        maximum_loss_difference = max(maximum_loss_difference, abs(left["trainBatchLoss"] - right["trainBatchLoss"]))
        maximum_gradient_difference = max(maximum_gradient_difference, abs(left["gradientL2Norm"] - right["gradientL2Norm"]))
        if left["batchIndicesSha256"] != right["batchIndicesSha256"]:
            mismatched_batches += 1
        if left["validTargets"] != right["validTargets"]:
            mismatched_targets += 1
        if first_mismatch is None and any((
            left["trainBatchLoss"] != right["trainBatchLoss"],
            left["gradientL2Norm"] != right["gradientL2Norm"],
            left["batchIndicesSha256"] != right["batchIndicesSha256"],
            left["validTargets"] != right["validTargets"],
        )):
            first_mismatch = left["absoluteStep"]
    return {
        "comparedUpdates": len(control),
        "maximumTrainLossDifference": maximum_loss_difference,
        "maximumGradientNormDifference": maximum_gradient_difference,
        "mismatchedBatchCount": mismatched_batches,
        "mismatchedValidTargetCount": mismatched_targets,
        "firstMismatchedStep": first_mismatch,
    }


def run_worker(phase: str, protocol_path: Path) -> None:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", phase, "--protocol", str(protocol_path)]
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--batching-evidence", type=Path, default=DEFAULT_BATCHING_EVIDENCE)
    parser.add_argument("--pretraining-evidence", type=Path, default=DEFAULT_PRETRAINING_EVIDENCE)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--worker", choices=("control", "resume"), help=argparse.SUPPRESS)
    parser.add_argument("--protocol", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        if args.protocol is None:
            raise SystemExit("Internal worker requires --protocol")
        worker(args.protocol, args.worker)
        return

    required = (args.tokenizer, args.batching_evidence, args.pretraining_evidence)
    if any(not path.is_file() for path in required):
        raise SystemExit("Lessons 4 and 6 learner-local artifacts are required")
    batching = json.loads(args.batching_evidence.read_text(encoding="utf-8"))
    pretraining = json.loads(args.pretraining_evidence.read_text(encoding="utf-8"))
    if batching.get("status") != "Complete" or pretraining.get("status") != "Complete":
        raise SystemExit("Lesson 4 batching and Lesson 6 pretraining must be complete")
    source = next((item for item in pretraining["checkpoints"] if item["step"] == SEGMENT_START), None)
    if source is None:
        raise SystemExit("Lesson 6 has no step-500 checkpoint")
    source_checkpoint = Path(source["checkpointFile"])
    if not source_checkpoint.is_file() or file_sha256(source_checkpoint) != source["checkpointSha256"]:
        raise SystemExit("Lesson 6 step-500 checkpoint failed its recorded checksum")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise SystemExit(f"Run directory already contains files: {args.run_dir}; move it aside before rerunning")

    inherited = pretraining["configuration"]
    configuration = {
        "sourceRunId": pretraining["experimentId"],
        "sourceAbsoluteStep": SEGMENT_START,
        "sourceCheckpointSha256": source["checkpointSha256"],
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
        "learningRate": inherited["learningRate"],
        "weightDecay": inherited["weightDecay"],
        "optimizerStateAtSegmentStart": "Freshly initialised; Lesson 6 did not save optimiser state",
        "dataOrderSeed": 4242,
        "segmentStartStep": SEGMENT_START,
        "interruptionStep": INTERRUPTION_STEP,
        "segmentEndStep": SEGMENT_END,
        "checkpointSchemaVersion": 1,
        "progressInterval": PROGRESS_INTERVAL,
        "numericalTolerances": {
            "trainBatchLoss": LOSS_TOLERANCE,
            "gradientL2Norm": GRADIENT_NORM_TOLERANCE,
            "finalWeight": WEIGHT_TOLERANCE,
            "finalOptimizerState": OPTIMIZER_STATE_TOLERANCE,
            "finalValidationLoss": VALIDATION_LOSS_TOLERANCE,
        },
        "generationSeed": inherited["seed"] + 1_000,
        "generationTokens": inherited["generation"]["maximumTokens"],
        "temperature": inherited["generation"]["temperature"],
    }
    args.run_dir.mkdir(parents=True)
    protocol_path = args.run_dir / "protocol.json"
    protocol = {
        "experimentId": args.run_dir.name,
        "createdAt": datetime.now(UTC).isoformat(),
        "protocolPath": str(protocol_path.resolve()),
        "runDir": str(args.run_dir.resolve()),
        "metricsPath": str(args.metrics.resolve()),
        "dataDir": str(args.data_dir.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "sourceCheckpoint": str(source_checkpoint.resolve()),
        "configuration": configuration,
    }
    atomic_json_write(protocol_path, protocol)
    initial_payload = {
        "status": "Running",
        "experimentId": args.run_dir.name,
        "updatedAt": datetime.now(UTC).isoformat(),
        "purpose": "Prove exact disk-backed continuation against an uninterrupted control",
        "configuration": configuration,
        "progress": {
            "phase": "Preparing uninterrupted control",
            "absoluteStep": SEGMENT_START,
            "segmentCompletedUpdates": 0,
            "segmentPlannedUpdates": SEGMENT_END - SEGMENT_START,
            "latestTrainLoss": None,
            "latestGradientL2Norm": None,
        },
        "liveTraces": {},
        "limitations": [
            "The Lesson 7 segment starts with fresh AdamW state because Lesson 6 saved weights only; it does not recreate the missing step-500 optimiser history.",
            "Checkpoint bytes round-trip exactly, but separate MLX processes can choose floating-point reduction paths that produce small numerical differences even with matching batches.",
            "The simulated interruption occurs after a completed checkpoint save, not during a partially written tensor file.",
            "Resumability prevents lost work but does not improve model quality by itself.",
            "The extra control branch doubles training computation for this teaching proof and is not required for every future production resume.",
        ],
        "nextExperiment": "Use the reliable checkpoint path to compare training-recipe changes under matched restart and monitoring rules.",
    }
    atomic_json_write(args.metrics, initial_payload)
    started = time.perf_counter()

    run_worker("control", protocol_path)
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    payload["progress"]["phase"] = "Reconstructing from disk in a fresh process"
    atomic_json_write(args.metrics, payload)
    run_worker("resume", protocol_path)

    control_dir = args.run_dir / "control"
    resumed_dir = args.run_dir / "resume"
    control_trace = json.loads((control_dir / "trace.json").read_text(encoding="utf-8"))
    resumed_trace = json.loads((resumed_dir / "trace.json").read_text(encoding="utf-8"))
    control_result = json.loads((control_dir / "control-result.json").read_text(encoding="utf-8"))
    resumed_result = json.loads((resumed_dir / "resume-result.json").read_text(encoding="utf-8"))
    # The first 50 records establish the shared prefix. Only the records after
    # the step-550 reload are an independently executed resume comparison.
    trace_check = trace_parity(
        [record for record in control_trace if record["absoluteStep"] > SAVE_STEP],
        [record for record in resumed_trace if record["absoluteStep"] > SAVE_STEP],
    )
    control_checkpoint = Path(control_result["checkpointDirectory"])
    resumed_checkpoint = Path(resumed_result["checkpointDirectory"])
    weight_difference = maximum_safetensor_difference(
        control_checkpoint / "model.safetensors",
        resumed_checkpoint / "model.safetensors",
    )
    optimiser_difference = maximum_safetensor_difference(
        control_checkpoint / "optimizer.safetensors",
        resumed_checkpoint / "optimizer.safetensors",
    )
    validation_difference = abs(
        control_result["validation"]["crossEntropyLoss"]
        - resumed_result["validation"]["crossEntropyLoss"]
    )
    bitwise_exact = all((
        trace_check["maximumTrainLossDifference"] == 0.0,
        trace_check["maximumGradientNormDifference"] == 0.0,
        trace_check["mismatchedBatchCount"] == 0,
        trace_check["mismatchedValidTargetCount"] == 0,
        weight_difference == 0.0,
        optimiser_difference == 0.0,
        validation_difference == 0.0,
        control_result["nextBatchIndicesSha256"] == resumed_result["nextBatchIndicesSha256"],
    ))
    checkpoint_round_trip = control_result["checkpointRoundTripAtStep550"]
    exact_state_round_trip = all((
        checkpoint_round_trip["maximumModelTensorDifference"] == 0.0,
        checkpoint_round_trip["maximumOptimizerTensorDifference"] == 0.0,
        checkpoint_round_trip["dataOrderStateExact"],
        checkpoint_round_trip["trainerStepExact"],
        checkpoint_round_trip["nextBatchIndicesMatch"],
    ))
    within_tolerance = all((
        trace_check["maximumTrainLossDifference"] <= LOSS_TOLERANCE,
        trace_check["maximumGradientNormDifference"] <= GRADIENT_NORM_TOLERANCE,
        trace_check["mismatchedBatchCount"] == 0,
        trace_check["mismatchedValidTargetCount"] == 0,
        weight_difference <= WEIGHT_TOLERANCE,
        optimiser_difference <= OPTIMIZER_STATE_TOLERANCE,
        validation_difference <= VALIDATION_LOSS_TOLERANCE,
        control_result["nextBatchIndicesSha256"] == resumed_result["nextBatchIndicesSha256"],
    ))
    resume_validated = exact_state_round_trip and within_tolerance
    if not resume_validated:
        raise RuntimeError("Interrupted/resumed path exceeded the declared continuation tolerances")

    final_payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    final_payload.update({
        "status": "Complete",
        "updatedAt": datetime.now(UTC).isoformat(),
        "progress": {
            "phase": "Parity proven",
            "absoluteStep": SEGMENT_END,
            "segmentCompletedUpdates": SEGMENT_END - SEGMENT_START,
            "segmentPlannedUpdates": SEGMENT_END - SEGMENT_START,
            "latestTrainLoss": resumed_trace[-1]["trainBatchLoss"],
            "latestGradientL2Norm": resumed_trace[-1]["gradientL2Norm"],
        },
        "checkpointContract": {
            "requiredFiles": ["model.safetensors", "optimizer.safetensors", "trainer-state.json", "manifest.json"],
            "manifestWrittenLast": True,
            "checksumVerificationBeforeLoad": True,
            "interruptionCheckpoint": {
                "checkpointDirectory": str(args.run_dir / "control" / f"checkpoint-{INTERRUPTION_STEP:04d}"),
                "manifest": json.loads((args.run_dir / "control" / f"checkpoint-{INTERRUPTION_STEP:04d}" / "manifest.json").read_text(encoding="utf-8")),
                "validTargetsSeenThisSegment": control_trace[INTERRUPTION_STEP - SEGMENT_START - 1]["validTargetsSeenThisSegment"],
            },
            "resumedFinalCheckpoint": resumed_result["manifest"],
            "exactStateRoundTrip": checkpoint_round_trip,
        },
        "parity": {
            **trace_check,
            "maximumFinalWeightDifference": weight_difference,
            "maximumFinalOptimizerStateDifference": optimiser_difference,
            "finalValidationLossDifference": validation_difference,
            "nextBatchIndicesMatch": control_result["nextBatchIndicesSha256"] == resumed_result["nextBatchIndicesSha256"],
            "bitwiseExactAcrossProcesses": bitwise_exact,
            "withinDeclaredTolerance": within_tolerance,
            "resumeValidated": resume_validated,
        },
        "validation": {
            "lesson6Step500Loss": source["validation"]["crossEntropyLoss"],
            "controlStep600": control_result["validation"],
            "resumedStep600": resumed_result["validation"],
        },
        "samples": resumed_result["samples"],
        "performance": {
            "control": control_result["performance"],
            "afterResume": resumed_result["performance"],
            "wholeExperimentElapsedSeconds": time.perf_counter() - started,
        },
        "environment": {
            "operatingSystem": platform.platform(),
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "mlx": mx.__version__,
            "tokenizers": __import__("tokenizers").__version__,
            "mlxDevice": str(mx.default_device()),
            "workerProcessBoundary": True,
        },
    })
    atomic_json_write(args.metrics, final_payload)
    print(json.dumps({
        "status": final_payload["status"],
        "resumeValidated": resume_validated,
        "bitwiseExactAcrossProcesses": bitwise_exact,
        "step500ValidationLoss": source["validation"]["crossEntropyLoss"],
        "step600ValidationLoss": resumed_result["validation"]["crossEntropyLoss"],
        "maximumFinalWeightDifference": weight_difference,
    }, indent=2))


if __name__ == "__main__":
    main()
