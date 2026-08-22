"""Measure TinyStories scaling choices, then train one laptop-feasible winner.

Five fresh-start probes spend the same 100-update budget while changing one
dimension: width, depth, context, or training-story coverage. Architecture
candidates are filtered by predeclared memory and speed ceilings, then selected
by complete held-out loss. The selected architecture is trained from the same
random seed for 700 updates with the already-tested learning-rate recipe.
"""

from __future__ import annotations

import argparse
import json
import math
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
from mlx.utils import tree_flatten
from tokenizers import Tokenizer

from paths import RAW_DATA_DIR, local_result, local_run
from shakespeare_transformer import TinyTransformerLanguageModel, atomic_json_write
from tinystories_checkpoint_resume import StatefulBatchOrder, atomic_safetensors, file_sha256
from tinystories_first_pretraining import gradient_l2_norm, masked_training_loss
from tinystories_random_baseline import (
    DEFAULT_PROMPTS,
    architecture_parameter_count,
    evaluate_validation,
    random_completion,
)
from tinystories_sequence_batching import build_examples, materialise_batch, sha256
from tinystories_transition import load_jsonl


DEFAULT_DATA_DIR = RAW_DATA_DIR / "tinystories-sample"
DEFAULT_TOKENIZER = local_run("tinystories-tokenizer-001") / "tokenizer.json"
DEFAULT_BASELINE = local_result("tinystories-random-baseline.json")
DEFAULT_RECIPE = local_result("tinystories-training-recipe.json")
DEFAULT_METRICS = local_result("tinystories-scaling-budget.json")
DEFAULT_RUN_DIR = local_run("tinystories-scaling-budget-001")
PROBE_UPDATES = 100
LONG_UPDATES = 700
PROGRESS_INTERVAL = 10
SEED = 42
BATCH_SEED = 4242
LEARNING_RATE = 3e-4
END_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
MAX_PEAK_BYTES = 4 * 1024**3
MAX_MEAN_UPDATE_MS = 250.0
MINIMUM_IMPROVEMENT = 0.01

PROBES = (
    {"id": "reference", "label": "Reference · 256 wide × 6 blocks", "modelSize": 256, "attentionHeads": 8, "transformerBlocks": 6, "contextSize": 128, "trainingFraction": 1.0, "axis": "control"},
    {"id": "wider", "label": "Wider · 320 wide × 6 blocks", "modelSize": 320, "attentionHeads": 8, "transformerBlocks": 6, "contextSize": 128, "trainingFraction": 1.0, "axis": "width"},
    {"id": "deeper", "label": "Deeper · 256 wide × 8 blocks", "modelSize": 256, "attentionHeads": 8, "transformerBlocks": 8, "contextSize": 128, "trainingFraction": 1.0, "axis": "depth"},
    {"id": "longer-context", "label": "Longer context · 256 wide × 6 blocks", "modelSize": 256, "attentionHeads": 8, "transformerBlocks": 6, "contextSize": 256, "trainingFraction": 1.0, "axis": "context"},
    {"id": "half-data", "label": "Half the training stories · reference model", "modelSize": 256, "attentionHeads": 8, "transformerBlocks": 6, "contextSize": 128, "trainingFraction": 0.5, "axis": "data coverage"},
)


def schedule(step: int, total: int) -> float:
    """Use constant 3e-4, then the Lesson 8 cosine policy for the final 100."""
    if total <= PROBE_UPDATES or step <= total - PROBE_UPDATES:
        return LEARNING_RATE
    progress = (step - (total - PROBE_UPDATES) - 1) / max(PROBE_UPDATES - 1, 1)
    return END_LEARNING_RATE + 0.5 * (LEARNING_RATE - END_LEARNING_RATE) * (
        1 + math.cos(math.pi * progress)
    )


def chosen_rows(rows: list[dict], fraction: float) -> list[dict]:
    if fraction >= 1:
        return rows
    count = max(1, round(len(rows) * fraction))
    rng = random.Random(1701)
    return [rows[index] for index in sorted(rng.sample(range(len(rows)), count))]


def verified_inputs(protocol: dict, candidate: dict) -> tuple[Tokenizer, list[dict], list[dict], int]:
    config = protocol["configuration"]
    data_dir = Path(protocol["dataDir"])
    tokenizer_path = Path(protocol["tokenizer"])
    observed = {split: sha256(data_dir / f"{split}.jsonl") for split in ("train", "validation")}
    if observed != config["dataHashes"] or sha256(tokenizer_path) != config["tokenizerSha256"]:
        raise RuntimeError("Frozen data or tokenizer changed after the scaling protocol was written")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    all_train_rows = load_jsonl(data_dir / "train.jsonl")
    train_rows = chosen_rows(all_train_rows, candidate["trainingFraction"])
    validation_rows = load_jsonl(data_dir / "validation.jsonl")
    train_examples, _ = build_examples(tokenizer, train_rows, candidate["contextSize"])
    validation_examples, _ = build_examples(tokenizer, validation_rows, candidate["contextSize"])
    return tokenizer, train_examples, validation_examples, len(train_rows)


def make_model(candidate: dict, vocabulary_size: int) -> TinyTransformerLanguageModel:
    mx.random.seed(SEED)
    model = TinyTransformerLanguageModel(
        vocabulary_size,
        candidate["contextSize"],
        candidate["modelSize"],
        candidate["attentionHeads"],
        candidate["transformerBlocks"],
    )
    mx.eval(model.parameters())
    model.train()
    return model


def live_update(metrics_path: Path, phase: str, candidate: dict, step: int, total: int, loss: float) -> None:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["updatedAt"] = datetime.now(UTC).isoformat()
    payload["progress"] = {
        "phase": phase,
        "candidateId": candidate["id"],
        "update": step,
        "plannedUpdates": total,
        "latestTrainLoss": loss,
    }
    atomic_json_write(metrics_path, payload)


def train(protocol: dict, candidate: dict, total_updates: int, output_dir: Path, phase: str) -> dict:
    tokenizer, train_examples, validation_examples, training_rows = verified_inputs(protocol, candidate)
    model = make_model(candidate, tokenizer.get_vocab_size())
    optimiser = optim.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    order = StatefulBatchOrder(len(train_examples), protocol["configuration"]["batchSize"], BATCH_SEED)
    loss_and_grad = nn.value_and_grad(model, masked_training_loss)
    model.eval()
    start_validation = evaluate_validation(model, validation_examples, protocol["configuration"]["batchSize"])
    model.train()
    trace: list[dict] = []
    update_times: list[float] = []
    valid_targets_seen = 0
    mx.reset_peak_memory()
    started = time.perf_counter()
    for step in range(1, total_updates + 1):
        indices = order.next_indices()
        inputs, targets, loss_mask = materialise_batch(train_examples, indices)
        rate = schedule(step, total_updates)
        optimiser.learning_rate = rate
        began = time.perf_counter()
        loss, gradients = loss_and_grad(model, inputs, targets, loss_mask)
        mx.eval(loss, gradients)
        norm = gradient_l2_norm(gradients)
        optimiser.update(model, gradients)
        mx.eval(model.parameters(), optimiser.state)
        elapsed = time.perf_counter() - began
        update_times.append(elapsed)
        targets_seen = sum(train_examples[index]["validTargetCount"] for index in indices)
        valid_targets_seen += targets_seen
        trace.append({
            "update": step,
            "trainBatchLoss": float(loss.item()),
            "gradientL2Norm": norm,
            "learningRate": rate,
            "validTargets": targets_seen,
            "validTargetsSeen": valid_targets_seen,
            "updateSeconds": elapsed,
        })
        if step % PROGRESS_INTERVAL == 0 or step == total_updates:
            live_update(Path(protocol["metricsPath"]), phase, candidate, step, total_updates, float(loss.item()))
            print(f"{phase:12s} {candidate['id']:15s} update={step:3d} loss={float(loss.item()):.5f}", flush=True)
    model.eval()
    final_validation = evaluate_validation(model, validation_examples, protocol["configuration"]["batchSize"])
    samples = [
        random_completion(model, tokenizer, prompt, seed=2042 + index, maximum_tokens=64, temperature=0.9)
        for index, prompt in enumerate(DEFAULT_PROMPTS)
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    model_path = output_dir / "model.safetensors"
    model.save_weights(str(model_path))
    performance = {
        "elapsedSeconds": time.perf_counter() - started,
        "meanUpdateMilliseconds": statistics.mean(update_times) * 1000,
        "validTargetsPerSecond": valid_targets_seen / sum(update_times),
        "peakMlxAllocationBytes": mx.get_peak_memory(),
    }
    result = {
        "candidate": candidate,
        "parameterCount": architecture_parameter_count(tokenizer.get_vocab_size(), candidate["contextSize"], candidate["modelSize"], candidate["transformerBlocks"]),
        "trainingRows": training_rows,
        "trainingWindows": len(train_examples),
        "validationWindows": len(validation_examples),
        "startingValidation": start_validation,
        "finalValidation": final_validation,
        "validationLossChange": final_validation["crossEntropyLoss"] - start_validation["crossEntropyLoss"],
        "trace": trace,
        "samples": samples,
        "performance": performance,
        "model": {"path": str(model_path.resolve()), "bytes": model_path.stat().st_size, "sha256": file_sha256(model_path)},
    }
    atomic_json_write(output_dir / "result.json", result)
    if phase == "long-run":
        atomic_safetensors(output_dir / "optimizer.safetensors", tree_flatten(optimiser.state, destination={}))
        atomic_json_write(output_dir / "trainer-state.json", {
            "schemaVersion": 1,
            "absoluteStep": total_updates,
            "dataOrder": order.state_dict(),
            "validTargetsSeen": valid_targets_seen,
            "candidateId": candidate["id"],
        })
        files = {
            filename: {"bytes": (output_dir / filename).stat().st_size, "sha256": file_sha256(output_dir / filename)}
            for filename in ("model.safetensors", "optimizer.safetensors", "trainer-state.json")
        }
        manifest = {"schemaVersion": 1, "createdAt": datetime.now(UTC).isoformat(), "absoluteStep": total_updates, "candidateId": candidate["id"], "files": files, "complete": True}
        atomic_json_write(output_dir / "manifest.json", manifest)
        result["checkpoint"] = manifest
        atomic_json_write(output_dir / "result.json", result)
    return result


def run_worker(protocol_path: Path, mode: str, candidate_id: str) -> None:
    subprocess.run([
        sys.executable, str(Path(__file__).resolve()), "--worker", mode,
        "--candidate", candidate_id, "--protocol", str(protocol_path),
    ], cwd=Path(__file__).resolve().parents[1], check=True)


def worker(protocol_path: Path, mode: str, candidate_id: str) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    candidate = next(item for item in protocol["candidates"] if item["id"] == candidate_id)
    if mode == "probe":
        train(protocol, candidate, PROBE_UPDATES, Path(protocol["runDir"]) / "probes" / candidate_id, "probe")
    else:
        train(protocol, candidate, LONG_UPDATES, Path(protocol["runDir"]) / "selected-long-run" / "checkpoint-0700", "long-run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--worker", choices=("probe", "long"), help=argparse.SUPPRESS)
    parser.add_argument("--candidate", choices=tuple(item["id"] for item in PROBES), help=argparse.SUPPRESS)
    parser.add_argument("--protocol", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.protocol or not args.candidate:
            raise SystemExit("Internal worker requires protocol and candidate")
        worker(args.protocol, args.worker, args.candidate)
        return
    for required in (args.tokenizer, args.baseline, args.recipe, args.data_dir / "train.jsonl", args.data_dir / "validation.jsonl"):
        if not required.is_file():
            raise SystemExit(f"Missing prerequisite: {required}")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise SystemExit(f"Run directory already contains files: {args.run_dir}; move it aside before rerunning")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    if baseline.get("status") != "Complete" or recipe.get("status") != "Complete":
        raise SystemExit("Lessons 5 and 8 must have complete learner-local evidence")
    inherited = baseline["configuration"]
    data_hashes = {split: sha256(args.data_dir / f"{split}.jsonl") for split in ("train", "validation")}
    if data_hashes != inherited["dataHashes"] or sha256(args.tokenizer) != inherited["tokenizerSha256"]:
        raise SystemExit("Dataset or tokenizer differs from the frozen baseline contract")
    vocabulary_size = inherited["vocabularySize"]
    candidates = []
    for item in PROBES:
        candidate = dict(item)
        candidate["parameterCount"] = architecture_parameter_count(vocabulary_size, item["contextSize"], item["modelSize"], item["transformerBlocks"])
        candidates.append(candidate)
    configuration = {
        "datasetId": inherited["datasetId"], "dataHashes": data_hashes,
        "tokenizerSha256": inherited["tokenizerSha256"], "vocabularySize": vocabulary_size,
        "batchSize": inherited["batchSize"], "seed": SEED, "batchSeed": BATCH_SEED,
        "probeUpdates": PROBE_UPDATES, "longRunUpdates": LONG_UPDATES,
        "optimiser": "AdamW", "learningRate": LEARNING_RATE, "weightDecay": WEIGHT_DECAY,
        "resourceCeilings": {"peakMlxAllocationBytes": MAX_PEAK_BYTES, "meanUpdateMilliseconds": MAX_MEAN_UPDATE_MS},
        "selectionRule": {"metric": "complete held-out cross-entropy after 100 updates", "minimumImprovementOverReference": MINIMUM_IMPROVEMENT, "fallback": "retain reference architecture"},
        "recipeEvidence": recipe["experimentId"],
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.run_dir / "protocol.json"
    protocol = {
        "experimentId": args.run_dir.name, "createdAt": datetime.now(UTC).isoformat(),
        "protocolPath": str(protocol_path.resolve()), "runDir": str(args.run_dir.resolve()),
        "metricsPath": str(args.metrics.resolve()), "dataDir": str(args.data_dir.resolve()),
        "tokenizer": str(args.tokenizer.resolve()), "configuration": configuration,
        "candidates": candidates,
    }
    atomic_json_write(protocol_path, protocol)
    payload = {
        "status": "Running", "experimentId": args.run_dir.name, "updatedAt": datetime.now(UTC).isoformat(),
        "purpose": "Measure quality and laptop cost before committing to one scaled TinyStories run",
        "configuration": configuration, "candidates": candidates,
        "progress": {"phase": "Protocol frozen", "update": 0, "plannedUpdates": PROBE_UPDATES},
        "probes": [], "limitations": [
            "A 100-update learning curve is a feasibility screen, not a final architecture verdict.",
            "The 1,000-story development sample is deliberately bounded and cannot establish full-corpus scaling behaviour.",
            "Context 256 changes window boundaries; every real validation target remains graded, but batches are not tensor-identical to context 128.",
            "Half-data changes both story diversity and how often examples repeat during a fixed update budget.",
            "One seed and passive laptop cooling make small quality and timing differences uncertain.",
        ],
        "nextExperiment": "Evaluate the selected complete checkpoint across loss, prompts, repetition and failure cases.",
    }
    atomic_json_write(args.metrics, payload)
    overall_started = time.perf_counter()
    for candidate in candidates:
        run_worker(protocol_path, "probe", candidate["id"])
        result = json.loads((args.run_dir / "probes" / candidate["id"] / "result.json").read_text(encoding="utf-8"))
        payload = json.loads(args.metrics.read_text(encoding="utf-8"))
        payload["probes"].append(result)
        atomic_json_write(args.metrics, payload)
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    architecture_probes = [item for item in payload["probes"] if item["candidate"]["axis"] != "data coverage"]
    for item in architecture_probes:
        item["feasible"] = (
            item["performance"]["peakMlxAllocationBytes"] <= MAX_PEAK_BYTES
            and item["performance"]["meanUpdateMilliseconds"] <= MAX_MEAN_UPDATE_MS
        )
    reference = next(item for item in architecture_probes if item["candidate"]["id"] == "reference")
    feasible = [item for item in architecture_probes if item["feasible"]]
    lowest = min(feasible, key=lambda item: item["finalValidation"]["crossEntropyLoss"])
    improvement = reference["finalValidation"]["crossEntropyLoss"] - lowest["finalValidation"]["crossEntropyLoss"]
    selected = lowest if lowest is not reference and improvement >= MINIMUM_IMPROVEMENT else reference
    selected_id = selected["candidate"]["id"]
    payload["selection"] = {
        "selectedCandidateId": selected_id, "selectedLabel": selected["candidate"]["label"],
        "referenceValidationLoss": reference["finalValidation"]["crossEntropyLoss"],
        "selectedProbeValidationLoss": selected["finalValidation"]["crossEntropyLoss"],
        "improvementOverReference": reference["finalValidation"]["crossEntropyLoss"] - selected["finalValidation"]["crossEntropyLoss"],
        "reason": "lowest feasible probe cleared the predeclared improvement threshold" if selected is lowest and lowest is not reference else "no feasible alternative cleared the threshold; retained the reference architecture",
    }
    payload["progress"] = {"phase": "Training selected architecture", "candidateId": selected_id, "update": 0, "plannedUpdates": LONG_UPDATES}
    atomic_json_write(args.metrics, payload)
    run_worker(protocol_path, "long", selected_id)
    long_dir = args.run_dir / "selected-long-run" / "checkpoint-0700"
    long_result = json.loads((long_dir / "result.json").read_text(encoding="utf-8"))
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    payload.update({
        "status": "Complete", "updatedAt": datetime.now(UTC).isoformat(),
        "progress": {"phase": "Scaling experiment complete", "candidateId": selected_id, "update": LONG_UPDATES, "plannedUpdates": LONG_UPDATES, "latestTrainLoss": long_result["trace"][-1]["trainBatchLoss"]},
        "longRun": long_result,
        "selection": {**payload["selection"], "selectedCheckpointDirectory": str(long_dir.resolve())},
        "performance": {"wholeExperimentElapsedSeconds": time.perf_counter() - overall_started},
        "environment": {"python": platform.python_version(), "mlx": getattr(mx, "__version__", "unknown"), "device": str(mx.default_device()), "machine": platform.machine(), "platform": platform.platform()},
    })
    atomic_json_write(args.metrics, payload)
    atomic_json_write(args.run_dir / "selection.json", {"experimentId": payload["experimentId"], "selection": payload["selection"], "configuration": configuration})
    print(json.dumps({"status": payload["status"], **payload["selection"], "finalValidation": long_result["finalValidation"]["crossEntropyLoss"]}, indent=2))


if __name__ == "__main__":
    main()
