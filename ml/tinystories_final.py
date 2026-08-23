"""Confirm the selected TinyStories recipe across independent training seeds.

Seed 42 reuses the exact clean Lesson 9 checkpoint. Seeds 43 and 44 rebuild
the same architecture from random weights and repeat all 700 updates with
independent model and batch-order seeds. Complete validation and the frozen
Lesson 10 behaviour protocol are then applied to every checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
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
from tokenizers import Tokenizer

from paths import RAW_DATA_DIR, ROOT, local_result, local_run
from shakespeare_transformer import TinyTransformerLanguageModel, atomic_json_write
from tinystories_checkpoint_resume import StatefulBatchOrder, file_sha256
from tinystories_first_pretraining import gradient_l2_norm, masked_training_loss
from tinystories_random_baseline import DEFAULT_PROMPTS, evaluate_validation, random_completion
from tinystories_scaling_budget import schedule
from tinystories_sequence_batching import build_examples, materialise_batch, sha256
from tinystories_story_evaluation import evaluate_candidate, training_overlap_index
from tinystories_transition import load_jsonl


DEFAULT_DATA_DIR = RAW_DATA_DIR / "tinystories-sample"
DEFAULT_TOKENIZER = local_run("tinystories-tokenizer-001") / "tokenizer.json"
DEFAULT_BASELINE = local_result("tinystories-random-baseline.json")
DEFAULT_SCALING = local_result("tinystories-scaling-budget.json")
DEFAULT_STORY_EVALUATION = local_result("tinystories-story-evaluation.json")
DEFAULT_TOKENIZER_EVIDENCE = local_result("tinystories-tokenizer.json")
DEFAULT_BATCHING = local_result("tinystories-batching.json")
DEFAULT_METRICS = local_result("tinystories-final.json")
DEFAULT_RUN_DIR = local_run("tinystories-final-001")
DEFAULT_SHAKESPEARE_FINAL = ROOT / "public" / "data" / "reference" / "shakespeare-final.json"
DEFAULT_SHAKESPEARE_SCALING = ROOT / "public" / "data" / "reference" / "shakespeare-scaling.json"
TRAINING_SEEDS = (42, 43, 44)
UPDATES = 700
PROGRESS_INTERVAL = 25
LEARNING_RATE = 3e-4
END_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01


def batch_seed(training_seed: int) -> int:
    """Retain Lesson 9's 4242 order while making later replicates independent."""
    return 4_200 + training_seed


def verified_inputs(protocol: dict) -> tuple[Tokenizer, list[dict], list[dict], list[dict]]:
    data_dir = Path(protocol["paths"]["dataDir"])
    tokenizer_path = Path(protocol["paths"]["tokenizer"])
    observed_hashes = {split: sha256(data_dir / f"{split}.jsonl") for split in ("train", "validation")}
    if observed_hashes != protocol["configuration"]["dataHashes"]:
        raise RuntimeError("TinyStories data differs from the frozen final protocol")
    if sha256(tokenizer_path) != protocol["configuration"]["tokenizerSha256"]:
        raise RuntimeError("TinyStories tokenizer differs from the frozen final protocol")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    training_rows = load_jsonl(data_dir / "train.jsonl")
    validation_rows = load_jsonl(data_dir / "validation.jsonl")
    context_size = protocol["configuration"]["architecture"]["contextSize"]
    training_examples, _ = build_examples(tokenizer, training_rows, context_size)
    validation_examples, _ = build_examples(tokenizer, validation_rows, context_size)
    return tokenizer, training_examples, validation_examples, training_rows


def make_model(configuration: dict, seed: int) -> TinyTransformerLanguageModel:
    architecture = configuration["architecture"]
    mx.random.seed(seed)
    model = TinyTransformerLanguageModel(
        architecture["vocabularySize"], architecture["contextSize"],
        architecture["modelSize"], architecture["attentionHeads"],
        architecture["transformerBlocks"],
    )
    mx.eval(model.parameters())
    model.train()
    return model


def update_live(metrics_path: Path, seed: int, update: int, loss: float) -> None:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    payload["updatedAt"] = datetime.now(UTC).isoformat()
    payload["progress"] = {
        "phase": "Training independent confirmation",
        "seed": seed,
        "completedUpdates": update,
        "plannedUpdates": UPDATES,
        "latestTrainLoss": loss,
        "completedSeeds": len(payload["seeds"]),
        "plannedSeeds": len(TRAINING_SEEDS),
    }
    atomic_json_write(metrics_path, payload)


def train_seed(protocol: dict, seed: int) -> dict:
    tokenizer, training_examples, validation_examples, training_rows = verified_inputs(protocol)
    configuration = protocol["configuration"]
    model = make_model(configuration, seed)
    optimiser = optim.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    order = StatefulBatchOrder(len(training_examples), configuration["batchSize"], batch_seed(seed))
    loss_and_grad = nn.value_and_grad(model, masked_training_loss)
    model.eval()
    starting_validation = evaluate_validation(model, validation_examples, configuration["batchSize"])
    model.train()
    trace: list[dict] = []
    update_seconds: list[float] = []
    valid_targets_seen = 0
    mx.reset_peak_memory()
    started = time.perf_counter()
    for update in range(1, UPDATES + 1):
        indices = order.next_indices()
        inputs, targets, loss_mask = materialise_batch(training_examples, indices)
        rate = schedule(update, UPDATES)
        optimiser.learning_rate = rate
        began = time.perf_counter()
        loss, gradients = loss_and_grad(model, inputs, targets, loss_mask)
        mx.eval(loss, gradients)
        gradient_norm = gradient_l2_norm(gradients)
        optimiser.update(model, gradients)
        mx.eval(model.parameters(), optimiser.state)
        elapsed = time.perf_counter() - began
        update_seconds.append(elapsed)
        targets_seen = sum(training_examples[index]["validTargetCount"] for index in indices)
        valid_targets_seen += targets_seen
        trace.append({
            "update": update, "trainBatchLoss": float(loss.item()),
            "gradientL2Norm": gradient_norm, "learningRate": rate,
            "validTargets": targets_seen, "validTargetsSeen": valid_targets_seen,
            "updateSeconds": elapsed,
        })
        if update % PROGRESS_INTERVAL == 0 or update == UPDATES:
            update_live(Path(protocol["paths"]["metrics"]), seed, update, float(loss.item()))
            print(f"seed={seed} update={update:3d}/{UPDATES} loss={float(loss.item()):.5f}", flush=True)

    model.eval()
    final_validation = evaluate_validation(model, validation_examples, configuration["batchSize"])
    samples = [
        random_completion(
            model, tokenizer, prompt, seed=2_000 + seed + index,
            maximum_tokens=64, temperature=0.9,
        )
        for index, prompt in enumerate(DEFAULT_PROMPTS)
    ]
    seed_dir = Path(protocol["paths"]["runDir"]) / f"seed-{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = seed_dir / "checkpoint-0700.safetensors"
    model.save_weights(str(checkpoint))
    performance = {
        "elapsedSeconds": time.perf_counter() - started,
        "meanUpdateMilliseconds": statistics.mean(update_seconds) * 1_000,
        "validTargetsPerSecond": valid_targets_seen / sum(update_seconds),
        "peakMlxAllocationBytes": mx.get_peak_memory(),
    }
    candidate = {
        "id": f"seed-{seed:03d}", "label": f"Independent training seed {seed}",
        "history": f"Fresh random weights; model seed {seed}; batch-order seed {batch_seed(seed)}; 700 updates",
        "checkpoint": str(checkpoint.resolve()), "checkpointSha256": file_sha256(checkpoint),
    }
    behaviour = evaluate_candidate(
        candidate, {**configuration["architecture"], "batchSize": configuration["batchSize"]},
        tokenizer, validation_examples, training_overlap_index(training_rows),
    )
    result = {
        "seed": seed, "batchSeed": batch_seed(seed), "reusedScalingRun": False,
        "startingValidation": starting_validation, "finalValidation": final_validation,
        "trace": trace, "validTargetsSeen": valid_targets_seen, "samples": samples,
        "behaviour": behaviour, "performance": performance,
        "checkpoint": {"path": str(checkpoint.resolve()), "bytes": checkpoint.stat().st_size, "sha256": file_sha256(checkpoint)},
    }
    atomic_json_write(seed_dir / "result.json", result)
    atomic_json_write(seed_dir / "config.json", {
        "experimentId": protocol["experimentId"], "trainingSeed": seed,
        "batchSeed": batch_seed(seed), "configuration": configuration,
        "createdAt": datetime.now(UTC).isoformat(),
    })
    return result


def run_worker(protocol_path: Path, seed: int) -> None:
    subprocess.run([
        sys.executable, str(Path(__file__).resolve()),
        "--worker-seed", str(seed), "--protocol", str(protocol_path),
    ], cwd=ROOT, check=True)


def reused_seed_result(scaling: dict, story_evaluation: dict) -> dict:
    clean = next(item for item in story_evaluation["results"] if item["candidate"]["id"] == "clean-700")
    long_run = scaling["longRun"]
    checkpoint = long_run["model"]
    return {
        "seed": 42, "batchSeed": batch_seed(42), "reusedScalingRun": True,
        "startingValidation": long_run["startingValidation"],
        "finalValidation": clean["validation"], "trace": long_run["trace"],
        "validTargetsSeen": long_run["trace"][-1]["validTargetsSeen"],
        "samples": long_run["samples"], "behaviour": clean,
        "performance": long_run["performance"],
        "checkpoint": {"path": checkpoint["path"], "bytes": checkpoint["bytes"], "sha256": checkpoint["sha256"]},
    }


def cross_project_comparison(
    protocol: dict, seeds: list[dict], baseline: dict,
    tokenizer_evidence: dict, batching: dict,
) -> dict:
    shakespeare_final_path = Path(protocol["paths"]["shakespeareFinal"])
    shakespeare_scaling_path = Path(protocol["paths"]["shakespeareScaling"])
    shakespeare_final = json.loads(shakespeare_final_path.read_text(encoding="utf-8"))
    shakespeare_scaling = json.loads(shakespeare_scaling_path.read_text(encoding="utf-8"))
    if shakespeare_final.get("status") != "Complete" or shakespeare_scaling.get("status") != "Complete":
        raise RuntimeError("Published Shakespeare comparison evidence is incomplete")
    selected_tokenizer = next(
        candidate for candidate in tokenizer_evidence["candidates"]
        if candidate["actualVocabularySize"] == protocol["configuration"]["architecture"]["vocabularySize"]
    )
    shakespeare_width = next(item for item in shakespeare_scaling["variants"] if item["factor"] == "Width")
    shakespeare_random_loss = shakespeare_width["checkpoints"][0]["validationLoss"]
    tiny_losses = [item["finalValidation"]["crossEntropyLoss"] for item in seeds]
    tiny_mean = statistics.mean(tiny_losses)
    tiny_random_loss = baseline["validation"]["crossEntropyLoss"]
    architecture = protocol["configuration"]["architecture"]
    shakespeare_architecture = shakespeare_final["architecture"]
    return {
        "comparisonBoundary": "Character-token and subword-token cross-entropy/perplexity use different vocabularies and prediction units, so their absolute values are not ranked against each other.",
        "shakespeareEvidence": {"kind": "committed reference", "path": str(shakespeare_final_path), "sha256": file_sha256(shakespeare_final_path)},
        "shakespeare": {
            "tokenUnit": "character", "vocabularySize": 65,
            "contextTokens": shakespeare_architecture["contextSize"],
            "approximateContextCharacters": shakespeare_architecture["contextSize"],
            "parameterCount": shakespeare_architecture["parameterCount"],
            "trainingUpdates": 3_000, "trainingExamples": "one 1,115,394-character Shakespeare corpus split 90/10",
            "randomValidationLoss": shakespeare_random_loss,
            "finalValidationLossMean": shakespeare_final["validationLossAcrossSeedsMean"],
            "withinProjectLossReductionFraction": 1 - shakespeare_final["validationLossAcrossSeedsMean"] / shakespeare_random_loss,
            "observedCapability": "Local Shakespeare-like spelling, dialogue layout, names, punctuation and short clause patterns",
            "boundary": "No instruction following, factual grounding, stable plot reasoning or evidence of understanding",
        },
        "tinyStories": {
            "tokenUnit": "byte-level BPE subword", "vocabularySize": architecture["vocabularySize"],
            "contextTokens": architecture["contextSize"],
            "approximateContextCharacters": architecture["contextSize"] * selected_tokenizer["validation"]["charactersPerToken"],
            "parameterCount": architecture["parameterCount"],
            "trainingUpdates": UPDATES, "trainingExamples": f"{batching['isolatedWindows']['train']['stories']:,} synthetic training stories and {batching['isolatedWindows']['train']['validTargetTokens']:,} valid targets per complete pass",
            "randomValidationLoss": tiny_random_loss, "finalValidationLossMean": tiny_mean,
            "withinProjectLossReductionFraction": 1 - tiny_mean / tiny_random_loss,
            "observedCapability": "Recognisable simple-story cadence, common names, dialogue and short event fragments",
            "boundary": "Unreliable prompt adherence, character/event continuity, grammar and word formation on the bounded development sample",
        },
        "ratios": {
            "tinyStoriesToShakespeareParameters": architecture["parameterCount"] / shakespeare_architecture["parameterCount"],
            "approximateContextCharacters": (architecture["contextSize"] * selected_tokenizer["validation"]["charactersPerToken"]) / shakespeare_architecture["contextSize"],
        },
    }


def add_token_exposure_summary(payload: dict) -> None:
    """Expose the small valid-target variation caused by shuffled padded windows."""
    counts = [item["validTargetsSeen"] for item in payload["seeds"]]
    payload["validTargetsSeenAcrossSeeds"] = {
        "minimum": min(counts), "maximum": max(counts),
        "range": max(counts) - min(counts), "bySeed": {
            str(item["seed"]): item["validTargetsSeen"] for item in payload["seeds"]
        },
        "explanation": "All runs use 700 batches. Different shuffled orders expose slightly different counts of non-padding targets because final story windows contain variable padding.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--scaling", type=Path, default=DEFAULT_SCALING)
    parser.add_argument("--story-evaluation", type=Path, default=DEFAULT_STORY_EVALUATION)
    parser.add_argument("--tokenizer-evidence", type=Path, default=DEFAULT_TOKENIZER_EVIDENCE)
    parser.add_argument("--batching", type=Path, default=DEFAULT_BATCHING)
    parser.add_argument("--shakespeare-final", type=Path, default=DEFAULT_SHAKESPEARE_FINAL)
    parser.add_argument("--shakespeare-scaling", type=Path, default=DEFAULT_SHAKESPEARE_SCALING)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--worker-seed", type=int, choices=TRAINING_SEEDS, help=argparse.SUPPRESS)
    parser.add_argument("--protocol", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--refresh-derived-summary", action="store_true", help="Recalculate derived fields from an already complete local result")
    args = parser.parse_args()
    if args.refresh_derived_summary:
        payload = json.loads(args.metrics.read_text(encoding="utf-8"))
        if payload.get("status") != "Complete" or len(payload.get("seeds", [])) != len(TRAINING_SEEDS):
            raise SystemExit("A complete three-seed result is required")
        add_token_exposure_summary(payload)
        if not any("non-padding target counts" in limitation for limitation in payload["limitations"]):
            payload["limitations"].append(
                "The fixed 700-batch budget exposes slightly different non-padding target counts across shuffled orders because story windows contain variable padding."
            )
        atomic_json_write(args.metrics, payload)
        atomic_json_write(args.run_dir / "result.json", payload)
        print(json.dumps(payload["validTargetsSeenAcrossSeeds"], indent=2))
        return
    if args.worker_seed is not None:
        if args.protocol is None or args.worker_seed == 42:
            raise SystemExit("Internal worker requires a protocol and a fresh seed")
        protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
        train_seed(protocol, args.worker_seed)
        return

    required = (
        args.tokenizer, args.baseline, args.scaling, args.story_evaluation,
        args.tokenizer_evidence, args.batching, args.shakespeare_final,
        args.shakespeare_scaling, args.data_dir / "train.jsonl", args.data_dir / "validation.jsonl",
    )
    for path in required:
        if not path.is_file():
            raise SystemExit(f"Missing prerequisite: {path}")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise SystemExit(f"Run directory already contains files: {args.run_dir}; move it aside before rerunning")

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    scaling = json.loads(args.scaling.read_text(encoding="utf-8"))
    story_evaluation = json.loads(args.story_evaluation.read_text(encoding="utf-8"))
    tokenizer_evidence = json.loads(args.tokenizer_evidence.read_text(encoding="utf-8"))
    batching = json.loads(args.batching.read_text(encoding="utf-8"))
    if any(item.get("status") != "Complete" for item in (baseline, scaling, story_evaluation, tokenizer_evidence, batching)):
        raise SystemExit("Lessons 4, 5, 9, and 10 must have complete learner-local evidence")
    if scaling["selection"]["selectedCandidateId"] != "reference":
        raise SystemExit("This final protocol expects Lesson 9 to retain the reference architecture")

    architecture = {
        key: baseline["configuration"][key]
        for key in ("vocabularySize", "contextSize", "modelSize", "attentionHeads", "transformerBlocks", "parameterCount")
    }
    clean = next(item for item in story_evaluation["protocol"]["candidates"] if item["id"] == "clean-700")
    seed42_checkpoint = Path(clean["checkpoint"])
    if not seed42_checkpoint.is_file() or file_sha256(seed42_checkpoint) != clean["checkpointSha256"]:
        raise SystemExit("Lesson 9 seed-42 checkpoint differs from the evaluated Lesson 10 checkpoint")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.run_dir / "protocol.json"
    configuration = {
        "datasetId": baseline["configuration"]["datasetId"],
        "dataHashes": baseline["configuration"]["dataHashes"],
        "tokenizerSha256": baseline["configuration"]["tokenizerSha256"],
        "architecture": architecture, "batchSize": baseline["configuration"]["batchSize"],
        "trainingSeeds": list(TRAINING_SEEDS),
        "batchSeedRule": "4200 + training seed; seed 42 therefore preserves Lesson 9 batch seed 4242",
        "optimiser": "AdamW", "weightDecay": WEIGHT_DECAY,
        "updates": UPDATES,
        "learningRateSchedule": "constant 3e-4 through update 600; cosine decay to 3e-5 over updates 601-700",
        "completeValidation": True,
        "behaviourProtocol": story_evaluation["protocol"]["generation"],
        "selectionRule": "lowest complete validation cross-entropy; lower seed breaks an exact tie; selection is practical and the three-seed distribution remains the architecture result",
    }
    protocol = {
        "experimentId": args.run_dir.name, "createdAt": datetime.now(UTC).isoformat(),
        "weightsUpdated": True, "configuration": configuration,
        "paths": {
            "dataDir": str(args.data_dir.resolve()), "tokenizer": str(args.tokenizer.resolve()),
            "metrics": str(args.metrics.resolve()), "runDir": str(args.run_dir.resolve()),
            "seed42Checkpoint": str(seed42_checkpoint.resolve()),
            "shakespeareFinal": str(args.shakespeare_final.resolve()),
            "shakespeareScaling": str(args.shakespeare_scaling.resolve()),
        },
        "reusedSeed": {"trainingSeed": 42, "source": scaling["experimentId"], "checkpointSha256": clean["checkpointSha256"]},
    }
    atomic_json_write(protocol_path, protocol)
    payload = {
        "status": "Running", "experimentId": args.run_dir.name,
        "updatedAt": datetime.now(UTC).isoformat(), "protocol": protocol,
        "progress": {"phase": "Protocol frozen", "completedSeeds": 0, "plannedSeeds": len(TRAINING_SEEDS)},
        "seeds": [],
        "limitations": [
            "Three seeds expose some training-path variation but do not estimate every possible run.",
            "The same 200-story validation split has guided prior development and is not an untouched final test set.",
            "The bounded 1,000-story training sample is not the complete TinyStories distribution.",
            "The behaviour protocol uses deterministic surface proxies and three generation seeds, not independent human semantic ratings.",
            "The selected checkpoint is the lowest observed validation loss, while the mean and spread—not that one minimum—summarise the recipe.",
            "Shakespeare and TinyStories losses are not compared numerically because character and subword prediction units differ.",
            "MLX peak allocation excludes total process memory, system memory, and energy use.",
        ],
        "nextExperiment": "Begin Project 3 by selecting and auditing a small pretrained base model for English-to-SQL adaptation.",
    }
    atomic_json_write(args.metrics, payload)
    started = time.perf_counter()

    payload["seeds"].append(reused_seed_result(scaling, story_evaluation))
    payload["progress"] = {"phase": "Seed 42 reused and verified", "completedSeeds": 1, "plannedSeeds": len(TRAINING_SEEDS)}
    atomic_json_write(args.metrics, payload)
    for seed in TRAINING_SEEDS[1:]:
        payload["progress"] = {"phase": "Launching independent confirmation", "seed": seed, "completedSeeds": len(payload["seeds"]), "plannedSeeds": len(TRAINING_SEEDS)}
        atomic_json_write(args.metrics, payload)
        run_worker(protocol_path, seed)
        result = json.loads((args.run_dir / f"seed-{seed:03d}" / "result.json").read_text(encoding="utf-8"))
        payload["seeds"].append(result)
        payload["updatedAt"] = datetime.now(UTC).isoformat()
        atomic_json_write(args.metrics, payload)

    losses = [item["finalValidation"]["crossEntropyLoss"] for item in payload["seeds"]]
    selected = min(payload["seeds"], key=lambda item: (item["finalValidation"]["crossEntropyLoss"], item["seed"]))
    payload.update({
        "status": "Complete", "updatedAt": datetime.now(UTC).isoformat(),
        "progress": {"phase": "Final confirmation complete", "completedSeeds": len(TRAINING_SEEDS), "plannedSeeds": len(TRAINING_SEEDS)},
        "validationLossAcrossSeedsMean": statistics.mean(losses),
        "validationLossAcrossSeedsStd": statistics.pstdev(losses),
        "validationLossAcrossSeedsRange": max(losses) - min(losses),
        "perplexityFromMeanLoss": math.exp(statistics.mean(losses)),
        "selectedSeed": selected["seed"], "selectedCheckpoint": selected["checkpoint"],
        "crossProject": cross_project_comparison(protocol, payload["seeds"], baseline, tokenizer_evidence, batching),
        "environment": {"python": platform.python_version(), "mlx": getattr(mx, "__version__", "unknown"), "device": str(mx.default_device()), "machine": platform.machine(), "platform": platform.platform()},
        "elapsedSeconds": time.perf_counter() - started,
        "conclusionBoundary": "The frozen recipe produced substantially better-than-random in-domain prediction across three seeds. It did not establish reliable story planning, broad language ability, safety, or unbiased generalisation.",
    })
    add_token_exposure_summary(payload)
    payload["limitations"].append(
        "The fixed 700-batch budget exposes slightly different non-padding target counts across shuffled orders because story windows contain variable padding."
    )
    atomic_json_write(args.metrics, payload)
    atomic_json_write(args.run_dir / "result.json", payload)
    print(json.dumps({
        "status": payload["status"], "lossMean": payload["validationLossAcrossSeedsMean"],
        "lossStd": payload["validationLossAcrossSeedsStd"], "selectedSeed": payload["selectedSeed"],
        "elapsedSeconds": payload["elapsedSeconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
