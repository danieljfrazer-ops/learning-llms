"""Prepare frozen WikiSQL chat data, train LoRA, and compare before/after behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ml.paths import ROOT, local_result, local_run
from ml.sql_base_model_selection import make_chat_prompt
from ml.sql_course import canonical_sql, evaluate_prediction, load_split, seeded_sample, user_prompt


BASE_RESULT = local_result("sql-base-model-selection.json")
PROMPT_RESULT = local_result("sql-prompt-formatting.json")
DEFAULT_METRICS = local_result("sql-lora-fine-tuning.json")
DEFAULT_RUN = local_run("sql-lora-fine-tuning-001")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chat_record(example: Any, style: str) -> dict[str, Any]:
    return {"messages": [
        {"role": "system", "content": "You translate questions to restricted WikiSQL and output SQL only."},
        {"role": "user", "content": user_prompt(example, style)},
        {"role": "assistant", "content": canonical_sql(example.sql)},
    ]}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def run_worker(args: argparse.Namespace) -> None:
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    specification = json.loads(args.specification.read_text(encoding="utf-8"))
    examples = seeded_sample(load_split("dev"), specification["evaluationSampleSize"], specification["evaluationSeed"])
    adapter_path = None if args.adapter == "base" else str(args.run_dir / args.adapter)
    mx.reset_peak_memory()
    model, tokenizer = load(specification["modelPath"], adapter_path=adapter_path)
    mx.eval(model.parameters())
    records = []
    for position, example in enumerate(examples, start=1):
        prompt = make_chat_prompt(tokenizer, user_prompt(example, specification["promptStyle"]))
        started = time.perf_counter()
        generated = generate(model, tokenizer, prompt=prompt, max_tokens=specification["maxTokens"], sampler=make_sampler(temp=0.0), verbose=False)
        evaluation = evaluate_prediction("dev", example, generated)
        records.append({"position": position, "sourceIndex": example.index, "question": example.question, "reference": canonical_sql(example.sql), "generated": generated.strip(), "generationSeconds": time.perf_counter() - started, **evaluation})
        print(f"{args.adapter} [{position}/{len(examples)}] execution={evaluation['execution']}", flush=True)
    atomic_json(args.worker_output, {
        "checkpoint": args.adapter,
        "sampleSize": len(records),
        "validSql": sum(row["valid"] for row in records),
        "logicalFormExact": sum(row["exact"] for row in records),
        "executionCorrect": sum(row["execution"] for row in records),
        "meanGenerationSeconds": mean(row["generationSeconds"] for row in records),
        "peakMemoryBytes": mx.get_peak_memory(),
        "records": records,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-examples", type=int, default=512)
    parser.add_argument("--validation-examples", type=int, default=64)
    parser.add_argument("--evaluation-sample-size", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--adapter", choices=("base", "random-adapter", "trained-adapter"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--specification", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.adapter:
        run_worker(args)
        return
    for dependency in (BASE_RESULT, PROMPT_RESULT):
        if not dependency.is_file():
            raise SystemExit(f"Missing prerequisite result: {dependency}")
    base = json.loads(BASE_RESULT.read_text(encoding="utf-8"))
    prompt = json.loads(PROMPT_RESULT.read_text(encoding="utf-8"))
    if base.get("status") != "complete" or prompt.get("status") != "complete":
        raise SystemExit("Lesson 2 and Lesson 3 must be complete")
    if base["protocol"].get("testSplitOpened") or prompt["protocol"].get("testSplitOpened"):
        raise SystemExit("A prerequisite violated the closed-test boundary")

    from huggingface_hub import snapshot_download

    selected_model = base["selectedModel"]
    model_path = snapshot_download(repo_id=selected_model["id"], revision=selected_model["revision"])
    style = prompt["selectedStyle"]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.run_dir / "data"
    train_examples = seeded_sample(load_split("train"), args.train_examples, args.seed)
    validation_examples = seeded_sample(load_split("dev"), args.validation_examples, args.seed + 1)
    write_jsonl(data_dir / "train.jsonl", [chat_record(row, style) for row in train_examples])
    write_jsonl(data_dir / "valid.jsonl", [chat_record(row, style) for row in validation_examples])
    protocol = {
        "runId": args.run_dir.name,
        "testSplitOpened": False,
        "model": selected_model,
        "modelPath": model_path,
        "promptStyle": style,
        "trainExamples": args.train_examples,
        "validationExamples": args.validation_examples,
        "evaluationSampleSize": args.evaluation_sample_size,
        "evaluationSeed": args.seed + 2,
        "maxTokens": 96,
        "training": {"type": "LoRA on quantized base", "rank": 8, "scale": 20, "layers": 8, "batchSize": 2, "iterations": args.iterations, "learningRate": 1e-5, "maskPrompt": True, "maxSequenceLength": 512},
        "comparison": ["base", "zero-effect random adapter", "trained adapter"],
    }
    specification = args.run_dir / "protocol.json"
    atomic_json(specification, protocol)

    logs = {}
    for adapter_name, iterations in (("random-adapter", 0), ("trained-adapter", args.iterations)):
        adapter_dir = args.run_dir / adapter_name
        command = ["uv", "run", "mlx_lm.lora", "--model", model_path, "--train", "--data", str(data_dir), "--fine-tune-type", "lora", "--mask-prompt", "--num-layers", "8", "--batch-size", "2", "--iters", str(iterations), "--val-batches", "8", "--learning-rate", "1e-5", "--steps-per-report", "10", "--steps-per-eval", "25", "--save-every", str(max(1, args.iterations)), "--max-seq-length", "512", "--seed", str(args.seed), "--adapter-path", str(adapter_dir)]
        print(f"Creating {adapter_name} ({iterations} optimizer updates)…", flush=True)
        completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        log_path = args.run_dir / f"{adapter_name}.log"
        log_path.write_text(completed.stdout, encoding="utf-8")
        print(completed.stdout, end="")
        if completed.returncode != 0:
            raise SystemExit(f"{adapter_name} exited {completed.returncode}; inspect {log_path}")
        logs[adapter_name] = completed.stdout

    evaluations = []
    for checkpoint in ("base", "random-adapter", "trained-adapter"):
        output = args.run_dir / f"evaluation-{checkpoint}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--adapter", checkpoint, "--run-dir", str(args.run_dir), "--worker-output", str(output), "--specification", str(specification)]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"evaluation worker for {checkpoint} exited {completed.returncode}")
        evaluations.append(json.loads(output.read_text(encoding="utf-8")))
    base_eval, random_eval, trained_eval = evaluations
    random_matches_base = all(a["generated"] == b["generated"] for a, b in zip(base_eval["records"], random_eval["records"], strict=True))
    loss_pattern = re.compile(r"Iter (\d+): (Val|Train) loss ([0-9.]+)")
    loss_trace = [{"iteration": int(match.group(1)), "kind": match.group(2).lower(), "loss": float(match.group(3))} for match in loss_pattern.finditer(logs["trained-adapter"])]
    trainable_line = next((line for line in logs["trained-adapter"].splitlines() if line.startswith("Trainable parameters:")), "Not reported")
    peak_training_memory = max((float(value) for value in re.findall(r"Peak mem ([0-9.]+) GB", logs["trained-adapter"])), default=0.0)
    adapter_file = args.run_dir / "trained-adapter" / "adapters.safetensors"
    payload = {
        "schemaVersion": 1,
        "status": "complete",
        "lesson": "lora-fine-tuning",
        "protocol": protocol,
        "data": {"trainSha256": sha256(data_dir / "train.jsonl"), "validationSha256": sha256(data_dir / "valid.jsonl")},
        "checkpoints": evaluations,
        "zeroEffectCheck": {"randomAdapterMatchesBaseOutputs": random_matches_base, "explanation": "MLX-LM initializes LoRA B matrices to zero, so the new low-rank path initially contributes zero."},
        "training": {"lossTrace": loss_trace, "trainableParameters": trainable_line, "peakMemoryGB": peak_training_memory, "adapterBytes": adapter_file.stat().st_size, "adapterSha256": sha256(adapter_file)},
        "limits": [
            "The 512-example, 100-update run is a bounded teaching experiment rather than a production fine-tune.",
            "Development execution accuracy guided training decisions; the untouched test split remains closed.",
            "A small adapter file does not imply that inference can run without the pinned base model.",
        ],
        "nextExperiment": "Open the test split once and evaluate the frozen base and trained-adapter cohort under the same restricted executor.",
    }
    atomic_json(args.run_dir / "result.json", payload)
    atomic_json(args.metrics, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
