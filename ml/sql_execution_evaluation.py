"""Evaluate the frozen base and LoRA cohort once on untouched WikiSQL test data."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ml.paths import local_result, local_run
from ml.sql_base_model_selection import make_chat_prompt
from ml.sql_course import canonical_sql, evaluate_prediction, load_split, seeded_sample, user_prompt


LORA_RESULT = local_result("sql-lora-fine-tuning.json")
DEFAULT_METRICS = local_result("sql-execution-evaluation.json")
DEFAULT_RUN = local_run("sql-execution-evaluation-001")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def failure_group(record: dict[str, Any]) -> str:
    if not record["valid"]:
        return "invalid restricted SQL"
    parsed = record["parsed"]
    reference = record["referenceLogical"]
    if parsed["sel"] != reference["sel"]:
        return "wrong selected column"
    if parsed["agg"] != reference["agg"]:
        return "wrong aggregation"
    if len(parsed["conds"]) != len(reference["conds"]):
        return "wrong condition count"
    if parsed["conds"] != reference["conds"]:
        return "wrong condition column, operator, or value"
    return "execution mismatch despite matching parsed fields"


def run_worker(args: argparse.Namespace) -> None:
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    examples = seeded_sample(load_split("test"), protocol["sampleSize"], protocol["seed"])
    adapter_path = None if args.checkpoint == "base" else str(protocol["trainedAdapterPath"])
    mx.reset_peak_memory()
    model, tokenizer = load(protocol["modelPath"], adapter_path=adapter_path)
    mx.eval(model.parameters())
    records = []
    for position, example in enumerate(examples, start=1):
        prompt = make_chat_prompt(tokenizer, user_prompt(example, protocol["promptStyle"]))
        started = time.perf_counter()
        generated = generate(model, tokenizer, prompt=prompt, max_tokens=protocol["maxTokens"], sampler=make_sampler(temp=0.0), verbose=False)
        evaluation = evaluate_prediction("test", example, generated)
        record = {"position": position, "sourceIndex": example.index, "tableId": example.table_id, "question": example.question, "reference": canonical_sql(example.sql), "referenceLogical": example.sql, "generated": generated.strip(), "generationSeconds": time.perf_counter() - started, **evaluation}
        record["failureGroup"] = None if evaluation["execution"] else failure_group(record)
        records.append(record)
        print(f"{args.checkpoint} test [{position}/{len(examples)}] execution={evaluation['execution']}", flush=True)
    atomic_json(args.output, {
        "checkpoint": args.checkpoint,
        "sampleSize": len(records),
        "validSql": sum(row["valid"] for row in records),
        "logicalFormExact": sum(row["exact"] for row in records),
        "executionCorrect": sum(row["execution"] for row in records),
        "meanGenerationSeconds": mean(row["generationSeconds"] for row in records),
        "peakMemoryBytes": mx.get_peak_memory(),
        "failureGroups": dict(Counter(row["failureGroup"] for row in records if row["failureGroup"])),
        "records": records,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--checkpoint", choices=("base", "trained-adapter"), help=argparse.SUPPRESS)
    parser.add_argument("--protocol", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.checkpoint:
        run_worker(args)
        return
    if not LORA_RESULT.is_file():
        raise SystemExit("Lesson 4 result is missing; run ml/sql_lora_fine_tuning.py first")
    lora = json.loads(LORA_RESULT.read_text(encoding="utf-8"))
    if lora.get("status") != "complete" or lora["protocol"].get("testSplitOpened"):
        raise SystemExit("Lesson 4 is incomplete or opened test data early")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    training_run = local_run(lora["protocol"]["runId"])
    adapter_path = training_run / "trained-adapter"
    protocol = {
        "runId": args.run_dir.name,
        "split": "test",
        "openedOnlyAfterSelectionAndTraining": True,
        "sampleSize": args.sample_size,
        "seed": args.seed,
        "model": lora["protocol"]["model"],
        "modelPath": lora["protocol"]["modelPath"],
        "trainedAdapterPath": str(adapter_path),
        "trainedAdapterSha256": lora["training"]["adapterSha256"],
        "promptStyle": lora["protocol"]["promptStyle"],
        "maxTokens": lora["protocol"]["maxTokens"],
        "decoding": "greedy",
        "primaryMetric": "execution accuracy",
        "secondaryMetrics": ["logical-form exact match", "valid restricted SQL", "failure taxonomy", "latency", "peak MLX allocation"],
        "noFurtherSelection": True,
    }
    protocol_path = args.run_dir / "protocol.json"
    atomic_json(protocol_path, protocol)
    cohort = []
    for checkpoint in ("base", "trained-adapter"):
        output = args.run_dir / f"{checkpoint}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--checkpoint", checkpoint, "--protocol", str(protocol_path), "--output", str(output)]
        completed = __import__("subprocess").run(command, cwd=REPOSITORY_ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"{checkpoint} evaluator exited {completed.returncode}")
        cohort.append(json.loads(output.read_text(encoding="utf-8")))
    base, trained = cohort
    payload = {
        "schemaVersion": 1,
        "status": "complete",
        "lesson": "execution-evaluation",
        "protocol": protocol,
        "cohort": cohort,
        "comparison": {
            "executionCorrectDelta": trained["executionCorrect"] - base["executionCorrect"],
            "exactMatchDelta": trained["logicalFormExact"] - base["logicalFormExact"],
            "validSqlDelta": trained["validSql"] - base["validSql"],
        },
        "limits": [
            f"The result estimates behavior on a seeded {args.sample_size}-example test sample, not all WikiSQL test examples.",
            "WikiSQL contains restricted single-table queries and does not establish behavior on joins, nested queries, private schemas, or adversarial prompts.",
            "Generated text is parsed into an allowlisted logical form and rebuilt as parameterized SQL; it is never sent directly to SQLite.",
            "No further model, prompt, or checkpoint selection is permitted after observing this test result.",
        ],
        "projectConclusion": "The complete course path now contrasts random-weight pretraining with parameter-efficient adaptation and execution-grounded evaluation.",
    }
    atomic_json(args.run_dir / "result.json", payload)
    atomic_json(args.metrics, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
