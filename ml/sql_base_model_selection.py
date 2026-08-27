"""Compare pinned MLX base models on a frozen WikiSQL development sample."""

from __future__ import annotations

import argparse
import json
import os
import platform
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
from ml.sql_course import canonical_sql, evaluate_prediction, load_split, seeded_sample, user_prompt


CANDIDATES = (
    {"id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "revision": "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3", "license": "Apache-2.0", "upstreamParametersBillions": 0.49, "repositoryBytes": 289601064, "upstreamContextTokens": 32768},
    {"id": "mlx-community/SmolLM2-360M-Instruct-6bit", "revision": "642affd1f9e387d1b56c745894afc83795aebe1d", "license": "Apache-2.0", "upstreamParametersBillions": 0.36, "repositoryBytes": 298985211, "upstreamContextTokens": 8192},
)
DEFAULT_METRICS = local_result("sql-base-model-selection.json")
DEFAULT_RUN = local_run("sql-base-model-selection-001")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def make_chat_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You translate questions to restricted WikiSQL and output SQL only."},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_worker(args: argparse.Namespace) -> None:
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    candidate = CANDIDATES[args.candidate]
    examples = seeded_sample(load_split("dev"), args.sample_size, args.seed)
    mx.reset_peak_memory()
    started = time.perf_counter()
    model, tokenizer = load(candidate["id"], revision=candidate["revision"])
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - started
    records = []
    generation_seconds = []
    for position, example in enumerate(examples, start=1):
        prompt = make_chat_prompt(tokenizer, user_prompt(example, "compact"))
        token_count = len(tokenizer.encode(prompt))
        generation_started = time.perf_counter()
        generated = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, sampler=make_sampler(temp=0.0), verbose=False)
        elapsed = time.perf_counter() - generation_started
        generation_seconds.append(elapsed)
        evaluation = evaluate_prediction("dev", example, generated)
        records.append({
            "position": position,
            "sourceIndex": example.index,
            "tableId": example.table_id,
            "question": example.question,
            "promptTokens": token_count,
            "reference": canonical_sql(example.sql),
            "generated": generated.strip(),
            **evaluation,
            "generationSeconds": elapsed,
        })
        print(f"[{position}/{len(examples)}] valid={evaluation['valid']} execution={evaluation['execution']}", flush=True)
    payload = {
        "candidate": candidate,
        "loadSeconds": load_seconds,
        "peakMemoryBytes": mx.get_peak_memory(),
        "sampleSize": len(records),
        "validSql": sum(row["valid"] for row in records),
        "logicalFormExact": sum(row["exact"] for row in records),
        "executionCorrect": sum(row["execution"] for row in records),
        "meanGenerationSeconds": mean(generation_seconds),
        "meanPromptTokens": mean(row["promptTokens"] for row in records),
        "tokenizerClass": tokenizer.__class__.__name__,
        "chatTemplateApplied": True,
        "records": records,
    }
    atomic_json(args.worker_output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--candidate", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.sample_size < 1 or args.max_tokens < 1:
        raise SystemExit("sample size and max tokens must be positive")
    if args.candidate is not None:
        if args.worker_output is None:
            raise SystemExit("worker output is required")
        run_worker(args)
        return

    args.run_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "runId": args.run_dir.name,
        "selectionSplit": "dev",
        "testSplitOpened": False,
        "sampleSize": args.sample_size,
        "seed": args.seed,
        "promptStyle": "compact-v1",
        "generation": {"decoding": "greedy", "maxNewTokens": args.max_tokens},
        "eligibility": "ungated pinned model, permissive recorded licence, valid MLX load, and peak MLX allocation below 8 GiB",
        "selectionRule": "Among eligible candidates, maximize development execution accuracy; break ties by logical-form exact match, then valid SQL, then lower peak memory, then model id.",
        "candidates": list(CANDIDATES),
    }
    atomic_json(args.run_dir / "protocol.json", protocol)
    results = []
    for index, candidate in enumerate(CANDIDATES):
        output = args.run_dir / f"candidate-{index}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--candidate", str(index), "--worker-output", str(output), "--sample-size", str(args.sample_size), "--seed", str(args.seed), "--max-tokens", str(args.max_tokens)]
        print(f"Benchmarking {candidate['id']} at {candidate['revision']}…", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0 or not output.is_file():
            results.append({"candidate": candidate, "eligible": False, "failure": f"worker exited {completed.returncode}"})
            continue
        result = json.loads(output.read_text(encoding="utf-8"))
        result["eligible"] = result["peakMemoryBytes"] < 8 * 1024**3
        results.append(result)
    eligible = [result for result in results if result.get("eligible")]
    if not eligible:
        raise SystemExit("No candidate passed the predeclared eligibility gates")
    selected = sorted(eligible, key=lambda row: (-row["executionCorrect"], -row["logicalFormExact"], -row["validSql"], row["peakMemoryBytes"], row["candidate"]["id"]))[0]
    payload = {
        "schemaVersion": 1,
        "status": "complete",
        "lesson": "base-model-selection",
        "protocol": protocol,
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "backend": "MLX"},
        "results": results,
        "selectedModel": selected["candidate"],
        "selectionEvidence": {
            "executionCorrect": selected["executionCorrect"],
            "logicalFormExact": selected["logicalFormExact"],
            "validSql": selected["validSql"],
            "sampleSize": selected["sampleSize"],
        },
        "limits": [
            "The seeded development sample is a model-selection screen, not a test-set result.",
            "The provisional compact prompt is held constant semantically, but each tokenizer applies its model-specific chat template.",
            "Execution equivalence on one frozen table does not prove universal SQL equivalence or production safety.",
        ],
        "nextExperiment": "Hold the selected model fixed and compare prompt serializations on a separate frozen development sample.",
    }
    atomic_json(args.run_dir / "selection.json", payload)
    atomic_json(args.metrics, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
