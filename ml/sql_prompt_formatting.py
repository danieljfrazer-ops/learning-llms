"""Compare schema prompt serializations while holding the selected base model fixed."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ml.paths import local_result, local_run
from ml.sql_base_model_selection import make_chat_prompt
from ml.sql_course import canonical_sql, evaluate_prediction, load_split, seeded_sample, user_prompt


STYLES = ("compact", "ddl", "json")
BASE_RESULT = local_result("sql-base-model-selection.json")
DEFAULT_METRICS = local_result("sql-prompt-formatting.json")
DEFAULT_RUN = local_run("sql-prompt-formatting-001")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--base-result", type=Path, default=BASE_RESULT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    if not args.base_result.is_file():
        raise SystemExit("Lesson 2 result is missing; run ml/sql_base_model_selection.py first")
    base = json.loads(args.base_result.read_text(encoding="utf-8"))
    if base.get("status") != "complete" or base["protocol"].get("testSplitOpened"):
        raise SystemExit("Lesson 2 result is incomplete or violates the closed-test boundary")
    selected_model = base["selectedModel"]
    protocol = {
        "runId": args.run_dir.name,
        "selectionSplit": "dev",
        "testSplitOpened": False,
        "sampleSize": args.sample_size,
        "seed": args.seed,
        "modelHeldFixed": selected_model,
        "styles": list(STYLES),
        "controlledFields": ["question", "column ids", "column names", "column types", "output grammar", "chat template", "greedy decoding", "max new tokens"],
        "changedField": "schema serialization only",
        "selectionRule": "Maximize development execution accuracy; break ties by exact match, valid SQL, fewer mean prompt tokens, then style name.",
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.run_dir / "protocol.json", protocol)

    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    examples = seeded_sample(load_split("dev"), args.sample_size, args.seed)
    mx.reset_peak_memory()
    model, tokenizer = load(selected_model["id"], revision=selected_model["revision"])
    mx.eval(model.parameters())
    results = []
    for style in STYLES:
        records = []
        for position, example in enumerate(examples, start=1):
            prompt = make_chat_prompt(tokenizer, user_prompt(example, style))
            started = time.perf_counter()
            generated = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, sampler=make_sampler(temp=0.0), verbose=False)
            evaluation = evaluate_prediction("dev", example, generated)
            records.append({
                "position": position,
                "sourceIndex": example.index,
                "question": example.question,
                "promptTokens": len(tokenizer.encode(prompt)),
                "reference": canonical_sql(example.sql),
                "generated": generated.strip(),
                "generationSeconds": time.perf_counter() - started,
                **evaluation,
            })
            print(f"{style} [{position}/{len(examples)}] execution={evaluation['execution']}", flush=True)
        results.append({
            "style": style,
            "sampleSize": len(records),
            "validSql": sum(row["valid"] for row in records),
            "logicalFormExact": sum(row["exact"] for row in records),
            "executionCorrect": sum(row["execution"] for row in records),
            "meanPromptTokens": mean(row["promptTokens"] for row in records),
            "meanGenerationSeconds": mean(row["generationSeconds"] for row in records),
            "records": records,
        })
    selected = sorted(results, key=lambda row: (-row["executionCorrect"], -row["logicalFormExact"], -row["validSql"], row["meanPromptTokens"], row["style"]))[0]
    payload = {
        "schemaVersion": 1,
        "status": "complete",
        "lesson": "prompt-formatting",
        "protocol": protocol,
        "results": results,
        "selectedStyle": selected["style"],
        "selectedEvidence": {key: selected[key] for key in ("sampleSize", "validSql", "logicalFormExact", "executionCorrect", "meanPromptTokens")},
        "peakMemoryBytes": mx.get_peak_memory(),
        "limits": [
            "This development comparison chooses a representation; it is not an untouched-test estimate.",
            "Column comments and JSON field names may interact differently with other tokenizers or models.",
            "No table rows, reference query, or expected answer entered any prompt.",
        ],
        "nextExperiment": "Freeze the selected serialization, build train/validation chat records, and compare the base model with zero-effect and trained LoRA adapters.",
    }
    atomic_json(args.run_dir / "selection.json", payload)
    atomic_json(args.metrics, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
