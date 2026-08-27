"""Audit WikiSQL's task contract, split integrity, and safe reference execution.

This lesson performs no model training. It reads the verified WikiSQL files,
checks their structured logical forms, and executes a deterministic sample of
reference queries assembled only from whitelisted operators and parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sqlite3
import statistics
import time
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from paths import RAW_DATA_DIR, ROOT, local_result, local_run
from shakespeare_transformer import atomic_json_write


DEFAULT_DATA_DIR = RAW_DATA_DIR / "wikisql-1.1"
DEFAULT_METRICS = local_result("sql-task-dataset-audit.json")
DEFAULT_RUN_DIR = local_run("sql-task-dataset-audit-001")
SPLITS = ("train", "dev", "test")
AGGREGATIONS = ("NONE", "MAX", "MIN", "COUNT", "SUM", "AVG")
CONDITIONS = ("=", ">", "<")
WORD_RE = re.compile(r"\w+", re.UNICODE)
TABLE_ID_RE = re.compile(r"^[0-9]+(?:-[0-9]+)+$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number}: expected an object")
            rows.append(value)
    return rows


def normalise(text: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def validate_example(example: dict, tables: dict[str, dict]) -> list[str]:
    errors = []
    table_id = example.get("table_id")
    table = tables.get(table_id)
    sql = example.get("sql")
    if not isinstance(example.get("question"), str) or not example["question"].strip():
        errors.append("missingQuestion")
    if table is None:
        return errors + ["missingTable"]
    if not isinstance(sql, dict):
        return errors + ["missingLogicalForm"]
    width = len(table.get("header", []))
    if not isinstance(sql.get("sel"), int) or not 0 <= sql["sel"] < width:
        errors.append("invalidSelectColumn")
    if not isinstance(sql.get("agg"), int) or not 0 <= sql["agg"] < len(AGGREGATIONS):
        errors.append("invalidAggregation")
    conditions = sql.get("conds")
    if not isinstance(conditions, list):
        return errors + ["invalidConditions"]
    for condition in conditions:
        if not isinstance(condition, list) or len(condition) != 3:
            errors.append("invalidConditionShape")
            continue
        column, operator, _ = condition
        if not isinstance(column, int) or not 0 <= column < width:
            errors.append("invalidConditionColumn")
        if not isinstance(operator, int) or not 0 <= operator < len(CONDITIONS):
            errors.append("invalidConditionOperator")
    return errors


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def build_query(example: dict, *, reverse_conditions: bool = False) -> tuple[str, list[object]]:
    table_id = example["table_id"]
    if not TABLE_ID_RE.fullmatch(table_id):
        raise ValueError(f"Unsafe table identifier: {table_id!r}")
    logical = example["sql"]
    aggregation = AGGREGATIONS[logical["agg"]]
    selected = quote_identifier(f"col{logical['sel']}")
    expression = selected if aggregation == "NONE" else f"{aggregation}({selected})"
    table = quote_identifier("table_" + table_id.replace("-", "_"))
    conditions = list(logical["conds"])
    if reverse_conditions:
        conditions.reverse()
    fragments, parameters = [], []
    for column, operator, value in conditions:
        fragments.append(f"{quote_identifier(f'col{column}')} {CONDITIONS[operator]} ?")
        parameters.append(value)
    where = " WHERE " + " AND ".join(fragments) if fragments else ""
    return f"SELECT {expression} FROM {table}{where}", parameters


def canonical_rows(rows: list[tuple]) -> list[str]:
    return sorted(repr(tuple(value for value in row)) for row in rows)


def execute_sample(data_dir: Path, split: str, examples: list[dict], sample_size: int, seed: int) -> dict:
    chosen = random.Random(seed).sample(examples, min(sample_size, len(examples)))
    successes, failures = 0, []
    with sqlite3.connect(f"file:{data_dir / f'{split}.db'}?mode=ro", uri=True) as database:
        for index, example in enumerate(chosen):
            try:
                query, parameters = build_query(example)
                database.execute("PRAGMA query_only = ON")
                database.execute(query, parameters).fetchall()
                successes += 1
            except Exception as error:  # preserved as evidence instead of hiding a bad row
                failures.append({"sampleIndex": index, "tableId": example.get("table_id"), "error": str(error)})
    return {"requested": sample_size, "executed": len(chosen), "succeeded": successes, "failures": failures}


def equivalence_example(data_dir: Path, examples: list[dict]) -> dict:
    with sqlite3.connect(f"file:{data_dir / 'dev.db'}?mode=ro", uri=True) as database:
        database.execute("PRAGMA query_only = ON")
        for example in examples:
            if len(example["sql"]["conds"]) < 2:
                continue
            original, original_parameters = build_query(example)
            original_rows = database.execute(original, original_parameters).fetchall()
            if original_rows:
                reordered, reordered_parameters = build_query(example, reverse_conditions=True)
                reordered_rows = database.execute(reordered, reordered_parameters).fetchall()
                break
        else:
            raise ValueError("No non-empty multi-condition development query was found")
    return {
        "question": example["question"],
        "tableId": example["table_id"],
        "logicalFormsDiffer": original != reordered,
        "resultsMatch": canonical_rows(original_rows) == canonical_rows(reordered_rows),
        "resultRowCount": len(original_rows),
        "comparison": "same structured reference query with AND conditions rendered in reverse order",
    }


def split_summary(examples: list[dict], tables: dict[str, dict]) -> dict:
    word_counts = [len(WORD_RE.findall(item["question"])) for item in examples]
    validation_errors = Counter(error for item in examples for error in validate_example(item, tables))
    conditions = [condition for item in examples for condition in item["sql"]["conds"]]
    grounded = sum(normalise(condition[2]) in normalise(item["question"]) for item in examples for condition in item["sql"]["conds"])
    return {
        "questions": len(examples),
        "tables": len(tables),
        "questionWords": {
            "median": statistics.median(word_counts),
            "p95": percentile(word_counts, 0.95),
            "maximum": max(word_counts),
        },
        "aggregationCounts": dict(Counter(AGGREGATIONS[item["sql"]["agg"]] for item in examples)),
        "conditionOperatorCounts": dict(Counter(CONDITIONS[item[1]] for item in conditions)),
        "questionsWithMultipleConditions": sum(len(item["sql"]["conds"]) > 1 for item in examples),
        "conditionValuesMentionedInQuestion": {"mentioned": grounded, "total": len(conditions)},
        "validationErrors": dict(validation_errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--execution-sample-per-split", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.execution_sample_per_split < 1:
        raise SystemExit("execution sample must be positive")

    manifest_path = args.data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing WikiSQL manifest: {manifest_path}\nRun scripts/download_wikisql.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    examples, tables = {}, {}
    observed_hashes = {}
    for split in SPLITS:
        examples[split] = load_jsonl(args.data_dir / f"{split}.jsonl")
        table_rows = load_jsonl(args.data_dir / f"{split}.tables.jsonl")
        tables[split] = {row["id"]: row for row in table_rows}
        for suffix in (".jsonl", ".tables.jsonl", ".db"):
            filename = f"{split}{suffix}"
            observed_hashes[filename] = sha256(args.data_dir / filename)
            if manifest["files"][filename]["sha256"] != observed_hashes[filename]:
                raise SystemExit(f"Dataset hash mismatch for {filename}")

    table_ids = {split: set(tables[split]) for split in SPLITS}
    question_sets = {split: {normalise(item["question"]) for item in examples[split]} for split in SPLITS}
    pairwise = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        pairwise[f"{left}Vs{right}"] = {
            "sharedTableIds": len(table_ids[left] & table_ids[right]),
            "exactNormalisedQuestions": len(question_sets[left] & question_sets[right]),
        }

    payload = {
        "status": "Complete",
        "experimentId": args.run_dir.name,
        "updatedAt": datetime.now(UTC).isoformat(),
        "taskContract": {
            "input": "Natural-language question plus one table schema; table contents are excluded from model input",
            "output": "One restricted single-table SELECT logical form",
            "primaryMetric": "execution accuracy on a frozen held-out split",
            "secondaryMetric": "logical-form exact match",
            "prohibitedUse": "Generated SQL must not be executed against production or writable databases",
            "modelTraining": "none",
        },
        "dataset": {
            "name": manifest["dataset"],
            "archiveSha256": manifest["archiveSha256"],
            "licensing": manifest["licensing"],
            "splits": {split: split_summary(examples[split], tables[split]) for split in SPLITS},
            "totalQuestions": sum(len(examples[split]) for split in SPLITS),
            "totalTables": sum(len(tables[split]) for split in SPLITS),
            "pairwiseLeakage": pairwise,
            "fileHashes": observed_hashes,
        },
        "referenceExecution": {
            "safety": "Read-only SQLite connections; query-only mode; whitelisted SELECT/aggregation/condition operators; parameterised values",
            "samples": {
                split: execute_sample(args.data_dir, split, examples[split], args.execution_sample_per_split, args.seed + index)
                for index, split in enumerate(SPLITS)
            },
            "equivalenceExample": equivalence_example(args.data_dir, examples["dev"]),
        },
        "elapsedSeconds": time.perf_counter() - started,
        "limitations": [
            "The upstream repository is archived and does not explicitly resolve whether its BSD-3-Clause code licence covers the dataset; redistribution remains blocked pending review.",
            "WikiSQL covers restricted single-table SELECT queries, not joins, nested queries, updates, transactions, or production schemas.",
            "The execution smoke test samples reference queries; it does not establish model accuracy because no model has been selected or invoked.",
            "Exact question matching cannot detect paraphrased leakage, and execution equivalence can hide semantically different queries that coincide on one table state.",
        ],
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "runId": args.run_dir.name,
        "dataset": manifest["dataset"],
        "archiveSha256": manifest["archiveSha256"],
        "seed": args.seed,
        "executionSamplePerSplit": args.execution_sample_per_split,
        "modelTraining": "none",
        "resultPath": str(args.metrics.relative_to(ROOT)) if args.metrics.is_relative_to(ROOT) else str(args.metrics),
    }
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    atomic_json_write(args.metrics, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
