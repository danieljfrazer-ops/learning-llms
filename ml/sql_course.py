"""Shared WikiSQL prompt, parsing, and read-only evaluation utilities."""

from __future__ import annotations

import json
import random
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.paths import RAW_DATA_DIR


DATA_DIR = RAW_DATA_DIR / "wikisql-1.1"
AGGREGATIONS = ("", "MAX", "MIN", "COUNT", "SUM", "AVG")
CONDITION_OPERATORS = ("=", ">", "<")
SELECT_RE = re.compile(
    r"SELECT\s+(?:(MAX|MIN|COUNT|SUM|AVG)\s*\(\s*)?(col\d+)\s*\)?\s+FROM\s+data"
    r"(?:\s+WHERE\s+(.+?))?\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)
CONDITION_RE = re.compile(r"^\s*(col\d+)\s*(=|>|<)\s*(.+?)\s*$", re.DOTALL)


@dataclass(frozen=True)
class SqlExample:
    index: int
    table_id: str
    question: str
    sql: dict[str, Any]
    table: dict[str, Any]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_split(split: str, data_dir: Path = DATA_DIR) -> list[SqlExample]:
    tables = {row["id"]: row for row in read_jsonl(data_dir / f"{split}.tables.jsonl")}
    return [
        SqlExample(index=index, table_id=row["table_id"], question=row["question"], sql=row["sql"], table=tables[row["table_id"]])
        for index, row in enumerate(read_jsonl(data_dir / f"{split}.jsonl"))
    ]


def seeded_sample(examples: list[SqlExample], count: int, seed: int) -> list[SqlExample]:
    if count > len(examples):
        raise ValueError(f"requested {count} examples from a split containing {len(examples)}")
    return [examples[index] for index in sorted(random.Random(seed).sample(range(len(examples)), count))]


def schema_text(table: dict[str, Any], style: str) -> str:
    columns = list(zip(table["header"], table["types"], strict=True))
    if style == "compact":
        return " | ".join(f"col{i}: {name} [{kind}]" for i, (name, kind) in enumerate(columns))
    if style == "ddl":
        fields = ",\n  ".join(f'col{i} {"REAL" if kind == "real" else "TEXT"} -- {name}' for i, (name, kind) in enumerate(columns))
        return f"CREATE TABLE data (\n  {fields}\n);"
    if style == "json":
        return json.dumps(
            {"table": "data", "columns": [{"id": f"col{i}", "name": name, "type": kind} for i, (name, kind) in enumerate(columns)]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    raise ValueError(f"unknown prompt style: {style}")


def user_prompt(example: SqlExample, style: str = "compact") -> str:
    return (
        "Translate the question into one WikiSQL query. Use table data and only colN identifiers from the schema. "
        "Allowed form: SELECT [MAX|MIN|COUNT|SUM|AVG](colN) or colN FROM data, optionally followed by WHERE "
        "conditions joined by AND with =, >, or <. Return SQL only.\n"
        f"Schema:\n{schema_text(example.table, style)}\nQuestion: {example.question}"
    )


def canonical_sql(logical_form: dict[str, Any]) -> str:
    selected = f"col{logical_form['sel']}"
    aggregation = AGGREGATIONS[logical_form["agg"]]
    expression = f"{aggregation}({selected})" if aggregation else selected
    query = f"SELECT {expression} FROM data"
    if logical_form["conds"]:
        parts = []
        for column, operator, value in logical_form["conds"]:
            parts.append(f"col{column} {CONDITION_OPERATORS[operator]} {json.dumps(value, ensure_ascii=False)}")
        query += " WHERE " + " AND ".join(parts)
    return query


def extract_sql(text: str) -> str:
    cleaned = re.sub(r"```(?:sql)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    match = re.search(r"\bSELECT\b", cleaned, flags=re.IGNORECASE)
    if not match:
        return cleaned
    candidate = cleaned[match.start():]
    if ";" in candidate:
        candidate = candidate.split(";", 1)[0] + ";"
    elif "\n" in candidate:
        candidate = candidate.splitlines()[0]
    return candidate.strip()


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
            return raw[1:-1]
        try:
            return float(raw) if "." in raw else int(raw)
        except ValueError:
            raise ValueError("condition value must be a quoted string or number") from None


def _split_conditions(where: str) -> list[str]:
    """Split grammar-level AND tokens while preserving quoted string values."""
    parts: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(where):
        character = where[index]
        if quote:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            index += 1
            continue
        if character in "'\"":
            quote = character
            index += 1
            continue
        if where[index:index + 5].upper() == " AND ":
            parts.append(where[start:index])
            index += 5
            start = index
            continue
        index += 1
    if quote:
        raise ValueError("condition contains an unterminated quoted value")
    parts.append(where[start:])
    return parts


def parse_sql(text: str, width: int) -> dict[str, Any]:
    candidate = extract_sql(text)
    match = SELECT_RE.fullmatch(candidate)
    if not match:
        raise ValueError("output does not match the restricted SELECT grammar")
    aggregation_name, selected_name, where = match.groups()
    selected = int(selected_name[3:])
    if selected >= width:
        raise ValueError("selected column is outside the schema")
    conditions: list[list[Any]] = []
    if where:
        for raw_condition in _split_conditions(where):
            condition = CONDITION_RE.fullmatch(raw_condition)
            if not condition:
                raise ValueError("condition does not match the restricted grammar")
            column_name, operator, raw_value = condition.groups()
            column = int(column_name[3:])
            if column >= width:
                raise ValueError("condition column is outside the schema")
            conditions.append([column, CONDITION_OPERATORS.index(operator), _parse_value(raw_value)])
    return {"sel": selected, "agg": AGGREGATIONS.index((aggregation_name or "").upper()), "conds": conditions}


def normalized_logical_form(logical_form: dict[str, Any]) -> tuple[Any, ...]:
    conditions = tuple(sorted((int(col), int(op), str(value).casefold()) for col, op, value in logical_form["conds"]))
    return int(logical_form["sel"]), int(logical_form["agg"]), conditions


def render_parameterized(logical_form: dict[str, Any], table_name: str) -> tuple[str, list[Any]]:
    selected = f'"col{int(logical_form["sel"])}"'
    aggregation = AGGREGATIONS[int(logical_form["agg"])]
    expression = f"{aggregation}({selected})" if aggregation else selected
    query = f'SELECT {expression} FROM "{table_name}"'
    parameters: list[Any] = []
    if logical_form["conds"]:
        clauses = []
        for column, operator, value in logical_form["conds"]:
            clauses.append(f'"col{int(column)}" {CONDITION_OPERATORS[int(operator)]} ?')
            parameters.append(value)
        query += " WHERE " + " AND ".join(clauses)
    return query, parameters


def execute_logical_form(split: str, example: SqlExample, logical_form: dict[str, Any], data_dir: Path = DATA_DIR) -> list[tuple[Any, ...]]:
    table_name = "table_" + example.table_id.replace("-", "_")
    query, parameters = render_parameterized(logical_form, table_name)
    uri = f"file:{(data_dir / f'{split}.db').resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.execute("PRAGMA query_only = ON")
        return connection.execute(query, parameters).fetchall()


def evaluate_prediction(split: str, example: SqlExample, generated: str) -> dict[str, Any]:
    try:
        predicted = parse_sql(generated, len(example.table["header"]))
    except ValueError as error:
        return {"valid": False, "exact": False, "execution": False, "error": str(error), "parsed": None}
    exact = normalized_logical_form(predicted) == normalized_logical_form(example.sql)
    try:
        predicted_rows = execute_logical_form(split, example, predicted)
        reference_rows = execute_logical_form(split, example, example.sql)
        execution = sorted(map(repr, predicted_rows)) == sorted(map(repr, reference_rows))
        return {"valid": True, "exact": exact, "execution": execution, "error": None, "parsed": predicted}
    except (sqlite3.Error, OverflowError) as error:
        return {"valid": True, "exact": exact, "execution": False, "error": f"SQLite: {error}", "parsed": predicted}
