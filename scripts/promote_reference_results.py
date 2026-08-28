#!/usr/bin/env python3
"""Deliberately replace committed reference JSON with validated local results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "public" / "data" / "local"
REFERENCE = ROOT / "public" / "data" / "reference"
LOCAL_RUNS = ROOT / "work" / "experiments"
REFERENCE_RUNS = ROOT / "experiments"
HOME = Path.home().resolve()
RUN_METADATA = ("config.json", "protocol.json", "selection.json")


def public_value(value: Any) -> Any:
    """Remove machine-specific absolute roots without changing measured evidence."""
    if isinstance(value, dict):
        return {key: public_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_value(item) for item in value]
    if not isinstance(value, str):
        return value

    candidate = Path(value)
    if not candidate.is_absolute():
        return value
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        pass
    try:
        relative_home = candidate.resolve().relative_to(HOME).as_posix()
    except ValueError as error:
        raise SystemExit(f"Refusing to publish an unrecognised absolute path: {value}") from error
    return f"<HOME>/{relative_home}"


def redact_sql_examples(value: Any) -> Any:
    """Publish SQL measurements without redistributing benchmark row content."""
    if isinstance(value, dict):
        return {
            key: redact_sql_examples(item)
            for key, item in value.items()
            if key not in {"records", "question", "tableId"}
        }
    if isinstance(value, list):
        return [redact_sql_examples(item) for item in value]
    return value


def write_public_json(source: Path, destination: Path) -> None:
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if any(part.startswith("sql-") for part in destination.parts):
        payload = redact_sql_examples(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(public_value(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="Acknowledge that committed course evidence will be overwritten")
    parser.add_argument("filenames", nargs="*", help="Specific local JSON filenames; default is every local JSON file")
    parser.add_argument("--run-id", action="append", default=[], help="Also promote one run's config/protocol/selection metadata, never its checkpoint weights")
    args = parser.parse_args()
    if not args.confirm:
        raise SystemExit("Refusing to overwrite reference evidence without --confirm")
    for filename in args.filenames:
        if Path(filename).name != filename or not filename.endswith(".json"):
            raise SystemExit(f"Unsafe result filename: {filename!r}")
    files = [LOCAL / filename for filename in args.filenames] if args.filenames else sorted(LOCAL.glob("*.json"))
    if not files:
        raise SystemExit(f"No local JSON results found in {LOCAL}")
    for source in files:
        if not source.is_file():
            raise SystemExit(f"Local result does not exist: {source}")
    REFERENCE.mkdir(parents=True, exist_ok=True)
    for source in files:
        write_public_json(source, REFERENCE / source.name)
        print(f"Promoted {source.name}")
    for run_id in args.run_id:
        if Path(run_id).name != run_id:
            raise SystemExit(f"Unsafe run identifier: {run_id!r}")
        sources = [LOCAL_RUNS / run_id / filename for filename in RUN_METADATA]
        sources = [source for source in sources if source.is_file()]
        if not sources:
            raise SystemExit(f"Local run metadata does not exist for: {run_id}")
        destination = REFERENCE_RUNS / run_id
        for source in sources:
            write_public_json(source, destination / source.name)
        names = ", ".join(source.name for source in sources)
        print(f"Promoted {names} for {run_id}; checkpoint weights remain local")


if __name__ == "__main__":
    main()
