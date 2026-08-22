#!/usr/bin/env python3
"""Deliberately replace committed reference JSON with validated local results."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "public" / "data" / "local"
REFERENCE = ROOT / "public" / "data" / "reference"
LOCAL_RUNS = ROOT / "work" / "experiments"
REFERENCE_RUNS = ROOT / "experiments"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="Acknowledge that committed course evidence will be overwritten")
    parser.add_argument("filenames", nargs="*", help="Specific local JSON filenames; default is every local JSON file")
    parser.add_argument("--run-id", action="append", default=[], help="Also promote one run's config.json, never its checkpoint weights")
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
        with source.open(encoding="utf-8") as handle:
            json.load(handle)
    REFERENCE.mkdir(parents=True, exist_ok=True)
    for source in files:
        shutil.copy2(source, REFERENCE / source.name)
        print(f"Promoted {source.name}")
    for run_id in args.run_id:
        if Path(run_id).name != run_id:
            raise SystemExit(f"Unsafe run identifier: {run_id!r}")
        source = LOCAL_RUNS / run_id / "config.json"
        if not source.is_file():
            raise SystemExit(f"Local run configuration does not exist: {source}")
        with source.open(encoding="utf-8") as handle:
            json.load(handle)
        destination = REFERENCE_RUNS / run_id
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / "config.json")
        print(f"Promoted configuration for {run_id}; checkpoint weights remain local")


if __name__ == "__main__":
    main()
