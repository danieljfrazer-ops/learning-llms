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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true", help="Acknowledge that committed course evidence will be overwritten")
    args = parser.parse_args()
    if not args.confirm:
        raise SystemExit("Refusing to overwrite reference evidence without --confirm")
    files = sorted(LOCAL.glob("*.json"))
    if not files:
        raise SystemExit(f"No local JSON results found in {LOCAL}")
    for source in files:
        with source.open(encoding="utf-8") as handle:
            json.load(handle)
    REFERENCE.mkdir(parents=True, exist_ok=True)
    for source in files:
        shutil.copy2(source, REFERENCE / source.name)
        print(f"Promoted {source.name}")


if __name__ == "__main__":
    main()
