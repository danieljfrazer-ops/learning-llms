#!/usr/bin/env python3
"""Download a small, reproducible TinyStories sample through the HF rows API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "tinystories-sample"
DATASET = "roneneldan/TinyStories"
API = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100


def fetch_page(split: str, offset: int, length: int) -> dict:
    query = urllib.parse.urlencode({
        "dataset": DATASET,
        "config": "default",
        "split": split,
        "offset": offset,
        "length": length,
    })
    request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": "LearningLLMs/0.1"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_split(split: str, count: int, seed: int) -> tuple[list[dict], int, list[int]]:
    first = fetch_page(split, 0, 1)
    total = int(first["num_rows_total"])
    page_count = (count + PAGE_SIZE - 1) // PAGE_SIZE
    starts = list(range(0, max(total - PAGE_SIZE + 1, 1), PAGE_SIZE))
    selected = sorted(random.Random(seed).sample(starts, page_count))
    rows: list[dict] = []
    for offset in selected:
        wanted = min(PAGE_SIZE, count - len(rows))
        payload = fetch_page(split, offset, wanted)
        for item in payload["rows"]:
            rows.append({"sourceRow": int(item["row_idx"]), "text": item["row"]["text"]})
        if len(rows) >= count:
            break
    if len(rows) != count:
        raise RuntimeError(f"Expected {count} {split} stories, received {len(rows)}")
    return rows, total, selected


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-stories", type=int, default=1_000)
    parser.add_argument("--validation-stories", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train_stories < 100 or args.validation_stories < 100:
        raise SystemExit("Use at least 100 stories in each split for a meaningful first audit")

    args.output.mkdir(parents=True, exist_ok=True)
    train_path = args.output / "train.jsonl"
    validation_path = args.output / "validation.jsonl"
    manifest_path = args.output / "manifest.json"

    if train_path.is_file() and validation_path.is_file() and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        same_request = (
            existing.get("seed") == args.seed
            and existing.get("train", {}).get("stories") == args.train_stories
            and existing.get("validation", {}).get("stories") == args.validation_stories
        )
        hashes_match = (
            existing.get("train", {}).get("sha256") == sha256(train_path)
            and existing.get("validation", {}).get("sha256") == sha256(validation_path)
        )
        if same_request and hashes_match:
            print("TinyStories sample already exists and matches its manifest; no download needed.")
            print(json.dumps(existing, indent=2))
            return

    train, train_total, train_offsets = sample_split("train", args.train_stories, args.seed)
    validation, validation_total, validation_offsets = sample_split("validation", args.validation_stories, args.seed + 1)
    atomic_jsonl(train_path, train)
    atomic_jsonl(validation_path, validation)
    manifest = {
        "dataset": DATASET,
        "source": "https://huggingface.co/datasets/roneneldan/TinyStories",
        "licenseRecordedByHost": "cdla-sharing-1.0",
        "retrievedAt": datetime.now(UTC).isoformat(),
        "sampling": "Seeded sample of non-overlapping contiguous API pages; sourceRow preserves provenance",
        "seed": args.seed,
        "train": {"stories": len(train), "sourceRowsTotal": train_total, "pageOffsets": train_offsets, "sha256": sha256(train_path)},
        "validation": {"stories": len(validation), "sourceRowsTotal": validation_total, "pageOffsets": validation_offsets, "sha256": sha256(validation_path)},
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
