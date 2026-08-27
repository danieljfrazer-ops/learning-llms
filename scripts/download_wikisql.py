#!/usr/bin/env python3
"""Download and safely unpack the verified WikiSQL 1.1 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "wikisql-1.1"
SOURCE = "https://raw.githubusercontent.com/salesforce/WikiSQL/master/data.tar.bz2"
ARCHIVE_SHA256 = "755c728ab188e364575705c8641f3fafd86fb089cb8b08e8c03f01832aae0881"
EXPECTED_FILES = tuple(
    f"{split}{suffix}"
    for split in ("train", "dev", "test")
    for suffix in (".jsonl", ".tables.jsonl", ".db")
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def download(path: Path) -> None:
    request = urllib.request.Request(SOURCE, headers={"User-Agent": "LearningLLMs/0.1"})
    temporary = path.with_suffix(".tmp")
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "data.tar.bz2"
    manifest_path = args.output / "manifest.json"

    if not archive.is_file() or sha256(archive) != ARCHIVE_SHA256:
        download(archive)
    observed_archive_hash = sha256(archive)
    if observed_archive_hash != ARCHIVE_SHA256:
        raise SystemExit(
            f"WikiSQL archive hash mismatch: expected {ARCHIVE_SHA256}, observed {observed_archive_hash}"
        )

    with tarfile.open(archive, "r:bz2") as bundle:
        available = {member.name: member for member in bundle.getmembers() if member.isfile()}
        for filename in EXPECTED_FILES:
            member_name = f"data/{filename}"
            member = available.get(member_name)
            if member is None:
                raise SystemExit(f"Verified archive is missing {member_name}")
            source = bundle.extractfile(member)
            if source is None:
                raise SystemExit(f"Could not read {member_name}")
            atomic_bytes(args.output / filename, source.read())

    files = {
        filename: {"bytes": (args.output / filename).stat().st_size, "sha256": sha256(args.output / filename)}
        for filename in EXPECTED_FILES
    }
    manifest = {
        "dataset": "WikiSQL 1.1",
        "source": SOURCE,
        "upstreamRepository": "https://github.com/salesforce/WikiSQL",
        "archiveSha256": observed_archive_hash,
        "retrievedAt": datetime.now(UTC).isoformat(),
        "files": files,
        "licensing": {
            "repositoryCodeLicense": "BSD-3-Clause",
            "datasetLicenseStatus": "Not explicitly resolved by the upstream repository; review required before redistribution",
            "upstreamIssue": "https://github.com/salesforce/WikiSQL/issues/91",
        },
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
