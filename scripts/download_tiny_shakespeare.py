#!/usr/bin/env python3
"""Download and verify the exact Tiny Shakespeare corpus used by the course."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "raw" / "tiny-shakespeare.txt"
SOURCE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
EXPECTED_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    if DESTINATION.exists() and digest(DESTINATION) == EXPECTED_SHA256:
        print(f"Dataset already verified: {DESTINATION}")
        return
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=DESTINATION.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        with urllib.request.urlopen(SOURCE, timeout=60) as response:
            while chunk := response.read(1024 * 1024):
                temporary.write(chunk)
    actual = digest(temporary_path)
    if actual != EXPECTED_SHA256:
        temporary_path.unlink(missing_ok=True)
        raise SystemExit(f"Dataset checksum mismatch: expected {EXPECTED_SHA256}, received {actual}")
    os.replace(temporary_path, DESTINATION)
    print(f"Downloaded and verified {DESTINATION} ({DESTINATION.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
