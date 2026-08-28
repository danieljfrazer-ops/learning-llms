#!/usr/bin/env python3
"""Scan committed Git blobs for common high-confidence credential signatures."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_SCAN_BYTES = 5 * 1024 * 1024
PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{50,255})"),
    "OpenAI API key": re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "AWS access key": re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "Google API key": re.compile(rb"AIza[0-9A-Za-z_-]{35}"),
    "Slack token": re.compile(rb"xox[baprs]-[0-9A-Za-z-]{20,}"),
    "credential-bearing URL": re.compile(rb"https?://[^\s/:]+:[^\s/@]+@[^\s]+"),
}
SENSITIVE_PATH = re.compile(r"(^|/)(?:\.env(?:\..+)?|id_(?:rsa|ed25519)|credentials(?:\..+)?|secrets?\.(?:json|ya?ml)|[^/]+\.(?:pem|p12|pfx|key))$")


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.DEVNULL)


def main() -> int:
    errors: set[str] = set()
    objects: dict[str, str] = {}
    for line in git("rev-list", "--objects", "--all").decode("utf-8", errors="replace").splitlines():
        oid, _, path = line.partition(" ")
        objects.setdefault(oid, path or "<unknown historical path>")
        if path and SENSITIVE_PATH.search(path):
            errors.add(f"sensitive filename exists in Git history: {path}")

    scanned = 0
    skipped_large = 0
    for oid, path in objects.items():
        if git("cat-file", "-t", oid).strip() != b"blob":
            continue
        size = int(git("cat-file", "-s", oid))
        if size > MAX_SCAN_BYTES:
            skipped_large += 1
            continue
        payload = git("cat-file", "blob", oid)
        scanned += 1
        for label, pattern in PATTERNS.items():
            if pattern.search(payload):
                errors.add(f"possible {label} in historical blob for {path} ({oid[:12]})")

    if errors:
        print("Git history credential audit failed:")
        for error in sorted(errors):
            print(f"- {error}")
        return 1
    print(f"Git history credential audit passed: {scanned} unique blobs scanned; {skipped_large} blobs over {MAX_SCAN_BYTES // (1024 * 1024)} MiB skipped and require independent scanner review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
