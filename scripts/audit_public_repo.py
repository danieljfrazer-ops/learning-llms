#!/usr/bin/env python3
"""Check public-readiness and reference/local evidence invariants."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
errors: list[str] = []

required = [
    "README.md", "AGENTS.md", "CONTRIBUTING.md",
    "docs/hardware.md", "docs/reproduce-course.md", "docs/evidence-modes.md",
    "docs/agent-workflow.md", "docs/independent-projects.md", "docs/public-release-checklist.md",
    "docs/publishing-github-and-cloudflare.md",
    "scripts/download_tiny_shakespeare.py", "scripts/system_report.py",
]
for relative in required:
    if not (ROOT / relative).is_file():
        errors.append(f"missing required public-readiness file: {relative}")

reference_files = sorted((ROOT / "public" / "data" / "reference").glob("*.json"))
if not reference_files:
    errors.append("public/data/reference contains no committed JSON evidence")

gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
for ignored in ("public/data/local/", "/work/", "data/raw/", "experiments/**/*.safetensors"):
    if ignored not in gitignore:
        errors.append(f".gitignore missing learner/generated path: {ignored}")

components = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "app" / "components").glob("*.tsx"))
if re.search(r"fetch\(`?/data/shakespeare", components):
    errors.append("dashboard fetch bypasses shared reference/local evidence mode")

for path in (ROOT / "ml").glob("shakespeare_*.py"):
    text = path.read_text(encoding="utf-8")
    if "public/data/reference" in text:
        errors.append(f"training code writes or references committed evidence directly: {path.name}")

readme = (ROOT / "README.md").read_text(encoding="utf-8")
for phrase in ("Reference results", "My lab", "Other hardware", "Before public release", "Cloudflare"):
    if phrase not in readme:
        errors.append(f"README missing public orientation: {phrase}")

if errors:
    print("Public repository audit failed:")
    for error in errors:
        print(f"- {error}")
    sys.exit(1)

print(f"Public repository audit passed: {len(required)} required files and {len(reference_files)} reference result files checked.")
