#!/usr/bin/env python3
"""Check public-clone hygiene, evidence completeness, and release-only gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 25 * 1024 * 1024

REQUIRED_FILES = [
    "README.md", "AGENTS.md", "CONTRIBUTING.md", "SECURITY.md",
    ".github/workflows/ci.yml", ".github/dependabot.yml", "public/_headers", "next.config.ts",
    "docs/hardware.md", "docs/reproduce-course.md", "docs/evidence-modes.md",
    "docs/agent-workflow.md", "docs/independent-projects.md", "docs/public-experience.md",
    "docs/public-release-checklist.md", "docs/publishing-github-and-cloudflare.md",
    "scripts/download_tiny_shakespeare.py", "scripts/download_tinystories_sample.py",
    "scripts/system_report.py", "scripts/audit_beginner_guidance.py",
    "scripts/audit_lesson_visuals.py", "scripts/audit_git_history.py",
    "scripts/scrub_private_build.mjs", "scripts/audit_shakespeare_browser.mjs",
    "LICENSE", "LICENSE-CONTENT.md", "NOTICE",
    "app/workers/shakespeare-browser.worker.ts", "public/models/shakespeare/manifest.json",
]

REQUIRED_REFERENCE_RESULTS = {
    "shakespeare-metrics.json", "shakespeare-context-metrics.json",
    "shakespeare-attention-metrics.json", "shakespeare-transformer-metrics.json",
    "shakespeare-evaluation.json", "shakespeare-training-improvements.json",
    "shakespeare-scaling.json", "shakespeare-final.json",
    "tinystories-transition.json", "tinystories-dataset-audit.json",
    "tinystories-tokenizer.json", "tinystories-batching.json",
    "tinystories-random-baseline.json", "tinystories-first-pretraining.json",
    "tinystories-checkpoints-dashboard.json", "tinystories-training-recipe.json",
    "tinystories-scaling-budget.json", "tinystories-story-evaluation.json",
    "tinystories-final.json", "sql-task-dataset-audit.json",
    "sql-base-model-selection.json", "sql-prompt-formatting.json",
    "sql-lora-fine-tuning.json", "sql-execution-evaluation.json",
}


def scalar_strings(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from scalar_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from scalar_strings(item)
    elif isinstance(value, str):
        yield value


def contains_key(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return bool(forbidden.intersection(value)) or any(contains_key(item, forbidden) for item in value.values())
    if isinstance(value, list):
        return any(contains_key(item, forbidden) for item in value)
    return False


def tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [ROOT / item.decode() for item in output.split(b"\0") if item]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="store_true", help="Also enforce owner decisions and browser-model gates required immediately before publication")
    args = parser.parse_args()
    errors: list[str] = []

    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required public-readiness file: {relative}")

    reference_dir = ROOT / "public" / "data" / "reference"
    reference_files = sorted(reference_dir.glob("*.json"))
    reference_names = {path.name for path in reference_files}
    for name in sorted(REQUIRED_REFERENCE_RESULTS - reference_names):
        errors.append(f"missing reviewed core reference result: public/data/reference/{name}")
    for path in reference_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as problem:
            errors.append(f"invalid reference JSON {path.relative_to(ROOT)}: {problem}")
            continue
        for value in scalar_strings(payload):
            if re.match(r"^(?:/Users/|/home/|[A-Za-z]:\\Users\\)", value):
                errors.append(f"machine-specific absolute path in {path.relative_to(ROOT)}")
                break
        if path.name.startswith("sql-") and contains_key(payload, {"records", "question", "tableId"}):
            errors.append(f"SQL reference evidence redistributes benchmark example content: {path.relative_to(ROOT)}")

    for path in sorted((ROOT / "experiments").glob("sql-*/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as problem:
            errors.append(f"invalid SQL run metadata {path.relative_to(ROOT)}: {problem}")
            continue
        if contains_key(payload, {"records", "question", "tableId"}):
            errors.append(f"SQL run metadata redistributes benchmark example content: {path.relative_to(ROOT)}")

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for ignored in ("public/data/local/", "/work/", "data/raw/", "experiments/**/*.safetensors", ".env*"):
        if ignored not in gitignore:
            errors.append(f".gitignore missing learner/generated path: {ignored}")

    forbidden_prefixes = ("public/data/local/", "work/", "data/raw/", ".wrangler/", "node_modules/", "dist/")
    forbidden_names = {".env", ".env.local", ".env.production", ".DS_Store"}
    for path in tracked_files():
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(forbidden_prefixes) or path.name in forbidden_names:
            errors.append(f"generated, private, or machine-local file is tracked: {relative}")
        if path.is_symlink() and not path.resolve().is_relative_to(ROOT):
            errors.append(f"tracked symlink escapes repository: {relative}")
        if path.is_file() and path.stat().st_size > MAX_TRACKED_BYTES:
            errors.append(f"tracked file exceeds Cloudflare's 25 MiB asset limit: {relative}")

    components = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "app" / "components").glob("*.tsx"))
    if re.search(r"fetch\(`?/data/shakespeare", components):
        errors.append("dashboard fetch bypasses shared reference/local evidence mode")
    for path in (ROOT / "ml").glob("shakespeare_*.py"):
        if "public/data/reference" in path.read_text(encoding="utf-8"):
            errors.append(f"training code writes or references committed evidence directly: {path.name}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for phrase in ("Reference results", "My lab", "Other hardware", "Before public release", "Cloudflare"):
        if phrase not in readme:
            errors.append(f"README missing public orientation: {phrase}")

    package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    if "scrub_private_build.mjs" not in package.get("scripts", {}).get("build", ""):
        errors.append("production build does not scrub learner-local evidence copies")

    manifest_path = ROOT / "public" / "models" / "shakespeare" / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            models = {model.get("id"): model for model in manifest.get("models", [])}
            for model_id in ("random", "minimal", "baseline", "final"):
                model = models.get(model_id)
                if not model:
                    errors.append(f"browser manifest missing required checkpoint stage: {model_id}")
                    continue
                model_path = manifest_path.parent / str(model.get("file", ""))
                if not model_path.is_file():
                    errors.append(f"browser model file is missing: {model_path.relative_to(ROOT)}")
                else:
                    if model_path.stat().st_size > MAX_TRACKED_BYTES:
                        errors.append(f"browser model exceeds Cloudflare's 25 MiB asset limit: {model_path.relative_to(ROOT)}")
                    if file_sha256(model_path) != model.get("sha256"):
                        errors.append(f"browser model checksum differs from manifest: {model_path.relative_to(ROOT)}")
                if not model.get("parameters") or any(not parameter.get("sourceSha256") for parameter in model.get("parameters", [])):
                    errors.append(f"browser model lacks layer-by-layer source checksums: {model_id}")
            if manifest.get("promptsStayOnDevice") is not True:
                errors.append("browser model manifest does not assert on-device prompt handling")
        except (OSError, json.JSONDecodeError, TypeError) as problem:
            errors.append(f"invalid browser model manifest: {problem}")

    if args.release:
        if not any((ROOT / name).is_file() for name in ("LICENSE", "LICENSE.md", "LICENSE.txt")):
            errors.append("release gate: repository owner has not selected and added a licence")
        if not (ROOT / "public" / "models" / "shakespeare" / "manifest.json").is_file():
            errors.append("release gate: browser-executable Shakespeare manifest is missing")

    if errors:
        print("Public repository audit failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    mode = "release" if args.release else "preparation"
    print(f"Public repository {mode} audit passed: {len(REQUIRED_FILES)} required files and {len(reference_files)} reference result files checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
