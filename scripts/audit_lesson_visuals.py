#!/usr/bin/env python3
"""Check that every project stage has a maintained conceptual visual."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    project_source = (ROOT / "lib" / "wiki-data.ts").read_text(encoding="utf-8")
    visual_source = (ROOT / "app" / "components" / "LearningVisual.tsx").read_text(encoding="utf-8")
    stage_slugs = set(re.findall(r"\{\s*slug:\s*'([^']+)',\s*name:", project_source))
    visual_block = visual_source.split("const lessonVisuals:", 1)[1].split("export function LessonVisual", 1)[0]
    visual_matches = re.findall(r"^  (?:'([^']+)'|([a-z][a-z0-9-]*)):\s*\{", visual_block, re.MULTILINE)
    visual_slugs = {quoted or bare for quoted, bare in visual_matches}
    missing = sorted(stage_slugs - visual_slugs)
    stale = sorted(visual_slugs - stage_slugs)
    for heading, slugs in (("Missing lesson visuals", missing), ("Visuals without a matching project stage", stale)):
        if slugs:
            print(f"{heading}:")
            for slug in slugs:
                print(f"  - {slug}")
    if missing or stale:
        return 1
    print(f"Lesson visual audit passed: {len(stage_slugs)} project stages covered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
