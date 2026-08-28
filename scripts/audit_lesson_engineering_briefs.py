#!/usr/bin/env python3
"""Require one engineering brief for every executable core-course lesson."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LESSON_FILES = (
    ROOT / "lib/shakespeare-lessons.tsx",
    ROOT / "lib/tinystories-lessons.tsx",
    ROOT / "lib/sql-lessons.tsx",
)
BRIEF_FILE = ROOT / "lib/lesson-engineering-briefs.ts"
PAGE_FILE = ROOT / "app/projects/[slug]/lessons/[lesson]/page.tsx"


def slugs(path: Path, pattern: str) -> set[str]:
    return set(re.findall(pattern, path.read_text(encoding="utf-8"), flags=re.MULTILINE))


def main() -> None:
    lessons = set().union(*(slugs(path, r"^    slug: '([^']+)'") for path in LESSON_FILES))
    briefs = slugs(BRIEF_FILE, r"^  '([^']+)': \{")
    missing = sorted(lessons - briefs)
    stale = sorted(briefs - lessons)

    if missing:
        raise SystemExit(f"Engineering brief audit failed; missing lessons: {', '.join(missing)}")
    if stale:
        raise SystemExit(f"Engineering brief audit failed; stale entries: {', '.join(stale)}")

    page = PAGE_FILE.read_text(encoding="utf-8")
    required_integration = (
        "lessonEngineeringBriefs[lessonSlug]",
        "<LessonEngineeringBrief brief={lessonEngineeringBriefs[lessonSlug]}",
    )
    absent = [fragment for fragment in required_integration if fragment not in page]
    if absent:
        raise SystemExit("Engineering brief audit failed; lesson-page integration is missing")

    print(f"Lesson engineering brief audit passed: {len(briefs)} executable lessons covered.")


if __name__ == "__main__":
    main()
