#!/usr/bin/env python3
"""Verify that lessons, section guidance, and glossary links stay aligned."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    lesson_sources = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "lib").glob("*-lessons.tsx"))
    }
    guidance = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "lib").glob("*beginner-guidance.ts"))
        if path.name != "planned-beginner-guidance.ts"
    )
    planned_path = ROOT / "lib" / "planned-beginner-guidance.ts"
    planned_guidance = planned_path.read_text(encoding="utf-8") if planned_path.exists() else ""
    wiki = (ROOT / "lib" / "wiki-data.ts").read_text(encoding="utf-8")

    guide_starts = list(re.finditer(r"^\s{2}(?:'([^']+)'|([a-z][\w-]*)):\s*\{", guidance, re.M))
    guide_blocks: dict[str, str] = {}
    for index, match in enumerate(guide_starts):
        end = guide_starts[index + 1].start() if index + 1 < len(guide_starts) else len(guidance)
        guide_blocks[match.group(1) or match.group(2)] = guidance[match.start():end]

    errors: list[str] = []
    total_sections = 0
    lesson_count = 0
    all_lessons = "\n".join(lesson_sources.values())
    for filename, lessons in lesson_sources.items():
        lesson_starts = list(re.finditer(r"^\s{4}slug:\s*'([^']+)'", lessons, re.M))
        lesson_count += len(lesson_starts)
        for index, match in enumerate(lesson_starts):
            slug = match.group(1)
            end = lesson_starts[index + 1].start() if index + 1 < len(lesson_starts) else lessons.index("\n];", match.start())
            section_ids = re.findall(r"\bid:\s*'([^']+)'", lessons[match.start():end])
            total_sections += len(section_ids)
            block = guide_blocks.get(slug)
            if not block:
                errors.append(f"{filename}/{slug}: missing lesson-level beginner guide")
                continue
            for field in ("background:", "picture:", "misconception:", "goal:"):
                if field not in block:
                    errors.append(f"{slug}: lesson guide missing {field[:-1]}")
            for section_id in section_ids:
                pattern = rf"^\s{{6}}(?:'{re.escape(section_id)}'|{re.escape(section_id)}):\s*note\((.+)\),$"
                section_note = re.search(pattern, block, re.M)
                if not section_note:
                    errors.append(f"{slug}/{section_id}: missing structured Beginner explanation")
                    continue
                strings = re.findall(r"'(?:[^'\\]|\\.)*'", section_note.group(1))
                if len(strings) < 3:
                    errors.append(f"{slug}/{section_id}: needs why, mechanism, and analogy")

    planned_slugs = re.findall(r"\{ slug: '([^']+)', name: '[^']+', state: 'planned'", wiki)
    for slug in planned_slugs:
        entry = re.search(rf"^\s{{2}}(?:'{re.escape(slug)}'|{re.escape(slug)}):\s*\{{([^\n]+)\}},$", planned_guidance, re.M)
        if not entry:
            errors.append(f"{slug}: missing planned-stage Beginner explanation")
            continue
        for field in ("purpose:", "mechanism:", "picture:", "boundary:"):
            if field not in entry.group(1):
                errors.append(f"{slug}: planned guide missing {field[:-1]}")

    term_ids = set(re.findall(r'<Term id="([^"]+)"', all_lessons))
    glossary_terms = re.findall(r"\['([^']+)',", wiki[wiki.index("export const glossary"):])
    glossary_ids = {term.lower().replace(" ", "-") for term in glossary_terms}
    for term_id in sorted(term_ids - glossary_ids):
        errors.append(f"glossary: missing target for {term_id}")

    if errors:
        print("Beginner Mode audit failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Beginner Mode audit passed: {lesson_count} completed lessons, {total_sections} structured section checks, {len(planned_slugs)} planned lessons, {len(term_ids)} glossary links.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
