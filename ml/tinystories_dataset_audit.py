"""Audit and freeze the bounded TinyStories development split.

This script does not train model weights. It verifies the Lesson 1 source
manifest, measures structure and duplication, checks the provisional tokenizer,
and records whether the sample is suitable for the next development lessons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from tokenizers import Tokenizer

from paths import RAW_DATA_DIR, ROOT, local_result, local_run
from shakespeare_transformer import atomic_json_write

DEFAULT_DATA_DIR = RAW_DATA_DIR / "tinystories-sample"
DEFAULT_TOKENIZER = local_run("tinystories-transition-001") / "provisional-tokenizer.json"
DEFAULT_METRICS = local_result("tinystories-dataset-audit.json")
DEFAULT_RUN_DIR = local_run("tinystories-dataset-audit-001")
WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
TEMPLATE_PHRASES = (
    "once upon a time",
    "one day",
    "lived happily ever after",
    "was very happy",
    "learned a valuable lesson",
)
CONTENT_SCREENS = {
    "injuryOrDeathTerms": re.compile(r"\b(?:die|dies|died|dead|death|kill|killed|blood|hurt|injured)\b"),
    "fearOrDangerTerms": re.compile(r"\b(?:afraid|scared|danger|dangerous|monster|fire)\b"),
    "conflictTerms": re.compile(r"\b(?:fight|fought|angry|steal|stole|punish|punished)\b"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}:{line_number} is invalid JSON: {error}") from error
            if not isinstance(row.get("sourceRow"), int) or not isinstance(row.get("text"), str):
                raise ValueError(f"{path.name}:{line_number} lacks integer sourceRow or string text")
            rows.append(row)
    return rows


def normalise(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def words(text: str) -> list[str]:
    return WORD_RE.findall(normalise(text))


def shingles(text: str, width: int = 5) -> set[tuple[str, ...]]:
    tokens = words(text)
    return {tuple(tokens[index:index + width]) for index in range(max(0, len(tokens) - width + 1))}


def jaccard(left: set, right: set) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def duplicate_counts(rows: list[dict]) -> dict:
    raw = Counter(row["text"] for row in rows)
    normalised = Counter(normalise(row["text"]) for row in rows)
    return {
        "exactDuplicateGroups": sum(count > 1 for count in raw.values()),
        "exactDuplicateRowsBeyondFirst": sum(count - 1 for count in raw.values()),
        "normalisedDuplicateGroups": sum(count > 1 for count in normalised.values()),
        "normalisedDuplicateRowsBeyondFirst": sum(count - 1 for count in normalised.values()),
    }


def cross_split_similarity(train_rows: list[dict], validation_rows: list[dict], threshold: float) -> dict:
    train_raw: dict[str, list[int]] = defaultdict(list)
    train_normalised: dict[str, list[int]] = defaultdict(list)
    for row in train_rows:
        train_raw[row["text"]].append(row["sourceRow"])
        train_normalised[normalise(row["text"])].append(row["sourceRow"])
    raw_exact_pairs = []
    normalised_exact_pairs = []
    for row in validation_rows:
        for train_source_row in train_raw.get(row["text"], []):
            raw_exact_pairs.append({"trainSourceRow": train_source_row, "validationSourceRow": row["sourceRow"]})
        for train_source_row in train_normalised.get(normalise(row["text"]), []):
            normalised_exact_pairs.append({"trainSourceRow": train_source_row, "validationSourceRow": row["sourceRow"]})

    train_shingles = [(row["sourceRow"], shingles(row["text"])) for row in train_rows]
    near_pairs = []
    strongest = {"similarity": 0.0, "trainSourceRow": None, "validationSourceRow": None}
    comparisons = 0
    for validation_row in validation_rows:
        validation_shingles = shingles(validation_row["text"])
        for train_source_row, train_story_shingles in train_shingles:
            comparisons += 1
            similarity = jaccard(train_story_shingles, validation_shingles)
            if similarity > strongest["similarity"]:
                strongest = {
                    "similarity": similarity,
                    "trainSourceRow": train_source_row,
                    "validationSourceRow": validation_row["sourceRow"],
                }
            if similarity >= threshold:
                near_pairs.append({
                    "similarity": similarity,
                    "trainSourceRow": train_source_row,
                    "validationSourceRow": validation_row["sourceRow"],
                })
    near_pairs.sort(key=lambda pair: pair["similarity"], reverse=True)
    return {
        "method": "exact match after NFKC/lowercase/whitespace normalisation plus word 5-gram Jaccard",
        "nearDuplicateThreshold": threshold,
        "comparisons": comparisons,
        "rawExactCrossSplitPairs": raw_exact_pairs,
        "normalisedExactCrossSplitPairs": normalised_exact_pairs,
        "nearDuplicatePairsAtOrAboveThreshold": near_pairs,
        "strongestPair": strongest,
    }


def structure(rows: list[dict]) -> dict:
    tokenised = [words(row["text"]) for row in rows]
    word_counts = [len(tokens) for tokens in tokenised]
    characters = [len(row["text"]) for row in rows]
    prefixes = Counter(" ".join(tokens[:5]) for tokens in tokenised if tokens)
    all_words = Counter(token for tokens in tokenised for token in tokens)
    lowercase_texts = [normalise(row["text"]) for row in rows]
    return {
        "stories": len(rows),
        "characters": sum(characters),
        "words": sum(word_counts),
        "distinctLowercaseWordForms": len(all_words),
        "storyWords": {
            "minimum": min(word_counts),
            "median": statistics.median(word_counts),
            "mean": statistics.mean(word_counts),
            "p95": sorted(word_counts)[math.ceil(0.95 * len(word_counts)) - 1],
            "maximum": max(word_counts),
        },
        "topFiveWordOpenings": [
            {"opening": opening, "stories": count} for opening, count in prefixes.most_common(8)
        ],
        "templatePhraseStories": {
            phrase: sum(phrase in text for text in lowercase_texts) for phrase in TEMPLATE_PHRASES
        },
        "mostCommonWords": [
            {"word": word, "occurrences": count} for word, count in all_words.most_common(12)
        ],
    }


def integrity(rows: list[dict]) -> dict:
    texts = [row["text"] for row in rows]
    return {
        "emptyStories": sum(not text.strip() for text in texts),
        "unicodeReplacementCharacters": sum(text.count("\ufffd") for text in texts),
        "unexpectedControlCharacters": sum(
            unicodedata.category(character) == "Cc" and character not in "\n\r\t"
            for text in texts for character in text
        ),
        "storiesUnder200Characters": sum(len(text) < 200 for text in texts),
        "storiesOver2000Characters": sum(len(text) > 2000 for text in texts),
        "duplicateSourceRowIds": len(rows) - len({row["sourceRow"] for row in rows}),
    }


def content_screen(rows: list[dict]) -> dict:
    texts = [normalise(row["text"]) for row in rows]
    return {
        name: sum(bool(pattern.search(text)) for text in texts)
        for name, pattern in CONTENT_SCREENS.items()
    }


def tokenizer_audit(tokenizer: Tokenizer, rows: list[dict]) -> dict:
    token_counts = []
    unknown_id = tokenizer.token_to_id("<unk>")
    unknown_tokens = 0
    roundTripMismatches = 0
    for row in rows:
        encoding = tokenizer.encode(row["text"])
        token_counts.append(len(encoding.ids))
        unknown_tokens += sum(token_id == unknown_id for token_id in encoding.ids)
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=True)
        if decoded != unicodedata.normalize("NFKC", row["text"]):
            roundTripMismatches += 1
    return {
        "stories": len(rows),
        "tokens": sum(token_counts),
        "tokensPerStoryMedian": statistics.median(token_counts),
        "tokensPerStoryP95": sorted(token_counts)[math.ceil(0.95 * len(token_counts)) - 1],
        "unknownTokens": unknown_tokens,
        "nfkcRoundTripMismatches": roundTripMismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.80)
    args = parser.parse_args()
    if not 0.5 <= args.near_duplicate_threshold <= 1.0:
        raise SystemExit("near-duplicate threshold must be between 0.5 and 1.0")
    if not args.tokenizer.is_file():
        raise SystemExit(f"Missing Lesson 1 tokenizer: {args.tokenizer}\nRun ml/tinystories_transition.py first.")

    started = time.perf_counter()
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    train_path = args.data_dir / "train.jsonl"
    validation_path = args.data_dir / "validation.jsonl"
    observed_hashes = {"train": sha256(train_path), "validation": sha256(validation_path)}
    expected_hashes = {"train": manifest["train"]["sha256"], "validation": manifest["validation"]["sha256"]}
    if observed_hashes != expected_hashes:
        raise SystemExit(f"Dataset hash mismatch: expected {expected_hashes}, observed {observed_hashes}")

    train_rows = load_rows(train_path)
    validation_rows = load_rows(validation_path)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    leakage = cross_split_similarity(train_rows, validation_rows, args.near_duplicate_threshold)
    payload = {
        "status": "Complete",
        "experimentId": args.run_dir.name,
        "updatedAt": datetime.now(UTC).isoformat(),
        "datasetDecision": {
            "datasetId": "tinystories-development-v1",
            "decision": "Frozen for tokenizer, batching, baseline, and short pretraining experiments",
            "notApprovedFor": "Final scaling run or claims about the complete TinyStories distribution",
            "trainStories": len(train_rows),
            "validationStories": len(validation_rows),
            "sourceManifestHashes": observed_hashes,
        },
        "integrity": {
            "train": integrity(train_rows),
            "validation": integrity(validation_rows),
        },
        "duplicates": {
            "train": duplicate_counts(train_rows),
            "validation": duplicate_counts(validation_rows),
            "crossSplit": leakage,
        },
        "structure": {
            "train": structure(train_rows),
            "validation": structure(validation_rows),
        },
        "contentTermScreen": {
            "method": "Case-insensitive keyword triage; presence is not a safety judgement and absence is not a safety guarantee",
            "train": content_screen(train_rows),
            "validation": content_screen(validation_rows),
        },
        "provisionalTokenizer": {
            "vocabularySize": tokenizer.get_vocab_size(),
            "train": tokenizer_audit(tokenizer, train_rows),
            "validation": tokenizer_audit(tokenizer, validation_rows),
        },
        "elapsedSeconds": time.perf_counter() - started,
        "limitations": [
            "The audit covers 1,200 sampled stories, not all 2.14 million source rows.",
            "Word 5-gram Jaccard can miss paraphrases and stories that share a template but use different words.",
            "Keyword screens locate text for human review; they do not classify harm, age suitability, bias, or factuality.",
            "The sample is frozen for development comparisons only; later data-scaling experiments require a larger separately versioned corpus.",
        ],
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "runId": args.run_dir.name,
        "datasetId": payload["datasetDecision"]["datasetId"],
        "trainStories": len(train_rows),
        "validationStories": len(validation_rows),
        "sourceManifestHashes": observed_hashes,
        "nearDuplicateMethod": leakage["method"],
        "nearDuplicateThreshold": args.near_duplicate_threshold,
        "modelTraining": "none",
        "tokenizerInput": str(args.tokenizer.relative_to(ROOT)),
    }
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    atomic_json_write(args.metrics, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
