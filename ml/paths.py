"""Shared filesystem locations for learner-owned runs.

Committed reference evidence is read-only course material. Training scripts
write to ignored local paths so cloning the repository starts with a blank lab
without overwriting the published comparison data.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = Path(os.environ.get("LEARNINGLLMS_RAW_DATA_DIR", ROOT / "data" / "raw"))
LOCAL_RESULTS_DIR = Path(os.environ.get("LEARNINGLLMS_RESULTS_DIR", ROOT / "public" / "data" / "local"))
LOCAL_EXPERIMENTS_DIR = Path(os.environ.get("LEARNINGLLMS_EXPERIMENTS_DIR", ROOT / "work" / "experiments"))
LEGACY_EXPERIMENTS_DIR = ROOT / "experiments"


def local_result(filename: str) -> Path:
    path = LOCAL_RESULTS_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def local_run(run_id: str) -> Path:
    return LOCAL_EXPERIMENTS_DIR / run_id


def available_run(run_id: str) -> Path:
    """Prefer a learner run, with the pre-migration local lab as fallback."""
    current = local_run(run_id)
    return current if current.exists() else LEGACY_EXPERIMENTS_DIR / run_id
