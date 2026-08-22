#!/usr/bin/env python3
"""Report whether this machine matches a documented LearningLLMs path."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys


def memory_gib() -> float | None:
    if sys.platform == "darwin":
        value = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=False).stdout.strip()
        return int(value) / 2**30 if value.isdigit() else None
    if sys.platform.startswith("linux"):
        try:
            line = next(line for line in open("/proc/meminfo", encoding="utf-8") if line.startswith("MemTotal:"))
            return int(line.split()[1]) * 1024 / 2**30
        except (OSError, StopIteration, ValueError):
            return None
    return None


def recommendation() -> tuple[str, str]:
    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        return "apple", "uv sync --extra apple"
    if sys.platform.startswith("linux") and shutil.which("nvidia-smi"):
        return "linux-nvidia", "Choose CUDA 12 or 13 after checking driver/toolkit compatibility; then run uv sync --extra linux-cuda12 (or linux-cuda13)."
    if sys.platform.startswith("linux"):
        return "linux-cpu", "uv sync --extra linux-cpu"
    if sys.platform == "win32":
        return "windows-adaptation", "Use WSL2 with a supported NVIDIA GPU, a Linux machine, or port the model code to PyTorch. Native Windows MLX is not documented."
    return "unsupported", "Use a documented Apple-silicon or Linux environment, or adapt the numerical backend."


def main() -> None:
    path, command = recommendation()
    print(json.dumps({
        "operatingSystem": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "memoryGiB": round(memory_gib(), 1) if memory_gib() is not None else None,
        "path": path,
        "recommendedSetup": command,
    }, indent=2))


if __name__ == "__main__":
    main()
