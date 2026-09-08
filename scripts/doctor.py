#!/usr/bin/env python3
"""Fail-fast environment diagnostics for the desktop launcher."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path


REQUIRED_IMPORTS = (
    "anyio", "httpx", "numpy", "pandas", "streamlit", "torch",
    "transformers", "soundfile", "librosa", "pydantic",
)


def main() -> int:
    failures = []
    for module in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module)
        except Exception as exc:
            failures.append(f"{module}: {exc}")
    if shutil.which("ffmpeg") is None:
        failures.append("ffmpeg executable is not on PATH")

    repo = Path(__file__).resolve().parent.parent
    try:
        installed = importlib.import_module("coherex")
        installed_root = Path(installed.__file__).resolve().parent.parent
        if installed_root != repo:
            failures.append(f"coherex resolves to {installed_root}, expected {repo}")
    except Exception as exc:
        failures.append(f"coherex: {exc}")

    if failures:
        print("CohereX environment is incomplete:")
        for failure in failures:
            print(f"  - {failure}")
        print(f'\nRepair with:\n  "{sys.executable}" -m pip install --upgrade -e "{repo}[dev,langid]"')
        return 1

    check = subprocess.run([sys.executable, "-m", "pip", "check"], check=False)
    if check.returncode:
        return check.returncode
    print("CohereX environment checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
