"""Run a public-package module while keeping ``src`` importable."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def run(module_name: str) -> None:
    source_dir = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(source_dir))
    runpy.run_module(module_name, run_name="__main__")
