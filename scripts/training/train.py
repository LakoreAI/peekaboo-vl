"""Convenience wrapper for the training entrypoint.

Runs standalone (`uv run python scripts/training/train.py ...`) by patching the
repo root onto sys.path, then delegating to `src.pipelines.train` so there is
exactly one implementation of the training loop / CLI.

Usage:
    uv run python scripts/training/train.py --config configs/train.yaml
    uv run python scripts/training/train.py --config configs/train.yaml --epochs 50 --lr 5e-4
"""

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    runpy.run_module("src.pipelines.train", run_name="__main__")
