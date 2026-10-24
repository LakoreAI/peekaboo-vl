"""Convenience wrapper for the prediction/evaluation entrypoint.

Runs standalone by patching the repo root onto sys.path, then delegating to
`src.pipelines.predict`, so there is exactly one implementation.

Usage:
    uv run python scripts/training/evaluate.py --ckpt checkpoints/<run>/best.pt \
        --split data/processed/splits/dev.json --tta cyclic --out_dir results/<run>/dev
"""

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    runpy.run_module("src.pipelines.predict", run_name="__main__")
