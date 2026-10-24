"""Download ReVA from the Hugging Face Hub into `data/raw/reva`.

The full release is 29.9 GB. On a small disk, download one source folder at
a time, extract frames, delete the mp4s, repeat (`--folders` + the
`--delete_videos` flag of extract_frames.py).

Usage:
    # annotations only (~14 MB): train.json, val.json, test.json, README.md
    uv run python scripts/data/fetch_data.py --annotations_only
    # everything, or selected source folders
    uv run python scripts/data/fetch_data.py
    uv run python scripts/data/fetch_data.py --folders VisDrone UAVDT
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.io_utils import load_env  # noqa: E402

REPO_ID = "ReVA-Benchmark/ReVA"
ANNOTATIONS = ["train.json", "val.json", "test.json", "README.md"]
FOLDERS = ["VisDrone", "UAVDT", "ERA_Select", "Hawk_UAV"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "raw" / "reva")
    parser.add_argument("--annotations_only", action="store_true")
    parser.add_argument("--folders", nargs="*", default=None, choices=FOLDERS)
    parser.add_argument(
        "--revision", default="main", help="pin a commit for reproducibility"
    )
    args = parser.parse_args()

    load_env(REPO_ROOT / ".env")
    # The dataset is public; a token only raises rate limits.
    if "HF_API_KEY" in os.environ and "HF_TOKEN" not in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HF_API_KEY"]
    from huggingface_hub import snapshot_download

    patterns = list(ANNOTATIONS)
    if not args.annotations_only:
        for folder in args.folders or FOLDERS:
            patterns.append(f"{folder}/**")
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.out,
        allow_patterns=patterns,
    )
    print(f"downloaded {patterns} -> {path}")


if __name__ == "__main__":
    main()
