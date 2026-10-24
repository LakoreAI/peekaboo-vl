"""Decode every ReVA clip once into a frame cache.

Writes `<frames_root>/<video_path without .mp4>/{000..N}.jpg` + `meta.json`
(fps, total frames, sampled indices). Training reads uniform subsets of the
cache, so extract the MOST frames you will ever want (default 64): you can
subsample later, but not re-extract once the mp4s are deleted.

Usage:
    uv run python scripts/data/extract_frames.py --workers 8
    # small disk: extract, then delete each mp4 after its frames are written
    uv run python scripts/data/extract_frames.py --delete_videos
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data import load_split  # noqa: E402
from src.video import (  # noqa: E402
    cache_dir_for,
    cache_frames,
    is_cached,
    limit_threads,
    read_video,
)


def extract_one(
    video_root: str, frames_root: str, video_path: str, num_frames: int, delete: bool
) -> str:
    src = Path(video_root) / video_path
    if is_cached(frames_root, video_path):
        status = "cached"
    else:
        frames = read_video(src, num_frames)
        cache_frames(frames, cache_dir_for(frames_root, video_path))
        status = f"{len(frames.frame_indices)} frames @ {frames.fps:.2f} fps"
    if delete and src.exists() and is_cached(frames_root, video_path):
        src.unlink()
    return status


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--reva_root", type=Path, default=REPO_ROOT / "data" / "raw" / "reva"
    )
    parser.add_argument(
        "--frames_root", type=Path, default=REPO_ROOT / "data" / "processed" / "frames"
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delete_videos", action="store_true")
    args = parser.parse_args()

    videos = sorted(
        {
            ex.video_path
            for s in args.splits
            for ex in load_split(args.reva_root / f"{s}.json")
        }
    )
    todo = [
        v
        for v in videos
        if (args.reva_root / v).exists() or is_cached(args.frames_root, v)
    ]
    missing = len(videos) - len(todo)
    print(
        f"{len(videos)} videos referenced, {len(todo)} available, {missing} not downloaded"
    )

    failures = []
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=limit_threads
    ) as pool:
        futures = {
            pool.submit(
                extract_one,
                str(args.reva_root),
                str(args.frames_root),
                v,
                args.num_frames,
                args.delete_videos,
            ): v
            for v in todo
        }
        for i, fut in enumerate(as_completed(futures), 1):
            v = futures[fut]
            try:
                status = fut.result()
            except Exception as e:  # noqa: BLE001 - report and continue
                failures.append((v, str(e)))
                status = f"FAILED: {e}"
            if i % 50 == 0 or status.startswith("FAILED"):
                print(f"[{i}/{len(todo)}] {v}: {status}")

    print(
        f"done: {len(todo) - len(failures)} ok, {len(failures)} failed, {missing} missing"
    )
    for v, err in failures[:20]:
        print(f"  {v}: {err}")


if __name__ == "__main__":
    main()
