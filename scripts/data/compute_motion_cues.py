"""Precompute the camera-motion cue for every cached clip.

Reads the frame cache, estimates frame-to-frame similarity transforms
(`src.modules.cues.motion`), and writes one JSON line per clip to
`<cues_root>/motion.jsonl`: {"video_path", "text", "summary", "steps"}.

Usage:
    uv run python scripts/data/compute_motion_cues.py --workers 8
    uv run python scripts/data/compute_motion_cues.py --num_frames 64 --limit 20   # inspect a few
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.modules.cues.motion import MotionConfig, motion_record  # noqa: E402
from src.utils.io_utils import load_json, write_jsonl  # noqa: E402
from src.video import is_cached, limit_threads, load_cached_frames  # noqa: E402


def one(frames_root: str, video_path: str, num_frames: int, cfg: MotionConfig) -> dict:
    fs = load_cached_frames(frames_root, video_path, num_frames)
    return {
        "video_path": video_path,
        **motion_record(list(fs.frames), fs.timestamps, cfg),
    }


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
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "cues" / "motion.jsonl",
    )
    parser.add_argument(
        "--num_frames", type=int, default=32, help="frames used for estimation"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--annotations",
        type=Path,
        nargs="+",
        default=None,
        help="take the clip list from these annotation files instead of ReVA's",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="only the first N clips, printed"
    )
    for f in fields(MotionConfig):
        parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    args = parser.parse_args()
    cfg = MotionConfig(**{f.name: getattr(args, f.name) for f in fields(MotionConfig)})

    files = args.annotations or [
        args.reva_root / f"{s}.json" for s in ("train", "val", "test")
    ]
    videos = sorted({rec["video_path"] for f in files for rec in load_json(f)["QA"]})
    videos = [v for v in videos if is_cached(args.frames_root, v)]
    if args.limit:
        videos = videos[: args.limit]
    print(f"{len(videos)} cached clips")

    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=limit_threads
    ) as pool:
        rows = list(
            pool.map(
                one,
                [str(args.frames_root)] * len(videos),
                videos,
                [args.num_frames] * len(videos),
                [cfg] * len(videos),
                chunksize=8,
            )
        )
    low = sum(r["summary"]["ok_fraction"] < 0.5 for r in rows)
    if args.limit:
        for r in rows:
            print(f"{r['video_path']}\n  {r['text']}")
        print(
            f"inspected {len(rows)} rows (--limit set: not writing {args.out})  "
            f"({low} low-confidence)"
        )
        return
    write_jsonl(rows, args.out)
    print(f"saved {len(rows)} rows -> {args.out}  ({low} low-confidence)")


if __name__ == "__main__":
    main()
