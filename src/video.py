"""Frame sampling and an on-disk frame cache.

Videos are decoded once by `scripts/data/extract_frames.py`: `cache_frames`
writes `max_frames` uniformly sampled JPEGs per clip plus a `meta.json` with
the source fps and the sampled frame indices. Training and inference then
read a uniform subset of the cache (`load_cached_frames`), so the mp4s can be
deleted after extraction (29.9 GB -> ~6 GB at 64 frames).

The frame indices and fps travel with the frames because Qwen3-VL / Qwen3.5
write a `<t seconds>` text timestamp before each temporal patch, computed from
`fps` and `frames_indices`. Without them, transformers falls back to 24 fps and
every timestamp is wrong (ReVA clips are not all 24 fps).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from src.utils.io_utils import load_json, save_json

META_FILE = "meta.json"


@dataclass
class FrameSet:
    """RGB frames `(T, H, W, 3)` uint8 plus where they sit in the source clip."""

    frames: np.ndarray
    frame_indices: List[int]
    fps: float
    total_frames: int

    @property
    def timestamps(self) -> List[float]:
        return [i / self.fps for i in self.frame_indices]

    @property
    def duration(self) -> float:
        return self.total_frames / self.fps if self.fps else 0.0


def uniform_indices(total: int, num: int, phase: float = 0.5) -> List[int]:
    """`num` indices, one per equal segment of `[0, total)`, at relative
    position `phase` inside each segment (0.5 = centre, the default).

    Segment centres (not `linspace(0, total-1)`) avoid always sampling the very
    first and last frames, matching the usual TSN-style uniform sampler. Other
    phases give temporally jittered views of the same clip (test-time
    augmentation).
    """
    if not 0.0 <= phase < 1.0:
        raise ValueError(f"phase must be in [0, 1), got {phase}")
    if total <= 0:
        raise ValueError(f"total must be > 0, got {total}")
    if num <= 0:
        raise ValueError(f"num must be > 0, got {num}")
    seg = total / num
    return [min(int(seg * i + seg * phase), total - 1) for i in range(num)]


def subsample(items: Sequence, num: int, phase: float = 0.5) -> List:
    """Uniform subset of `num` items (all of them when `num >= len`)."""
    if num >= len(items):
        return list(items)
    return [items[i] for i in uniform_indices(len(items), num, phase)]


def limit_threads() -> None:
    """Make this process single-threaded for OpenCV / BLAS.

    Use as the `initializer` of a process pool. With N worker processes each
    spawning a full-machine thread pool, a container's task cap is hit fast:
    on the GPU server 48 workers produced "Can't spawn new thread" and
    half-decoded clips.
    """
    import os

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    import cv2

    cv2.setNumThreads(1)


def read_video(path: str | Path, num_frames: int, decoder_threads: int = 1) -> FrameSet:
    """Decode `num_frames` uniformly sampled RGB frames with OpenCV.

    The FFmpeg decoder is single-threaded by default: extraction parallelizes
    across clips instead (see `limit_threads`).
    """
    import cv2

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")
    # CAP_PROP_N_THREADS is absent on some OpenCV builds; opening with no extra
    # params is the safe fallback.
    params = (
        [cv2.CAP_PROP_N_THREADS, decoder_threads]
        if hasattr(cv2, "CAP_PROP_N_THREADS")
        else []
    )
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, params)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        if total <= 0:
            raise ValueError(f"could not read frame count of {path}")
        wanted = uniform_indices(total, num_frames)
        frames, kept = [], []
        # Sequential decode + grab() is faster and more reliable than seeking
        # for short clips (CAP_PROP_POS_FRAMES is inexact on some codecs).
        targets = set(wanted)
        for idx in range(max(wanted) + 1):
            if idx in targets:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                kept.append(idx)
            elif not cap.grab():
                break
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    # A decode that stops early (resource errors, corrupt stream) must fail
    # loudly rather than be cached as a shorter clip.
    expected = len(set(wanted))
    if len(set(kept)) < 0.9 * expected:
        raise ValueError(
            f"{path}: decoded {len(set(kept))} of {expected} sampled frames"
        )
    return FrameSet(np.stack(frames), kept, fps, total)


def cache_dir_for(cache_root: str | Path, video_path: str) -> Path:
    """`VisDrone/uav..._01.mp4` -> `<cache_root>/VisDrone/uav..._01/`."""
    return Path(cache_root) / Path(video_path).with_suffix("")


def cache_frames(frameset: FrameSet, out_dir: str | Path, quality: int = 95) -> None:
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, frame in enumerate(frameset.frames):
        cv2.imwrite(
            str(out_dir / f"{k:03d}.jpg"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
    save_json(
        {
            "fps": frameset.fps,
            "total_frames": frameset.total_frames,
            "frame_indices": frameset.frame_indices,
            "height": int(frameset.frames.shape[1]),
            "width": int(frameset.frames.shape[2]),
        },
        out_dir / META_FILE,
    )


def is_cached(cache_root: str | Path, video_path: str) -> bool:
    return (cache_dir_for(cache_root, video_path) / META_FILE).exists()


def load_cached_frames(
    cache_root: str | Path,
    video_path: str,
    num_frames: Optional[int] = None,
    phase: float = 0.5,
) -> FrameSet:
    """Read a uniform subset of `num_frames` cached frames (all if None)."""
    import cv2

    d = cache_dir_for(cache_root, video_path)
    meta_path = d / META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(
            f"no frame cache for {video_path} under {cache_root}; "
            "run scripts/data/extract_frames.py first"
        )
    meta = load_json(meta_path)
    positions = list(range(len(meta["frame_indices"])))
    if num_frames is not None:
        positions = subsample(positions, num_frames, phase)
    frames = []
    for k in positions:
        img = cv2.imread(str(d / f"{k:03d}.jpg"))
        if img is None:
            raise FileNotFoundError(f"missing cached frame {d / f'{k:03d}.jpg'}")
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return FrameSet(
        frames=np.stack(frames),
        frame_indices=[meta["frame_indices"][k] for k in positions],
        fps=float(meta["fps"]),
        total_frames=int(meta["total_frames"]),
    )


def load_frames(
    video_path: str,
    num_frames: int,
    cache_root: Optional[str | Path] = None,
    video_root: Optional[str | Path] = None,
    phase: float = 0.5,
) -> FrameSet:
    """Prefer the frame cache; fall back to decoding the mp4."""
    if cache_root is not None and is_cached(cache_root, video_path):
        return load_cached_frames(cache_root, video_path, num_frames, phase)
    if video_root is None:
        raise FileNotFoundError(
            f"{video_path}: not in frame cache {cache_root} and no video_root set"
        )
    return read_video(Path(video_root) / video_path, num_frames)


def overlay_time_bar(
    frames: np.ndarray,
    timestamps: Sequence[float],
    bar_color: tuple = (235, 64, 52),
    text_color: tuple = (255, 255, 255),
    bar_height: int = 16,
) -> np.ndarray:
    """Draw a progress bar + the true time on each frame (a VTimeCoT-style cue).

    The bar fills left-to-right with time and prints the frame's timestamp, so a
    model can *read* when a frame is rather than infer it from position. Purely
    visual and training-free; applies to a (T, H, W, 3) RGB uint8 array.
    """
    import cv2

    out = np.array(frames, copy=True)
    t, h, w = out.shape[:3]
    if t == 0 or h <= bar_height:
        return out
    t_end = max(float(timestamps[-1]), 1e-6)
    for i in range(t):
        top = h - bar_height
        cv2.rectangle(out[i], (0, top), (w, h), (0, 0, 0), -1)
        x = int(min(max(float(timestamps[i]) / t_end, 0.0), 1.0) * w)
        cv2.rectangle(out[i], (0, top), (x, h), bar_color, -1)
        cv2.putText(
            out[i], f"{float(timestamps[i]):.1f}s", (4, h - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA,
        )
    return out
