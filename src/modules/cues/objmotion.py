"""Object motion as text, with the camera's own motion removed.

The ReVA paper's main named failure is confusing camera ego-motion with
object motion. We compute dense optical flow per frame pair, fit the
background's image motion as a RANSAC homography on that flow, subtract it,
and whatever still moves is an independently moving object (vehicle, person,
boat). A homography, not the similarity of `motion.py`: oblique drone views
have strong perspective (near ground streams faster than far ground), which a
similarity leaves behind as large false residuals.

    Farneback flow F(p)  -  (H p - p)  =  residual r(p)
    |r| above a rate threshold -> moving mask -> connected blobs
    blobs binned into a 3x3 grid of frame regions, direction = dominant axis
    of the area-weighted residual ("left to right", "top to bottom", ...)

Rendered in the questions' own vocabulary, e.g.

    Object motion (camera motion removed): top: moving left to right
    (0.0-8.2s); bottom-left: moving top to bottom and bottom to top
    (2.0-6.0s). About 4 moving objects at a time.

Targets Change Detection and Trend/Pattern ("how does the blue car move",
"what is the motion direction of vehicles on the main street"). Parallax of
tall structures (roofs, stands) under a moving camera also leaves residual
flow, but as large regions: blobs above `max_blob_frac` of the frame are
dropped, and a region must hold motion in `min_steps` steps to be reported.

Cost: ~5 ms per frame pair at 320 px on CPU. Precomputed once per clip.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

REGIONS = (
    ("top-left", "top", "top-right"),
    ("left", "center", "right"),
    ("bottom-left", "bottom", "bottom-right"),
)
# Direction bins of the residual (image y points down).
DIRECTIONS = ("left to right", "right to left", "top to bottom", "bottom to top")


@dataclass
class ObjMotionConfig:
    # Processing width (height keeps the aspect ratio).
    proc_width: int = 320
    # A pixel moves if its residual exceeds both thresholds (proc pixels).
    min_residual_px: float = 1.5
    min_rate_px_s: float = 6.0
    # Ignore a border band where flow is unreliable (fraction of each side).
    border: float = 0.04
    min_blob_area: int = 6  # proc pixels
    # Larger "blobs" are parallax or failed compensation, not vehicles/people.
    max_blob_frac: float = 0.02
    # A region is reported if it holds motion in at least this many steps.
    min_steps: int = 3
    # Background homography: flow sampled every `grid` px, RANSAC threshold px.
    grid: int = 4
    ransac_px: float = 1.0
    min_inlier_ratio: float = 0.3
    # Track check: a blob counts only if the next step has a blob within
    # `track_radius` (frame fraction) of where it should have moved to, going
    # the same way (cosine >= `track_cos`). Removes flicker from flow noise.
    track_radius: float = 0.06
    track_cos: float = 0.5
    # A direction is named if it has at least this share of the region's motion.
    dir_share: float = 0.3
    max_regions: int = 4


@dataclass
class Blob:
    cx: float  # centroid, fraction of width
    cy: float  # centroid, fraction of height
    area: int
    rx: float  # mean residual, proc px per second
    ry: float


def _gray_small(frame: np.ndarray, width: int) -> np.ndarray:
    import cv2

    g = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    h, w = g.shape[:2]
    if w != width:
        g = cv2.resize(g, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)
    return g


def background_homography(flow: np.ndarray, cfg: ObjMotionConfig) -> Optional[np.ndarray]:
    """RANSAC homography of the dominant (background) motion in a flow field."""
    import cv2

    h, w = flow.shape[:2]
    ys, xs = np.mgrid[cfg.grid : h - cfg.grid : cfg.grid, cfg.grid : w - cfg.grid : cfg.grid]
    p0 = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
    p1 = p0 + flow[ys.ravel(), xs.ravel()]
    if len(p0) < 8:
        return None
    H, inl = cv2.findHomography(p0, p1, cv2.RANSAC, cfg.ransac_px)
    if H is None or inl is None or inl.mean() < cfg.min_inlier_ratio:
        return None
    return H


def residual_blobs(
    prev: np.ndarray, nxt: np.ndarray, dt: float, cfg: ObjMotionConfig
) -> Optional[List[Blob]]:
    """Independently moving blobs between two frames; None if the background
    motion cannot be fit (then the residual is meaningless)."""
    import cv2

    g0, g1 = _gray_small(prev, cfg.proc_width), _gray_small(nxt, cfg.proc_width)
    h, w = g0.shape[:2]
    # Small window: vehicles are a few pixels wide at this resolution.
    flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 9, 3, 5, 1.1, 0)
    H = background_homography(flow, cfg)
    if H is None:
        return None
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    den = H[2, 0] * xs + H[2, 1] * ys + H[2, 2]
    pred_x = (H[0, 0] * xs + H[0, 1] * ys + H[0, 2]) / den - xs
    pred_y = (H[1, 0] * xs + H[1, 1] * ys + H[1, 2]) / den - ys
    rx, ry = flow[..., 0] - pred_x, flow[..., 1] - pred_y
    mag = np.hypot(rx, ry)
    dt = max(dt, 1e-6)
    mask = (mag >= cfg.min_residual_px) & (mag / dt >= cfg.min_rate_px_s)
    bx, by = int(cfg.border * w), int(cfg.border * h)
    mask[:by] = mask[h - by :] = False
    mask[:, :bx] = mask[:, w - bx :] = False
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < cfg.min_blob_area or area > cfg.max_blob_frac * w * h:
            continue
        sel = labels == i
        blobs.append(
            Blob(
                cx=float(cents[i][0] / w),
                cy=float(cents[i][1] / h),
                area=area,
                rx=float(rx[sel].mean() / dt),
                ry=float(ry[sel].mean() / dt),
            )
        )
    return blobs


def direction_bin(rx: float, ry: float) -> int:
    if abs(rx) >= abs(ry):
        return 0 if rx > 0 else 1
    return 2 if ry > 0 else 3


def region_of(cx: float, cy: float) -> str:
    col = min(2, int(cx * 3))
    row = min(2, int(cy * 3))
    return REGIONS[row][col]


def confirmed(
    steps: Sequence[Optional[List[Blob]]],
    timestamps: Sequence[float],
    cfg: ObjMotionConfig,
    width: int,
    height: int,
) -> List[Optional[List[Blob]]]:
    """Keep blobs that continue into the next step (see `track_radius`)."""
    out: List[Optional[List[Blob]]] = []
    for i, blobs in enumerate(steps):
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        if blobs is None:
            out.append(None)
            continue
        if not nxt:
            out.append([])
            continue
        dt = float(timestamps[i + 1] - timestamps[i])
        keep = []
        for b in blobs:
            px, py = b.cx + b.rx * dt / width, b.cy + b.ry * dt / height
            nb = math.hypot(b.rx, b.ry) or 1.0
            for c in nxt:
                cos = (b.rx * c.rx + b.ry * c.ry) / (nb * (math.hypot(c.rx, c.ry) or 1.0))
                if math.hypot(c.cx - px, c.cy - py) <= cfg.track_radius and cos >= cfg.track_cos:
                    keep.append(b)
                    break
        out.append(keep)
    return out


def describe_objmotion(
    steps: Sequence[Optional[List[Blob]]],
    timestamps: Sequence[float],
    cfg: Optional[ObjMotionConfig] = None,
) -> dict:
    """Aggregate per-step blobs into one prompt line plus a summary."""
    cfg = cfg or ObjMotionConfig()
    regions: dict = {}
    counts = []
    ok = 0
    for i, blobs in enumerate(steps):
        if blobs is None:
            continue
        ok += 1
        counts.append(len(blobs))
        t0, t1 = float(timestamps[i]), float(timestamps[i + 1])
        for b in blobs:
            r = regions.setdefault(
                region_of(b.cx, b.cy),
                {"dirs": [0.0] * 4, "steps": set(), "t0": t0, "t1": t1, "weight": 0.0},
            )
            r["dirs"][direction_bin(b.rx, b.ry)] += b.area
            r["steps"].add(i)
            r["t0"], r["t1"] = min(r["t0"], t0), max(r["t1"], t1)
            r["weight"] += b.area
    kept = [
        (name, r) for name, r in regions.items() if len(r["steps"]) >= cfg.min_steps
    ]
    kept.sort(key=lambda kv: kv[1]["weight"] * len(kv[1]["steps"]), reverse=True)
    kept = kept[: cfg.max_regions]
    head = "Object motion (camera motion removed):"
    ok_fraction = ok / len(steps) if steps else 0.0
    if not kept:
        text = f"{head} no clearly moving objects."
    else:
        parts = []
        for name, r in kept:
            total = sum(r["dirs"]) or 1.0
            order = sorted(range(4), key=lambda k: r["dirs"][k], reverse=True)
            names = [DIRECTIONS[k] for k in order if r["dirs"][k] / total >= cfg.dir_share]
            names = names[:2] or [DIRECTIONS[order[0]]]
            parts.append(f"{name}: moving {' and '.join(names)} ({r['t0']:.1f}-{r['t1']:.1f}s)")
        # Typical number of moving blobs at a time, over steps that had any.
        busy = [c for c in counts if c > 0]
        typical = int(round(float(np.median(busy)))) if busy else 0
        text = f"{head} " + "; ".join(parts) + "."
        if typical >= 2:
            text += f" About {typical} moving objects at a time."
    if ok_fraction < 0.5:
        text += " (Low-confidence estimate.)"
    return {
        "text": text,
        "summary": {
            "ok_fraction": ok_fraction,
            "regions": {name: len(r["steps"]) for name, r in regions.items()},
            "max_blobs": max(counts) if counts else 0,
        },
    }


def objmotion_record(
    frames: Sequence[np.ndarray],
    timestamps: Sequence[float],
    cfg: Optional[ObjMotionConfig] = None,
) -> dict:
    """Everything `compute_objmotion_cues.py` stores per clip."""
    cfg = cfg or ObjMotionConfig()
    if len(frames) != len(timestamps):
        raise ValueError("frames and timestamps must have equal length")
    steps = [
        residual_blobs(
            frames[i], frames[i + 1], float(timestamps[i + 1] - timestamps[i]), cfg
        )
        for i in range(len(frames) - 1)
    ]
    h, w = frames[0].shape[:2]
    pw, ph = cfg.proc_width, round(h * cfg.proc_width / w)
    return describe_objmotion(confirmed(steps, timestamps, cfg, pw, ph), timestamps, cfg)
