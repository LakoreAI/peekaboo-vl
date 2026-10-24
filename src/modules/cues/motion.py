"""Camera ego-motion as text: "Tell, don't learn".

Drone footage is close to planar, so the frame-to-frame image motion of the
background is well described by a 2-D similarity transform (translation,
isotropic scale, rotation). For each consecutive frame pair we:

    ORB keypoints -> brute-force Hamming matches -> RANSAC similarity M
    M maps frame t onto frame t+1; read off, at the image centre:
        shift  (dx, dy)  -> pan / truck direction  (scene moves opposite)
        scale  s         -> zoom in / descend (s > 1), zoom out / ascend (s < 1)
        angle  theta     -> yaw (scene rotates opposite to the camera)

Steps are labelled with thresholds on per-second rates, consecutive equal
labels are merged into segments, and the result is rendered as one line that
is prepended to the prompt (train and test alike), e.g.

    Camera motion: 0.0-4.1s pans right, slow; 4.1-9.6s zooms out (ascending);
    9.6-15.0s static. Overall: camera moved 0.42 frame-widths right, view
    shrunk x0.81 (ascended / zoomed out).

The RANSAC inlier ratio doubles as a crude "how much moves independently of
the camera" signal; it is kept in the per-step record but not rendered yet.

Cost: a few ms per frame pair on CPU. Precomputed for every clip by
`scripts/data/compute_motion_cues.py`.
"""

import math
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class MotionConfig:
    n_features: int = 1500
    # Lowe ratio test on the two nearest Hamming neighbours.
    ratio: float = 0.75
    min_matches: int = 25
    ransac_threshold: float = 3.0
    # Labelling thresholds, all per second. Shift is in frame diagonals.
    static_shift: float = 0.015
    fast_shift: float = 0.08
    zoom_rate: float = 0.015  # |log scale| per second
    yaw_rate: float = 1.5  # degrees per second
    # Segments shorter than this (seconds) are absorbed by a neighbour.
    min_segment: float = 0.8
    # "image": image-space wording ("pans toward the top of the frame and
    # zooms in"). "camera": the dataset's answer vocabulary ("moving forward",
    # "panning right", "stationary") plus the dominant motion.
    phrasing: str = "image"


@dataclass
class MotionStep:
    t0: float
    t1: float
    ok: bool
    # Scene shift of the image centre, in pixels, from t0 to t1. Positive dx:
    # scene moved right (camera panned left). Positive dy: scene moved down.
    dx: float = 0.0
    dy: float = 0.0
    scale: float = 1.0
    # Counter-clockwise rotation of the scene on screen, in degrees.
    rot_ccw: float = 0.0
    inlier_ratio: float = 0.0
    n_matches: int = 0


def _to_gray(frame: np.ndarray) -> np.ndarray:
    import cv2

    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def estimate_pair(
    prev: np.ndarray, nxt: np.ndarray, cfg: MotionConfig, detector=None
) -> Optional[dict]:
    """Similarity transform taking `prev` to `nxt`; None if it cannot be fit."""
    import cv2

    g0, g1 = _to_gray(prev), _to_gray(nxt)
    orb = detector or cv2.ORB_create(nfeatures=cfg.n_features)
    k0, d0 = orb.detectAndCompute(g0, None)
    k1, d1 = orb.detectAndCompute(g1, None)
    if (
        d0 is None
        or d1 is None
        or len(k0) < cfg.min_matches
        or len(k1) < cfg.min_matches
    ):
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = []
    for pair in matcher.knnMatch(d0, d1, k=2):
        if len(pair) == 2 and pair[0].distance < cfg.ratio * pair[1].distance:
            good.append(pair[0])
    if len(good) < cfg.min_matches:
        return None
    p0 = np.float32([k0[m.queryIdx].pt for m in good])
    p1 = np.float32([k1[m.trainIdx].pt for m in good])
    M, inliers = cv2.estimateAffinePartial2D(
        p0, p1, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_threshold
    )
    if M is None:
        return None
    h, w = g0.shape[:2]
    centre = np.array([w / 2.0, h / 2.0, 1.0])
    moved = M @ centre
    return {
        "dx": float(moved[0] - centre[0]),
        "dy": float(moved[1] - centre[1]),
        "scale": float(math.hypot(M[0, 0], M[1, 0])),
        # Image y points down, so atan2(M10, M00) is clockwise on screen.
        "rot_ccw": float(-math.degrees(math.atan2(M[1, 0], M[0, 0]))),
        "inlier_ratio": float(inliers.mean()) if inliers is not None else 0.0,
        "n_matches": len(good),
    }


def estimate_motion(
    frames: Sequence[np.ndarray],
    timestamps: Sequence[float],
    cfg: Optional[MotionConfig] = None,
) -> List[MotionStep]:
    """One `MotionStep` per consecutive frame pair."""
    import cv2

    cfg = cfg or MotionConfig()
    if len(frames) != len(timestamps):
        raise ValueError("frames and timestamps must have equal length")
    orb = cv2.ORB_create(nfeatures=cfg.n_features)
    steps = []
    for i in range(len(frames) - 1):
        est = estimate_pair(frames[i], frames[i + 1], cfg, detector=orb)
        t0, t1 = float(timestamps[i]), float(timestamps[i + 1])
        steps.append(
            MotionStep(t0, t1, ok=False)
            if est is None
            else MotionStep(t0, t1, ok=True, **est)
        )
    return steps


# --- labelling and rendering --------------------------------------------------


def label_step(step: MotionStep, diag: float, cfg: MotionConfig) -> str:
    """Short phrase for one step, e.g. "pans right, slow" or "static"."""
    if not step.ok:
        return "unclear"
    dt = max(step.t1 - step.t0, 1e-6)
    parts = []

    shift = math.hypot(step.dx, step.dy) / diag / dt
    if shift >= cfg.static_shift:
        # The camera moves opposite to the scene. "up"/"down" are frame
        # directions; for a forward-tilted drone camera "up" is forward.
        cam_x, cam_y = -step.dx, -step.dy
        if abs(cam_x) >= abs(cam_y):
            direction = "right" if cam_x > 0 else "left"
        else:
            direction = (
                "toward the top of the frame"
                if cam_y < 0
                else "toward the bottom of the frame"
            )
        speed = "fast" if shift >= cfg.fast_shift else "slow"
        parts.append(f"pans {direction}, {speed}")

    zoom = math.log(max(step.scale, 1e-6)) / dt
    if zoom >= cfg.zoom_rate:
        parts.append("zooms in (descending)")
    elif zoom <= -cfg.zoom_rate:
        parts.append("zooms out (ascending)")

    yaw = step.rot_ccw / dt
    if abs(yaw) >= cfg.yaw_rate:
        # Scene turning counter-clockwise means the camera yaws clockwise.
        parts.append("rotates clockwise" if yaw > 0 else "rotates counter-clockwise")

    return " and ".join(parts) if parts else "static"


def camera_label(step: MotionStep, diag: float, cfg: MotionConfig) -> tuple:
    """(primary motion, phrase) in camera terms, e.g. ("forward", "moving
    forward quickly while rotating clockwise").

    A forward-tilted drone camera flying forward sees the scene stream toward
    the bottom of the frame and expand; that is "moving forward", not
    "panning up and zooming in"."""
    if not step.ok:
        return "unclear", "unclear"
    dt = max(step.t1 - step.t0, 1e-6)
    shift = math.hypot(step.dx, step.dy) / diag / dt
    zoom = math.log(max(step.scale, 1e-6)) / dt
    yaw = step.rot_ccw / dt
    cam_x, cam_y = -step.dx, -step.dy
    moving = shift >= cfg.static_shift
    lateral = moving and abs(cam_x) >= abs(cam_y)
    speed = "quickly" if shift >= cfg.fast_shift else "slowly"

    if lateral:
        primary = "pan_right" if cam_x > 0 else "pan_left"
        phrase = f"panning {'right' if cam_x > 0 else 'left'} {speed}"
        if zoom >= cfg.zoom_rate:
            phrase += " while moving forward"
        elif zoom <= -cfg.zoom_rate:
            phrase += " while moving backward"
    elif (moving and cam_y < 0) or zoom >= cfg.zoom_rate:
        primary, phrase = "forward", f"moving forward {speed if moving else 'slowly'}"
        if not moving:
            phrase = "moving closer (descending or zooming in)"
    elif (moving and cam_y > 0) or zoom <= -cfg.zoom_rate:
        primary, phrase = "backward", f"moving backward {speed if moving else 'slowly'}"
        if not moving:
            phrase = "pulling away (ascending or zooming out)"
    else:
        primary, phrase = "stationary", "stationary"

    if abs(yaw) >= cfg.yaw_rate:
        # Scene turning counter-clockwise means the camera yaws clockwise.
        turn = "clockwise" if yaw > 0 else "counter-clockwise"
        if primary == "stationary":
            primary, phrase = "rotating", f"rotating {turn}"
        else:
            phrase += f" while rotating {turn}"
    return primary, phrase


def describe_camera(
    steps: Sequence[MotionStep], width: int, height: int, cfg: MotionConfig
) -> str:
    """Camera-vocabulary rendering (`phrasing="camera"`)."""
    diag = math.hypot(width, height)
    labelled = [camera_label(s, diag, cfg) for s in steps]
    segs = segment(steps, [phrase for _, phrase in labelled], cfg)
    body = "; ".join(f"{s['t0']:.1f}-{s['t1']:.1f}s {s['label']}" for s in segs)
    # Dominant motion by duration, from the primary class of each step.
    share: dict = {}
    for step, (primary, _) in zip(steps, labelled):
        share[primary] = share.get(primary, 0.0) + (step.t1 - step.t0)
    total = sum(share.values()) or 1.0
    names = {
        "forward": "moving forward",
        "backward": "moving backward",
        "pan_right": "panning right",
        "pan_left": "panning left",
        "rotating": "rotating",
        "stationary": "stationary",
        "unclear": "unclear",
    }
    top, dur = max(share.items(), key=lambda kv: kv[1])
    text = f"Camera motion: {body}. Mostly {names[top]} ({100 * dur / total:.0f}% of the clip)."
    if summarize(steps, width, height)["ok_fraction"] < 0.5:
        text += " (Low-confidence estimate.)"
    return text


def segment(
    steps: Sequence[MotionStep], labels: Sequence[str], cfg: MotionConfig
) -> List[dict]:
    """Merge runs of equal labels; absorb segments shorter than `min_segment`."""
    segs: List[dict] = []
    for step, label in zip(steps, labels):
        if segs and segs[-1]["label"] == label:
            segs[-1]["t1"] = step.t1
        else:
            segs.append({"t0": step.t0, "t1": step.t1, "label": label})
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, s in enumerate(segs):
            if s["t1"] - s["t0"] < cfg.min_segment:
                # Give the short segment to its longer neighbour.
                left = segs[i - 1] if i > 0 else None
                right = segs[i + 1] if i + 1 < len(segs) else None
                target = max(
                    (n for n in (left, right) if n is not None),
                    key=lambda n: n["t1"] - n["t0"],
                )
                target["t0"] = min(target["t0"], s["t0"])
                target["t1"] = max(target["t1"], s["t1"])
                segs.pop(i)
                changed = True
                break
    # Neighbours may now share a label.
    merged: List[dict] = []
    for s in segs:
        if merged and merged[-1]["label"] == s["label"]:
            merged[-1]["t1"] = s["t1"]
        else:
            merged.append(dict(s))
    return merged


def summarize(steps: Sequence[MotionStep], width: int, height: int) -> dict:
    """Net motion over the clip, from the successfully estimated steps."""
    ok = [s for s in steps if s.ok]
    return {
        "dx_frac": sum(s.dx for s in ok) / width if width else 0.0,
        "dy_frac": sum(s.dy for s in ok) / height if height else 0.0,
        "scale": float(np.prod([s.scale for s in ok])) if ok else 1.0,
        "rot_ccw": sum(s.rot_ccw for s in ok),
        "ok_fraction": len(ok) / len(steps) if steps else 0.0,
        "mean_inlier_ratio": float(np.mean([s.inlier_ratio for s in ok]))
        if ok
        else 0.0,
    }


def describe(
    steps: Sequence[MotionStep],
    width: int,
    height: int,
    cfg: Optional[MotionConfig] = None,
) -> str:
    """Render the per-step estimates as one prompt line."""
    cfg = cfg or MotionConfig()
    if not steps:
        return ""
    if cfg.phrasing == "camera":
        return describe_camera(steps, width, height, cfg)
    diag = math.hypot(width, height)
    labels = [label_step(s, diag, cfg) for s in steps]
    segs = segment(steps, labels, cfg)
    body = "; ".join(f"{s['t0']:.1f}-{s['t1']:.1f}s {s['label']}" for s in segs)

    summ = summarize(steps, width, height)
    # Everything is phrased from the camera's point of view, like the
    # segments: the camera moves opposite to the scene's image motion.
    overall = []
    cam_x, cam_y = -summ["dx_frac"], -summ["dy_frac"]
    if abs(cam_x) >= 0.05:
        overall.append(
            f"camera moved {abs(cam_x):.2f} frame-widths {'right' if cam_x > 0 else 'left'}"
        )
    if abs(cam_y) >= 0.05:
        overall.append(
            f"{abs(cam_y):.2f} frame-heights "
            f"{'toward the bottom' if cam_y > 0 else 'toward the top'}"
        )
    scale = summ["scale"]
    if abs(math.log(max(scale, 1e-6))) >= 0.05:
        overall.append(
            f"view {'magnified' if scale > 1 else 'shrunk'} x{scale:.2f} "
            f"({'descended / zoomed in' if scale > 1 else 'ascended / zoomed out'})"
        )
    if abs(summ["rot_ccw"]) >= 5:
        # Scene turning counter-clockwise means the camera yawed clockwise.
        overall.append(
            f"rotated {abs(summ['rot_ccw']):.0f} degrees "
            f"{'clockwise' if summ['rot_ccw'] > 0 else 'counter-clockwise'}"
        )
    text = f"Camera motion: {body}."
    if overall:
        text += " Overall: " + ", ".join(overall) + "."
    if summ["ok_fraction"] < 0.5:
        text += " (Low-confidence estimate.)"
    return text


def motion_record(
    frames: Sequence[np.ndarray],
    timestamps: Sequence[float],
    cfg: Optional[MotionConfig] = None,
) -> dict:
    """Everything `compute_motion_cues.py` stores per clip: steps, summary, text."""
    cfg = cfg or MotionConfig()
    h, w = frames[0].shape[:2]
    steps = estimate_motion(frames, timestamps, cfg)
    return {
        "text": describe(steps, w, h, cfg),
        "summary": summarize(steps, w, h),
        "steps": [asdict(s) for s in steps],
    }
