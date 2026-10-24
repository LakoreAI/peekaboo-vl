import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.modules.cues.motion import (  # noqa: E402
    MotionConfig,
    MotionStep,
    describe,
    estimate_pair,
    label_step,
    motion_record,
    segment,
)

H, W = 180, 320


def world(seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (H * 3 // 8, W * 3 // 8, 3), dtype=np.uint8)
    return cv2.GaussianBlur(
        cv2.resize(base, (W * 3, H * 3), interpolation=cv2.INTER_NEAREST), (3, 3), 0
    )


def view(img, cx, cy, scale=1.0, angle=0.0):
    """Window of size W x H centred on (cx, cy); angle in degrees CCW."""
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += W / 2 - cx
    M[1, 2] += H / 2 - cy
    return cv2.warpAffine(img, M, (W, H))


@pytest.fixture(scope="module")
def tex():
    return world()


def test_pan_sign(tex):
    cfg = MotionConfig()
    a = view(tex, W * 1.5, H * 1.5)
    b = view(tex, W * 1.5 + 12, H * 1.5)  # camera moved right
    est = estimate_pair(a, b, cfg)
    assert est is not None
    assert est["dx"] == pytest.approx(-12, abs=1.0)  # scene shifts left
    assert abs(est["dy"]) < 1.0 and est["scale"] == pytest.approx(1.0, abs=0.01)


def test_zoom_and_rotation_signs(tex):
    cfg = MotionConfig()
    a = view(tex, W * 1.5, H * 1.5)
    zoom = estimate_pair(a, view(tex, W * 1.5, H * 1.5, scale=1.1), cfg)
    assert zoom["scale"] == pytest.approx(1.1, abs=0.01)
    rot = estimate_pair(a, view(tex, W * 1.5, H * 1.5, angle=5.0), cfg)
    assert rot["rot_ccw"] == pytest.approx(5.0, abs=0.5)


def test_blank_frames_fail_gracefully():
    blank = np.zeros((H, W, 3), dtype=np.uint8)
    assert estimate_pair(blank, blank, MotionConfig()) is None


def test_label_step_directions():
    cfg, diag = MotionConfig(), math.hypot(W, H)
    step = lambda **kw: MotionStep(0.0, 1.0, ok=True, **kw)  # noqa: E731
    assert label_step(step(dx=-15.0), diag, cfg) == "pans right, slow"  # 0.04 diag/s
    assert label_step(step(dx=200.0), diag, cfg) == "pans left, fast"
    assert label_step(step(dy=15.0), diag, cfg).startswith("pans toward the top")
    assert label_step(step(scale=1.1), diag, cfg) == "zooms in (descending)"
    assert label_step(step(scale=0.9), diag, cfg) == "zooms out (ascending)"
    assert label_step(step(rot_ccw=5.0), diag, cfg) == "rotates clockwise"
    assert label_step(step(), diag, cfg) == "static"
    assert label_step(MotionStep(0.0, 1.0, ok=False), diag, cfg) == "unclear"


def test_segment_merges_and_absorbs_short_runs():
    cfg = MotionConfig(min_segment=0.8)
    steps = [MotionStep(t * 0.5, (t + 1) * 0.5, ok=True) for t in range(8)]
    labels = ["static"] * 3 + ["pans left, slow"] + ["static"] * 4
    segs = segment(steps, labels, cfg)
    assert segs == [{"t0": 0.0, "t1": 4.0, "label": "static"}]


def test_describe_end_to_end(tex):
    frames = [view(tex, W * 1.5 + 8 * t, H * 1.5) for t in range(9)]
    times = [0.5 * t for t in range(9)]
    rec = motion_record(frames, times)
    assert rec["text"].startswith("Camera motion: 0.0-4.0s pans right")
    assert "camera moved" in rec["text"] and "frame-widths right" in rec["text"]
    assert rec["summary"]["ok_fraction"] == 1.0
    assert describe([], W, H) == ""


def test_segments_and_overall_share_the_camera_frame():
    # Scene turns counter-clockwise and drifts left: camera yaws clockwise, moves right.
    steps = [MotionStep(t, t + 1.0, ok=True, dx=-20.0, rot_ccw=4.0) for t in range(4)]
    text = describe(steps, W, H)
    body, overall = text.split(" Overall: ")
    assert "pans right" in body and "rotates clockwise" in body
    assert "camera moved" in overall and "frame-widths right" in overall
    assert "rotated 16 degrees clockwise" in overall


def test_camera_phrasing_uses_answer_vocabulary():
    cfg = MotionConfig(phrasing="camera")
    # Scene streams down + expands: forward flight of a tilted camera.
    fwd = [MotionStep(t, t + 1.0, ok=True, dy=30.0, scale=1.05) for t in range(4)]
    text = describe(fwd, W, H, cfg)
    assert (
        "moving forward" in text and "Mostly moving forward (100% of the clip)" in text
    )
    assert "pans" not in text and "zooms" not in text
    pan = [MotionStep(t, t + 1.0, ok=True, dx=-30.0) for t in range(4)]
    assert "panning right" in describe(pan, W, H, cfg)
    still = [MotionStep(t, t + 1.0, ok=True) for t in range(4)]
    assert describe(still, W, H, cfg).endswith("Mostly stationary (100% of the clip).")
    # Image phrasing is unchanged by default.
    assert "pans" in describe(pan, W, H)
