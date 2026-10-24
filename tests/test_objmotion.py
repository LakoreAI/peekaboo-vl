import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.modules.cues.objmotion import (  # noqa: E402
    Blob,
    ObjMotionConfig,
    background_homography,
    describe_objmotion,
    direction_bin,
    objmotion_record,
    region_of,
)

H, W = 180, 320


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (H * 3 // 8, W * 3 // 8, 3), dtype=np.uint8)
    return cv2.GaussianBlur(
        cv2.resize(base, (W * 3, H * 3), interpolation=cv2.INTER_NEAREST), (3, 3), 0
    )


def view(img, cx, cy):
    M = np.float32([[1, 0, W / 2 - cx], [0, 1, H / 2 - cy]])
    return cv2.warpAffine(img, M, (W, H))


def clip(world, cam_dx, obj_dx, n=8):
    """Camera pans `cam_dx` px per frame; a bright car near the top of the
    frame moves `obj_dx` px per frame on the ground."""
    frames = []
    for i in range(n):
        f = view(world, W * 1.5 + cam_dx * i, H * 1.5).copy()
        # Car position in ground coordinates, drawn into the camera view.
        gx = 60 + obj_dx * i - cam_dx * i
        cv2.rectangle(f, (int(gx), 30), (int(gx) + 10, 36), (255, 255, 255), -1)
        frames.append(f)
    return frames, [0.5 * i for i in range(n)]


def test_background_homography_recovers_pan(world):
    a = view(world, W * 1.5, H * 1.5)
    b = view(world, W * 1.5 + 6, H * 1.5)  # camera right -> scene shifts left
    flow = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY),
        None, 0.5, 3, 9, 3, 5, 1.1, 0,
    )
    H_ = background_homography(flow, ObjMotionConfig())
    assert H_ is not None
    assert H_[0, 2] / H_[2, 2] == pytest.approx(-6, abs=0.5)


def test_direction_and_region_bins():
    assert direction_bin(3, 1) == 0 and direction_bin(-3, 1) == 1
    assert direction_bin(0.5, 2) == 2 and direction_bin(0.5, -2) == 3
    assert region_of(0.1, 0.1) == "top-left" and region_of(0.5, 0.9) == "bottom"


def test_object_moving_right_under_camera_pan(world):
    # Camera pans right (scene shifts left); the car drives right on the ground.
    frames, ts = clip(world, cam_dx=6, obj_dx=5)
    rec = objmotion_record(frames, ts)
    assert "left to right" in rec["text"], rec["text"]
    assert rec["text"].startswith("Object motion (camera motion removed): top")


def test_static_scene_has_no_objects(world):
    frames = [view(world, W * 1.5 + 6 * i, H * 1.5) for i in range(8)]
    rec = objmotion_record(frames, [0.5 * i for i in range(8)])
    assert "no clearly moving objects" in rec["text"], rec["text"]


def test_describe_needs_persistent_regions():
    one = [Blob(0.5, 0.5, 20, 10.0, 0.0)]
    rec = describe_objmotion([one, [], [], []], [0, 1, 2, 3, 4], ObjMotionConfig())
    assert "no clearly moving objects" in rec["text"]
    rec = describe_objmotion([one, one, None, one, []], [0, 1, 2, 3, 4, 5], ObjMotionConfig())
    assert "center: moving left to right (0.0-4.0s)" in rec["text"]
