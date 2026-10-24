import numpy as np
import pytest

from src.video import (
    FrameSet,
    cache_dir_for,
    cache_frames,
    is_cached,
    load_cached_frames,
    load_frames,
    overlay_time_bar,
    read_video,
    subsample,
    uniform_indices,
)


def test_uniform_indices_are_segment_centres():
    assert uniform_indices(100, 4) == [12, 37, 62, 87]
    assert uniform_indices(3, 8) == [0, 0, 0, 1, 1, 2, 2, 2]
    with pytest.raises(ValueError):
        uniform_indices(0, 4)


def test_subsample():
    items = list(range(64))
    assert subsample(items, 100) == items
    assert len(subsample(items, 16)) == 16
    assert subsample(items, 2) == [16, 48]


def _frameset(n=8, fps=10.0):
    frames = np.stack([np.full((36, 64, 3), 30 * i, dtype=np.uint8) for i in range(n)])
    return FrameSet(frames, [5 * i for i in range(n)], fps, 5 * n)


def test_cache_roundtrip(tmp_path):
    fs = _frameset()
    vp = "VisDrone/clip_01.mp4"
    cache_frames(fs, cache_dir_for(tmp_path, vp))
    assert is_cached(tmp_path, vp)
    back = load_cached_frames(tmp_path, vp, 4)
    assert back.frames.shape == (4, 36, 64, 3)
    assert back.frame_indices == [5, 15, 25, 35]
    assert back.timestamps == pytest.approx([0.5, 1.5, 2.5, 3.5])
    assert back.fps == 10.0 and back.total_frames == 40
    # JPEG is lossy but flat frames survive
    assert abs(int(back.frames[0].mean()) - 30) <= 2


def test_load_frames_without_cache_or_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_frames("X/missing.mp4", 4, cache_root=tmp_path)


def test_read_video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "v.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (64, 48))
    if not out.isOpened():
        pytest.skip("no mp4v encoder in this OpenCV build")
    for i in range(24):
        out.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    out.release()
    fs = read_video(path, 6)
    assert fs.frames.shape[0] == 6 and fs.fps == pytest.approx(8.0)
    assert fs.frame_indices == uniform_indices(24, 6)


def test_uniform_indices_phase():
    assert uniform_indices(100, 4) == uniform_indices(100, 4, 0.5)
    assert uniform_indices(100, 4, 0.0) == [0, 25, 50, 75]
    assert uniform_indices(100, 4, 0.9) == [22, 47, 72, 97]
    with pytest.raises(ValueError):
        uniform_indices(100, 4, 1.0)
    items = list(range(64))
    assert subsample(items, 16, 0.25) != subsample(items, 16, 0.75)


def test_overlay_time_bar_draws_bottom_strip_only():
    frames = np.full((4, 64, 96, 3), 128, dtype=np.uint8)
    timestamps = [0.0, 1.0, 2.0, 3.0]
    out = overlay_time_bar(frames, timestamps)
    assert out.shape == frames.shape and out.dtype == frames.dtype
    assert not np.array_equal(out, frames)  # something was drawn
    assert np.array_equal(out[:, :-16], frames[:, :-16])  # top untouched
    assert not np.array_equal(out[:, -16:], frames[:, -16:])  # bar present
