"""Smoke tests — no GPU, no dataset, no network required.

Covers the pieces the demo depends on:
  - device auto-fallback
  - ImageSequenceSource synchronized reads
  - cross-camera union-find (the FEAT-7 bug fix)
  - mp4 VideoWriter + grid montage
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import resolve_device, TrackerConfig
from src.tracking.tracker import MultiCameraTracker, Track
from src.tracking.video_writer import MultiCameraVideoWriter, compose_grid


def test_resolve_device_cpu_fallback():
    # On a CPU-only box, cuda requests must downgrade to cpu.
    assert resolve_device("cpu") == "cpu"
    dev = resolve_device("cuda:0")
    assert dev in ("cpu", "cuda:0", "cuda")


def test_image_sequence_source(tmp_path):
    import cv2
    from src.data.sources import ImageSequenceSource

    cam_dirs = {}
    for cam in (1, 2):
        d = tmp_path / f"C{cam}"
        d.mkdir()
        for i in range(3):
            cv2.imwrite(str(d / f"{i:04d}.png"), np.full((16, 16, 3), cam * 30, np.uint8))
        cam_dirs[cam] = str(d)

    src = ImageSequenceSource(cam_dirs)
    assert src.camera_ids == [1, 2]
    n = 0
    for frames in src:
        assert set(frames.keys()) == {1, 2}
        n += 1
    assert n == 3


def test_cross_camera_union_find_links_first_appearance():
    """Two views of the same person on their FIRST frame must share a global id.

    The legacy single-pass logic could not link a first-appearance pair (the
    partner had to already own a global id). FEAT-7 union-find fixes this.
    """
    mct = MultiCameraTracker(TrackerConfig(use_reid=True), num_cameras=2)
    same = np.ones(512, np.float32)
    same /= np.linalg.norm(same)
    other = np.zeros(512, np.float32)
    other[0] = 1.0

    t1 = Track(track_id=10, bbox=np.array([0, 0, 10, 20.0]), embedding=same.copy())
    t2 = Track(track_id=20, bbox=np.array([0, 0, 10, 20.0]), embedding=same.copy())
    t3 = Track(track_id=30, bbox=np.array([5, 5, 15, 25.0]), embedding=other)

    mct.associate_cross_camera({1: [t1], 2: [t2, t3]})

    assert t1.global_id == t2.global_id          # linked
    assert t3.global_id != t1.global_id          # separate person


def test_video_writer_produces_readable_mp4(tmp_path):
    import cv2

    cam_ids = [1, 2]
    writer = MultiCameraVideoWriter(
        output_dir=str(tmp_path), camera_ids=cam_ids, fps=5.0,
    )
    with writer:
        for _ in range(5):
            imgs = {c: np.full((32, 48, 3), c * 20, np.uint8) for c in cam_ids}
            writer.write(imgs)
    counts = writer.release()  # idempotent-ish; ensures files flushed
    grid = Path(tmp_path) / "grid.mp4"
    assert grid.exists() and grid.stat().st_size > 0

    cap = cv2.VideoCapture(str(grid))
    read = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        read += 1
    cap.release()
    assert read >= 5


def test_compose_grid_shape():
    imgs = {1: np.zeros((20, 30, 3), np.uint8), 2: np.zeros((20, 30, 3), np.uint8)}
    canvas = compose_grid(imgs, [1, 2, 3, 4], cell_size=(30, 20), cols=2)
    assert canvas.shape == (40, 60, 3)  # 2 rows x 2 cols
