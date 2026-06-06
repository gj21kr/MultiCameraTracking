"""Frame sources — unify file/image-sequence/WILDTRACK inputs (ADR-002).

The tracking pipeline consumes a stream of *synchronized* multi-camera frames
without knowing whether they come from live USB cameras, video files, or an
image-sequence dataset such as WILDTRACK. Every source yields the same
``Dict[int, Frame]`` (camera_id -> Frame) and returns ``None`` when exhausted.
"""

import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..tracking.camera import Frame

logger = logging.getLogger(__name__)

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


class FrameSource(ABC):
    """Abstract synchronized multi-camera frame source."""

    camera_ids: List[int]
    fps: float

    @abstractmethod
    def read(self) -> Optional[Dict[int, Frame]]:
        """Return one synchronized frame per camera, or None when exhausted."""

    def __iter__(self):
        return self

    def __next__(self) -> Dict[int, Frame]:
        frames = self.read()
        if frames is None:
            raise StopIteration
        return frames

    def release(self) -> None:  # pragma: no cover - default no-op
        """Release underlying resources."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class ImageSequenceSource(FrameSource):
    """Each camera is a directory of frame images, read in lockstep.

    This is the native layout of WILDTRACK and many MTMC datasets.
    """

    def __init__(
        self,
        camera_dirs: Dict[int, str],
        max_frames: Optional[int] = None,
        step: int = 1,
        fps: float = 2.0,
    ):
        if not camera_dirs:
            raise ValueError("camera_dirs must not be empty")

        self.camera_ids = sorted(camera_dirs.keys())
        self.fps = fps
        self.step = max(1, step)

        # Build sorted file lists per camera.
        self._files: Dict[int, List[Path]] = {}
        for cam_id in self.camera_ids:
            d = Path(camera_dirs[cam_id])
            if not d.is_dir():
                raise FileNotFoundError(f"Camera {cam_id} dir not found: {d}")
            files = sorted(
                p for p in d.iterdir()
                if p.suffix.lower() in _IMAGE_EXTS
            )
            if not files:
                raise FileNotFoundError(f"No images in camera {cam_id} dir: {d}")
            self._files[cam_id] = files[::self.step]

        # Synchronize on the shortest sequence.
        n = min(len(v) for v in self._files.values())
        if max_frames is not None:
            n = min(n, max_frames)
        self._n = n
        self._idx = 0
        logger.info(
            "ImageSequenceSource: %d cameras, %d synchronized frames (step=%d)",
            len(self.camera_ids), self._n, self.step,
        )

    def __len__(self) -> int:
        return self._n

    def read(self) -> Optional[Dict[int, Frame]]:
        if self._idx >= self._n:
            return None
        ts = time.time()
        frames: Dict[int, Frame] = {}
        for cam_id in self.camera_ids:
            path = self._files[cam_id][self._idx]
            image = cv2.imread(str(path))
            if image is None:
                logger.warning("Failed to read %s; skipping camera %d", path, cam_id)
                continue
            frames[cam_id] = Frame(
                camera_id=cam_id,
                frame_id=self._idx,
                timestamp=ts,
                image=image,
            )
        self._idx += 1
        return frames if frames else None


class VideoFileSource(FrameSource):
    """Each camera is a video file, read frame-by-frame in lockstep."""

    def __init__(
        self,
        video_paths: Dict[int, str],
        max_frames: Optional[int] = None,
        step: int = 1,
    ):
        if not video_paths:
            raise ValueError("video_paths must not be empty")

        self.camera_ids = sorted(video_paths.keys())
        self.step = max(1, step)
        self._caps: Dict[int, cv2.VideoCapture] = {}
        fpss = []
        for cam_id in self.camera_ids:
            path = Path(video_paths[cam_id])
            if not path.is_file():
                raise FileNotFoundError(f"Video for camera {cam_id} not found: {path}")
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video for camera {cam_id}: {path}")
            self._caps[cam_id] = cap
            fpss.append(cap.get(cv2.CAP_PROP_FPS) or 0)

        self.fps = float(np.median([f for f in fpss if f > 0]) or 25.0) / self.step
        self.max_frames = max_frames
        self._idx = 0

    def read(self) -> Optional[Dict[int, Frame]]:
        if self.max_frames is not None and self._idx >= self.max_frames:
            return None
        ts = time.time()
        frames: Dict[int, Frame] = {}
        for cam_id in self.camera_ids:
            cap = self._caps[cam_id]
            image = None
            for _ in range(self.step):
                ret, image = cap.read()
                if not ret:
                    image = None
                    break
            if image is None:
                # One stream ended -> stop the whole synchronized read.
                return None
            frames[cam_id] = Frame(
                camera_id=cam_id,
                frame_id=self._idx,
                timestamp=ts,
                image=image,
            )
        self._idx += 1
        return frames

    def release(self) -> None:
        for cap in self._caps.values():
            cap.release()


def from_wildtrack(
    root: str,
    cameras: Optional[List[int]] = None,
    max_frames: Optional[int] = None,
    step: int = 1,
) -> ImageSequenceSource:
    """Build an ImageSequenceSource from a WILDTRACK dataset root.

    Expected layout (official WILDTRACK / Chavdarova toolkit)::

        <root>/Image_subsets/C1/00000000.png ...
        <root>/Image_subsets/C2/...
        ...                   C7/...

    Args:
        root: Path to the extracted WILDTRACK dataset.
        cameras: 1-based camera indices to use (default: all C1..C7 found).
        max_frames: Cap synchronized frames (None = all 400).
        step: Take every ``step``-th frame.
    """
    root_path = Path(root)
    # Tolerate either <root>/Image_subsets/Cx or <root>/Cx layouts.
    base = root_path / "Image_subsets"
    if not base.is_dir():
        base = root_path
    if not base.is_dir():
        raise FileNotFoundError(f"WILDTRACK root not found: {root}")

    # Discover camera subdirs named C1, C2, ... (case-insensitive).
    found: Dict[int, Path] = {}
    for d in base.iterdir():
        if d.is_dir() and len(d.name) >= 2 and d.name[0] in ("C", "c"):
            try:
                idx = int(d.name[1:])
            except ValueError:
                continue
            found[idx] = d
    if not found:
        raise FileNotFoundError(
            f"No camera subdirs (C1..Cn) under {base}. "
            "Is this a WILDTRACK dataset? See README for the download guide."
        )

    if cameras is not None:
        missing = [c for c in cameras if c not in found]
        if missing:
            raise FileNotFoundError(f"Requested cameras {missing} not found in {base}")
        camera_dirs = {c: str(found[c]) for c in cameras}
    else:
        camera_dirs = {c: str(found[c]) for c in sorted(found)}

    logger.info("WILDTRACK: using cameras %s from %s", sorted(camera_dirs), base)
    return ImageSequenceSource(
        camera_dirs=camera_dirs, max_frames=max_frames, step=step, fps=2.0
    )


def make_synthetic_source(
    base_image: np.ndarray,
    num_cameras: int = 4,
    num_frames: int = 40,
    out_dir: Optional[str] = None,
    fps: float = 10.0,
) -> ImageSequenceSource:
    """Generate a small synthetic multi-camera image sequence for self-tests.

    Each "camera" is a distinct view (flip / brightness / crop) of ``base_image``
    that pans across frames, so a real detector finds the real people in the
    photo while their boxes move frame-to-frame. This proves the full
    source -> detect -> track -> video pipeline without the ~12GB WILDTRACK
    download (strategy §7 partial-subset mitigation).

    Returns an ImageSequenceSource over the generated frames.
    """
    out = Path(out_dir or "data/_synthetic")
    out.mkdir(parents=True, exist_ok=True)
    H, W = base_image.shape[:2]

    # View transforms per camera (deterministic, no randomness).
    def view(img, cam):
        v = img.copy()
        if cam % 2 == 1:
            v = cv2.flip(v, 1)
        beta = -25 * (cam % 3)
        v = cv2.convertScaleAbs(v, alpha=1.0, beta=beta)
        return v

    camera_dirs: Dict[int, str] = {}
    pan = max(1, W // (num_frames * 3))  # pixels panned per frame
    for cam in range(1, num_cameras + 1):
        cam_dir = out / f"C{cam}"
        if cam_dir.exists():
            for old in cam_dir.glob("*.png"):
                old.unlink()  # avoid stale frames from a prior run
        cam_dir.mkdir(parents=True, exist_ok=True)
        v = view(base_image, cam)
        # Pad horizontally so we can pan a viewport across the scene.
        pad = pan * num_frames + 2
        canvas = cv2.copyMakeBorder(v, 0, 0, pad, pad, cv2.BORDER_REFLECT)
        for f in range(num_frames):
            x0 = f * pan + (cam * 3)
            crop = canvas[:, x0:x0 + W]
            if crop.shape[1] != W:
                crop = cv2.resize(crop, (W, H))
            cv2.imwrite(str(cam_dir / f"{f:08d}.png"), crop)
        camera_dirs[cam] = str(cam_dir)

    logger.info(
        "Synthetic source: %d cameras x %d frames at %s", num_cameras, num_frames, out
    )
    return ImageSequenceSource(camera_dirs=camera_dirs, fps=fps)
