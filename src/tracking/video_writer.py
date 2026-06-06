"""Video output — per-camera annotated mp4 + N-view grid montage (FEAT-5/6).

Uses the ``mp4v`` FourCC, which produces a readable .mp4 on Windows 11 with the
stock OpenCV FFMPEG backend (no extra codec install needed — verified in MIL-0).
"""

import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def compose_grid(
    images: Dict[int, np.ndarray],
    camera_ids: List[int],
    cell_size: Tuple[int, int],
    cols: int,
    label: bool = True,
) -> np.ndarray:
    """Tile per-camera images into a single grid canvas.

    Args:
        images: {camera_id: BGR image}. Missing cameras render as black cells.
        camera_ids: Fixed ordering of cameras (defines cell positions).
        cell_size: (width, height) of each cell.
        cols: Number of columns.
        label: Draw "Cam N" on each cell.

    Returns:
        Grid canvas (BGR).
    """
    cw, ch = cell_size
    n = len(camera_ids)
    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)

    for i, cam_id in enumerate(camera_ids):
        r, c = divmod(i, cols)
        y0, x0 = r * ch, c * cw
        img = images.get(cam_id)
        if img is not None:
            cell = cv2.resize(img, (cw, ch))
        else:
            cell = np.zeros((ch, cw, 3), dtype=np.uint8)
        if label:
            cv2.rectangle(cell, (0, 0), (110, 28), (0, 0, 0), -1)
            cv2.putText(
                cell, f"Cam {cam_id}", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
        canvas[y0:y0 + ch, x0:x0 + cw] = cell
    return canvas


class MultiCameraVideoWriter:
    """Write annotated frames to per-camera mp4s and/or a single grid mp4."""

    def __init__(
        self,
        output_dir: str,
        camera_ids: List[int],
        fps: float = 10.0,
        fourcc: str = "mp4v",
        write_per_camera: bool = True,
        write_grid: bool = True,
        grid_cols: Optional[int] = None,
        grid_width: int = 1280,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera_ids = list(camera_ids)
        self.fps = float(fps)
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.write_per_camera = write_per_camera
        self.write_grid = write_grid
        self.grid_cols = grid_cols or math.ceil(math.sqrt(len(self.camera_ids)))
        self.grid_width = grid_width

        self._cam_writers: Dict[int, cv2.VideoWriter] = {}
        self._grid_writer: Optional[cv2.VideoWriter] = None
        self._cell_size: Optional[Tuple[int, int]] = None
        self._frame_counts: Dict[int, int] = {c: 0 for c in self.camera_ids}
        self._grid_count = 0
        self.paths: Dict[str, str] = {}

    def _ensure_cam_writer(self, cam_id: int, image: np.ndarray) -> cv2.VideoWriter:
        if cam_id not in self._cam_writers:
            h, w = image.shape[:2]
            path = self.output_dir / f"cam{cam_id}.mp4"
            vw = cv2.VideoWriter(str(path), self.fourcc, self.fps, (w, h))
            if not vw.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter: {path}")
            self._cam_writers[cam_id] = vw
            self.paths[f"cam{cam_id}"] = str(path)
            logger.info("Opened per-camera writer %s (%dx%d)", path, w, h)
        return self._cam_writers[cam_id]

    def _ensure_grid_writer(self, sample: np.ndarray) -> cv2.VideoWriter:
        if self._grid_writer is None:
            h, w = sample.shape[:2]
            cw = self.grid_width // self.grid_cols
            ch = int(cw * h / w)
            self._cell_size = (cw, ch)
            rows = math.ceil(len(self.camera_ids) / self.grid_cols)
            gw, gh = cw * self.grid_cols, ch * rows
            path = self.output_dir / "grid.mp4"
            vw = cv2.VideoWriter(str(path), self.fourcc, self.fps, (gw, gh))
            if not vw.isOpened():
                raise RuntimeError(f"Failed to open grid VideoWriter: {path}")
            self._grid_writer = vw
            self.paths["grid"] = str(path)
            logger.info("Opened grid writer %s (%dx%d, %d cols)", path, gw, gh, self.grid_cols)
        return self._grid_writer

    def write(self, annotated: Dict[int, np.ndarray]) -> None:
        """Write one timestep of annotated per-camera images."""
        if not annotated:
            return

        if self.write_per_camera:
            for cam_id, img in annotated.items():
                vw = self._ensure_cam_writer(cam_id, img)
                vw.write(img)
                self._frame_counts[cam_id] = self._frame_counts.get(cam_id, 0) + 1

        if self.write_grid:
            sample = next(iter(annotated.values()))
            self._ensure_grid_writer(sample)
            grid = compose_grid(
                annotated, self.camera_ids, self._cell_size, self.grid_cols
            )
            self._grid_writer.write(grid)
            self._grid_count += 1

    def release(self) -> Dict[str, int]:
        """Close all writers. Returns frame counts per output."""
        for vw in self._cam_writers.values():
            vw.release()
        if self._grid_writer is not None:
            self._grid_writer.release()
        counts = dict(self._frame_counts)
        counts["grid"] = self._grid_count
        logger.info("Closed writers. Frame counts: %s", counts)
        return counts

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
