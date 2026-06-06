"""WILDTRACK camera calibration + ground-plane projection (ADR-003).

Cross-camera association by *appearance* (ReID) is fragile under viewpoint and
lighting change. WILDTRACK ships full camera calibration, so the principled
approach is geometric: project each detection's foot point to the common ground
plane (Z=0) and link detections from different cameras that land near the same
world location. This is more stable and explainable than ReID similarity.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# WILDTRACK camera id (1-based, == viewNum+1) -> calibration file stem.
WILDTRACK_CAM_NAMES = {
    1: "CVLab1", 2: "CVLab2", 3: "CVLab3", 4: "CVLab4",
    5: "IDIAP1", 6: "IDIAP2", 7: "IDIAP3",
}


def _read_extrinsic(path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse rvec/tvec from a WILDTRACK extrinsic XML (plain number sequences)."""
    import xml.etree.ElementTree as ET
    root = ET.parse(str(path)).getroot()
    def vec(tag):
        node = root.find(tag)
        if node is None:
            raise ValueError(f"{tag} not found in {path}")
        return np.array([float(x) for x in node.text.split()], dtype=np.float64)
    return vec("rvec"), vec("tvec")


class CameraCalib:
    """Single camera: intrinsics, distortion, extrinsics, ground homography."""

    def __init__(self, K, dist, rvec, tvec):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(dist, dtype=np.float64).reshape(-1)
        self.rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        self.tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        R, _ = cv2.Rodrigues(self.rvec)
        # World point on Z=0 plane: p ~ K [r1 r2 t] [X Y 1]^T  => homography.
        H_w2i = self.K @ np.column_stack([R[:, 0], R[:, 1], self.tvec[:, 0]])
        self.H_i2w = np.linalg.inv(H_w2i)

    def image_to_ground(self, uv: Tuple[float, float]) -> np.ndarray:
        """Map an image pixel (u, v) to ground-plane world coords (X, Y)."""
        pts = np.array([[[uv[0], uv[1]]]], dtype=np.float64)
        und = cv2.undistortPoints(pts, self.K, self.dist, P=self.K)[0, 0]
        w = self.H_i2w @ np.array([und[0], und[1], 1.0])
        return w[:2] / w[2]


class WildtrackCalibration:
    """All 7 WILDTRACK cameras."""

    def __init__(self, cams: Dict[int, CameraCalib]):
        self.cams = cams

    @classmethod
    def from_dir(cls, calib_root: str) -> "WildtrackCalibration":
        root = Path(calib_root)
        intr = root / "intrinsic_original"
        extr = root / "extrinsic"
        if not intr.is_dir():
            # some distributions name it intrinsic_zero
            alt = root / "intrinsic_zero"
            if alt.is_dir():
                intr = alt
        cams: Dict[int, CameraCalib] = {}
        for cam_id, name in WILDTRACK_CAM_NAMES.items():
            fi = intr / f"intr_{name}.xml"
            fe = extr / f"extr_{name}.xml"
            if not fi.exists() or not fe.exists():
                continue
            fsi = cv2.FileStorage(str(fi), cv2.FILE_STORAGE_READ)
            K = fsi.getNode("camera_matrix").mat()
            dist = fsi.getNode("distortion_coefficients").mat()
            fsi.release()
            # Extrinsics store rvec/tvec as plain number sequences (not typed
            # opencv-matrix nodes), so FileStorage.mat() fails -> parse the XML.
            rvec, tvec = _read_extrinsic(fe)
            cams[cam_id] = CameraCalib(K, dist, rvec, tvec)
        if not cams:
            raise FileNotFoundError(f"No calibration XMLs found under {calib_root}")
        logger.info("Loaded calibration for cameras %s", sorted(cams))
        return cls(cams)

    def foot_to_ground(self, cam_id: int, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Project a detection's foot point (bottom-center of bbox) to ground."""
        cam = self.cams.get(cam_id)
        if cam is None:
            return None
        x1, y1, x2, y2 = bbox
        foot = ((x1 + x2) / 2.0, y2)
        return cam.image_to_ground(foot)


def assign_global_ids_geometric(
    all_tracks: Dict[int, List],
    calib: WildtrackCalibration,
    global_id_map: Dict[Tuple[int, int], int],
    next_id_box: List[int],
    dist_thresh_cm: float = 100.0,
) -> None:
    """Assign cross-camera global ids via ground-plane proximity (union-find).

    Mutates ``track.global_id`` in place and persists assignments in
    ``global_id_map`` (keyed by (cam_id, track_id)) so ids stay stable over time.
    ``next_id_box`` is a 1-element list holding the next free global id.
    """
    items = []  # (cam_id, track, ground_xy)
    for cam_id, tracks in all_tracks.items():
        for t in tracks:
            g = calib.foot_to_ground(cam_id, t.bbox)
            if g is not None and np.all(np.isfinite(g)):
                items.append((cam_id, t, g))

    n = len(items)
    if n == 0:
        return
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        ci, _, gi = items[i]
        for j in range(i + 1, n):
            cj, _, gj = items[j]
            if ci == cj:
                continue
            if np.linalg.norm(gi - gj) <= dist_thresh_cm:
                union(i, j)

    from collections import defaultdict
    comps = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)

    for members in comps.values():
        existing = None
        for i in members:
            cam, t, _ = items[i]
            prior = global_id_map.get((cam, t.track_id))
            if prior is not None:
                existing = prior
                break
        if existing is None:
            existing = next_id_box[0]
            next_id_box[0] += 1
        for i in members:
            cam, t, _ = items[i]
            global_id_map[(cam, t.track_id)] = existing
            t.global_id = existing
