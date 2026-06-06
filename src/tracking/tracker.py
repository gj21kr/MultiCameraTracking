"""Multi-object tracker with ByteTrack algorithm."""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import logging

from .detector import Detection
from ..utils.config import TrackerConfig

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Track data class."""
    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))  # [vx, vy]
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    state: str = "tentative"  # tentative, confirmed, deleted
    embedding: Optional[np.ndarray] = None
    camera_id: int = -1
    global_id: Optional[int] = None
    confidence: float = 0.0


class KalmanFilter:
    """Constant-velocity Kalman filter on bbox state [cx, cy, w, h, vx, vy, vw, vh].

    A proper recursive estimator with covariance (replaces the previous
    fixed-gain alpha=0.5 smoother): predict propagates state + covariance under
    a constant-velocity model; update applies the optimal Kalman gain. This
    reduces positional jitter and ID switches versus the fixed-gain version.
    """

    def __init__(self, bbox: np.ndarray, dt: float = 1.0):
        self.dt = dt
        # State transition (constant velocity): position += velocity * dt.
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = dt
        # Measurement matrix: observe [cx, cy, w, h].
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)

        # Noise covariances (tuned for pixel-scale boxes).
        self.Q = np.eye(8)
        self.Q[4:, 4:] *= 0.01   # velocity process noise (small, smooth motion)
        self.Q[:4, :4] *= 1.0
        self.R = np.eye(4) * 10.0  # measurement noise

        cx, cy, w, h = self._to_z(bbox)
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)
        self.P = np.eye(8)
        self.P[4:, 4:] *= 1000.0  # high initial velocity uncertainty
        self.P *= 10.0

    @staticmethod
    def _to_z(bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=float)

    def _to_bbox(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        w = max(w, 1.0)
        h = max(h, 1.0)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self._to_bbox()

    def update(self, bbox: np.ndarray) -> np.ndarray:
        z = self._to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        return self._to_bbox()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[4:6].copy()

    def current_bbox(self) -> np.ndarray:
        return self._to_bbox()


class ByteTracker:
    """ByteTrack multi-object tracker."""

    def __init__(self, config: TrackerConfig):
        """
        Initialize ByteTrack tracker.

        Args:
            config: Tracker configuration
        """
        self.config = config
        self.tracks: List[Track] = []
        self.next_track_id = 0
        self.kalman_filters: Dict[int, KalmanFilter] = {}
        self.frame_count = 0

    def update(
        self,
        detections: List[Detection],
        frame: Optional[np.ndarray] = None
    ) -> List[Track]:
        """
        Update tracks with new detections.

        Args:
            detections: List of detections
            frame: Original frame (for ReID)

        Returns:
            List of active tracks
        """
        self.frame_count += 1

        # Separate high and low confidence detections
        high_dets = [
            d for d in detections
            if d.confidence >= self.config.track_thresh
        ]
        low_dets = [
            d for d in detections
            if d.confidence < self.config.track_thresh
        ]

        # Predict all tracks
        for track in self.tracks:
            if track.track_id in self.kalman_filters:
                kf = self.kalman_filters[track.track_id]
                track.bbox = kf.predict()
                track.velocity = kf.velocity

        # First association with high confidence detections
        matched, unmatched_tracks, unmatched_dets = self._associate(
            self.tracks, high_dets
        )

        # Update matched tracks
        for track_idx, det_idx in matched:
            track = self.tracks[track_idx]
            det = high_dets[det_idx]
            self._update_track(track, det)

        # Second association with low confidence detections
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
        matched2, unmatched_tracks2, _ = self._associate(
            remaining_tracks, low_dets
        )

        # Update second-round matches
        for track_idx, det_idx in matched2:
            track = remaining_tracks[track_idx]
            det = low_dets[det_idx]
            self._update_track(track, det)

        # Create new tracks from unmatched high confidence detections
        for det_idx in unmatched_dets:
            det = high_dets[det_idx]
            if self._is_valid_detection(det):
                self._create_track(det)

        # Mark unmatched tracks
        all_unmatched = set(unmatched_tracks) | {
            remaining_tracks[i].track_id
            for i in unmatched_tracks2
        }

        for track in self.tracks:
            if track.track_id in all_unmatched:
                track.time_since_update += 1
            else:
                track.time_since_update = 0

        # Remove deleted tracks
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.config.track_buffer
        ]

        # Update track states
        for track in self.tracks:
            if track.hits >= 3 and track.state == "tentative":
                track.state = "confirmed"

        # Return confirmed tracks only
        return [t for t in self.tracks if t.state == "confirmed"]

    def _associate(
        self,
        tracks: List[Track],
        detections: List[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate tracks with detections using IoU.

        Returns:
            - matched: List of (track_idx, det_idx) pairs
            - unmatched_tracks: List of track indices
            - unmatched_detections: List of detection indices
        """
        if len(tracks) == 0 or len(detections) == 0:
            return (
                [],
                list(range(len(tracks))),
                list(range(len(detections)))
            )

        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(track.bbox, det.bbox)

        # Optionally fuse appearance (ReID) into the cost. sim is NaN where an
        # embedding is missing on either side.
        sim_matrix = None
        if self.config.use_reid:
            sim_matrix = np.full((len(tracks), len(detections)), np.nan)
            for i, track in enumerate(tracks):
                if track.embedding is None:
                    continue
                for j, det in enumerate(detections):
                    if det.embedding is None:
                        continue
                    sim_matrix[i, j] = self._cos(track.embedding, det.embedding)

        if sim_matrix is not None:
            w = self.config.appearance_weight
            have = ~np.isnan(sim_matrix)
            app_cost = 1.0 - np.where(have, sim_matrix, 0.0)
            iou_cost = 1.0 - iou_matrix
            cost = np.where(have, w * app_cost + (1.0 - w) * iou_cost, iou_cost)
        else:
            cost = 1.0 - iou_matrix

        # Hungarian algorithm on the (lower-is-better) cost matrix
        row_ind, col_ind = linear_sum_assignment(cost)

        # Accept a match if EITHER spatial overlap OR appearance is convincing
        # (appearance still needs a minimal IoU gate to stay plausible).
        matched = []
        for i, j in zip(row_ind, col_ind):
            ok_iou = iou_matrix[i, j] >= self.config.match_thresh
            ok_app = (
                sim_matrix is not None
                and not np.isnan(sim_matrix[i, j])
                and sim_matrix[i, j] >= self.config.appearance_thresh
                and iou_matrix[i, j] >= self.config.appearance_iou_gate
            )
            if ok_iou or ok_app:
                matched.append((i, j))

        # Find unmatched
        matched_track_idx = {i for i, _ in matched}
        matched_det_idx = {j for _, j in matched}

        unmatched_tracks = [
            i for i in range(len(tracks))
            if i not in matched_track_idx
        ]
        unmatched_dets = [
            j for j in range(len(detections))
            if j not in matched_det_idx
        ]

        return matched, unmatched_tracks, unmatched_dets

    def _update_track(self, track: Track, detection: Detection):
        """Update track with matched detection."""
        # Update Kalman filter
        if track.track_id not in self.kalman_filters:
            self.kalman_filters[track.track_id] = KalmanFilter(track.bbox)

        kf = self.kalman_filters[track.track_id]
        track.bbox = kf.update(detection.bbox)
        track.velocity = kf.velocity
        track.confidence = detection.confidence
        track.hits += 1
        track.age += 1
        track.time_since_update = 0

        # Update embedding if available
        if detection.embedding is not None:
            if track.embedding is None:
                track.embedding = detection.embedding
            else:
                # Exponential moving average
                alpha = 0.9
                track.embedding = (
                    alpha * track.embedding +
                    (1 - alpha) * detection.embedding
                )

    def _create_track(self, detection: Detection) -> Track:
        """Create new track from detection."""
        track = Track(
            track_id=self.next_track_id,
            bbox=detection.bbox.copy(),
            confidence=detection.confidence,
            hits=1,
            age=1,
            embedding=detection.embedding.copy() if detection.embedding is not None else None
        )

        self.tracks.append(track)
        self.kalman_filters[track.track_id] = KalmanFilter(detection.bbox)
        self.next_track_id += 1

        return track

    def _is_valid_detection(self, detection: Detection) -> bool:
        """Check if detection is valid for tracking."""
        x1, y1, x2, y2 = detection.bbox
        area = (x2 - x1) * (y2 - y1)
        return area >= self.config.min_box_area

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two (already ~normalized) embeddings."""
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

class MultiCameraTracker:
    """Multi-camera tracker with cross-camera association."""

    def __init__(self, config: TrackerConfig, num_cameras: int):
        """
        Initialize multi-camera tracker.

        Args:
            config: Tracker configuration
            num_cameras: Number of cameras
        """
        self.config = config
        self.num_cameras = num_cameras

        # Per-camera trackers are created lazily keyed by the *actual* camera id
        # (datasets like WILDTRACK use 1-based ids C1..C7, not 0..n-1).
        self.trackers: Dict[int, ByteTracker] = {}

        self.next_global_id = 0
        self.global_id_map: Dict[Tuple[int, int], int] = {}  # (camera_id, track_id) -> global_id

    def update(
        self,
        detections: Dict[int, List[Detection]],
        frames: Dict[int, np.ndarray]
    ) -> Dict[int, List[Track]]:
        """
        Update all camera tracks.

        Args:
            detections: {camera_id: [Detection]}
            frames: {camera_id: frame}

        Returns:
            {camera_id: [Track]}
        """
        all_tracks = {}

        # Update each camera independently
        for camera_id, dets in detections.items():
            if camera_id not in self.trackers:
                self.trackers[camera_id] = ByteTracker(self.config)
            frame = frames.get(camera_id)
            tracks = self.trackers[camera_id].update(dets, frame)

            # Assign camera ID
            for track in tracks:
                track.camera_id = camera_id

            all_tracks[camera_id] = tracks

        # Cross-camera association
        if self.config.use_reid:
            self.associate_cross_camera(all_tracks)

        return all_tracks

    def associate_cross_camera(
        self,
        all_tracks: Dict[int, List[Track]],
        sim_thresh: float = 0.7,
    ):
        """Associate tracks across cameras using ReID embeddings (ADR-003).

        Uses a 2-pass union-find over all (camera, track) pairs so that the
        *first appearance* of a person in two views is linked symmetrically.
        The previous single-pass logic required the match partner to already
        own a global id, so first-appearance pairs were never connected
        (tracker.py legacy bug). Global ids are persisted in ``global_id_map``
        keyed by ``(camera_id, track_id)`` to stay stable across frames.

        Args:
            all_tracks: {camera_id: [Track]}
            sim_thresh: Cosine-similarity threshold to link two tracks.
        """
        items: List[Tuple[int, Track]] = []
        for camera_id, tracks in all_tracks.items():
            for track in tracks:
                if track.embedding is not None:
                    items.append((camera_id, track))

        n = len(items)
        if n == 0:
            return

        # --- Pass 1: union-find across cameras by embedding similarity ---
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            cam_i, trk_i = items[i]
            for j in range(i + 1, n):
                cam_j, trk_j = items[j]
                if cam_i == cam_j:
                    continue  # never merge two tracks within the same view
                if self._cosine_similarity(trk_i.embedding, trk_j.embedding) > sim_thresh:
                    union(i, j)

        # --- Pass 2: assign a global id per component (reuse persistent ids) ---
        components: Dict[int, List[int]] = defaultdict(list)
        for i in range(n):
            components[find(i)].append(i)

        for members in components.values():
            existing = None
            for i in members:
                cam, trk = items[i]
                prior = self.global_id_map.get((cam, trk.track_id))
                if prior is not None:
                    existing = prior
                    break
            if existing is None:
                existing = self.next_global_id
                self.next_global_id += 1
            for i in members:
                cam, trk = items[i]
                self.global_id_map[(cam, trk.track_id)] = existing
                trk.global_id = existing

    @staticmethod
    def _cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity between two feature vectors."""
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
