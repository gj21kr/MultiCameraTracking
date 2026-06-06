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
    """Simple Kalman filter for 2D bounding box tracking."""

    def __init__(self):
        """Initialize Kalman filter with constant velocity model."""
        # State: [x, y, vx, vy, w, h]
        self.dt = 1.0  # time step
        self.state = np.zeros(6)

        # Process noise
        self.Q = np.eye(6) * 0.01

        # Measurement noise
        self.R = np.eye(4) * 1.0

    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Predict next state.

        Args:
            state: Current state [x, y, vx, vy, w, h]

        Returns:
            Predicted state
        """
        x, y, vx, vy, w, h = state

        # Constant velocity model
        x_pred = x + vx * self.dt
        y_pred = y + vy * self.dt

        return np.array([x_pred, y_pred, vx, vy, w, h])

    def update(
        self,
        state: np.ndarray,
        measurement: np.ndarray
    ) -> np.ndarray:
        """
        Update state with measurement.

        Args:
            state: Predicted state [x, y, vx, vy, w, h]
            measurement: Measurement [x, y, w, h]

        Returns:
            Updated state
        """
        x, y, vx, vy, w, h = state
        x_meas, y_meas, w_meas, h_meas = measurement

        # Compute velocity from measurement
        vx_new = x_meas - x
        vy_new = y_meas - y

        # Simple weighted update (Kalman gain = 0.5)
        alpha = 0.5

        x_updated = x + alpha * (x_meas - x)
        y_updated = y + alpha * (y_meas - y)
        vx_updated = vx + alpha * (vx_new - vx)
        vy_updated = vy + alpha * (vy_new - vy)
        w_updated = w + alpha * (w_meas - w)
        h_updated = h + alpha * (h_meas - h)

        return np.array([
            x_updated, y_updated, vx_updated, vy_updated, w_updated, h_updated
        ])


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
                state = self._bbox_to_state(track.bbox, track.velocity)
                pred_state = kf.predict(state)
                track.bbox = self._state_to_bbox(pred_state)
                track.velocity = pred_state[2:4]

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

        # Compute IoU cost matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(track.bbox, det.bbox)

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)

        # Filter by IoU threshold
        matched = []
        for i, j in zip(row_ind, col_ind):
            if iou_matrix[i, j] >= self.config.match_thresh:
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
            self.kalman_filters[track.track_id] = KalmanFilter()

        kf = self.kalman_filters[track.track_id]
        state = self._bbox_to_state(track.bbox, track.velocity)
        measurement = self._bbox_to_measurement(detection.bbox)
        updated_state = kf.update(state, measurement)

        # Update track
        track.bbox = self._state_to_bbox(updated_state)
        track.velocity = updated_state[2:4]
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
        self.kalman_filters[track.track_id] = KalmanFilter()
        self.next_track_id += 1

        return track

    def _is_valid_detection(self, detection: Detection) -> bool:
        """Check if detection is valid for tracking."""
        x1, y1, x2, y2 = detection.bbox
        area = (x2 - x1) * (y2 - y1)
        return area >= self.config.min_box_area

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

    @staticmethod
    def _bbox_to_state(bbox: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Convert bbox to Kalman state [x, y, vx, vy, w, h]."""
        x1, y1, x2, y2 = bbox
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        vx, vy = velocity
        return np.array([x, y, vx, vy, w, h])

    @staticmethod
    def _bbox_to_measurement(bbox: np.ndarray) -> np.ndarray:
        """Convert bbox to measurement [x, y, w, h]."""
        x1, y1, x2, y2 = bbox
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.array([x, y, w, h])

    @staticmethod
    def _state_to_bbox(state: np.ndarray) -> np.ndarray:
        """Convert Kalman state to bbox [x1, y1, x2, y2]."""
        x, y, _, _, w, h = state
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return np.array([x1, y1, x2, y2])


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
