"""Data drift detection for monitoring distribution changes."""

import numpy as np
from typing import Dict, List, Optional
from collections import deque
from scipy.stats import ks_2samp
import logging

from ..utils.config import DriftConfig

logger = logging.getLogger(__name__)


class DriftDetector:
    """Data drift detector using statistical tests."""

    def __init__(self, config: DriftConfig):
        """
        Initialize drift detector.

        Args:
            config: Drift detection configuration
        """
        self.config = config

        # Reference distributions (baseline)
        self.reference_data: Dict[str, deque] = {
            'image_mean': deque(maxlen=config.reference_window),
            'image_std': deque(maxlen=config.reference_window),
            'bbox_size': deque(maxlen=config.reference_window),
            'detection_count': deque(maxlen=config.reference_window),
        }

        # Detection window (current data)
        self.detection_data: Dict[str, deque] = {
            'image_mean': deque(maxlen=config.detection_window),
            'image_std': deque(maxlen=config.detection_window),
            'bbox_size': deque(maxlen=config.detection_window),
            'detection_count': deque(maxlen=config.detection_window),
        }

        self.drift_scores: Dict[str, float] = {}
        self.is_drifted_flag = False
        self.frame_count = 0

        logger.info("Drift detector initialized")

    def update(
        self,
        image: np.ndarray,
        detections: List,
        embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Update drift detector with new frame data.

        Args:
            image: Input image
            detections: List of detections
            embeddings: ReID embeddings (optional)

        Returns:
            Dictionary of drift scores
        """
        self.frame_count += 1

        # Extract features
        features = self._extract_features(image, detections, embeddings)

        # Update reference window (first N frames)
        if self.frame_count <= self.config.reference_window:
            for key, value in features.items():
                if key in self.reference_data:
                    self.reference_data[key].append(value)
            return {}

        # Update detection window
        for key, value in features.items():
            if key in self.detection_data:
                self.detection_data[key].append(value)

        # Compute drift every check_interval frames
        if self.frame_count % self.config.check_interval == 0:
            self.drift_scores = self._compute_drift_scores()
            self.is_drifted_flag = any(
                score > self.config.threshold
                for score in self.drift_scores.values()
            )

            if self.is_drifted_flag:
                logger.warning(f"Drift detected! Scores: {self.drift_scores}")

        return self.drift_scores

    def _extract_features(
        self,
        image: np.ndarray,
        detections: List,
        embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Extract features for drift detection.

        Args:
            image: Input image
            detections: List of detections
            embeddings: ReID embeddings

        Returns:
            Feature dictionary
        """
        features = {}

        # Image statistics
        features['image_mean'] = float(np.mean(image))
        features['image_std'] = float(np.std(image))

        # Detection statistics
        features['detection_count'] = float(len(detections))

        if len(detections) > 0:
            # Average bounding box size
            bbox_sizes = []
            for det in detections:
                if hasattr(det, 'bbox'):
                    x1, y1, x2, y2 = det.bbox
                    area = (x2 - x1) * (y2 - y1)
                    bbox_sizes.append(area)

            if bbox_sizes:
                features['bbox_size'] = float(np.mean(bbox_sizes))
            else:
                features['bbox_size'] = 0.0
        else:
            features['bbox_size'] = 0.0

        return features

    def _compute_drift_scores(self) -> Dict[str, float]:
        """
        Compute drift scores using Kolmogorov-Smirnov test.

        Returns:
            Dictionary of drift scores (0-1, higher = more drift)
        """
        drift_scores = {}

        for feature_name in self.reference_data.keys():
            ref_data = list(self.reference_data[feature_name])
            det_data = list(self.detection_data[feature_name])

            if len(ref_data) < 10 or len(det_data) < 10:
                drift_scores[feature_name] = 0.0
                continue

            try:
                # Kolmogorov-Smirnov test
                statistic, p_value = ks_2samp(ref_data, det_data)

                # Convert p-value to drift score (lower p-value = higher drift)
                drift_score = 1.0 - p_value

                drift_scores[feature_name] = float(drift_score)

            except Exception as e:
                logger.error(f"Failed to compute drift for {feature_name}: {e}")
                drift_scores[feature_name] = 0.0

        return drift_scores

    def is_drifted(self) -> bool:
        """
        Check if drift has been detected.

        Returns:
            True if drift detected
        """
        return self.is_drifted_flag

    def get_drift_scores(self) -> Dict[str, float]:
        """
        Get current drift scores.

        Returns:
            Dictionary of drift scores
        """
        return self.drift_scores

    def get_retrain_trigger(self) -> Optional[str]:
        """
        Get retrain trigger reason if drift detected.

        Returns:
            Reason string or None
        """
        if not self.is_drifted_flag:
            return None

        # Find feature with highest drift
        max_drift_feature = max(
            self.drift_scores.items(),
            key=lambda x: x[1]
        )

        return (
            f"Drift detected in {max_drift_feature[0]} "
            f"(score: {max_drift_feature[1]:.3f}, "
            f"threshold: {self.config.threshold})"
        )

    def reset_reference(self):
        """Reset reference distribution to current detection window."""
        logger.info("Resetting reference distribution")

        for key in self.reference_data.keys():
            # Copy detection window to reference
            self.reference_data[key] = deque(
                self.detection_data[key],
                maxlen=self.config.reference_window
            )
            # Clear detection window
            self.detection_data[key].clear()

        self.is_drifted_flag = False
        self.drift_scores = {}

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for reference and detection windows.

        Returns:
            Dictionary of statistics
        """
        stats = {}

        for feature_name in self.reference_data.keys():
            ref_data = list(self.reference_data[feature_name])
            det_data = list(self.detection_data[feature_name])

            stats[feature_name] = {
                'ref_mean': float(np.mean(ref_data)) if ref_data else 0.0,
                'ref_std': float(np.std(ref_data)) if ref_data else 0.0,
                'det_mean': float(np.mean(det_data)) if det_data else 0.0,
                'det_std': float(np.std(det_data)) if det_data else 0.0,
            }

        return stats
