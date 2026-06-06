"""Prometheus metrics collector.

NOTE: prometheus_client is an optional dependency (frozen Non-goal, see strategy
ADR-001). The import is guarded so the package stays importable without it.
"""

from typing import Dict, Optional
import logging

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    _PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    Counter = Gauge = Histogram = None
    start_http_server = None
    _PROMETHEUS_AVAILABLE = False

from ..utils.config import PrometheusConfig

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Prometheus metrics collector for tracking pipeline."""

    def __init__(self, config: PrometheusConfig, enabled: bool = True):
        """
        Initialize metrics collector.

        Args:
            config: Prometheus configuration
            enabled: Whether metrics collection is enabled
        """
        self.config = config
        self.enabled = enabled

        if not enabled:
            logger.info("Metrics collection disabled")
            return

        if not _PROMETHEUS_AVAILABLE:
            logger.warning(
                "Metrics requested but 'prometheus_client' is not installed; "
                "disabling metrics. Run: pip install prometheus-client"
            )
            self.enabled = False
            return

        # Define metrics
        self._define_metrics()

        # Start HTTP server for Prometheus scraping
        try:
            start_http_server(config.port)
            logger.info(f"Prometheus metrics server started on port {config.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")
            self.enabled = False

    def _define_metrics(self):
        """Define all Prometheus metrics."""
        # Gauge metrics (current values)
        self.tracking_fps = Gauge(
            'tracking_fps_gauge',
            'Current tracking FPS',
            ['camera_id']
        )

        self.active_tracks = Gauge(
            'active_tracks_gauge',
            'Number of active tracks',
            ['camera_id']
        )

        self.drift_score = Gauge(
            'drift_score_gauge',
            'Current drift score',
            ['feature']
        )

        # Histogram metrics (distributions)
        self.detection_latency = Histogram(
            'detection_latency_seconds',
            'Detection inference latency',
            buckets=[0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5]
        )

        self.tracking_latency = Histogram(
            'tracking_latency_seconds',
            'Tracking update latency',
            buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        )

        self.reid_latency = Histogram(
            'reid_latency_seconds',
            'ReID extraction latency',
            buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
        )

        self.total_latency = Histogram(
            'total_latency_seconds',
            'End-to-end pipeline latency',
            buckets=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
        )

        # Counter metrics (cumulative)
        self.frames_processed = Counter(
            'frames_processed_total',
            'Total frames processed',
            ['camera_id']
        )

        self.tracks_created = Counter(
            'tracks_created_total',
            'Total tracks created',
            ['camera_id']
        )

        self.detections_total = Counter(
            'detections_total',
            'Total detections',
            ['camera_id', 'class']
        )

    def update_fps(self, camera_id: int, fps: float):
        """
        Update FPS metric.

        Args:
            camera_id: Camera ID
            fps: Current FPS
        """
        if not self.enabled:
            return

        try:
            self.tracking_fps.labels(camera_id=camera_id).set(fps)
        except Exception as e:
            logger.error(f"Failed to update FPS metric: {e}")

    def update_active_tracks(self, camera_id: int, count: int):
        """
        Update active tracks count.

        Args:
            camera_id: Camera ID
            count: Number of active tracks
        """
        if not self.enabled:
            return

        try:
            self.active_tracks.labels(camera_id=camera_id).set(count)
        except Exception as e:
            logger.error(f"Failed to update active tracks metric: {e}")

    def update_drift_score(self, feature: str, score: float):
        """
        Update drift score.

        Args:
            feature: Feature name
            score: Drift score
        """
        if not self.enabled:
            return

        try:
            self.drift_score.labels(feature=feature).set(score)
        except Exception as e:
            logger.error(f"Failed to update drift score metric: {e}")

    def observe_latency(self, latency_dict: Dict[str, float]):
        """
        Observe latency metrics.

        Args:
            latency_dict: Dictionary of latencies
        """
        if not self.enabled:
            return

        try:
            if 'detection' in latency_dict:
                self.detection_latency.observe(latency_dict['detection'])

            if 'tracking' in latency_dict:
                self.tracking_latency.observe(latency_dict['tracking'])

            if 'reid' in latency_dict:
                self.reid_latency.observe(latency_dict['reid'])

            if 'total' in latency_dict:
                self.total_latency.observe(latency_dict['total'])
        except Exception as e:
            logger.error(f"Failed to observe latency: {e}")

    def increment_frames_processed(self, camera_id: int):
        """
        Increment frames processed counter.

        Args:
            camera_id: Camera ID
        """
        if not self.enabled:
            return

        try:
            self.frames_processed.labels(camera_id=camera_id).inc()
        except Exception as e:
            logger.error(f"Failed to increment frames processed: {e}")

    def increment_tracks_created(self, camera_id: int, count: int = 1):
        """
        Increment tracks created counter.

        Args:
            camera_id: Camera ID
            count: Number of tracks created
        """
        if not self.enabled:
            return

        try:
            self.tracks_created.labels(camera_id=camera_id).inc(count)
        except Exception as e:
            logger.error(f"Failed to increment tracks created: {e}")

    def increment_detections(
        self,
        camera_id: int,
        class_name: str,
        count: int = 1
    ):
        """
        Increment detections counter.

        Args:
            camera_id: Camera ID
            class_name: Detection class name
            count: Number of detections
        """
        if not self.enabled:
            return

        try:
            self.detections_total.labels(
                camera_id=camera_id,
                class_name=class_name
            ).inc(count)
        except Exception as e:
            logger.error(f"Failed to increment detections: {e}")
