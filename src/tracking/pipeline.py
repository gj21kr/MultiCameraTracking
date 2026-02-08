"""Main tracking pipeline integrating all components."""

import time
import numpy as np
from typing import Dict, List, Optional
import logging

from .camera import CameraManager, Frame
from .detector import Detector, Detection
from .tracker import MultiCameraTracker, Track
from .reid import ReIDExtractor
from ..utils.config import Config

logger = logging.getLogger(__name__)


class TrackingPipeline:
    """End-to-end multi-camera tracking pipeline."""

    def __init__(self, config: Config):
        """
        Initialize tracking pipeline.

        Args:
            config: Main configuration object
        """
        self.config = config

        # Initialize components
        logger.info("Initializing tracking pipeline...")

        # Camera manager
        self.camera_manager = CameraManager(config.cameras)

        # Detector
        self.detector = Detector(config.detection)

        # Tracker
        num_cameras = len(config.cameras)
        self.tracker = MultiCameraTracker(config.tracking, num_cameras)

        # ReID extractor (optional)
        self.reid_extractor = None
        if config.tracking.use_reid:
            try:
                self.reid_extractor = ReIDExtractor(config.reid)
            except Exception as e:
                logger.warning(f"Failed to load ReID model: {e}")
                logger.warning("Continuing without ReID")

        # Performance metrics
        self.fps_dict: Dict[str, float] = {}
        self.latency_dict: Dict[str, float] = {}

        logger.info("Tracking pipeline initialized")

    def start(self):
        """Start camera capture."""
        self.camera_manager.start_all()
        logger.info("Pipeline started")

    def stop(self):
        """Stop camera capture."""
        self.camera_manager.stop_all()
        logger.info("Pipeline stopped")

    def process_frame(
        self,
        frames: Dict[int, Frame],
        extract_reid: bool = True
    ) -> Dict[int, List[Track]]:
        """
        Process a batch of synchronized frames.

        Args:
            frames: {camera_id: Frame}
            extract_reid: Whether to extract ReID features

        Returns:
            {camera_id: [Track]}
        """
        # Extract images
        images = {cam_id: frame.image for cam_id, frame in frames.items()}

        # Detection
        t0 = time.time()
        detections = self._detect_batch(images)
        t1 = time.time()
        self.latency_dict['detection'] = t1 - t0

        # ReID feature extraction
        if extract_reid and self.reid_extractor is not None:
            t2 = time.time()
            detections = self._extract_reid_features(images, detections)
            t3 = time.time()
            self.latency_dict['reid'] = t3 - t2

        # Tracking
        t4 = time.time()
        tracks = self.tracker.update(detections, images)
        t5 = time.time()
        self.latency_dict['tracking'] = t5 - t4

        # Total latency
        self.latency_dict['total'] = t5 - t0

        return tracks

    def run(
        self,
        max_frames: Optional[int] = None,
        display: bool = False,
        output_video: Optional[str] = None
    ):
        """
        Run pipeline continuously.

        Args:
            max_frames: Maximum number of frames to process (None = infinite)
            display: Whether to display results
            output_video: Path to save output video
        """
        self.start()

        frame_count = 0
        fps_timer = time.time()
        fps_frame_count = 0

        try:
            while True:
                # Capture frames
                frames = self.camera_manager.capture_synchronized(timeout=1.0)

                if frames is None:
                    logger.warning("Failed to capture synchronized frames")
                    continue

                # Process
                tracks = self.process_frame(frames)

                # Update FPS
                fps_frame_count += 1
                if time.time() - fps_timer >= 1.0:
                    self.fps_dict['overall'] = fps_frame_count / (time.time() - fps_timer)
                    fps_frame_count = 0
                    fps_timer = time.time()

                # Display
                if display:
                    self._display_results(frames, tracks)

                # Save video
                if output_video:
                    # TODO: Implement video writer
                    pass

                frame_count += 1

                # Check max frames
                if max_frames is not None and frame_count >= max_frames:
                    break

                # Check for exit
                if display and cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            self.stop()
            if display:
                cv2.destroyAllWindows()

    def _detect_batch(
        self,
        images: Dict[int, np.ndarray]
    ) -> Dict[int, List[Detection]]:
        """
        Run detection on batch of images.

        Args:
            images: {camera_id: image}

        Returns:
            {camera_id: [Detection]}
        """
        # Convert to list maintaining order
        camera_ids = list(images.keys())
        image_list = [images[cam_id] for cam_id in camera_ids]

        # Batch detection
        detection_lists = self.detector.detect_batch(image_list)

        # Convert back to dict
        detections = {
            cam_id: dets
            for cam_id, dets in zip(camera_ids, detection_lists)
        }

        return detections

    def _extract_reid_features(
        self,
        images: Dict[int, np.ndarray],
        detections: Dict[int, List[Detection]]
    ) -> Dict[int, List[Detection]]:
        """
        Extract ReID features for all detections.

        Args:
            images: {camera_id: image}
            detections: {camera_id: [Detection]}

        Returns:
            Updated detections with embeddings
        """
        if self.reid_extractor is None:
            return detections

        for cam_id, dets in detections.items():
            if len(dets) == 0:
                continue

            image = images[cam_id]
            bboxes = [det.bbox for det in dets]

            # Extract features
            embeddings = self.reid_extractor.extract_batch(image, bboxes)

            # Assign to detections
            for det, emb in zip(dets, embeddings):
                det.embedding = emb

        return detections

    def _display_results(
        self,
        frames: Dict[int, Frame],
        tracks: Dict[int, List[Track]]
    ):
        """
        Display tracking results.

        Args:
            frames: {camera_id: Frame}
            tracks: {camera_id: [Track]}
        """
        import cv2

        for cam_id, frame in frames.items():
            image = frame.image.copy()
            cam_tracks = tracks.get(cam_id, [])

            # Draw tracks
            for track in cam_tracks:
                x1, y1, x2, y2 = track.bbox.astype(int)

                # Color based on global ID or track ID
                color_id = track.global_id if track.global_id is not None else track.track_id
                color = self._get_color(color_id)

                # Draw box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label = f"ID:{track.track_id}"
                if track.global_id is not None:
                    label += f" G:{track.global_id}"

                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

            # Draw FPS
            fps_text = f"FPS: {self.fps_dict.get('overall', 0):.1f}"
            cv2.putText(
                image,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            # Draw latency
            lat_text = f"Lat: {self.latency_dict.get('total', 0)*1000:.1f}ms"
            cv2.putText(
                image,
                lat_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Show
            cv2.imshow(f"Camera {cam_id}", image)

    @staticmethod
    def _get_color(track_id: int) -> tuple:
        """Get consistent color for track ID."""
        np.random.seed(track_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color

    def get_metrics(self) -> Dict[str, float]:
        """
        Get current performance metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            **self.fps_dict,
            **self.latency_dict
        }

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
