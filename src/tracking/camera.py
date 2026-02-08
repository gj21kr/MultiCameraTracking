"""Multi-camera capture and frame synchronization."""

import cv2
import time
import threading
import numpy as np
from queue import Queue, Empty
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

from ..utils.config import CameraConfig

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    """Frame data with metadata."""
    camera_id: int
    frame_id: int
    timestamp: float
    image: np.ndarray


class CameraCapture:
    """Single camera capture thread."""

    def __init__(self, config: CameraConfig):
        """
        Initialize camera capture.

        Args:
            config: Camera configuration

        Raises:
            RuntimeError: If camera cannot be opened
        """
        self.config = config
        self.camera_id = config.device_id

        # Open camera
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")

        # Set properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        self.cap.set(cv2.CAP_PROP_FPS, config.fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.codec))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

        # Verify settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        logger.info(
            f"Camera {self.camera_id} opened: {actual_width}x{actual_height} @ {actual_fps}fps"
        )

        # Threading
        self.frame_queue = Queue(maxsize=config.buffer_size)
        self.running = False
        self.thread = None
        self.frame_count = 0

    def start(self):
        """Start capture thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info(f"Camera {self.camera_id} capture started")

    def stop(self):
        """Stop capture thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        logger.info(f"Camera {self.camera_id} capture stopped")

    def _capture_loop(self):
        """Main capture loop (runs in thread)."""
        while self.running:
            ret, image = self.cap.read()

            if not ret:
                logger.warning(f"Camera {self.camera_id} read failed")
                time.sleep(0.01)
                continue

            frame = Frame(
                camera_id=self.camera_id,
                frame_id=self.frame_count,
                timestamp=time.time(),
                image=image
            )

            # Drop old frames if queue is full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass

            self.frame_queue.put(frame)
            self.frame_count += 1

    def get_frame(self, timeout: float = 0.1) -> Optional[Frame]:
        """
        Get latest frame from queue.

        Args:
            timeout: Timeout in seconds

        Returns:
            Frame object or None if timeout
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except Empty:
            return None

    def release(self):
        """Release camera resources."""
        self.stop()
        if self.cap:
            self.cap.release()
        logger.info(f"Camera {self.camera_id} released")


class CameraManager:
    """Multi-camera manager with frame synchronization."""

    def __init__(self, configs: List[CameraConfig]):
        """
        Initialize camera manager.

        Args:
            configs: List of camera configurations

        Raises:
            RuntimeError: If any camera fails to initialize
        """
        self.configs = configs
        self.cameras: Dict[int, CameraCapture] = {}

        # Initialize all cameras
        for config in configs:
            try:
                camera = CameraCapture(config)
                self.cameras[config.device_id] = camera
                logger.info(f"Initialized camera {config.device_id}")
            except Exception as e:
                logger.error(f"Failed to initialize camera {config.device_id}: {e}")
                # Cleanup already initialized cameras
                self.release()
                raise RuntimeError(f"Camera initialization failed: {e}")

    def start_all(self):
        """Start all camera capture threads."""
        for camera in self.cameras.values():
            camera.start()
        logger.info(f"Started {len(self.cameras)} cameras")

    def stop_all(self):
        """Stop all camera capture threads."""
        for camera in self.cameras.values():
            camera.stop()
        logger.info("Stopped all cameras")

    def capture_synchronized(
        self,
        timeout: float = 0.5,
        max_time_diff: float = 0.033  # ~1 frame at 30fps
    ) -> Optional[Dict[int, Frame]]:
        """
        Capture synchronized frames from all cameras.

        Args:
            timeout: Max time to wait for all cameras
            max_time_diff: Max timestamp difference for sync (seconds)

        Returns:
            Dictionary of {camera_id: Frame} or None if sync failed
        """
        frames: Dict[int, Frame] = {}
        start_time = time.time()

        # Collect one frame from each camera
        for camera_id, camera in self.cameras.items():
            frame = camera.get_frame(timeout=timeout)

            if frame is None:
                logger.warning(f"Camera {camera_id} timeout")
                return None

            frames[camera_id] = frame

            # Check timeout
            if time.time() - start_time > timeout:
                logger.warning("Synchronized capture timeout")
                return None

        # Check timestamp synchronization
        timestamps = [f.timestamp for f in frames.values()]
        time_span = max(timestamps) - min(timestamps)

        if time_span > max_time_diff:
            logger.debug(f"Frame time span: {time_span*1000:.1f}ms (above threshold)")
            # Still return frames, but log warning
            # In production, you might want to drop and retry

        return frames

    def capture_async(self) -> Dict[int, Optional[Frame]]:
        """
        Capture latest frames from all cameras (non-blocking).

        Returns:
            Dictionary of {camera_id: Frame or None}
        """
        frames = {}
        for camera_id, camera in self.cameras.items():
            frames[camera_id] = camera.get_frame(timeout=0.001)
        return frames

    def get_camera_ids(self) -> List[int]:
        """Get list of camera IDs."""
        return list(self.cameras.keys())

    def release(self):
        """Release all camera resources."""
        self.stop_all()
        for camera in self.cameras.values():
            camera.release()
        self.cameras.clear()
        logger.info("Released all cameras")

    def __enter__(self):
        """Context manager entry."""
        self.start_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
