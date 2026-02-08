"""Test script for camera capture."""

import sys
import cv2
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import CameraConfig
from src.tracking.camera import CameraCapture


def main():
    parser = argparse.ArgumentParser(description="Test camera capture")
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1920,
        help='Frame width (default: 1920)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=1080,
        help='Frame height (default: 1080)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Target FPS (default: 30)'
    )
    parser.add_argument(
        '--display',
        action='store_true',
        help='Display captured frames'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create camera config
    config = CameraConfig(
        device_id=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps
    )

    # Initialize camera
    logger.info(f"Initializing camera {args.device}...")
    try:
        camera = CameraCapture(config)
        camera.start()
    except Exception as e:
        logger.error(f"Failed to initialize camera: {e}")
        return

    logger.info("Camera started. Press 'q' to quit.")

    try:
        import time
        frame_count = 0
        start_time = time.time()

        while True:
            # Get frame
            frame = camera.get_frame(timeout=1.0)

            if frame is None:
                logger.warning("Failed to get frame")
                continue

            frame_count += 1

            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Display info
            if frame_count % 30 == 0:
                logger.info(
                    f"Frame {frame_count} | "
                    f"FPS: {fps:.1f} | "
                    f"Shape: {frame.image.shape} | "
                    f"Timestamp: {frame.timestamp:.3f}"
                )

            # Display frame
            if args.display:
                display_frame = frame.image.copy()

                # Draw FPS
                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(f"Camera {args.device}", display_frame)

                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        # Cleanup
        logger.info("Releasing camera...")
        camera.release()

        if args.display:
            cv2.destroyAllWindows()

        logger.info(
            f"Test complete. Processed {frame_count} frames "
            f"at {fps:.1f} FPS"
        )


if __name__ == '__main__':
    main()
