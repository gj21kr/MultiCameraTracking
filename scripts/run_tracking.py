"""Main script to run the tracking pipeline."""

import sys
import time
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.tracking.pipeline import TrackingPipeline
from src.mlops.experiment import ExperimentTracker
from src.mlops.metrics import MetricsCollector
from src.mlops.drift import DriftDetector


def setup_logging(log_level: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('tracking.log')
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Real-time Multi-Camera Tracking System"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--cameras',
        type=str,
        help='Comma-separated camera IDs (overrides config)'
    )
    parser.add_argument(
        '--display',
        action='store_true',
        help='Display tracking results'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        help='Maximum frames to process'
    )
    parser.add_argument(
        '--mlflow-tracking',
        action='store_true',
        help='Enable MLflow tracking'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        help='MLflow experiment name'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--api-only',
        action='store_true',
        help='Run API server without display'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override camera IDs if specified
    if args.cameras:
        camera_ids = [int(c) for c in args.cameras.split(',')]
        logger.info(f"Using cameras: {camera_ids}")
        # Update config (simplified, you may want to keep other camera settings)
        from src.utils.config import CameraConfig
        config.cameras = [
            CameraConfig(device_id=cam_id) for cam_id in camera_ids
        ]

    # Override experiment name if specified
    if args.experiment_name:
        config.mlops.mlflow.experiment_name = args.experiment_name

    # Initialize MLOps components
    experiment_tracker = ExperimentTracker(
        config.mlops.mlflow,
        enabled=args.mlflow_tracking
    )

    metrics_collector = MetricsCollector(
        config.mlops.prometheus,
        enabled=config.mlops.prometheus.enabled
    )

    drift_detector = DriftDetector(config.mlops.drift)

    # Initialize tracking pipeline
    logger.info("Initializing tracking pipeline...")
    pipeline = TrackingPipeline(config)

    # Start MLflow run
    if args.mlflow_tracking:
        experiment_tracker.start_run(
            run_name=f"tracking_run_{int(time.time())}",
            tags={
                'num_cameras': str(len(config.cameras)),
                'model': config.detection.model_type,
                'use_reid': str(config.tracking.use_reid)
            }
        )

        # Log parameters
        experiment_tracker.log_params({
            'detection_model': config.detection.model_type,
            'detection_confidence': config.detection.confidence_threshold,
            'tracking_algorithm': config.tracking.algorithm,
            'track_buffer': config.tracking.track_buffer,
            'use_reid': config.tracking.use_reid,
            'num_cameras': len(config.cameras)
        })

    # Run pipeline
    try:
        logger.info("Starting tracking pipeline...")
        pipeline.start()

        frame_count = 0

        while True:
            # Capture frames
            frames = pipeline.camera_manager.capture_synchronized(timeout=1.0)

            if frames is None:
                logger.warning("Failed to capture synchronized frames")
                continue

            # Process
            tracks = pipeline.process_frame(frames)

            # Update metrics
            latencies = pipeline.get_metrics()
            metrics_collector.observe_latency(latencies)

            for camera_id, cam_tracks in tracks.items():
                metrics_collector.update_active_tracks(camera_id, len(cam_tracks))
                metrics_collector.increment_frames_processed(camera_id)

            # Drift detection
            for camera_id, frame in frames.items():
                dets = pipeline.detector.detect(frame.image)
                drift_scores = drift_detector.update(frame.image, dets)

                for feature, score in drift_scores.items():
                    metrics_collector.update_drift_score(feature, score)

            # Display
            if args.display and not args.api_only:
                pipeline._display_results(frames, tracks)

                # Exit on 'q'
                import cv2
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Log to MLflow periodically
            if args.mlflow_tracking and frame_count % 100 == 0:
                experiment_tracker.log_metrics({
                    'fps': latencies.get('overall', 0),
                    'latency_ms': latencies.get('total', 0) * 1000,
                    'detection_latency_ms': latencies.get('detection', 0) * 1000,
                    'tracking_latency_ms': latencies.get('tracking', 0) * 1000,
                }, step=frame_count)

            frame_count += 1

            # Check max frames
            if args.max_frames and frame_count >= args.max_frames:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    except Exception as e:
        logger.error(f"Error in pipeline: {e}", exc_info=True)

    finally:
        # Cleanup
        logger.info("Shutting down...")
        pipeline.stop()

        if args.mlflow_tracking:
            experiment_tracker.end_run()

        if args.display:
            import cv2
            cv2.destroyAllWindows()

        logger.info("Shutdown complete")


if __name__ == '__main__':
    main()
