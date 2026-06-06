"""MLflow experiment tracking integration.

NOTE: MLflow is an optional dependency (frozen Non-goal, see strategy ADR-001).
The import is guarded so the package remains importable without mlflow installed;
enabling tracking without it raises a clear error.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging

try:
    import mlflow
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None

from ..utils.config import MLflowConfig

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """MLflow experiment tracker."""

    def __init__(self, config: MLflowConfig, enabled: bool = True):
        """
        Initialize experiment tracker.

        Args:
            config: MLflow configuration
            enabled: Whether tracking is enabled
        """
        self.config = config
        self.enabled = enabled
        self.run_id: Optional[str] = None

        if not enabled:
            logger.info("Experiment tracking disabled")
            return

        if mlflow is None:
            logger.warning(
                "MLflow tracking requested but 'mlflow' is not installed; "
                "disabling tracking. Run: pip install mlflow"
            )
            self.enabled = False
            return

        # Set tracking URI
        mlflow.set_tracking_uri(config.tracking_uri)

        # Set experiment
        try:
            mlflow.set_experiment(config.experiment_name)
            logger.info(
                f"MLflow experiment tracking enabled: {config.experiment_name} "
                f"at {config.tracking_uri}"
            )
        except Exception as e:
            logger.warning(f"Failed to set MLflow experiment: {e}")
            logger.warning("Experiment tracking will be disabled")
            self.enabled = False

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Start new MLflow run.

        Args:
            run_name: Name for this run
            tags: Tags for this run

        Returns:
            Run ID or None if disabled
        """
        if not self.enabled:
            return None

        try:
            run = mlflow.start_run(run_name=run_name, tags=tags)
            self.run_id = run.info.run_id
            logger.info(f"Started MLflow run: {self.run_id}")
            return self.run_id
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            return None

    def end_run(self):
        """End current MLflow run."""
        if not self.enabled or self.run_id is None:
            return

        try:
            mlflow.end_run()
            logger.info(f"Ended MLflow run: {self.run_id}")
            self.run_id = None
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters.

        Args:
            params: Parameter dictionary
        """
        if not self.enabled or self.run_id is None:
            return

        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.error(f"Failed to log params: {e}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log metrics.

        Args:
            metrics: Metric dictionary
            step: Step number
        """
        if not self.enabled or self.run_id is None:
            return

        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

    def log_artifact(self, file_path: str):
        """
        Log artifact file.

        Args:
            file_path: Path to file
        """
        if not self.enabled or self.run_id is None:
            return

        try:
            mlflow.log_artifact(file_path)
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")

    def log_model(
        self,
        model_path: str,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None
    ):
        """
        Log model artifact.

        Args:
            model_path: Path to model file
            artifact_path: Artifact path in MLflow
            registered_model_name: Name for model registry
        """
        if not self.enabled or self.run_id is None:
            return

        try:
            mlflow.log_artifact(model_path, artifact_path=artifact_path)

            if registered_model_name is not None:
                # Register model
                model_uri = f"runs:/{self.run_id}/{artifact_path}"
                mlflow.register_model(model_uri, registered_model_name)
                logger.info(f"Registered model: {registered_model_name}")
        except Exception as e:
            logger.error(f"Failed to log model: {e}")

    def set_tags(self, tags: Dict[str, str]):
        """
        Set tags for current run.

        Args:
            tags: Tag dictionary
        """
        if not self.enabled or self.run_id is None:
            return

        try:
            mlflow.set_tags(tags)
        except Exception as e:
            logger.error(f"Failed to set tags: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()
        return False
