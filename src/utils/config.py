"""Configuration management for Tracking_101."""

import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    """
    Resolve a requested torch device against actual hardware availability.

    Falls back to CPU when CUDA is requested but unavailable (ADR-005).

    Args:
        requested: Requested device string (e.g. "cuda:0", "cuda", "cpu").

    Returns:
        A usable device string ("cuda:0"/"cuda" if available, else "cpu").
    """
    if requested is None or str(requested).lower() == "cpu":
        return "cpu"

    try:
        import torch
        if torch.cuda.is_available():
            return requested
        logger.warning(
            "Requested device '%s' but CUDA is not available; falling back to CPU.",
            requested,
        )
        return "cpu"
    except ImportError:
        logger.warning("torch not importable; using CPU.")
        return "cpu"


@dataclass
class CameraConfig:
    """Camera configuration."""
    device_id: int
    width: int = 1920
    height: int = 1080
    fps: int = 30
    buffer_size: int = 5
    codec: str = "MJPG"


@dataclass
class DetectorConfig:
    """Object detector configuration."""
    model_type: str = "yolo26s"  # yolo26n, yolo26s, yolo26m, yolo26l, yolo26x
    model_path: Optional[str] = None
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    input_size: tuple = (640, 640)
    device: str = "cuda:0"
    fp16: bool = True
    int8: bool = False  # INT8 quantization for Jetson
    classes: List[int] = field(default_factory=lambda: [0])  # person class


@dataclass
class TrackerConfig:
    """Multi-object tracker configuration."""
    algorithm: str = "bytetrack"
    track_thresh: float = 0.5
    match_thresh: float = 0.8
    track_buffer: int = 30
    min_box_area: int = 10
    use_reid: bool = True


@dataclass
class ReIDConfig:
    """ReID model configuration."""
    model_name: str = "osnet_x1_0"
    model_path: Optional[str] = None
    embedding_dim: int = 512
    device: str = "cuda:0"


@dataclass
class MLflowConfig:
    """MLflow configuration."""
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "tracking_101"
    artifact_location: str = "./mlruns"


@dataclass
class PrometheusConfig:
    """Prometheus configuration."""
    port: int = 8000
    enabled: bool = True


@dataclass
class DriftConfig:
    """Drift detection configuration."""
    threshold: float = 0.1
    check_interval: int = 100
    reference_window: int = 1000
    detection_window: int = 100


@dataclass
class MLOpsConfig:
    """MLOps configuration."""
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class SystemConfig:
    """System-level configuration."""
    log_level: str = "INFO"
    num_workers: int = 4
    device: str = "cuda:0"


@dataclass
class Config:
    """Main configuration class."""
    system: SystemConfig = field(default_factory=SystemConfig)
    cameras: List[CameraConfig] = field(default_factory=list)
    detection: DetectorConfig = field(default_factory=DetectorConfig)
    tracking: TrackerConfig = field(default_factory=TrackerConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    mlops: MLOpsConfig = field(default_factory=MLOpsConfig)
    api: APIConfig = field(default_factory=APIConfig)


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Config object

    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If config file is invalid
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Parse system config
    system_dict = config_dict.get('system', {})
    system_config = SystemConfig(**system_dict)

    # Parse camera configs
    cameras_list = config_dict.get('cameras', [])
    camera_configs = [CameraConfig(**cam) for cam in cameras_list]

    # Parse detection config
    detection_dict = config_dict.get('detection', {})
    detection_config = DetectorConfig(**detection_dict)

    # Parse tracking config
    tracking_dict = config_dict.get('tracking', {})
    tracking_config = TrackerConfig(**tracking_dict)

    # Parse ReID config
    reid_dict = config_dict.get('reid', {})
    reid_config = ReIDConfig(**reid_dict)

    # Parse MLOps config
    mlops_dict = config_dict.get('mlops', {})
    mlflow_dict = mlops_dict.get('mlflow', {})
    prometheus_dict = mlops_dict.get('prometheus', {})
    drift_dict = mlops_dict.get('drift', {})

    mlops_config = MLOpsConfig(
        mlflow=MLflowConfig(**mlflow_dict),
        prometheus=PrometheusConfig(**prometheus_dict),
        drift=DriftConfig(**drift_dict)
    )

    # Parse API config
    api_dict = config_dict.get('api', {})
    api_config = APIConfig(**api_dict)

    return Config(
        system=system_config,
        cameras=camera_configs,
        detection=detection_config,
        tracking=tracking_config,
        reid=reid_config,
        mlops=mlops_config,
        api=api_config
    )


def save_config(config: Config, config_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Config object
        config_path: Path to save YAML file
    """
    import dataclasses

    def dataclass_to_dict(obj):
        """Convert dataclass to dict recursively."""
        if dataclasses.is_dataclass(obj):
            return {
                k: dataclass_to_dict(v)
                for k, v in dataclasses.asdict(obj).items()
            }
        elif isinstance(obj, list):
            return [dataclass_to_dict(item) for item in obj]
        else:
            return obj

    config_dict = dataclass_to_dict(config)

    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
