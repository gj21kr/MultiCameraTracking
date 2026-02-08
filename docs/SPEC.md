# Technical Specification

## Real-time Multi-Camera Tracking System with MLOps

---

## 1. System Requirements

### 1.1 Hardware Requirements

#### Development Environment
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Intel i5 / Ryzen 5 | Intel i7 / Ryzen 7 |
| RAM | 16GB | 32GB |
| GPU | GTX 1660 (6GB) | RTX 3060 (12GB) |
| Storage | 256GB SSD | 512GB NVMe |
| Camera | 2x USB 2.0 | 4x USB 3.0 |

#### Production Environment (Jetson)
| Component | Specification |
|-----------|---------------|
| Device | NVIDIA Jetson Orin Nano / AGX |
| RAM | 8GB+ unified memory |
| Storage | 64GB+ eMMC/NVMe |
| Camera | 4x USB 3.0 / MIPI CSI-2 |
| Power | 15W (Nano) / 60W (AGX) |

### 1.2 Software Requirements

```yaml
Operating System:
  - Ubuntu 22.04 LTS (x86_64)
  - JetPack 6.0+ (Jetson)

Python: 3.10+

CUDA: 12.x (desktop) / JetPack bundled (Jetson)

Docker: 24.0+ with NVIDIA Container Toolkit
```

---

## 2. Module Specifications

### 2.1 Camera Module (`src/tracking/camera.py`)

```python
class CameraConfig:
    """카메라 설정 스펙"""
    device_id: int          # /dev/video{N}
    width: int = 1920       # 해상도 너비
    height: int = 1080      # 해상도 높이
    fps: int = 30           # 목표 FPS
    buffer_size: int = 5    # 프레임 버퍼 크기
    codec: str = "MJPG"     # V4L2 코덱

class CameraManager:
    """멀티 카메라 관리자"""

    def __init__(self, configs: List[CameraConfig]):
        """
        Args:
            configs: 카메라별 설정 리스트

        Raises:
            CameraInitError: 카메라 초기화 실패
        """

    async def capture(self) -> Dict[int, Frame]:
        """
        모든 카메라에서 동기화된 프레임 캡처

        Returns:
            {camera_id: Frame} 딕셔너리

        Timing:
            < 33ms (30 FPS 유지)
        """

    def release(self) -> None:
        """리소스 해제"""
```

### 2.2 Detection Module (`src/tracking/detector.py`)

```python
class DetectorConfig:
    """탐지 모델 설정"""
    model_path: str                    # ONNX/TensorRT 모델 경로
    model_type: Literal["yolo26n", "yolo26s", "yolo26m"] = "yolo26s"
    confidence_threshold: float = 0.5  # 신뢰도 임계값
    nms_threshold: float = 0.45        # NMS 임계값
    input_size: Tuple[int, int] = (640, 640)  # 입력 크기
    device: str = "cuda:0"             # 추론 장치
    fp16: bool = True                  # FP16 사용 여부

class Detection:
    """탐지 결과 데이터 클래스"""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    confidence: float         # 신뢰도 점수
    class_id: int            # 클래스 ID
    class_name: str          # 클래스 이름
    embedding: np.ndarray    # ReID 특징 벡터 (optional)

class Detector:
    """YOLO26 기반 객체 탐지기"""

    def __init__(self, config: DetectorConfig):
        """
        TensorRT 엔진 로드 또는 ONNX에서 변환
        """

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        단일 프레임에서 객체 탐지

        Args:
            frame: BGR 이미지 (H, W, 3)

        Returns:
            Detection 리스트

        Performance:
            < 15ms on RTX 3060
            < 25ms on Jetson Orin Nano
        """

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """배치 추론 (멀티카메라용)"""
```

### 2.3 Tracker Module (`src/tracking/tracker.py`)

```python
class TrackerConfig:
    """트래커 설정"""
    algorithm: Literal["bytetrack", "deepsort"] = "bytetrack"
    track_thresh: float = 0.5      # 추적 시작 임계값
    match_thresh: float = 0.8      # 매칭 임계값
    track_buffer: int = 30         # 트랙 유지 프레임 수
    min_box_area: int = 10         # 최소 박스 면적
    use_reid: bool = True          # ReID 사용 여부

class Track:
    """트랙 데이터 클래스"""
    track_id: int                 # 고유 트랙 ID
    bbox: np.ndarray              # 현재 바운딩 박스
    velocity: np.ndarray          # 추정 속도 [vx, vy]
    age: int                      # 트랙 수명 (프레임)
    hits: int                     # 탐지 매칭 횟수
    time_since_update: int        # 마지막 업데이트 이후 프레임
    state: Literal["tentative", "confirmed", "deleted"]
    embedding: np.ndarray         # ReID 특징 (평균)
    camera_id: int               # 소속 카메라 ID
    global_id: Optional[int]     # 크로스 카메라 ID

class MultiCameraTracker:
    """멀티 카메라 통합 트래커"""

    def __init__(self, config: TrackerConfig, num_cameras: int):
        """
        Args:
            config: 트래커 설정
            num_cameras: 카메라 개수
        """

    def update(
        self,
        detections: Dict[int, List[Detection]],
        frames: Dict[int, np.ndarray]
    ) -> Dict[int, List[Track]]:
        """
        전체 카메라 트랙 업데이트

        Args:
            detections: {camera_id: [Detection]} 탐지 결과
            frames: {camera_id: frame} 원본 프레임

        Returns:
            {camera_id: [Track]} 트랙 결과
        """

    def associate_cross_camera(self) -> None:
        """크로스 카메라 트랙 연관"""
```

### 2.4 ReID Module (`src/tracking/reid.py`)

```python
class ReIDConfig:
    """ReID 모델 설정"""
    model_name: str = "osnet_x1_0"    # 모델 아키텍처
    model_path: Optional[str] = None  # 사전학습 가중치
    embedding_dim: int = 512          # 특징 벡터 차원
    device: str = "cuda:0"

class ReIDExtractor:
    """ReID 특징 추출기"""

    def __init__(self, config: ReIDConfig):
        """torchreid 모델 로드"""

    def extract(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        단일 객체 ReID 특징 추출

        Args:
            image: 전체 프레임
            bbox: 객체 바운딩 박스 [x1, y1, x2, y2]

        Returns:
            정규화된 특징 벡터 (embedding_dim,)
        """

    def extract_batch(
        self,
        image: np.ndarray,
        bboxes: List[np.ndarray]
    ) -> np.ndarray:
        """
        배치 ReID 특징 추출

        Returns:
            (N, embedding_dim) 특징 행렬
        """

    @staticmethod
    def cosine_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """코사인 거리 계산"""
```

---

## 3. MLOps Specifications

### 3.1 MLflow Integration (`src/mlops/experiment.py`)

```python
class ExperimentConfig:
    """MLflow 실험 설정"""
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "tracking_101"
    artifact_location: str = "./mlruns"

class ExperimentTracker:
    """MLflow 실험 추적기"""

    def __init__(self, config: ExperimentConfig):
        """MLflow 클라이언트 초기화"""

    def start_run(self, run_name: str, tags: Dict[str, str] = None) -> str:
        """
        새 실험 실행 시작

        Returns:
            run_id
        """

    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        """
        메트릭 로깅

        Required Metrics:
            - mAP: Mean Average Precision
            - MOTA: Multi-Object Tracking Accuracy
            - IDF1: ID F1 Score
            - fps: Frames Per Second
            - latency_ms: End-to-end latency
        """

    def log_model(self, model_path: str, model_name: str) -> None:
        """모델 아티팩트 로깅"""

    def register_model(
        self,
        model_name: str,
        stage: Literal["Staging", "Production"]
    ) -> None:
        """모델 레지스트리 등록"""
```

### 3.2 Prometheus Metrics (`src/mlops/metrics.py`)

```python
# 메트릭 정의
METRICS = {
    # Gauge: 현재 값
    "tracking_fps": Gauge(
        "tracking_fps_gauge",
        "Current tracking FPS",
        ["camera_id"]
    ),
    "active_tracks": Gauge(
        "active_tracks_gauge",
        "Number of active tracks",
        ["camera_id"]
    ),

    # Histogram: 분포
    "detection_latency": Histogram(
        "detection_latency_seconds",
        "Detection inference latency",
        buckets=[0.01, 0.02, 0.03, 0.05, 0.1]
    ),
    "tracking_latency": Histogram(
        "tracking_latency_seconds",
        "Tracking update latency",
        buckets=[0.001, 0.005, 0.01, 0.02, 0.05]
    ),

    # Counter: 누적
    "frames_processed": Counter(
        "frames_processed_total",
        "Total frames processed",
        ["camera_id"]
    ),
    "tracks_created": Counter(
        "tracks_created_total",
        "Total tracks created"
    )
}
```

### 3.3 Drift Detection (`src/mlops/drift.py`)

```python
class DriftConfig:
    """드리프트 감지 설정"""
    reference_window: int = 1000      # 참조 윈도우 크기
    detection_window: int = 100       # 감지 윈도우 크기
    threshold: float = 0.1            # KL-divergence 임계값
    features: List[str] = [           # 모니터링 특징
        "image_mean",
        "image_std",
        "bbox_size_dist",
        "reid_embedding_cluster"
    ]

class DriftDetector:
    """데이터 드리프트 감지기"""

    def __init__(self, config: DriftConfig):
        """참조 분포 초기화"""

    def update(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        드리프트 점수 업데이트

        Returns:
            {feature_name: drift_score}
        """

    def is_drifted(self) -> bool:
        """드리프트 발생 여부"""

    def get_retrain_trigger(self) -> Optional[str]:
        """재학습 트리거 이유 반환"""
```

---

## 4. API Specifications

### 4.1 REST API Endpoints

```yaml
# Health & Status
GET /api/v1/health:
  response:
    status: "healthy" | "degraded" | "unhealthy"
    uptime_seconds: float
    cameras_active: int

GET /api/v1/status:
  response:
    fps: Dict[camera_id, float]
    active_tracks: Dict[camera_id, int]
    latency_ms: float
    drift_score: float

# Tracks
GET /api/v1/tracks:
  query:
    camera_id: Optional[int]
    limit: int = 100
  response:
    tracks: List[Track]
    timestamp: datetime

GET /api/v1/tracks/{track_id}:
  response:
    track_id: int
    history: List[TrackSnapshot]
    first_seen: datetime
    last_seen: datetime
    cameras: List[int]

GET /api/v1/tracks/{track_id}/trajectory:
  query:
    duration_seconds: int = 60
  response:
    points: List[Point2D]
    timestamps: List[datetime]

# Configuration
GET /api/v1/config:
  response:
    detection: DetectorConfig
    tracking: TrackerConfig
    mlops: ExperimentConfig

PATCH /api/v1/config:
  body:
    detection: Optional[Partial[DetectorConfig]]
    tracking: Optional[Partial[TrackerConfig]]
  response:
    updated: bool
    config: FullConfig

# Metrics
GET /api/v1/metrics:
  response:
    format: "prometheus"
    content: str  # Prometheus exposition format
```

### 4.2 WebSocket API

```yaml
WS /api/v1/stream:
  # Client → Server
  subscribe:
    type: "subscribe"
    channels: ["tracks", "metrics", "alerts"]
    camera_ids: Optional[List[int]]

  # Server → Client (tracks channel)
  track_update:
    type: "track_update"
    camera_id: int
    tracks: List[Track]
    timestamp: datetime

  # Server → Client (metrics channel)
  metrics_update:
    type: "metrics_update"
    fps: Dict[int, float]
    latency_ms: float
    interval_ms: 1000

  # Server → Client (alerts channel)
  drift_alert:
    type: "drift_alert"
    feature: str
    score: float
    threshold: float
    recommendation: str
```

---

## 5. Configuration Schema

### 5.1 Main Configuration (`configs/config.yaml`)

```yaml
# 시스템 설정
system:
  log_level: INFO
  num_workers: 4
  device: "cuda:0"

# 카메라 설정
cameras:
  - device_id: 0
    width: 1920
    height: 1080
    fps: 30
  - device_id: 2
    width: 1920
    height: 1080
    fps: 30

# 탐지 설정
detection:
  model_type: yolo26s
  model_path: models/yolo26s.engine
  confidence_threshold: 0.5
  nms_threshold: 0.45
  input_size: [640, 640]
  fp16: true
  classes: [0]  # person only

# 트래킹 설정
tracking:
  algorithm: bytetrack
  track_thresh: 0.5
  match_thresh: 0.8
  track_buffer: 30
  use_reid: true

# ReID 설정
reid:
  model_name: osnet_x1_0
  model_path: models/osnet.pth
  embedding_dim: 512

# MLOps 설정
mlops:
  mlflow:
    tracking_uri: http://localhost:5000
    experiment_name: tracking_101
  prometheus:
    port: 8000
  drift:
    threshold: 0.1
    check_interval: 100

# API 설정
api:
  host: 0.0.0.0
  port: 8080
  cors_origins: ["*"]
```

---

## 6. Testing Specifications

### 6.1 Unit Tests

```python
# tests/test_detector.py
def test_detector_initialization():
    """모델 로드 테스트"""

def test_detection_output_format():
    """출력 포맷 검증"""

def test_detection_confidence_filter():
    """신뢰도 필터링 테스트"""

# tests/test_tracker.py
def test_track_creation():
    """새 트랙 생성 테스트"""

def test_track_association():
    """탐지-트랙 연관 테스트"""

def test_track_deletion():
    """트랙 삭제 조건 테스트"""
```

### 6.2 Integration Tests

```python
# tests/test_pipeline.py
def test_end_to_end_pipeline():
    """전체 파이프라인 통합 테스트"""
    # Given: 테스트 비디오 입력
    # When: 파이프라인 실행
    # Then: 유효한 트랙 출력

def test_mlflow_logging():
    """MLflow 로깅 통합 테스트"""

def test_prometheus_metrics():
    """Prometheus 메트릭 노출 테스트"""
```

### 6.3 Performance Tests

```python
# tests/test_performance.py
@pytest.mark.benchmark
def test_detection_latency():
    """탐지 지연시간 < 30ms"""

@pytest.mark.benchmark
def test_tracking_throughput():
    """30 FPS 처리량 유지"""

@pytest.mark.benchmark
def test_memory_usage():
    """메모리 사용량 < 4GB"""
```

---

## 7. Error Handling

### 7.1 Error Codes

| Code | Name | Description |
|------|------|-------------|
| E001 | CameraInitError | 카메라 초기화 실패 |
| E002 | CameraReadError | 프레임 읽기 실패 |
| E003 | ModelLoadError | 모델 로드 실패 |
| E004 | InferenceError | 추론 실패 |
| E005 | ConfigError | 설정 파싱 실패 |
| E006 | MLflowError | MLflow 연결 실패 |

### 7.2 Recovery Strategies

```python
# 자동 복구 전략
RECOVERY_STRATEGIES = {
    "CameraReadError": {
        "max_retries": 3,
        "backoff_ms": 100,
        "fallback": "skip_frame"
    },
    "InferenceError": {
        "max_retries": 1,
        "backoff_ms": 0,
        "fallback": "return_empty"
    },
    "MLflowError": {
        "max_retries": 5,
        "backoff_ms": 1000,
        "fallback": "local_logging"
    }
}
```
