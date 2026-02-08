# CLAUDE.md

This file provides guidance to Claude Code when working with the **Tracking_101** project.

---

## Overview

**Real-time Multi-Camera Tracking System with MLOps**

포트폴리오용 1인 프로젝트로, 4개 USB 카메라로부터 실시간 영상을 수집하여 사람/물체를 추적하고,
MLOps 파이프라인을 통해 모델 성능을 지속적으로 모니터링하는 End-to-End 시스템입니다.

### Key Features
- **Multi-Camera Tracking**: 4채널 동시 30fps 처리
- **MLOps Pipeline**: MLflow + Prometheus + Grafana 통합
- **Drift Detection**: 데이터 분포 변화 감지 및 재학습 트리거
- **Production Ready**: Docker 배포, CI/CD, A/B 테스트

### Technology Stack
```
Vision:    YOLO26 (detection) + ByteTrack (tracking) + OSNet (ReID)
MLOps:     MLflow, DVC, Prometheus, Grafana
API:       FastAPI, WebSocket
Deploy:    Docker, NVIDIA Container Runtime
CI/CD:     GitHub Actions, pytest
```

---

## Documentation Structure

```
docs/
├── ARCHITECTURE.md    # 시스템 아키텍처 (레이어별 상세)
├── SPEC.md           # 기술 명세 (API, 모듈, 설정)
├── ROADMAP.md        # 구현 로드맵 (6주 계획)
└── TROUBLESHOOTING.md # 문제 해결 가이드
```

**중요**: 아키텍처 이해를 위해 먼저 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)를 읽으세요.

---

## Environment Setup

### Prerequisites
```bash
# System requirements
- Ubuntu 22.04 LTS (or compatible Linux)
- Python 3.10+
- CUDA 12.x (for GPU acceleration)
- Docker 24.0+ with NVIDIA Container Runtime
```

### Installation

#### 1. Create Conda Environment
```bash
# Development environment
conda create -n tracking101 python=3.10
conda activate tracking101

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
cd /workspace/Code/Tracking_101
pip install -r requirements.txt
```

#### 2. Download Pre-trained Models
```bash
# YOLO26 detection model
python scripts/download_models.py --model yolo26s

# OSNet ReID model
python scripts/download_models.py --model osnet_x1_0

# Or manually:
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolo26s.pt -O models/yolo26s.pt
```

#### 3. Setup MLOps Stack (Docker)
```bash
# Start MLflow, Prometheus, Grafana
docker-compose up -d

# Verify services
curl http://localhost:5000  # MLflow
curl http://localhost:9090  # Prometheus
curl http://localhost:3000  # Grafana (admin/admin)
```

#### 4. Initialize DVC (Optional)
```bash
# Initialize DVC for data versioning
dvc init
dvc remote add -d myremote s3://mybucket/tracking101
dvc remote modify myremote region us-west-2
```

---

## Common Commands

### Development

#### Run Single Camera Test
```bash
# Test camera capture
python scripts/test_camera.py --device 0

# Test with display
python scripts/test_camera.py --device 0 --display
```

#### Run Detection Pipeline
```bash
# Single image inference
python src/tracking/detector.py --image data/test.jpg --output output/

# Batch inference on folder
python src/tracking/detector.py --folder data/images/ --output output/ --batch-size 8
```

#### Run Full Tracking Pipeline
```bash
# 2 cameras tracking
python scripts/run_tracking.py --cameras 0,2 --display

# 4 cameras with MLflow logging
python scripts/run_tracking.py \
    --cameras 0,1,2,3 \
    --config configs/config.yaml \
    --mlflow-tracking \
    --experiment-name tracking_demo

# Production mode (no display, API only)
python scripts/run_tracking.py --cameras 0,1,2,3 --api-only
```

### MLOps

#### MLflow Experiment Tracking
```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Run experiment with logging
python scripts/train_reid.py \
    --data-dir data/market1501/ \
    --mlflow-tracking \
    --experiment-name reid_training

# Compare runs
mlflow runs compare --experiment-id 0

# Register model to production
python scripts/register_model.py \
    --run-id <run_id> \
    --model-name tracking_model \
    --stage Production
```

#### Model Evaluation
```bash
# Evaluate on MOT17 test set
python scripts/evaluate.py \
    --dataset mot17 \
    --split test \
    --config configs/config.yaml \
    --output results/mot17_eval.json

# Metrics: MOTA, IDF1, FP, FN, ID switches
```

#### Drift Detection
```bash
# Run drift detection on new data
python src/mlops/drift.py \
    --reference-data data/reference/ \
    --new-data data/new/ \
    --threshold 0.1

# Output: drift_report.json
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_tracker.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run performance benchmarks
pytest tests/test_performance.py -v --benchmark-only
```

### Docker Deployment

```bash
# Build Docker image
docker build -t tracking101:latest .

# Run container with GPU support
docker run --gpus all \
    -p 8080:8080 \
    -v /dev/video0:/dev/video0 \
    -v /dev/video1:/dev/video1 \
    -v /dev/video2:/dev/video2 \
    -v /dev/video3:/dev/video3 \
    tracking101:latest

# Full stack deployment
docker-compose -f docker-compose.prod.yaml up -d

# Check logs
docker-compose logs -f tracker
```

### API Usage

```bash
# Health check
curl http://localhost:8080/api/v1/health

# Get current tracks
curl http://localhost:8080/api/v1/tracks

# Get specific track history
curl http://localhost:8080/api/v1/tracks/42

# Update configuration
curl -X PATCH http://localhost:8080/api/v1/config \
    -H "Content-Type: application/json" \
    -d '{"detection": {"confidence_threshold": 0.6}}'

# WebSocket streaming (Python client)
python scripts/ws_client.py --url ws://localhost:8080/api/v1/stream
```

---

## Project Structure

```
Tracking_101/
├── configs/
│   ├── config.yaml              # 메인 설정 파일
│   ├── prometheus.yml           # Prometheus 설정
│   └── grafana_dashboard.json   # Grafana 대시보드
│
├── data/                        # 데이터 디렉토리 (gitignored)
│   ├── videos/                  # 테스트 비디오
│   ├── images/                  # 테스트 이미지
│   └── datasets/                # MOT17, Market-1501 등
│
├── docs/
│   ├── ARCHITECTURE.md          # 시스템 아키텍처
│   ├── SPEC.md                  # 기술 명세
│   ├── ROADMAP.md               # 구현 로드맵
│   └── TROUBLESHOOTING.md       # 문제 해결
│
├── models/                      # 모델 가중치 (DVC tracked)
│   ├── yolo26s.pt
│   ├── yolo26s.engine           # TensorRT
│   └── osnet_x1_0.pth
│
├── src/
│   ├── tracking/
│   │   ├── camera.py            # 카메라 캡처
│   │   ├── detector.py          # YOLO26 탐지
│   │   ├── tracker.py           # ByteTrack 추적
│   │   ├── reid.py              # OSNet ReID
│   │   └── pipeline.py          # 통합 파이프라인
│   │
│   ├── mlops/
│   │   ├── experiment.py        # MLflow 통합
│   │   ├── metrics.py           # Prometheus 메트릭
│   │   ├── drift.py             # 드리프트 감지
│   │   └── ab_testing.py        # A/B 테스트
│   │
│   ├── api/
│   │   ├── server.py            # FastAPI 서버
│   │   ├── routes.py            # API 라우트
│   │   └── websocket.py         # WebSocket 핸들러
│   │
│   └── utils/
│       ├── config.py            # 설정 로드
│       ├── visualization.py     # 시각화 유틸
│       └── metrics.py           # 평가 메트릭
│
├── scripts/
│   ├── download_models.py       # 모델 다운로드
│   ├── test_camera.py           # 카메라 테스트
│   ├── run_tracking.py          # 메인 실행 스크립트
│   ├── evaluate.py              # 모델 평가
│   └── register_model.py        # 모델 레지스트리
│
├── tests/
│   ├── test_camera.py
│   ├── test_detector.py
│   ├── test_tracker.py
│   ├── test_reid.py
│   ├── test_integration.py
│   └── test_performance.py
│
├── .github/
│   └── workflows/
│       ├── test.yml             # CI 테스트
│       └── deploy.yml           # CD 배포
│
├── docker-compose.yaml          # 개발 환경
├── docker-compose.prod.yaml     # 프로덕션 환경
├── Dockerfile
├── requirements.txt
├── dvc.yaml                     # DVC 파이프라인
├── .dvcignore
└── README.md
```

---

## Module Dependencies

### Core Tracking Pipeline
```
camera.py → detector.py → tracker.py → reid.py → pipeline.py
   ↓           ↓             ↓           ↓
   └───────────┴─────────────┴───────────┘
                     ↓
              mlops/metrics.py (Prometheus)
```

### MLOps Pipeline
```
pipeline.py → experiment.py (MLflow logging)
             → drift.py (drift detection)
             → ab_testing.py (model comparison)
```

### API Layer
```
server.py → routes.py → pipeline.py
         → websocket.py → pipeline.py
```

---

## Configuration Guide

### Main Config (`configs/config.yaml`)

```yaml
# 카메라 설정
cameras:
  - device_id: 0      # /dev/video0
    width: 1920
    height: 1080
    fps: 30

# 탐지 설정
detection:
  model_type: yolo26s
  confidence_threshold: 0.5
  nms_threshold: 0.45
  classes: [0]  # 0 = person

# 트래킹 설정
tracking:
  algorithm: bytetrack
  track_thresh: 0.5
  match_thresh: 0.8
  use_reid: true

# MLOps 설정
mlops:
  mlflow:
    tracking_uri: http://localhost:5000
    experiment_name: tracking_101
  drift:
    threshold: 0.1
    check_interval: 100
```

### Runtime Configuration Override
```bash
# CLI arguments override config file
python scripts/run_tracking.py \
    --config configs/config.yaml \
    --detection-confidence 0.6 \
    --tracking-algorithm deepsort
```

---

## Performance Optimization

### GPU Optimization
```bash
# Export to TensorRT for faster inference
python scripts/export_tensorrt.py \
    --model models/yolo26s.pt \
    --output models/yolo26s.engine \
    --fp16

# Use TensorRT model
python scripts/run_tracking.py \
    --detector-model models/yolo26s.engine \
    --detector-backend tensorrt
```

### Multi-Processing
```python
# pipeline.py uses multi-threading by default
# For CPU-bound tasks, use multi-processing
WORKERS = 4  # Adjust based on CPU cores
```

### Memory Management
```bash
# Monitor memory usage
python scripts/run_tracking.py --profile-memory

# Reduce batch size if OOM
# Edit configs/config.yaml:
detection:
  batch_size: 4  # Reduce from 8
```

---

## Monitoring & Debugging

### Grafana Dashboard

Access: http://localhost:3000 (admin/admin)

**Key Metrics**:
- Tracking FPS per camera
- Detection latency (P50, P95, P99)
- Active track count
- Drift score over time

### Prometheus Queries

```promql
# Average FPS across all cameras
avg(tracking_fps_gauge)

# 95th percentile detection latency
histogram_quantile(0.95, detection_latency_seconds_bucket)

# Total frames processed
sum(rate(frames_processed_total[5m]))
```

### Logging

```python
# Enable debug logging
python scripts/run_tracking.py --log-level DEBUG

# Log to file
python scripts/run_tracking.py --log-file tracking.log
```

---

## Common Issues & Solutions

### Camera Not Found
```bash
# Check available cameras
v4l2-ctl --list-devices

# Check permissions
sudo usermod -aG video $USER
```

### CUDA Out of Memory
```bash
# Reduce batch size
export BATCH_SIZE=4

# Use smaller model
python scripts/run_tracking.py --model yolo26n  # nano model
```

### Low FPS
```bash
# Check bottleneck
python scripts/run_tracking.py --profile

# Optimize:
# 1. Use TensorRT
# 2. Reduce resolution
# 3. Disable visualization
# 4. Use smaller model
```

### MLflow Connection Error
```bash
# Check if MLflow server is running
curl http://localhost:5000

# Restart MLflow
docker-compose restart mlflow

# Use local logging fallback
python scripts/run_tracking.py --mlflow-fallback-local
```

---

## Development Workflow

### Adding a New Feature

1. **Branch**: `git checkout -b feature/my-feature`
2. **Implement**: Write code in `src/`
3. **Test**: Add tests in `tests/`
4. **Run Tests**: `pytest tests/ -v`
5. **Document**: Update relevant docs
6. **Commit**: `git commit -m "feat: add my feature"`
7. **Push**: `git push origin feature/my-feature`
8. **PR**: Create pull request (CI runs automatically)

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/

# Run all quality checks
pre-commit run --all-files
```

---

## Benchmarks (Target)

| Metric | Target | Actual |
|--------|--------|--------|
| Detection FPS | 30 | TBD |
| Tracking FPS | 30 | TBD |
| End-to-End Latency | < 100ms | TBD |
| MOTA (MOT17) | > 60% | TBD |
| IDF1 (MOT17) | > 55% | TBD |
| Memory Usage | < 4GB | TBD |

**Update after implementation**: Run `scripts/benchmark.py` and update table.

---

## Dataset Information

### MOT17 (Tracking Evaluation)
```bash
# Download
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip -d data/

# Evaluate
python scripts/evaluate.py --dataset mot17 --split test
```

### Market-1501 (ReID Evaluation)
```bash
# Download
python scripts/download_datasets.py --dataset market1501

# Train ReID model
python scripts/train_reid.py --data-dir data/market1501/
```

### Custom Dataset
```bash
# Convert to standard format
python scripts/convert_dataset.py \
    --input data/raw/ \
    --output data/processed/ \
    --format mot17

# Track with DVC
dvc add data/processed/
git add data/processed.dvc
```

---

## CI/CD Pipeline

### GitHub Actions Workflow

**.github/workflows/test.yml** (On every PR):
```yaml
- Lint & format check
- Unit tests
- Integration tests
- Performance benchmarks
```

**.github/workflows/deploy.yml** (On merge to main):
```yaml
- Build Docker image
- Run model validation
- Deploy if mAP > 60%
- Update MLflow registry
```

### Manual Deployment
```bash
# Tag release
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0

# Deploy
./scripts/deploy.sh --environment production --tag v1.0.0
```

---

## Contributing

This is a solo portfolio project, but contributions are welcome!

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

---

## License

MIT License - See LICENSE file for details

---

## Contact & Links

- **GitHub**: [Your GitHub URL]
- **Portfolio**: [Your Portfolio URL]
- **Demo Video**: [YouTube/Vimeo URL]
- **Documentation**: See `docs/` folder

---

## Next Steps

1. **First Time Setup**: Follow [Environment Setup](#environment-setup)
2. **Understand Architecture**: Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
3. **Implementation**: Follow [docs/ROADMAP.md](docs/ROADMAP.md)
4. **Development**: Start with [scripts/test_camera.py](scripts/test_camera.py)

**Questions?** Check [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) or open an issue.
