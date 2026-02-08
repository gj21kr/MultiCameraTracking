# Tracking_101: Real-time Multi-Camera Tracking System with MLOps

포트폴리오용 1인 프로젝트: 4개 USB 카메라로 실시간 객체 추적 및 MLOps 파이프라인 통합

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Features

- **Multi-Camera Tracking**: 4채널 동시 30fps 처리
- **State-of-the-Art Models**: YOLO26 detection + ByteTrack + OSNet ReID
- **MLOps Pipeline**: MLflow experiment tracking + Prometheus monitoring + Grafana dashboard
- **Drift Detection**: 실시간 데이터 분포 변화 감지 및 알림
- **Production Ready**: Docker 배포, REST API, CI/CD

## Architecture

```
Camera → Detection → Tracking → ReID → Cross-Camera Association
  ↓         ↓          ↓         ↓              ↓
MLflow   Prometheus  Grafana   DVC         A/B Testing
```

자세한 내용은 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)를 참조하세요.

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/Tracking_101.git
cd Tracking_101

# Create conda environment
conda create -n tracking101 python=3.10
conda activate tracking101

# Install dependencies
pip install -r requirements.txt
```

### 2. Start MLOps Stack

```bash
# Start MLflow, Prometheus, Grafana
docker-compose up -d

# Verify services
curl http://localhost:5000  # MLflow
curl http://localhost:9090  # Prometheus
open http://localhost:3000  # Grafana (admin/admin)
```

### 3. Run Tracking

```bash
# Test with 2 cameras
python scripts/run_tracking.py --cameras 0,2 --display

# Production mode with MLflow tracking
python scripts/run_tracking.py \
    --cameras 0,1,2,3 \
    --config configs/config.yaml \
    --mlflow-tracking \
    --experiment-name production_run
```

## Configuration

Edit `configs/config.yaml`:

```yaml
cameras:
  - device_id: 0
    width: 1920
    height: 1080
    fps: 30

detection:
  model_type: yolo26s
  confidence_threshold: 0.5

tracking:
  algorithm: bytetrack
  use_reid: true

mlops:
  mlflow:
    tracking_uri: http://localhost:5000
  drift:
    threshold: 0.1
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) - System design & data flow
- [Technical Spec](docs/SPEC.md) - API reference & module specs
- [Roadmap](docs/ROADMAP.md) - Implementation timeline (6 weeks)
- [Claude Guide](claude.md) - Development guide for Claude Code

## Project Structure

```
Tracking_101/
├── src/
│   ├── tracking/       # Camera, detector, tracker, ReID
│   ├── mlops/          # MLflow, Prometheus, drift detection
│   ├── api/            # FastAPI server (TODO)
│   └── utils/          # Configuration, visualization
├── configs/            # YAML configs
├── docs/               # Documentation
├── scripts/            # Run scripts
├── tests/              # Unit & integration tests
├── models/             # Model weights
└── data/               # Datasets (gitignored)
```

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| FPS (4 cameras) | 30 | ⏳ TBD |
| End-to-End Latency | < 100ms | ⏳ TBD |
| MOTA (MOT17) | > 60% | ⏳ TBD |
| IDF1 (MOT17) | > 55% | ⏳ TBD |

## Development Status

- [x] Project structure & documentation
- [x] Core tracking pipeline (camera, detector, tracker, ReID)
- [x] MLOps integration (MLflow, Prometheus, drift)
- [ ] API server (FastAPI + WebSocket)
- [ ] Unit & integration tests
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Docker optimization
- [ ] Jetson deployment
- [ ] Performance benchmarks
- [ ] Demo video

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run benchmarks
pytest tests/test_performance.py --benchmark-only
```

## Monitoring

### Grafana Dashboard

Access: http://localhost:3000 (admin/admin)

**Metrics**:
- Real-time FPS per camera
- Detection/tracking latency (P50, P95, P99)
- Active track count
- Drift scores

### Prometheus Queries

```promql
# Average FPS
avg(tracking_fps_gauge)

# 95th percentile latency
histogram_quantile(0.95, detection_latency_seconds_bucket)
```

## Contributing

This is a portfolio project, but contributions are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) file

## Acknowledgments

- [YOLO26](https://github.com/ultralytics/ultralytics) - Object detection
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - Multi-object tracking
- [torchreid](https://github.com/KaiyangZhou/deep-person-reid) - Person re-identification
- [MLflow](https://mlflow.org/) - Experiment tracking
- [Prometheus](https://prometheus.io/) + [Grafana](https://grafana.com/) - Monitoring

## Contact

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Portfolio**: [your-portfolio.com](https://your-portfolio.com)
- **Email**: your.email@example.com

---

**Status**: 🚧 Active Development | **Start Date**: 2026-02 | **Target**: Production-ready in 6 weeks
