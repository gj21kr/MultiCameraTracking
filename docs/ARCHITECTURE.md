# System Architecture

## Real-time Multi-Camera Tracking System with MLOps

### Overview

이 시스템은 4개의 USB 카메라로부터 실시간 영상을 수집하여 사람/물체를 추적하고,
MLOps 파이프라인을 통해 모델 성능을 지속적으로 모니터링하는 End-to-End 솔루션입니다.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                           │
│  │ Camera 0 │ │ Camera 1 │ │ Camera 2 │ │ Camera 3 │  USB 3.0 / GigE           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘                           │
│       │            │            │            │                                   │
│       └────────────┴────────────┴────────────┘                                  │
│                          │                                                       │
│                    ┌─────▼─────┐                                                │
│                    │  V4L2 /   │                                                │
│                    │  GStreamer│                                                │
│                    └─────┬─────┘                                                │
├──────────────────────────┼──────────────────────────────────────────────────────┤
│                    PROCESSING LAYER                                              │
│                          │                                                       │
│              ┌───────────▼───────────┐                                          │
│              │   Frame Synchronizer   │                                          │
│              │   (Multi-thread Queue) │                                          │
│              └───────────┬───────────┘                                          │
│                          │                                                       │
│    ┌─────────────────────┼─────────────────────┐                                │
│    │                     │                     │                                │
│    ▼                     ▼                     ▼                                │
│ ┌──────────┐      ┌──────────────┐      ┌───────────┐                          │
│ │ Detection│      │   Tracking   │      │  Re-ID    │                          │
│ │ (YOLO26) │──────│ (ByteTrack)  │──────│ (OSNet)   │                          │
│ └──────────┘      └──────────────┘      └───────────┘                          │
│       │                  │                    │                                 │
│       └──────────────────┼────────────────────┘                                 │
│                          │                                                       │
│              ┌───────────▼───────────┐                                          │
│              │   Cross-Camera        │                                          │
│              │   Track Association   │                                          │
│              └───────────┬───────────┘                                          │
├──────────────────────────┼──────────────────────────────────────────────────────┤
│                    MLOPS LAYER                                                   │
│                          │                                                       │
│    ┌─────────────────────┼─────────────────────┐                                │
│    │                     │                     │                                │
│    ▼                     ▼                     ▼                                │
│ ┌──────────┐      ┌──────────────┐      ┌───────────┐                          │
│ │  MLflow  │      │  Prometheus  │      │   DVC     │                          │
│ │ Tracking │      │   Metrics    │      │  Data Ver │                          │
│ └────┬─────┘      └──────┬───────┘      └─────┬─────┘                          │
│      │                   │                    │                                 │
│      └───────────────────┼────────────────────┘                                 │
│                          │                                                       │
│              ┌───────────▼───────────┐                                          │
│              │    Drift Detection    │                                          │
│              │    & Auto Retrain     │                                          │
│              └───────────┬───────────┘                                          │
├──────────────────────────┼──────────────────────────────────────────────────────┤
│                    OUTPUT LAYER                                                  │
│                          │                                                       │
│    ┌─────────────────────┼─────────────────────┐                                │
│    │                     │                     │                                │
│    ▼                     ▼                     ▼                                │
│ ┌──────────┐      ┌──────────────┐      ┌───────────┐                          │
│ │ REST API │      │   Grafana    │      │  WebSocket│                          │
│ │ (FastAPI)│      │  Dashboard   │      │  Stream   │                          │
│ └──────────┘      └──────────────┘      └───────────┘                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Input Layer

| Component | Technology | Specification |
|-----------|------------|---------------|
| Camera Interface | V4L2 / GStreamer | 4x USB 3.0 cameras |
| Resolution | 1920x1080 | 30 FPS target |
| Color Format | BGR / YUV420 | OpenCV compatible |
| Buffer | Ring Buffer | 5 frames per camera |

### 2. Processing Layer

#### 2.1 Frame Synchronizer
```python
# Multi-producer, Single-consumer pattern
class FrameSynchronizer:
    - Thread-safe queue per camera
    - Timestamp-based synchronization (±16ms tolerance)
    - Adaptive frame dropping under load
```

#### 2.2 Detection Module
- **Model**: YOLO26n/s/m/l/x (configurable)
- **Input**: 640x640 (letterbox)
- **Output**: Bounding boxes + confidence + class
- **Optimization**: TensorRT FP16/INT8 on Jetson
- **Improvements**: Better mAP (+3-5%), lower latency, improved small object detection vs YOLO26

#### 2.3 Tracking Module
- **Algorithm**: ByteTrack (primary) / DeepSORT (fallback)
- **State**: Kalman Filter (position, velocity)
- **Association**: IoU + ReID feature distance

#### 2.4 Cross-Camera Association
```
Camera Overlap Matrix:
       Cam0  Cam1  Cam2  Cam3
Cam0   -     20%   0%    10%
Cam1   20%   -     15%   0%
Cam2   0%    15%   -     25%
Cam3   10%   0%    25%   -
```

### 3. MLOps Layer

#### 3.1 MLflow Integration
```yaml
Tracking:
  - experiment_id: tracking_101
  - metrics: [mAP, MOTA, IDF1, FPS]
  - artifacts: [model.pt, config.yaml]

Model Registry:
  - staging → production workflow
  - automatic A/B testing
```

#### 3.2 Prometheus Metrics
```
# Custom metrics
tracking_fps_gauge
detection_latency_histogram
track_count_gauge
reid_distance_histogram
data_drift_score_gauge
```

#### 3.3 Drift Detection
- **Input Distribution**: Image statistics (mean, std, histogram)
- **Feature Distribution**: ReID embedding space clustering
- **Threshold**: KL-divergence > 0.1 triggers alert

### 4. Output Layer

#### 4.1 REST API (FastAPI)
```
GET  /api/v1/tracks         # Current active tracks
GET  /api/v1/tracks/{id}    # Specific track history
POST /api/v1/config         # Update runtime config
GET  /api/v1/metrics        # System metrics
WS   /api/v1/stream         # Real-time track updates
```

#### 4.2 Grafana Dashboard
- Real-time FPS per camera
- Detection/Tracking latency percentiles
- Track count over time
- Drift score trends

---

## Data Flow Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Raw Frame  │────▶│  Detection  │────▶│   Tracks    │
│  (4 cams)   │     │   Results   │     │   Output    │
└─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │
      ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    DVC      │     │   MLflow    │     │ Prometheus  │
│  (storage)  │     │  (logging)  │     │ (metrics)   │
└─────────────┘     └─────────────┘     └─────────────┘
```

---

## Deployment Architecture

### Development (Local)
```
┌─────────────────────────────────────┐
│         Docker Compose              │
│  ┌─────────┐  ┌─────────┐          │
│  │ Tracker │  │ MLflow  │          │
│  └─────────┘  └─────────┘          │
│  ┌─────────┐  ┌─────────┐          │
│  │Prometheus│  │ Grafana │          │
│  └─────────┘  └─────────┘          │
└─────────────────────────────────────┘
```

### Production (Jetson + Server)
```
┌─────────────────┐     ┌─────────────────┐
│  Jetson Edge    │     │   Cloud Server  │
│  ┌───────────┐  │     │  ┌───────────┐  │
│  │ Inference │  │────▶│  │  MLflow   │  │
│  │ Pipeline  │  │     │  │  Server   │  │
│  └───────────┘  │     │  └───────────┘  │
│                 │     │  ┌───────────┐  │
│                 │     │  │ Prometheus│  │
│                 │     │  │ + Grafana │  │
│                 │     │  └───────────┘  │
└─────────────────┘     └─────────────────┘
```

---

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| End-to-End Latency | < 100ms | Frame capture → Track output |
| Detection FPS | 30 FPS | Per camera |
| Tracking FPS | 30 FPS | All cameras combined |
| MOTA (MOT17) | > 60% | Multi-Object Tracking Accuracy |
| IDF1 (MOT17) | > 55% | ID F1 Score |
| Model Update | < 5 min | Auto-deploy on accuracy threshold |

---

## Technology Stack Summary

| Layer | Technology | Purpose |
|-------|------------|---------|
| Camera | V4L2, GStreamer | Frame capture |
| Detection | YOLO26, TensorRT | Object detection (latest, 2024) |
| Tracking | ByteTrack, Kalman | Multi-object tracking |
| ReID | OSNet, torchreid | Re-identification |
| MLOps | MLflow, DVC | Experiment tracking |
| Monitoring | Prometheus, Grafana | Metrics & visualization |
| API | FastAPI, WebSocket | Service interface |
| Deployment | Docker, Jetson | Containerization |
| CI/CD | GitHub Actions | Automation |
