"""YOLO26 object detector with TensorRT support."""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

from ..utils.config import DetectorConfig, resolve_device

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Detection result."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    embedding: Optional[np.ndarray] = None  # ReID feature (optional)


class Detector:
    """YOLO26 object detector."""

    def __init__(self, config: DetectorConfig):
        """
        Initialize detector.

        Args:
            config: Detector configuration

        Raises:
            RuntimeError: If model loading fails
        """
        self.config = config
        self.device = torch.device(resolve_device(config.device))

        # Load model
        model_path = config.model_path or f"models/{config.model_type}.pt"
        self.model = self._load_model(model_path)

        logger.info(
            f"Loaded {config.model_type} on {config.device} "
            f"(FP16: {config.fp16})"
        )

        # COCO class names (80 classes)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    def _load_model(self, model_path: str):
        """Load YOLO26 model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "ultralytics not installed. Run: pip install ultralytics>=8.3.0"
            )

        model_file = Path(model_path)

        if model_file.exists():
            model = YOLO(model_path)
        else:
            # Auto-download by model name, with a robust fallback.
            # yolo26* requires a recent ultralytics; if it cannot be resolved
            # (older ultralytics / asset missing) fall back to yolo11s. (FEAT-2)
            primary = f"{self.config.model_type}.pt"
            fallback = "yolo11s.pt"
            logger.warning(f"Model file not found: {model_path}")
            logger.info(f"Attempting auto-download of '{primary}'...")
            try:
                model = YOLO(primary)
                self.weight_name = primary
            except Exception as e:
                if primary == fallback:
                    raise
                logger.warning(
                    f"Could not resolve '{primary}' ({type(e).__name__}: {e}); "
                    f"falling back to '{fallback}'."
                )
                model = YOLO(fallback)
                self.weight_name = fallback
        if not hasattr(self, "weight_name"):
            self.weight_name = model_path

        # Move to device
        model.to(self.device)

        # Set FP16 if requested
        if self.config.fp16 and self.device.type == 'cuda':
            model.model.half()

        logger.info(
            f"YOLO26 model loaded: {self.config.model_type} "
            f"(params: {sum(p.numel() for p in model.model.parameters())/1e6:.1f}M)"
        )

        return model

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a single frame.

        Args:
            frame: BGR image (H, W, 3)

        Returns:
            List of Detection objects
        """
        # Run inference
        results = self.model.predict(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.nms_threshold,
            classes=self.config.classes,
            verbose=False,
            imgsz=self.config.input_size[0]
        )

        detections = []

        # Parse results
        if len(results) > 0:
            result = results[0]  # Single image

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                for i in range(len(boxes)):
                    # Get box data
                    bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(boxes.conf[i].cpu())
                    cls_id = int(boxes.cls[i].cpu())

                    # Create detection
                    detection = Detection(
                        bbox=bbox,
                        confidence=conf,
                        class_id=cls_id,
                        class_name=self.class_names[cls_id]
                    )
                    detections.append(detection)

        return detections

    def detect_tiled(
        self,
        frame: np.ndarray,
        rows: int = 2,
        cols: int = 3,
        overlap: float = 0.2,
        nms_iou: float = 0.6,
    ) -> List[Detection]:
        """SAHI-style sliced inference for small/distant objects (no training).

        Splits the frame into an overlapping ``rows x cols`` grid, runs detection
        on each tile (in one batch), maps boxes back to full-image coordinates,
        and removes duplicates from overlap regions with a global NMS. Recovers
        small pedestrians that vanish when a 1080p frame is squashed to imgsz.
        """
        H, W = frame.shape[:2]
        tw, th = W / cols, H / rows
        ox, oy = tw * overlap, th * overlap

        tiles, origins = [], []
        for r in range(rows):
            for c in range(cols):
                x1 = max(0, int(c * tw - ox)); y1 = max(0, int(r * th - oy))
                x2 = min(W, int((c + 1) * tw + ox)); y2 = min(H, int((r + 1) * th + oy))
                tiles.append(frame[y1:y2, x1:x2])
                origins.append((x1, y1))

        results = self.model.predict(
            tiles, conf=self.config.confidence_threshold,
            iou=self.config.nms_threshold, classes=self.config.classes,
            verbose=False, imgsz=self.config.input_size[0],
        )

        dets: List[Detection] = []
        for res, (ox0, oy0) in zip(results, origins):
            if res.boxes is None:
                continue
            b = res.boxes
            for i in range(len(b)):
                x1, y1, x2, y2 = b.xyxy[i].cpu().numpy()
                cls = int(b.cls[i].cpu())
                dets.append(Detection(
                    bbox=np.array([x1 + ox0, y1 + oy0, x2 + ox0, y2 + oy0]),
                    confidence=float(b.conf[i].cpu()),
                    class_id=cls, class_name=self.class_names[cls],
                ))

        return self._nms(dets, nms_iou)

    @staticmethod
    def _nms(dets: List[Detection], iou_thresh: float) -> List[Detection]:
        """Global class-agnostic NMS over a Detection list (merges tile overlaps)."""
        if len(dets) <= 1:
            return dets
        boxes = [[int(d.bbox[0]), int(d.bbox[1]),
                  int(d.bbox[2] - d.bbox[0]), int(d.bbox[3] - d.bbox[1])] for d in dets]
        scores = [float(d.confidence) for d in dets]
        keep = cv2.dnn.NMSBoxes(boxes, scores, 0.0, iou_thresh)
        if len(keep) == 0:
            return []
        idxs = keep.flatten() if hasattr(keep, "flatten") else [int(k) for k in keep]
        return [dets[i] for i in idxs]

    def detect_batch(
        self,
        frames: List[np.ndarray]
    ) -> List[List[Detection]]:
        """
        Detect objects in a batch of frames.

        Args:
            frames: List of BGR images

        Returns:
            List of detection lists (one per frame)
        """
        if len(frames) == 0:
            return []

        # Run batch inference
        results = self.model.predict(
            frames,
            conf=self.config.confidence_threshold,
            iou=self.config.nms_threshold,
            classes=self.config.classes,
            verbose=False,
            imgsz=self.config.input_size[0]
        )

        all_detections = []

        for result in results:
            detections = []

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu())
                    cls_id = int(boxes.cls[i].cpu())

                    detection = Detection(
                        bbox=bbox,
                        confidence=conf,
                        class_id=cls_id,
                        class_name=self.class_names[cls_id]
                    )
                    detections.append(detection)

            all_detections.append(detections)

        return all_detections

    def warmup(self, num_iterations: int = 10):
        """
        Warmup model with dummy inputs.

        Args:
            num_iterations: Number of warmup iterations
        """
        logger.info(f"Warming up detector ({num_iterations} iterations)...")

        dummy_input = np.zeros(
            (self.config.input_size[1], self.config.input_size[0], 3),
            dtype=np.uint8
        )

        for _ in range(num_iterations):
            _ = self.detect(dummy_input)

        logger.info("Detector warmup complete")

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: List[Detection],
        thickness: int = 2,
        font_scale: float = 0.5
    ) -> np.ndarray:
        """
        Draw detections on image.

        Args:
            image: BGR image
            detections: List of detections
            thickness: Box line thickness
            font_scale: Text font scale

        Returns:
            Image with drawn detections
        """
        output = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)

            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Background for text
            cv2.rectangle(
                output,
                (x1, y1 - label_size[1] - 4),
                (x1 + label_size[0], y1),
                color,
                -1
            )

            # Text
            cv2.putText(
                output,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness
            )

        return output


def letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """
    Letterbox image for YOLO26 input.

    Args:
        image: Input image (H, W, 3)
        new_shape: Target shape (width, height)
        color: Padding color

    Returns:
        - Letterboxed image
        - Scale ratio (width_ratio, height_ratio)
        - Padding (dw, dh)
    """
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return image, (r, r), (dw, dh)
