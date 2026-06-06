"""ReID (Re-Identification) feature extractor."""

import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2
from typing import List, Optional
from pathlib import Path
import logging

from ..utils.config import ReIDConfig, resolve_device

logger = logging.getLogger(__name__)


class ReIDExtractor:
    """ReID feature extractor using OSNet."""

    def __init__(self, config: ReIDConfig):
        """
        Initialize ReID extractor.

        Args:
            config: ReID configuration

        Raises:
            RuntimeError: If model loading fails
        """
        self.config = config
        self.device = torch.device(resolve_device(config.device))

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Image transforms
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),  # OSNet input size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        logger.info(f"Loaded {config.model_name} ReID model on {config.device}")

    def _load_model(self) -> nn.Module:
        """Load OSNet model."""
        self.is_dummy = False
        try:
            import torchreid
        except ImportError:
            self.is_dummy = True
            logger.warning(
                "=" * 70 + "\n"
                "  torchreid NOT installed -> using DUMMY ReID (random embeddings).\n"
                "  Cross-camera association will be MEANINGLESS in this mode.\n"
                "  For real cross-camera IDs: pip install torchreid  (see FEAT-8)\n"
                + "=" * 70
            )
            return self._create_dummy_model()

        # Build model
        model = torchreid.models.build_model(
            name=self.config.model_name,
            num_classes=1000,  # Not used for feature extraction
            loss='softmax',
            pretrained=True
        )

        # Load custom weights if provided
        if self.config.model_path is not None:
            model_file = Path(self.config.model_path)
            if model_file.exists():
                checkpoint = torch.load(model_file, map_location=self.device)
                model.load_state_dict(checkpoint['state_dict'])
                logger.info(f"Loaded custom weights from {self.config.model_path}")

        model = model.to(self.device)
        return model

    def _create_dummy_model(self) -> nn.Module:
        """Create dummy model for testing without torchreid."""
        class DummyReID(nn.Module):
            def __init__(self, embedding_dim=512):
                super().__init__()
                self.embedding_dim = embedding_dim

            def forward(self, x):
                batch_size = x.size(0)
                # Return random embeddings
                return torch.randn(batch_size, self.embedding_dim)

        return DummyReID(self.config.embedding_dim).to(self.device)

    @torch.no_grad()
    def extract(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Extract ReID feature from single detection.

        Args:
            image: Full frame (BGR)
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Normalized feature vector (embedding_dim,)
        """
        # Crop image
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            # Invalid bbox, return zero vector
            return np.zeros(self.config.embedding_dim)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return np.zeros(self.config.embedding_dim)

        # Convert BGR to RGB
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Transform
        crop_tensor = self.transform(crop).unsqueeze(0).to(self.device)

        # Extract feature
        feature = self.model(crop_tensor)

        # Normalize
        feature = feature.cpu().numpy()[0]
        feature = feature / (np.linalg.norm(feature) + 1e-8)

        return feature

    @torch.no_grad()
    def extract_batch(
        self,
        image: np.ndarray,
        bboxes: List[np.ndarray]
    ) -> np.ndarray:
        """
        Extract ReID features from batch of detections.

        Args:
            image: Full frame (BGR)
            bboxes: List of bounding boxes

        Returns:
            Feature matrix (N, embedding_dim)
        """
        if len(bboxes) == 0:
            return np.zeros((0, self.config.embedding_dim))

        # Crop all boxes
        crops = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

            if x2 > x1 and y2 > y1:
                crop = image[y1:y2, x1:x2]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crops.append(crop)
            else:
                # Invalid crop, use black image
                crops.append(np.zeros((128, 64, 3), dtype=np.uint8))

        # Transform batch
        crop_tensors = torch.stack([self.transform(crop) for crop in crops])
        crop_tensors = crop_tensors.to(self.device)

        # Extract features
        features = self.model(crop_tensors)

        # Normalize
        features = features.cpu().numpy()
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

        return features

    @staticmethod
    def cosine_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compute cosine distance between two features.

        Args:
            feat1: Feature vector 1
            feat2: Feature vector 2

        Returns:
            Cosine distance (0 = identical, 2 = opposite)
        """
        similarity = np.dot(feat1, feat2) / (
            np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8
        )
        return 1.0 - similarity

    @staticmethod
    def euclidean_distance(feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two features.

        Args:
            feat1: Feature vector 1
            feat2: Feature vector 2

        Returns:
            Euclidean distance
        """
        return np.linalg.norm(feat1 - feat2)
