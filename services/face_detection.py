"""Face detection with InsightFace SCRFD (buffalo_l det_10g)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from models.insightface_model import InsightFaceModel
from services.config import UniqueCountConfig

logger = logging.getLogger(__name__)


class FaceDetectionError(RuntimeError):
    """Raised when the face detector itself fails (not when it finds no faces)."""


@dataclass
class FaceDetection:
    bbox: np.ndarray        # (4,) x1, y1, x2, y2 in image pixels
    kps: np.ndarray         # (5, 2) eyes, nose, mouth corners - used for alignment
    det_score: float


class FaceDetector:
    def __init__(self, model: InsightFaceModel, cfg: UniqueCountConfig):
        self.model = model
        self.cfg = cfg

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """Detect all faces in a BGR image and drop unreliable detections."""
        try:
            bboxes, kpss = self.model.detector.detect(image, max_num=0)
        except Exception as exc:  # noqa: BLE001
            raise FaceDetectionError(f"Face detector failed: {exc}") from exc

        faces: list[FaceDetection] = []
        dropped = 0
        for i in range(bboxes.shape[0]):
            score = float(bboxes[i, 4])
            box = bboxes[i, :4].astype(np.float32)
            min_side = min(box[2] - box[0], box[3] - box[1])
            # Landmarks are mandatory: without them alignment (and therefore
            # the ArcFace embedding) is meaningless.
            if score < self.cfg.det_thresh or min_side < self.cfg.min_face_size or kpss is None:
                dropped += 1
                continue
            faces.append(FaceDetection(bbox=box, kps=kpss[i].astype(np.float32), det_score=score))
        if dropped:
            logger.debug("Dropped %d low-confidence/too-small faces", dropped)
        return faces
