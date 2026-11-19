"""Face alignment + ArcFace embedding extraction and face quality scoring."""

from __future__ import annotations

import cv2
import numpy as np
from insightface.utils import face_align

from models.insightface_model import InsightFaceModel
from services.config import UniqueCountConfig
from services.face_detection import FaceDetection
from utils.similarity import l2_normalize


class FaceRecognizer:
    def __init__(self, model: InsightFaceModel, cfg: UniqueCountConfig, batch_size: int = 32):
        self.rec = model.recognizer
        self.cfg = cfg
        self.batch_size = batch_size
        self.input_size = int(self.rec.input_size[0])  # 112 for ArcFace

    def align(self, image: np.ndarray, face: FaceDetection) -> np.ndarray:
        """Similarity-transform the face onto the canonical ArcFace 5-point template."""
        return face_align.norm_crop(image, landmark=face.kps, image_size=self.input_size)

    def embed(self, image: np.ndarray, faces: list[FaceDetection]) -> tuple[np.ndarray, np.ndarray]:
        """Return (embeddings (N, 512) L2-normalised, quality scores (N,))."""
        if not faces:
            return np.zeros((0, 512), np.float32), np.zeros((0,), np.float32)
        aligned = [self.align(image, f) for f in faces]
        feats = [
            np.asarray(self.rec.get_feat(aligned[i:i + self.batch_size]), dtype=np.float32)
            for i in range(0, len(aligned), self.batch_size)
        ]
        emb = l2_normalize(np.concatenate(feats).reshape(len(faces), -1))
        quality = np.array([self.quality(f, a) for f, a in zip(faces, aligned)], dtype=np.float32)
        return emb, quality

    def quality(self, face: FaceDetection, aligned: np.ndarray) -> float:
        """Heuristic 0..1 face quality: size, detector confidence, frontalness, sharpness.

        Low-quality faces (tiny, profile, blurred) give compressed similarity
        scores, so the matcher treats their face evidence as less decisive.
        """
        x1, y1, x2, y2 = face.bbox
        min_side = float(min(x2 - x1, y2 - y1))
        size_s = np.clip((min_side - self.cfg.min_face_size) / max(1.0, 112.0 - self.cfg.min_face_size), 0, 1)
        det_s = np.clip((face.det_score - 0.5) / 0.4, 0, 1)

        # Yaw proxy: horizontal offset of the nose from the eye midpoint,
        # relative to the inter-ocular distance (0 = frontal, ~0.5+ = profile).
        le, re, nose = face.kps[0], face.kps[1], face.kps[2]
        eye_dist = float(np.linalg.norm(re - le)) + 1e-6
        yaw = abs(float(nose[0] - (le[0] + re[0]) / 2.0)) / eye_dist
        frontal_s = np.clip(1.0 - yaw / 0.6, 0, 1)

        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        sharp_s = np.clip(cv2.Laplacian(gray, cv2.CV_64F).var() / 150.0, 0, 1)

        return float(0.30 * size_s + 0.25 * det_s + 0.30 * frontal_s + 0.15 * sharp_s)
