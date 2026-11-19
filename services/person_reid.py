"""Person detection (existing YOLO detector) + OSNet body embeddings + face association."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from detector import PersonDetector
from models.osnet_model import OSNetModel
from services.config import UniqueCountConfig
from utils.image_processing import crop
from utils.similarity import box_area, containment, intersection

logger = logging.getLogger(__name__)


@dataclass
class PersonDetection:
    bbox: np.ndarray            # (4,) x1, y1, x2, y2
    confidence: float
    # False when the box overlaps another person so much that the crop mixes
    # two people's appearance; such embeddings are not used for matching.
    reliable: bool = True


class PersonReIdentifier:
    def __init__(self, person_detector: PersonDetector, osnet: OSNetModel, cfg: UniqueCountConfig):
        self.person_detector = person_detector
        self.osnet = osnet
        self.cfg = cfg

    def detect_and_embed(self, image: np.ndarray) -> tuple[list[PersonDetection], np.ndarray]:
        """Detect person boxes with the shared YOLO model and embed each crop with OSNet."""
        raw = self.person_detector.detect(image, self.cfg.use_person_tiles)
        persons: list[PersonDetection] = []
        crops: list[np.ndarray] = []
        for d in raw:
            box = np.asarray(d["box"], dtype=np.float32)
            if box[3] - box[1] < self.cfg.min_person_height:
                continue
            c = crop(image, box)
            if c is None:
                continue
            persons.append(PersonDetection(bbox=box, confidence=float(d["confidence"])))
            crops.append(c)

        # Mark heavily overlapped boxes as unreliable (crowded group photos).
        for i, p in enumerate(persons):
            for j, q in enumerate(persons):
                if i != j and intersection(p.bbox, q.bbox) > 0.5 * box_area(p.bbox):
                    p.reliable = False
                    break

        emb = self.osnet.embed(crops, self.cfg.reid_batch_size)
        return persons, emb

    @staticmethod
    def associate(face_boxes: list[np.ndarray], persons: list[PersonDetection]) -> dict[int, int]:
        """One-to-one face -> person assignment using spatial geometry.

        A face belongs to a person box when the face lies (mostly) inside the
        box, sits in its upper part, is horizontally near its centre and has a
        plausible width relative to the body. Hungarian matching resolves
        conflicts when boxes overlap.
        """
        if not face_boxes or not persons:
            return {}
        score = np.zeros((len(face_boxes), len(persons)), dtype=np.float32)
        for i, f in enumerate(face_boxes):
            fw = float(f[2] - f[0])
            cx, cy = (f[0] + f[2]) / 2.0, (f[1] + f[3]) / 2.0
            for j, p in enumerate(persons):
                b = p.bbox
                pw, ph = float(b[2] - b[0]), float(b[3] - b[1])
                c = containment(f, b)
                rel_y = (cy - b[1]) / max(ph, 1.0)
                rel_dx = abs(cx - (b[0] + b[2]) / 2.0) / max(pw, 1.0)
                width_ratio = fw / max(pw, 1.0)
                if c < 0.6 or not (0.0 <= rel_y <= 0.55) or rel_dx > 0.35 or not (0.08 <= width_ratio <= 0.9):
                    continue
                score[i, j] = c * (1.0 - rel_y) * (1.0 - rel_dx)

        rows, cols = linear_sum_assignment(-score)
        return {int(r): int(c) for r, c in zip(rows, cols) if score[r, c] > 0}
