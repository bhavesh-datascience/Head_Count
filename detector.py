"""Person detection with overlap-aware counting.

Strategy:
  1. Full-image YOLO pass at high resolution (catches large / near people).
  2. Sliced pass over overlapping tiles (catches small / distant people in crowds).
  3. Merge both sets: standard IoU-NMS with a fairly high threshold so that
     people standing close together / partially occluding each other are kept
     as separate detections, plus an "intersection over smaller box" check to
     remove duplicates where a tile only saw part of a person.
"""

import os
import threading
from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLO

PERSON_CLASS = 0
_HAS_CUDA = torch.cuda.is_available()


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class DetectorConfig:
    model_path: str = os.getenv("PERSON_MODEL", "yolo11m.pt")
    conf: float = 0.30          # minimum confidence for a person
    nms_iou: float = 0.60       # high => overlapping people are not merged
    merge_iou: float = 0.55     # IoU for merging full-image + tile detections
    merge_ios: float = 0.85     # intersection / smaller-box area for dedup
    imgsz: int = int(os.getenv("PERSON_IMGSZ", "1280"))
    tile_size: int = 640
    tile_overlap: float = 0.25
    use_tiles: bool = _env_bool("PERSON_TILES", True)
    # Tiles sent to the model per forward pass. A 12 MP photo has ~48 tiles;
    # running them in one batch needed ~6 GB, so they go in small chunks
    # (results are identical, peak memory is bounded).
    tile_batch: int = int(os.getenv("PERSON_TILE_BATCH", "16" if _HAS_CUDA else "2"))


def _iou_and_ios(box: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    iou = inter / (area + areas - inter + 1e-9)
    ios = inter / (np.minimum(area, areas) + 1e-9)
    return iou, ios


def merge_detections(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, ios_thr: float) -> np.ndarray:
    """Greedy NMS that also suppresses boxes mostly contained in a stronger box."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        iou, ios = _iou_and_ios(boxes[i], boxes[order[1:]])
        # Contained-box suppression only applies to similarly sized boxes; a small
        # person standing in front of a larger one should not be removed.
        rest = boxes[order[1:]]
        area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
        areas = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        ratio = np.minimum(area_i, areas) / (np.maximum(area_i, areas) + 1e-9)
        dup = (iou > iou_thr) | ((ios > ios_thr) & (ratio > 0.35))
        order = order[1:][~dup]
    return np.array(keep, dtype=int)


def _tiles(w: int, h: int, size: int, overlap: float):
    step = max(1, int(size * (1 - overlap)))
    xs = list(range(0, max(w - size, 0) + 1, step))
    ys = list(range(0, max(h - size, 0) + 1, step))
    if xs[-1] + size < w:
        xs.append(w - size)
    if ys[-1] + size < h:
        ys.append(h - size)
    for y in ys:
        for x in xs:
            yield x, y, min(x + size, w), min(y + size, h)


class PersonDetector:
    def __init__(self, config: DetectorConfig | None = None):
        self.cfg = config or DetectorConfig()
        self.device = 0 if _HAS_CUDA else "cpu"
        self.model = YOLO(self.cfg.model_path)
        # The Ultralytics predictor keeps per-call state and is not safe to use
        # from several threads at once; /count, /number-count and /unique-count
        # share this instance, so calls are serialised.
        self._lock = threading.Lock()

    def _predict(self, images, imgsz: int):
        return self.model.predict(
            images,
            classes=[PERSON_CLASS],
            conf=self.cfg.conf,
            iou=self.cfg.nms_iou,
            imgsz=imgsz,
            max_det=1000,
            device=self.device,
            verbose=False,
        )

    def detect(self, image: np.ndarray, use_tiles: bool | None = None) -> list[dict]:
        """image: BGR numpy array. Returns list of {box: [x1,y1,x2,y2], confidence}. Thread-safe."""
        with self._lock:
            return self._detect(image, use_tiles)

    def _detect(self, image: np.ndarray, use_tiles: bool | None = None) -> list[dict]:
        use_tiles = self.cfg.use_tiles if use_tiles is None else use_tiles
        h, w = image.shape[:2]
        all_boxes, all_scores = [], []

        res = self._predict(image, self.cfg.imgsz)[0]
        all_boxes.append(res.boxes.xyxy.cpu().numpy())
        all_scores.append(res.boxes.conf.cpu().numpy())

        # Tiling only helps when the image is noticeably larger than a tile.
        if use_tiles and max(w, h) > self.cfg.tile_size * 1.5:
            tiles = list(_tiles(w, h, self.cfg.tile_size, self.cfg.tile_overlap))
            crops = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in tiles]
            step = max(1, self.cfg.tile_batch)
            results = []
            for i in range(0, len(crops), step):
                results.extend(self._predict(crops[i:i + step], self.cfg.tile_size))
            for (x1, y1, x2, y2), r in zip(tiles, results):
                b = r.boxes.xyxy.cpu().numpy()
                s = r.boxes.conf.cpu().numpy()
                if not len(b):
                    continue
                # Drop boxes cut by an inner tile edge; the full-image pass or
                # a neighbouring tile sees that person whole.
                m = 2
                cut = (
                    ((b[:, 0] <= m) & (x1 > 0))
                    | ((b[:, 1] <= m) & (y1 > 0))
                    | ((b[:, 2] >= (x2 - x1) - m) & (x2 < w))
                    | ((b[:, 3] >= (y2 - y1) - m) & (y2 < h))
                )
                b, s = b[~cut], s[~cut]
                b[:, [0, 2]] += x1
                b[:, [1, 3]] += y1
                all_boxes.append(b)
                all_scores.append(s)

        boxes = np.concatenate(all_boxes) if all_boxes else np.zeros((0, 4))
        scores = np.concatenate(all_scores) if all_scores else np.zeros((0,))
        if not len(boxes):
            return []

        keep = merge_detections(boxes, scores, self.cfg.merge_iou, self.cfg.merge_ios)
        return [
            {"box": [round(float(v), 1) for v in boxes[i]], "confidence": round(float(scores[i]), 3)}
            for i in keep
        ]
