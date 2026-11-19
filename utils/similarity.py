"""Vector similarity and box geometry helpers."""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Cosine similarity between rows of a and rows of b (both get normalised)."""
    a = l2_normalize(a)
    b = a if b is None else l2_normalize(b)
    return np.clip(a @ b.T, -1.0, 1.0).astype(np.float32)


def box_area(box: np.ndarray | list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def intersection(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    w = min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))
    h = min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1]))
    return max(0.0, w) * max(0.0, h)


def containment(inner: np.ndarray | list[float], outer: np.ndarray | list[float]) -> float:
    """Fraction of `inner`'s area that lies inside `outer` (0..1)."""
    area = box_area(inner)
    return intersection(inner, outer) / area if area > 0 else 0.0
