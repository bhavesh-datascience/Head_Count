"""In-memory store of every observation (face and/or body) for one job.

Each detected face becomes one observation; a person box with no associated
face becomes a body-only observation. Embeddings are kept in contiguous
float32 NumPy arrays (one row per observation) for vectorised similarity.
Images themselves are never retained - only embeddings and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Observation:
    obs_id: int
    image_id: str
    image_index: int
    face_id: str | None = None
    face_bbox: list[float] | None = None
    det_score: float | None = None
    face_quality: float = 0.0
    face_row: int | None = None            # row in EmbeddingManager.face_matrix
    person_bbox: list[float] | None = None
    person_confidence: float | None = None
    body_row: int | None = None            # row in EmbeddingManager.body_matrix
    body_reliable: bool = False

    @property
    def has_face(self) -> bool:
        return self.face_row is not None


@dataclass
class EmbeddingManager:
    observations: list[Observation] = field(default_factory=list)
    _faces: list[np.ndarray] = field(default_factory=list)
    _bodies: list[np.ndarray] = field(default_factory=list)
    _face_matrix: np.ndarray | None = None
    _body_matrix: np.ndarray | None = None

    def add(self, obs: Observation, face_emb: np.ndarray | None, body_emb: np.ndarray | None) -> Observation:
        obs.obs_id = len(self.observations)
        if face_emb is not None:
            obs.face_row = len(self._faces)
            self._faces.append(np.asarray(face_emb, np.float32))
        if body_emb is not None:
            obs.body_row = len(self._bodies)
            self._bodies.append(np.asarray(body_emb, np.float32))
        self.observations.append(obs)
        self._face_matrix = self._body_matrix = None  # invalidate cache
        return obs

    @property
    def face_matrix(self) -> np.ndarray:
        if self._face_matrix is None:
            self._face_matrix = np.stack(self._faces) if self._faces else np.zeros((0, 512), np.float32)
        return self._face_matrix

    @property
    def body_matrix(self) -> np.ndarray:
        if self._body_matrix is None:
            self._body_matrix = np.stack(self._bodies) if self._bodies else np.zeros((0, 512), np.float32)
        return self._body_matrix

    @property
    def face_observations(self) -> list[Observation]:
        return [o for o in self.observations if o.has_face]

    @property
    def body_only_observations(self) -> list[Observation]:
        return [o for o in self.observations if not o.has_face and o.body_row is not None]

    @property
    def total_faces(self) -> int:
        return len(self._faces)
