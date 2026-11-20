"""InsightFace (buffalo_l) model wrapper.

Only the detection (SCRFD) and recognition (ArcFace) modules are loaded;
landmark/gender-age models are not needed for identity matching. Runs on the
CUDA execution provider of onnxruntime-gpu when available, otherwise CPU.

Lifecycle (load once, reuse, thread safety) is owned by
``services.model_manager.ModelManager``.
"""

from __future__ import annotations

import logging

import numpy as np

from models.gpu import ort_providers, session_device
from services.config import UniqueCountConfig

logger = logging.getLogger(__name__)


class ModelInitializationError(RuntimeError):
    """Raised when a model cannot be loaded."""


class InsightFaceModel:
    def __init__(self, cfg: UniqueCountConfig):
        from insightface.app import FaceAnalysis  # heavy import, keep lazy

        providers, ctx_id, sess_options = ort_providers(cfg)
        names = [p[0] if isinstance(p, tuple) else p for p in providers]
        logger.info("Loading InsightFace '%s' (providers=%s)", cfg.face_model_name, names)
        try:
            self.app = FaceAnalysis(
                name=cfg.face_model_name,
                root=cfg.face_model_root,
                allowed_modules=["detection", "recognition"],
                providers=providers,
                **({"sess_options": sess_options} if sess_options is not None else {}),
            )
            self.app.prepare(ctx_id=ctx_id, det_thresh=cfg.det_thresh, det_size=cfg.det_sizes)
        except Exception as exc:  # noqa: BLE001 - surface any loader failure uniformly
            raise ModelInitializationError(f"InsightFace initialisation failed: {exc}") from exc

        if "recognition" not in self.app.models:
            raise ModelInitializationError(f"Model pack '{cfg.face_model_name}' has no recognition model")
        self.detector = self.app.det_model
        self.recognizer = self.app.models["recognition"]

        # Verify where the sessions really run: ORT silently falls back to
        # CPU when CUDA/cuDNN libraries cannot be loaded.
        self.detector_device = session_device(getattr(self.detector, "session", None))
        self.recognizer_device = session_device(getattr(self.recognizer, "session", None))
        if cfg.require_gpu and "cpu" in (self.detector_device, self.recognizer_device):
            raise ModelInitializationError(
                "UC_REQUIRE_GPU is set but InsightFace is running on CPU "
                f"(detector={self.detector_device}, recognizer={self.recognizer_device}); "
                "check that onnxruntime-gpu and the CUDA 13 / cuDNN 9 libraries are installed"
            )
        logger.info("InsightFace ready (detector=%s, recognizer=%s, det sizes=%s, det_thresh=%.2f)",
                    self.detector_device, self.recognizer_device, cfg.det_sizes, cfg.det_thresh)

    def warmup(self) -> None:
        """Run once at the largest detector input size so kernels/arenas are initialised."""
        h, w = max(s[1] for s in self.app.det_size), max(s[0] for s in self.app.det_size)
        self.detector.detect(np.zeros((h, w, 3), np.uint8), max_num=0)
        self.recognizer.get_feat([np.zeros((112, 112, 3), np.uint8)] * 4)
