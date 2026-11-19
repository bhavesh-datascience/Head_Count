"""Model lifecycle for /unique-count: load once, warm up, report state.

State machine::

    NOT_LOADED --load()--> LOADING --ok--> READY
                                   \\--error--> FAILED --load() again--> LOADING ...

* Loading is idempotent and thread-safe: concurrent callers wait on the same
  load instead of loading twice.
* A FAILED load is retried on the next request (e.g. a transient download
  error), so the server never needs a restart to recover.
* InsightFace is mandatory; OSNet Re-ID is optional - if it fails the
  pipeline continues on faces only and reports why.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum

import numpy as np

from detector import PersonDetector
from models.gpu import gpu_info, release_gpu_memory
from models.insightface_model import InsightFaceModel, ModelInitializationError
from models.osnet_model import OSNetModel
from services.config import UniqueCountConfig
from services.face_detection import FaceDetector
from services.face_recognition import FaceRecognizer
from services.person_reid import PersonReIdentifier

logger = logging.getLogger(__name__)


class ModelState(str, Enum):
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class ModelManager:
    def __init__(self, cfg: UniqueCountConfig, person_detector: PersonDetector | None):
        self.cfg = cfg
        self.person_detector = person_detector
        self._lock = threading.Lock()
        self._loaded = threading.Condition(self._lock)
        self.state = ModelState.NOT_LOADED
        self.error: str | None = None
        self.load_seconds: float | None = None

        self.face_model: InsightFaceModel | None = None
        self.face_detector: FaceDetector | None = None
        self.face_recognizer: FaceRecognizer | None = None
        self.reid: PersonReIdentifier | None = None
        self.osnet: OSNetModel | None = None
        self.reid_error: str | None = None

    # ------------------------------------------------------------------
    def ensure_ready(self) -> None:
        """Block until models are READY; raise ModelInitializationError on failure."""
        with self._lock:
            while self.state is ModelState.LOADING:
                self._loaded.wait()
            if self.state is ModelState.READY:
                return
            self.state, self.error = ModelState.LOADING, None
        # Load outside the lock so status() stays responsive meanwhile.
        try:
            self._load()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            logger.exception("Model loading failed")
            with self._lock:
                self.state, self.error = ModelState.FAILED, msg
                self._loaded.notify_all()
            if isinstance(exc, ModelInitializationError):
                raise
            raise ModelInitializationError(msg) from exc
        with self._lock:
            self.state = ModelState.READY
            self._loaded.notify_all()

    def _load(self) -> None:
        t0 = time.perf_counter()
        face_model = InsightFaceModel(self.cfg)
        face_model.warmup()

        reid, osnet, reid_error = None, None, None
        if not self.cfg.reid_enabled:
            reid_error = "disabled by configuration (UC_REID_ENABLED=false)"
        elif self.person_detector is None:
            reid_error = "person detector is not initialised"
        else:
            try:
                osnet = OSNetModel(self.cfg)
                osnet.warmup()
                reid = PersonReIdentifier(self.person_detector, osnet, self.cfg)
                # First YOLO call on GPU builds kernels for the full-image and
                # tile input shapes; do it now instead of on the first request.
                self.person_detector.detect(np.zeros((1080, 1920, 3), np.uint8), self.cfg.use_person_tiles)
            except ModelInitializationError as exc:
                reid_error = str(exc)
        if reid_error:
            logger.warning("Person Re-ID unavailable, matching on faces only: %s", reid_error)

        self.face_model, self.osnet, self.reid, self.reid_error = face_model, osnet, reid, reid_error
        self.face_detector = FaceDetector(face_model, self.cfg)
        self.face_recognizer = FaceRecognizer(face_model, self.cfg)
        release_gpu_memory()
        self.load_seconds = round(time.perf_counter() - t0, 2)
        logger.info("All /unique-count models ready in %.1fs", self.load_seconds)

    # ------------------------------------------------------------------
    def status(self) -> dict:
        with self._lock:
            state, error = self.state, self.error
        devices = {}
        if self.face_model is not None:
            devices["face_detector"] = self.face_model.detector_device
            devices["face_recognizer"] = self.face_model.recognizer_device
        if self.osnet is not None:
            devices["osnet"] = str(self.osnet.device) + (" (fp16)" if self.osnet.half else "")
        if self.person_detector is not None:
            devices["person_detector"] = f"cuda:{self.person_detector.device}" \
                if isinstance(self.person_detector.device, int) else str(self.person_detector.device)
        return {
            "state": state.value,
            "error": error,
            "load_seconds": self.load_seconds,
            "reid_enabled": self.reid is not None,
            "reid_error": self.reid_error,
            "devices": devices,
            "gpu": gpu_info(self.cfg.gpu_device_id),
        }
