"""Orchestrates the /unique-count pipeline.

Workflow per job (runs on the dedicated GPU worker thread, see job_manager):
  1. Images arrive already validated (format) and spooled to disk.
  2. They are processed in FIFO order, one at a time: decode -> faces (SCRFD)
     -> aligned ArcFace embeddings -> persons (YOLO) -> OSNet embeddings ->
     face<->person association. Pixels are discarded after each image; only
     embeddings and metadata are kept.
  3. Only after *every* image is processed, global identity matching runs
     over the whole dataset and the unique count is computed.

Process-wide state lives in ``UniqueCountRuntime`` (one per server):
ModelManager (models + their load state) and JobManager (queue, workers, jobs).
"""

from __future__ import annotations

import logging
import threading
import time

from detector import PersonDetector
from models.insightface_model import ModelInitializationError
from services.config import UniqueCountConfig, config as default_config
from services.embedding_manager import EmbeddingManager, Observation
from services.face_detection import FaceDetectionError
from services.identity_matching import Identity, IdentityMatcher
from services.job_manager import ImageStatus, ImageTask, Job, JobManager, JobStatus, UniqueCountError
from services.model_manager import ModelManager
from utils.image_processing import ImageValidationError, decode_image

logger = logging.getLogger(__name__)

__all__ = ["UniqueCountRuntime", "UniqueCountError", "ImageTask", "Job", "JobStatus",
           "get_runtime", "start_runtime", "shutdown_runtime"]


def _round_box(box) -> list[float]:
    return [round(float(v), 1) for v in box]


class UniqueCountService:
    """Stateless pipeline logic; all mutable state is in Job / ModelManager."""

    def __init__(self, models: ModelManager, cfg: UniqueCountConfig):
        self.models = models
        self.cfg = cfg
        self._matcher = IdentityMatcher(cfg)

    def run(self, job: Job) -> dict:
        started = time.perf_counter()
        logger.info("[job %s] started: %d images", job.job_id, len(job.tasks))

        job.update(status=JobStatus.LOADING_MODELS)
        try:
            self.models.ensure_ready()
        except ModelInitializationError as exc:
            raise UniqueCountError(503, "MODEL_INITIALIZATION_FAILED", str(exc)) from exc

        job.update(status=JobStatus.PROCESSING)
        store = EmbeddingManager()
        for index, task in enumerate(job.tasks):   # FIFO queue order
            self._process_image(job, task, index, store)

        processed = [t for t in job.tasks if t.status is ImageStatus.PROCESSED]
        if not processed:
            raise UniqueCountError(422, "NO_VALID_IMAGES", "None of the uploaded images could be processed",
                                   {"images": [t.summary() for t in job.tasks]})

        # Global matching only after every image has been processed.
        job.update(status=JobStatus.MATCHING, current_image=None)
        t0 = time.perf_counter()
        result = self._matcher.match(store)
        logger.info("[job %s] matching done in %.0f ms", job.job_id, (time.perf_counter() - t0) * 1000)

        identities = sorted(result.identities, key=lambda i: i.first_seen)
        failed = [t for t in job.tasks if t.status is ImageStatus.FAILED]
        warnings = []
        if self.models.reid_error:
            warnings.append(f"Person Re-ID unavailable, matching used face embeddings only: {self.models.reid_error}")
        if store.total_faces == 0:
            warnings.append("No faces were detected in any processed image")
        if result.unassigned_body_only:
            warnings.append(
                f"{len(result.unassigned_body_only)} person detection(s) without a usable face could not be "
                "matched reliably and are not included in unique_person_count"
                + (" (set UC_COUNT_BODY_ONLY=true to count them)" if not self.cfg.count_body_only else "")
            )

        response = self._format(job, processed, failed, store, identities, result, warnings)
        response["processing_time_ms"] = round((time.perf_counter() - started) * 1000, 1)
        logger.info("[job %s] %d images, %d faces -> %d unique persons (%d uncertain pairs) in %.1fs",
                    job.job_id, len(processed), store.total_faces, len(identities),
                    len(result.uncertain_matches), response["processing_time_ms"] / 1000)
        return response

    def _process_image(self, job: Job, task: ImageTask, image_index: int, store: EmbeddingManager) -> None:
        m = self.models
        job.set_image(task, status=ImageStatus.PROCESSING)
        t0 = time.perf_counter()
        try:
            image, scale = decode_image(task.path.read_bytes(), self.cfg.max_image_side)
        except (ImageValidationError, OSError) as exc:
            code = exc.code if isinstance(exc, ImageValidationError) else "UNREADABLE_FILE"
            msg = exc.message if isinstance(exc, ImageValidationError) else f"Could not read upload: {exc}"
            job.set_image(task, status=ImageStatus.FAILED, error=msg)
            logger.warning("[job %s] %s (%s): %s", job.job_id, task.image_id, task.filename, msg)
            if not job.skip_invalid:
                raise UniqueCountError(422, code, f"{task.filename}: {msg}",
                                       {"image_id": task.image_id, "filename": task.filename}) from exc
            return
        finally:
            task.path.unlink(missing_ok=True)   # upload no longer needed once read
        job.set_image(task, height=image.shape[0], width=image.shape[1])

        try:
            faces = m.face_detector.detect(image)
            face_emb, quality = m.face_recognizer.embed(image, faces)
            persons, body_emb, assoc = [], None, {}
            if m.reid is not None:
                persons, body_emb = m.reid.detect_and_embed(image)
                assoc = m.reid.associate([f.bbox for f in faces], persons)
        except FaceDetectionError as exc:
            job.set_image(task, status=ImageStatus.FAILED, error=str(exc))
            logger.exception("[job %s] face detection failed on %s", job.job_id, task.image_id)
            raise UniqueCountError(500, "FACE_DETECTION_FAILED", str(exc), {"image_id": task.image_id}) from exc
        except Exception as exc:  # noqa: BLE001
            job.set_image(task, status=ImageStatus.FAILED, error=str(exc))
            logger.exception("[job %s] processing failed on %s", job.job_id, task.image_id)
            raise UniqueCountError(500, "PROCESSING_ERROR", f"Unexpected error while processing {task.filename}",
                                   {"image_id": task.image_id, "reason": str(exc)}) from exc
        finally:
            del image  # never keep full-resolution pixels around

        # Store observations (coordinates in original image pixels).
        inv = 1.0 / scale
        used_persons = set(assoc.values())
        for fi, face in enumerate(faces):
            pj = assoc.get(fi)
            p = persons[pj] if pj is not None else None
            store.add(Observation(
                obs_id=-1, image_id=task.image_id, image_index=image_index,
                face_id=f"{task.image_id}_FACE_{fi + 1:03d}",
                face_bbox=_round_box(face.bbox * inv), det_score=round(face.det_score, 4),
                face_quality=round(float(quality[fi]), 4),
                person_bbox=_round_box(p.bbox * inv) if p else None,
                person_confidence=round(p.confidence, 4) if p else None,
                body_reliable=bool(p and p.reliable),
            ), face_emb[fi], body_emb[pj] if p is not None else None)
        for pj, p in enumerate(persons):
            if pj not in used_persons:
                store.add(Observation(
                    obs_id=-1, image_id=task.image_id, image_index=image_index,
                    person_bbox=_round_box(p.bbox * inv), person_confidence=round(p.confidence, 4),
                    body_reliable=p.reliable,
                ), None, body_emb[pj])

        ms = round((time.perf_counter() - t0) * 1000, 1)
        job.set_image(task, status=ImageStatus.PROCESSED, faces_detected=len(faces),
                      persons_detected=len(persons), processing_ms=ms)
        logger.info("[job %s] %s processed: %d faces, %d persons, %d associated (%.0f ms)",
                    job.job_id, task.image_id, len(faces), len(persons), len(assoc), ms)

    # ---------------------------------------------------------- response
    def _format(self, job: Job, processed, failed, store, identities: list[Identity], result, warnings) -> dict:
        person_ids = {id(ident): f"PERSON_{n:03d}" for n, ident in enumerate(identities, 1)}
        obs_person = {o.obs_id: person_ids[id(ident)] for ident in identities for o in ident.observations}

        out_identities = []
        for ident in identities:
            entry = {
                "person_id": person_ids[id(ident)],
                "appearance_count": len(ident.image_ids),
                "image_ids": ident.image_ids,
                "confidence": ident.confidence,
                "match_basis": ident.match_basis,
            }
            if job.include_details:
                entry["cohesion"] = ident.cohesion
                entry["nearest_other_similarity"] = ident.nearest_other
                entry["observations"] = [self._obs_dict(o) for o in ident.observations]
            out_identities.append(entry)

        uncertain = []
        for u in result.uncertain_matches:
            a, b = u["observation_a"], u["observation_b"]
            uncertain.append({
                "person_id_a": obs_person.get(a.obs_id),
                "person_id_b": obs_person.get(b.obs_id),
                "observation_a": {"image_id": a.image_id, "face_id": a.face_id, "bbox": a.face_bbox},
                "observation_b": {"image_id": b.image_id, "face_id": b.face_id, "bbox": b.face_bbox},
                "face_similarity": u["face_similarity"],
                "body_similarity": u["body_similarity"],
                "fused_similarity": u["fused_similarity"],
                "min_face_quality": u["min_face_quality"],
                "reason": u["reason"],
                "decision": "kept_separate",
            })

        with job.lock:
            images = [t.summary() for t in job.tasks]
        return {
            "success": True,
            "job_id": job.job_id,
            "total_images": len(job.tasks),
            "processed_images": len(processed),
            "failed_images": len(failed),
            "total_detected_faces": store.total_faces,
            "total_detected_persons": sum(t.persons_detected for t in job.tasks),
            "unique_person_count": len(identities),
            "processing_status": "completed" if not failed else "completed_with_errors",
            "reid_enabled": self.models.reid is not None,
            "devices": self.models.status()["devices"],
            "identities": out_identities,
            "uncertain_matches": uncertain,
            "unmatched_body_detections": len(result.unassigned_body_only),
            "images": images,
            "thresholds": {
                "face_strong": self.cfg.face_strong_threshold,
                "face_weak": self.cfg.face_weak_threshold,
                "cluster": self.cfg.cluster_threshold,
                "member_min": self.cfg.member_min_similarity,
                "body_only_attach": self.cfg.body_only_attach_threshold,
            },
            "warnings": warnings,
        }

    @staticmethod
    def _obs_dict(o: Observation) -> dict:
        return {
            "image_id": o.image_id,
            "face_id": o.face_id,
            "face_bbox": o.face_bbox,
            "det_score": o.det_score,
            "face_quality": o.face_quality if o.has_face else None,
            "person_bbox": o.person_bbox,
            "evidence": "face" if o.has_face else "body_reid",
        }


# ----------------------------------------------------------------------------
# Process-wide runtime
# ----------------------------------------------------------------------------

class UniqueCountRuntime:
    def __init__(self, person_detector: PersonDetector | None, cfg: UniqueCountConfig = default_config):
        self.cfg = cfg
        self.models = ModelManager(cfg, person_detector)
        self.service = UniqueCountService(self.models, cfg)
        self.jobs = JobManager(cfg, self.service.run)

    def preload(self) -> None:
        """Load + warm up models on the GPU worker thread without blocking startup."""
        def _load():
            try:
                self.models.ensure_ready()
            except ModelInitializationError:
                pass  # state is FAILED and visible via /unique-count/status; retried on next job
        self.jobs.submit_callable(_load)

    def status(self) -> dict:
        return {"models": self.models.status(), "queue": self.jobs.stats()}


_runtime: UniqueCountRuntime | None = None
_runtime_lock = threading.Lock()


def get_runtime(person_detector: PersonDetector | None = None) -> UniqueCountRuntime:
    """Return the singleton runtime, creating it lazily if startup did not."""
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = UniqueCountRuntime(person_detector)
    return _runtime


def start_runtime(person_detector: PersonDetector | None) -> UniqueCountRuntime:
    rt = get_runtime(person_detector)
    if rt.cfg.preload_models:
        logger.info("Preloading /unique-count models in the background")
        rt.preload()
    return rt


def shutdown_runtime() -> None:
    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            _runtime.jobs.shutdown()
            _runtime = None
