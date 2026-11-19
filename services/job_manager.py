"""Job queue, worker threads and job state for /unique-count.

Threading model
---------------
* All model work runs on a dedicated ``ThreadPoolExecutor`` whose size is
  ``UC_MAX_CONCURRENT_JOBS`` (default 1). The event loop never runs model
  code, and FastAPI's shared threadpool is not used for GPU work, so the
  /count routes stay responsive while a large job runs.
* The queue is bounded (``UC_MAX_PENDING_JOBS``); beyond it new submissions
  are rejected with HTTP 429 instead of piling up unbounded GPU work.
* Every ``Job`` has its own lock; workers mutate state through ``Job.update``
  / ``Job.set_image`` and readers take consistent ``snapshot()`` copies.

Uploads are spooled to a per-job temp directory so a job can outlive the
HTTP request (async mode) and images are read one at a time from disk. The
directory is deleted when the job finishes.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from models.gpu import release_gpu_memory
from services.config import UniqueCountConfig

logger = logging.getLogger(__name__)


class UniqueCountError(Exception):
    """Error that maps directly onto an HTTP error response."""

    def __init__(self, status_code: int, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


class JobStatus(str, Enum):
    QUEUED = "queued"
    LOADING_MODELS = "loading_models"
    PROCESSING = "processing"
    MATCHING = "matching"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def finished(self) -> bool:
        return self in (JobStatus.COMPLETED, JobStatus.FAILED)


class ImageStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


@dataclass
class ImageTask:
    image_id: str
    filename: str
    path: Path                     # spooled upload on disk, read only when processed
    status: ImageStatus = ImageStatus.QUEUED
    faces_detected: int = 0
    persons_detected: int = 0
    width: int | None = None
    height: int | None = None
    error: str | None = None
    processing_ms: float | None = None

    def summary(self) -> dict:
        return {
            "image_id": self.image_id,
            "filename": self.filename,
            "status": self.status.value,
            "faces_detected": self.faces_detected,
            "persons_detected": self.persons_detected,
            "width": self.width,
            "height": self.height,
            "processing_ms": self.processing_ms,
            "error": self.error,
        }


@dataclass
class Job:
    job_id: str
    tasks: list[ImageTask]
    workdir: Path
    skip_invalid: bool = False
    include_details: bool = False
    status: JobStatus = JobStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    current_image: str | None = None
    result: dict | None = None
    error: UniqueCountError | None = None
    future: Future | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **fields) -> None:
        with self.lock:
            for k, v in fields.items():
                setattr(self, k, v)

    def set_image(self, task: ImageTask, **fields) -> None:
        with self.lock:
            for k, v in fields.items():
                setattr(task, k, v)
            if fields.get("status") is ImageStatus.PROCESSING:
                self.current_image = task.image_id

    def snapshot(self, include_result: bool = True) -> dict:
        with self.lock:
            processed = sum(t.status is ImageStatus.PROCESSED for t in self.tasks)
            failed = sum(t.status is ImageStatus.FAILED for t in self.tasks)
            now = self.finished_at or time.time()
            snap = {
                "job_id": self.job_id,
                "status": self.status.value,
                "progress": {
                    "total_images": len(self.tasks),
                    "processed_images": processed,
                    "failed_images": failed,
                    "current_image": self.current_image if not self.status.finished else None,
                    "percent": round(100.0 * (processed + failed) / max(1, len(self.tasks)), 1),
                },
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "queued_seconds": round((self.started_at or now) - self.created_at, 2),
                "elapsed_seconds": round(now - self.started_at, 2) if self.started_at else None,
                "images": [t.summary() for t in self.tasks],
            }
            if self.error is not None:
                snap["error"] = self.error.to_dict()
            if include_result and self.result is not None:
                snap["result"] = self.result
            return snap


class JobManager:
    def __init__(self, cfg: UniqueCountConfig, runner: Callable[[Job], dict]):
        self.cfg = cfg
        self._runner = runner
        self._executor = ThreadPoolExecutor(max_workers=max(1, cfg.max_concurrent_jobs),
                                            thread_name_prefix="unique-count-gpu")
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.RLock()
        self._closed = False
        self._root = Path(cfg.upload_dir) if cfg.upload_dir else Path(tempfile.gettempdir()) / "unique_count_jobs"
        self._root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- creation
    def new_workdir(self) -> tuple[str, Path]:
        job_id = uuid.uuid4().hex[:12]
        workdir = self._root / job_id
        workdir.mkdir(parents=True, exist_ok=False)
        return job_id, workdir

    def submit(self, job: Job) -> Job:
        with self._lock:
            if self._closed:
                raise UniqueCountError(503, "SHUTTING_DOWN", "Server is shutting down")
            self._prune()
            pending = sum(1 for j in self._jobs.values() if j.status is JobStatus.QUEUED)
            if pending >= self.cfg.max_pending_jobs:
                raise UniqueCountError(429, "QUEUE_FULL",
                                       f"{pending} jobs already waiting; retry later",
                                       {"max_pending_jobs": self.cfg.max_pending_jobs})
            self._jobs[job.job_id] = job
            job.future = self._executor.submit(self._execute, job)
        logger.info("[job %s] queued (%d images, %d jobs waiting)", job.job_id, len(job.tasks), pending)
        return job

    def submit_callable(self, fn: Callable[[], object]) -> Future:
        """Run arbitrary model work (e.g. warm-up) on the GPU worker thread."""
        return self._executor.submit(fn)

    # ---------------------------------------------------------- execution
    def _execute(self, job: Job) -> None:
        job.update(status=JobStatus.LOADING_MODELS, started_at=time.time())
        try:
            result = self._runner(job)
            job.update(result=result, status=JobStatus.COMPLETED)
        except UniqueCountError as exc:
            job.update(error=exc, status=JobStatus.FAILED)
        except Exception as exc:  # noqa: BLE001 - never let a worker thread die silently
            logger.exception("[job %s] unexpected failure", job.job_id)
            job.update(error=UniqueCountError(500, "INTERNAL_ERROR", "Unexpected error while processing images",
                                              {"reason": str(exc)}),
                       status=JobStatus.FAILED)
        finally:
            job.update(finished_at=time.time(), current_image=None)
            shutil.rmtree(job.workdir, ignore_errors=True)
            release_gpu_memory()
            logger.info("[job %s] %s in %.1fs", job.job_id, job.status.value,
                        job.finished_at - (job.started_at or job.created_at))

    # ---------------------------------------------------------- queries
    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._prune()
            return self._jobs.get(job_id)

    def stats(self) -> dict:
        with self._lock:
            counts: dict[str, int] = {}
            for j in self._jobs.values():
                counts[j.status.value] = counts.get(j.status.value, 0) + 1
            return {
                "workers": self.cfg.max_concurrent_jobs,
                "max_pending_jobs": self.cfg.max_pending_jobs,
                "jobs_by_status": counts,
                "stored_jobs": len(self._jobs),
            }

    def _prune(self) -> None:
        """Forget finished jobs after their TTL, and the oldest beyond the cap."""
        now = time.time()
        for jid in [jid for jid, j in self._jobs.items()
                    if j.status.finished and j.finished_at and now - j.finished_at > self.cfg.job_ttl_seconds]:
            del self._jobs[jid]
        while len(self._jobs) > self.cfg.max_stored_jobs:
            oldest = next((jid for jid, j in self._jobs.items() if j.status.finished), None)
            if oldest is None:
                break
            del self._jobs[oldest]

    # ---------------------------------------------------------- shutdown
    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            for j in self._jobs.values():
                if j.status is JobStatus.QUEUED and j.future is not None and j.future.cancel():
                    j.update(status=JobStatus.FAILED, finished_at=time.time(),
                             error=UniqueCountError(503, "SHUTTING_DOWN", "Server shut down before the job started"))
                    shutil.rmtree(j.workdir, ignore_errors=True)
        self._executor.shutdown(wait=False, cancel_futures=True)
