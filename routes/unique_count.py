"""HTTP layer for /unique-count.

Handles only request validation, job submission and response formatting; all
processing runs on the GPU worker thread managed by the service layer.

Endpoints
  POST /unique-count                 submit N images; waits for the result
                                     (or returns 202 + job id with async_mode=true)
  GET  /unique-count/jobs/{job_id}   job state, per-image progress, result when done
  GET  /unique-count/status          model state, GPU/device info, queue stats
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from services.config import config
from services.unique_count_service import (ImageTask, Job, JobStatus, UniqueCountError, get_runtime)
from utils.image_processing import has_allowed_extension, sniff_format

logger = logging.getLogger(__name__)

router = APIRouter(tags=["unique-count"])


def _configure_logging() -> None:
    """Make pipeline stage logs visible under uvicorn without touching the root logger."""
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s")
    for name in ("routes", "services", "models"):
        lg = logging.getLogger(name)
        if not lg.handlers:
            h = logging.StreamHandler()
            h.setFormatter(fmt)
            lg.addHandler(h)
            lg.setLevel(logging.INFO)
            lg.propagate = False


_configure_logging()


def _error(status: int, code: str, message: str, details: dict | None = None, job_id: str | None = None) -> JSONResponse:
    body = {
        "success": False,
        "processing_status": "failed",
        "error": {"code": code, "message": message, "details": details or {}},
    }
    if job_id:
        body["job_id"] = job_id
    return JSONResponse(status_code=status, content=body)


def _spool(src, dest: Path) -> None:
    """Copy an upload to disk in chunks (never the whole file in memory)."""
    src.seek(0)
    with open(dest, "wb") as out:
        shutil.copyfileobj(src, out, length=1024 * 1024)


def _runtime(request: Request):
    return get_runtime(getattr(request.app.state, "person_detector", None))


@router.post("/unique-count", summary="Count unique people across N images")
async def unique_count(
    request: Request,
    files: list[UploadFile] | None = File(None, description="One or more images (jpg, jpeg, png, webp)"),
    skip_invalid: bool = Query(False, description="Skip corrupted images instead of failing the whole request"),
    include_details: bool = Query(False, description="Include per-face observations for each identity"),
    async_mode: bool = Query(False, description="Return 202 with a job id immediately; poll /unique-count/jobs/{job_id}"),
):
    if not files:
        return _error(400, "NO_IMAGES", "No images uploaded. Send one or more files in the 'files' form field.")
    if len(files) > config.max_images:
        return _error(413, "TOO_MANY_IMAGES", f"At most {config.max_images} images per request (got {len(files)})")

    # 1) Cheap validation of every file before any work is queued.
    accepted: list[tuple[str, str, UploadFile]] = []
    for n, f in enumerate(files, 1):
        image_id = f"IMAGE_{n:03d}"
        name = f.filename or image_id
        info = {"image_id": image_id, "filename": name}
        if not has_allowed_extension(f.filename, config.allowed_extensions):
            return _error(415, "UNSUPPORTED_FORMAT",
                          f"{name}: unsupported file type. Allowed: {', '.join(config.allowed_extensions)}", info)
        if f.size is not None and f.size > config.max_upload_bytes:
            return _error(413, "FILE_TOO_LARGE", f"{name}: larger than {config.max_upload_bytes // (1024 * 1024)} MB", info)
        header = await f.read(16)
        await f.seek(0)
        if not header:
            if not skip_invalid:
                return _error(422, "EMPTY_FILE", f"{name}: file is empty", info)
            logger.warning("Upload %s is empty; it will be reported as failed", name)
        elif sniff_format(header) is None:
            return _error(415, "UNSUPPORTED_FORMAT", f"{name}: content is not a JPEG, PNG or WEBP image", info)
        accepted.append((image_id, name, f))

    # 2) Spool uploads into a per-job directory so the job can outlive the request.
    rt = _runtime(request)
    job_id, workdir = rt.jobs.new_workdir()
    try:
        tasks = []
        for image_id, name, f in accepted:
            dest = workdir / f"{image_id}{Path(name).suffix.lower()}"
            await run_in_threadpool(_spool, f.file, dest)
            tasks.append(ImageTask(image_id=image_id, filename=name, path=dest))
        job = rt.jobs.submit(Job(job_id=job_id, tasks=tasks, workdir=workdir,
                                 skip_invalid=skip_invalid, include_details=include_details))
    except UniqueCountError as exc:
        shutil.rmtree(workdir, ignore_errors=True)
        return _error(exc.status_code, exc.code, exc.message, exc.details)
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(workdir, ignore_errors=True)
        logger.exception("Failed to queue /unique-count job")
        return _error(500, "INTERNAL_ERROR", "Could not queue the images for processing", {"reason": str(exc)})

    status_url = str(request.url_for("unique_count_job", job_id=job.job_id))
    if async_mode:
        return JSONResponse(status_code=202, content={
            "success": True, "job_id": job.job_id, "status": job.status.value,
            "total_images": len(tasks), "status_url": status_url,
        })

    # 3) Sync mode: wait without blocking the event loop. If the client
    #    disconnects the job still finishes and stays pollable.
    await asyncio.wrap_future(job.future)
    if job.status is JobStatus.COMPLETED:
        return job.result
    err = job.error or UniqueCountError(500, "INTERNAL_ERROR", "Job failed")
    return _error(err.status_code, err.code, err.message, err.details, job_id=job.job_id)


@router.get("/unique-count/jobs/{job_id}", name="unique_count_job", summary="Status / result of a unique-count job")
async def unique_count_job(request: Request, job_id: str,
                           include_result: bool = Query(True, description="Include the result once completed")):
    job = _runtime(request).jobs.get(job_id)
    if job is None:
        return _error(404, "JOB_NOT_FOUND", f"Unknown or expired job id '{job_id}'")
    return {"success": True, **job.snapshot(include_result=include_result)}


@router.get("/unique-count/status", summary="Model state, device placement and queue statistics")
async def unique_count_status(request: Request):
    rt = _runtime(request)
    return await run_in_threadpool(rt.status)
