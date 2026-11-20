"""CPU / thread / memory helpers shared by every model.

On a small CPU-only server (e.g. AWS t3.large: 2 vCPU, 8 GB) the defaults of
PyTorch, ONNX Runtime and OpenCV hurt badly:

* Each library sizes its thread pool from the *host* CPU count, not the CPUs
  this process may actually use (affinity / cgroup quota), so several pools
  of many threads fight over 2 cores.
* ONNX Runtime threads spin-wait between ops, burning CPU (and burstable
  instance CPU credits) while doing nothing.
* glibc keeps freed memory in per-thread arenas, so the resident size keeps
  the peak of the largest image ever processed.

This module fixes those once at startup and after each job.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import sys
import threading

logger = logging.getLogger(__name__)

_configured = False
_lock = threading.Lock()


def has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _cgroup_cpu_limit() -> float | None:
    """CPU quota from cgroup v2/v1 (Docker --cpus, systemd CPUQuota), if any."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()[:2]
        if quota != "max":
            return int(quota) / int(period)
    except Exception:  # noqa: BLE001
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        if quota > 0:
            return quota / period
    except Exception:  # noqa: BLE001
        pass
    return None


def available_cpus() -> int:
    """CPUs this process may really use (affinity and cgroup quota aware)."""
    n = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        n = len(os.sched_getaffinity(0))
    else:
        try:
            import psutil
            n = len(psutil.Process().cpu_affinity()) or n
        except Exception:  # noqa: BLE001
            pass
    quota = _cgroup_cpu_limit()
    if quota:
        n = min(n, max(1, int(quota + 0.5)))
    return max(1, n)


def cpu_threads() -> int:
    """Threads per inference library; UC_CPU_THREADS overrides (0 = auto)."""
    try:
        override = int(os.getenv("UC_CPU_THREADS", "0"))
    except ValueError:
        override = 0
    return override if override > 0 else available_cpus()


def configure_threads() -> int:
    """Size the PyTorch and OpenCV thread pools to the usable CPUs (idempotent)."""
    global _configured
    with _lock:
        n = cpu_threads()
        if _configured:
            return n
        import cv2
        import torch

        torch.set_num_threads(n)
        try:
            # Only allowed before any inter-op parallel work has started.
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        cv2.setNumThreads(n)
        _configured = True
        logger.info("CPU threads per library: %d (usable CPUs: %d, cuda: %s)", n, available_cpus(), has_cuda())
        return n


def ort_cpu_session_options(threads: int, mem_arena: bool):
    """ONNX Runtime session options for CPU inference on few cores."""
    import onnxruntime

    so = onnxruntime.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    # The arena keeps the buffers of the largest input ever seen; without it
    # memory goes back to the allocator after each run (slightly slower).
    so.enable_cpu_mem_arena = mem_arena
    # Idle worker threads sleep instead of spinning (saves CPU / burst credits).
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return so


def _malloc_trim() -> None:
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:  # noqa: BLE001 - musl or no glibc
            pass


def release_memory() -> None:
    """Free cached memory after a job: Python garbage, CUDA cache, glibc arenas."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    _malloc_trim()


def rss_mb() -> int | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss >> 20
    except Exception:  # noqa: BLE001
        return None
