"""GPU runtime helpers shared by every model in the /unique-count pipeline.

* Makes CUDA / cuDNN DLLs available to ONNX Runtime (onnxruntime-gpu does not
  ship them; they come from the ``nvidia-*`` wheels and PyTorch).
* Builds ONNX Runtime CUDA provider options.
* Reports which device each model actually ended up on.
"""

from __future__ import annotations

import logging
import threading

import torch  # imported before onnxruntime so its CUDA DLLs are loaded first

import onnxruntime

from models.runtime import cpu_threads, ort_cpu_session_options, release_memory
from services.config import UniqueCountConfig

logger = logging.getLogger(__name__)

_runtime_lock = threading.Lock()
_runtime_ready = False


def init_cuda_runtime() -> None:
    """Preload the CUDA/cuDNN DLLs for onnxruntime-gpu (idempotent)."""
    global _runtime_ready
    with _runtime_lock:
        if _runtime_ready:
            return
        preload = getattr(onnxruntime, "preload_dlls", None)
        if preload is not None and "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            try:
                preload()
            except Exception as exc:  # noqa: BLE001 - fall back to whatever is on PATH
                logger.warning("onnxruntime.preload_dlls() failed: %s", exc)
        _runtime_ready = True


def torch_cuda_available() -> bool:
    return torch.cuda.is_available()


def ort_cuda_available() -> bool:
    return "CUDAExecutionProvider" in onnxruntime.get_available_providers()


def ort_providers(cfg: UniqueCountConfig) -> tuple[list, int, object | None]:
    """ONNX Runtime providers, InsightFace ctx_id (-1 = CPU) and session options."""
    init_cuda_runtime()
    # onnxruntime-gpu lists the CUDA provider even on machines with no usable
    # GPU; check for a real device first so we fall back to CPU cleanly
    # instead of ORT printing EP errors and retrying.
    if ort_cuda_available() and not torch_cuda_available():
        logger.warning("No CUDA GPU detected - running face models on CPU (much slower)")
    elif ort_cuda_available():
        opts: dict[str, object] = {
            "device_id": cfg.gpu_device_id,
            # Grow the arena only by what is requested -> lower peak memory.
            "arena_extend_strategy": "kSameAsRequested",
            # SCRFD runs at several input sizes; exhaustive cuDNN search per
            # new shape is slow, the heuristic is near-optimal.
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": True,
        }
        if cfg.ort_gpu_mem_limit_mb > 0:
            opts["gpu_mem_limit"] = cfg.ort_gpu_mem_limit_mb * 1024 * 1024
        return [("CUDAExecutionProvider", opts), "CPUExecutionProvider"], cfg.gpu_device_id, None
    return ["CPUExecutionProvider"], -1, ort_cpu_session_options(cpu_threads(), cfg.ort_cpu_arena)


def session_device(session) -> str:
    """'cuda' if an ORT session really runs on the CUDA provider, else 'cpu'."""
    try:
        return "cuda" if session.get_providers()[0] == "CUDAExecutionProvider" else "cpu"
    except Exception:  # noqa: BLE001
        return "unknown"


def gpu_info(device_id: int = 0) -> dict:
    info: dict[str, object] = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.cuda.is_available(),
        "onnxruntime_version": onnxruntime.__version__,
        "onnxruntime_providers": onnxruntime.get_available_providers(),
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(device_id)
        info.update({
            "gpu_name": torch.cuda.get_device_name(device_id),
            "gpu_memory_total_mb": round(total / 2**20),
            "gpu_memory_free_mb": round(free / 2**20),
            "torch_allocated_mb": round(torch.cuda.memory_allocated(device_id) / 2**20),
            "torch_reserved_mb": round(torch.cuda.memory_reserved(device_id) / 2**20),
        })
    return info


def release_gpu_memory() -> None:
    """Free cached memory after a job (CUDA cache, Python garbage, glibc arenas)."""
    release_memory()
