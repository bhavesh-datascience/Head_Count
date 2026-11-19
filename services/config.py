"""Configuration for the /unique-count pipeline.

Every threshold can be overridden with an environment variable (prefix
``UC_``) so the matching behaviour can be tuned per deployment without code
changes, following the same pattern as ``detector.DetectorConfig``.

Similarity values refer to cosine similarity of L2-normalised embeddings:
  * Face: InsightFace buffalo_l ArcFace (w600k_r50). Same person is usually
    0.45-0.85, different people usually below 0.25.
  * Body: OSNet x1_0 trained on MSMT17. Values are compressed into a higher
    range than faces; same person/same clothes is usually > 0.75.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _b(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _sizes(name: str, default: str) -> list[tuple[int, int]]:
    """Parse "640,1280" into [(640, 640), (1280, 1280)] (multi-scale SCRFD)."""
    raw = os.getenv(name, default)
    return [(int(v), int(v)) for v in raw.replace(" ", "").split(",") if v]


@dataclass
class UniqueCountConfig:
    # ---- Upload / queue -------------------------------------------------
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    max_upload_bytes: int = field(default_factory=lambda: _i("UC_MAX_UPLOAD_MB", 20) * 1024 * 1024)
    max_images: int = field(default_factory=lambda: _i("UC_MAX_IMAGES", 500))
    # Images larger than this (longest side, px) are downscaled after decoding
    # to bound memory; 0 disables. Group photos need resolution for small faces.
    max_image_side: int = field(default_factory=lambda: _i("UC_MAX_IMAGE_SIDE", 4096))
    # Number of dedicated worker threads that run the GPU pipeline, i.e. how
    # many jobs execute at once. 1 keeps GPU memory predictable.
    max_concurrent_jobs: int = field(default_factory=lambda: _i("UC_MAX_CONCURRENT_JOBS", 1))
    # Jobs allowed to wait in the queue; further submissions get HTTP 429.
    max_pending_jobs: int = field(default_factory=lambda: _i("UC_MAX_PENDING_JOBS", 8))
    # Finished jobs (and their results) are kept for polling this long.
    job_ttl_seconds: int = field(default_factory=lambda: _i("UC_JOB_TTL_SECONDS", 3600))
    max_stored_jobs: int = field(default_factory=lambda: _i("UC_MAX_STORED_JOBS", 200))
    # Where uploads are spooled while a job is queued/processing.
    upload_dir: str = field(default_factory=lambda: os.getenv("UC_UPLOAD_DIR", ""))

    # ---- GPU ------------------------------------------------------------
    gpu_device_id: int = field(default_factory=lambda: _i("UC_GPU_DEVICE", 0))
    # Fail model initialisation instead of silently running on CPU.
    require_gpu: bool = field(default_factory=lambda: _b("UC_REQUIRE_GPU", False))
    # ONNX Runtime CUDA arena cap in MB (0 = no cap).
    ort_gpu_mem_limit_mb: int = field(default_factory=lambda: _i("UC_ORT_GPU_MEM_MB", 0))
    # OSNet in FP16 on GPU (negligible accuracy impact, ~2x faster).
    osnet_fp16: bool = field(default_factory=lambda: _b("UC_OSNET_FP16", True))
    # Load and warm up all models in the background at server start.
    preload_models: bool = field(default_factory=lambda: _b("UC_PRELOAD_MODELS", True))

    # ---- InsightFace ----------------------------------------------------
    face_model_name: str = field(default_factory=lambda: os.getenv("UC_FACE_MODEL", "buffalo_l"))
    face_model_root: str = field(default_factory=lambda: os.getenv("UC_FACE_MODEL_ROOT", "~/.insightface"))
    det_sizes: list[tuple[int, int]] = field(default_factory=lambda: _sizes("UC_DET_SIZES", "640,1280"))
    det_thresh: float = field(default_factory=lambda: _f("UC_DET_THRESH", 0.50))
    min_face_size: int = field(default_factory=lambda: _i("UC_MIN_FACE_SIZE", 20))
    # Faces whose quality score (0..1) is below this are kept (they are still
    # people) but their embeddings are treated as unreliable during matching.
    min_face_quality: float = field(default_factory=lambda: _f("UC_MIN_FACE_QUALITY", 0.35))

    # ---- Person Re-ID (OSNet) -------------------------------------------
    reid_enabled: bool = field(default_factory=lambda: _b("UC_REID_ENABLED", True))
    osnet_weights: str = field(default_factory=lambda: os.getenv("UC_OSNET_WEIGHTS", "weights/osnet_x1_0_msmt17.pth"))
    osnet_weights_url: str = field(default_factory=lambda: os.getenv(
        "UC_OSNET_WEIGHTS_URL",
        "https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
        "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    ))
    reid_batch_size: int = field(default_factory=lambda: _i("UC_REID_BATCH", 32))
    use_person_tiles: bool = field(default_factory=lambda: _b("UC_PERSON_TILES", True))
    min_person_height: int = field(default_factory=lambda: _i("UC_MIN_PERSON_HEIGHT", 48))

    # ---- Identity matching ---------------------------------------------
    # Face similarity at/above which two faces are the same person on face
    # evidence alone. Body evidence never overrides this.
    face_strong_threshold: float = field(default_factory=lambda: _f("UC_FACE_STRONG", 0.55))
    # Below this the faces are different people; body similarity is ignored
    # (similar clothing alone never proves identity).
    face_weak_threshold: float = field(default_factory=lambda: _f("UC_FACE_WEAK", 0.30))
    # When either face is low quality the weak bound is lowered by this much,
    # letting Re-ID support pairs whose face evidence is degraded.
    low_quality_weak_relax: float = field(default_factory=lambda: _f("UC_LOW_QUALITY_RELAX", 0.05))
    # Average-linkage threshold: clusters merge only while the mean fused
    # similarity across *all* member pairs stays above this value.
    cluster_threshold: float = field(default_factory=lambda: _f("UC_CLUSTER_THRESHOLD", 0.42))
    # A member is ejected from its cluster if its mean similarity to the other
    # members drops below this (cluster consistency check).
    member_min_similarity: float = field(default_factory=lambda: _f("UC_MEMBER_MIN_SIM", 0.36))
    # Body similarity used when faces are in the uncertain band.
    body_neutral: float = field(default_factory=lambda: _f("UC_BODY_NEUTRAL", 0.60))
    body_weight: float = field(default_factory=lambda: _f("UC_BODY_WEIGHT", 0.50))
    body_max_adjust: float = field(default_factory=lambda: _f("UC_BODY_MAX_ADJUST", 0.12))
    # Body-only observations (no usable face, e.g. back turned) attach to an
    # identity only above this OSNet similarity *and* with a clear margin.
    body_only_attach_threshold: float = field(default_factory=lambda: _f("UC_BODY_ONLY_ATTACH", 0.82))
    body_only_margin: float = field(default_factory=lambda: _f("UC_BODY_ONLY_MARGIN", 0.05))
    # Whether body-only observations that match no identity count as new people.
    count_body_only: bool = field(default_factory=lambda: _b("UC_COUNT_BODY_ONLY", False))
    # Pairs of *different* final identities whose best face similarity falls
    # in [face_weak_threshold, cluster_threshold) are reported for review.
    report_uncertain: bool = field(default_factory=lambda: _b("UC_REPORT_UNCERTAIN", True))
    max_uncertain_reported: int = field(default_factory=lambda: _i("UC_MAX_UNCERTAIN", 200))
    # Number of best-quality faces averaged into an identity's representative embedding.
    representative_top_k: int = field(default_factory=lambda: _i("UC_REPRESENTATIVE_TOP_K", 5))


config = UniqueCountConfig()
