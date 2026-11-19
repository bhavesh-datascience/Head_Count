"""Image validation, decoding and cropping helpers for /unique-count."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Magic-byte signatures of the accepted formats. Checked in addition to the
# file extension so a renamed text/PDF file is rejected before decoding.
_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    "jpeg": (b"\xff\xd8\xff",),
    "png": (b"\x89PNG\r\n\x1a\n",),
    "webp": (b"RIFF",),  # plus b"WEBP" at offset 8, checked below
}


class ImageValidationError(ValueError):
    """Raised for unsupported or unreadable image uploads."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def has_allowed_extension(filename: str | None, allowed: tuple[str, ...]) -> bool:
    return bool(filename) and Path(filename).suffix.lower() in allowed


def sniff_format(header: bytes) -> str | None:
    """Return "jpeg" / "png" / "webp" from the first bytes of a file, else None."""
    if header.startswith(_SIGNATURES["jpeg"][0]):
        return "jpeg"
    if header.startswith(_SIGNATURES["png"][0]):
        return "png"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "webp"
    return None


def decode_image(data: bytes, max_side: int = 0) -> tuple[np.ndarray, float]:
    """Decode bytes into a BGR image.

    Returns the image and the scale applied (1.0 when not resized). Raises
    ImageValidationError when the data is empty or corrupted.
    """
    if not data:
        raise ImageValidationError("EMPTY_FILE", "Uploaded file is empty")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ImageValidationError("CORRUPTED_IMAGE", "Image could not be decoded (corrupted or truncated)")

    scale = 1.0
    h, w = image.shape[:2]
    if max_side and max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return image, scale


def clip_box(box: np.ndarray | list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(v) for v in box)
    return (
        int(max(0, min(width - 1, round(x1)))),
        int(max(0, min(height - 1, round(y1)))),
        int(max(0, min(width, round(x2)))),
        int(max(0, min(height, round(y2)))),
    )


def crop(image: np.ndarray, box: np.ndarray | list[float]) -> np.ndarray | None:
    """Crop a box (x1, y1, x2, y2) from an image; None if the crop is empty."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clip_box(box, w, h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return image[y1:y2, x1:x2]
