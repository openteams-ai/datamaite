"""Shared MAITE image decoding helpers for still-image tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from datamaite._io import decode_image_bytes, decode_image_path
from datamaite._upath import sanitized_uri
from datamaite.records import ImageRecord


def decode_image(
    sample: ImageRecord,
    *,
    task_name: str = "image",
    extra: str = "all",
    storage_options: Mapping[str, Any] | None = None,
    base_cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Decode an image sample to a ``(C, H, W)`` ``uint8`` RGB array.

    If ``sample.region`` is set (a ``(left, top, width, height)`` crop rectangle,
    used by loaders that derive classification crops from detection boxes, e.g.
    VisDrone IC), the decoded image is cropped to that region after clamping it to
    the image bounds. A region that is empty after clamping (fully outside the
    image, or sub-pixel) raises ``ValueError``.

    OpenCV is imported lazily so importing/loading datasets does not require the
    optional task extras; only MAITE-style indexing does.
    """
    try:
        __import__("cv2")
    except ImportError as exc:
        raise ImportError(
            f"Indexing a {task_name} dataset as a MAITE dataset decodes images and needs OpenCV. "
            f"Install it with: pip install datamaite[{extra}]"
        ) from exc

    if sample.image_bytes is not None:
        bgr = decode_image_bytes(sample.image_bytes, color=True)
    elif sample.path_or_uri is not None:
        cache_key = sample.path_or_uri
        if base_cache is not None and cache_key in base_cache:
            bgr = base_cache[cache_key]
        else:
            bgr = decode_image_path(sample.path_or_uri, storage_options, color=True)
            if bgr is not None and base_cache is not None:
                base_cache.clear()
                base_cache[cache_key] = bgr
    else:
        raise ValueError(f"image sample {sample.image_id!r} has neither path_or_uri nor image_bytes")
    if bgr is None:
        location = sanitized_uri(sample.path_or_uri) if sample.path_or_uri is not None else "embedded bytes"
        raise OSError(f"could not decode image for sample {sample.image_id!r} ({location})")
    region = getattr(sample, "region", None)
    if region is not None:
        img_h, img_w = bgr.shape[:2]
        left, top, box_w, box_h = region
        x1 = max(0, round(left))
        y1 = max(0, round(top))
        x2 = min(img_w, round(left + box_w))
        y2 = min(img_h, round(top + box_h))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"crop region {region} for sample {sample.image_id!r} is empty after clamping to image bounds"
            )
        bgr = bgr[y1:y2, x1:x2]
    rgb = bgr[:, :, ::-1]  # BGR -> RGB
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))  # HWC -> CHW
