"""Shared MAITE image decoding helpers for still-image tasks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from datamaite.records import ImageRecord


def decode_image(sample: ImageRecord, *, task_name: str = "image", extra: str = "all") -> np.ndarray:
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
        import cv2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            f"Indexing a {task_name} dataset as a MAITE dataset decodes images and needs OpenCV. "
            f"Install it with: pip install datamaite[{extra}]"
        ) from exc

    if sample.image_bytes is not None:
        buf = np.frombuffer(sample.image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif sample.path_or_uri is not None:
        bgr = cv2.imread(str(Path(sample.path_or_uri)), cv2.IMREAD_COLOR)
    else:
        raise ValueError(f"image sample {sample.image_id!r} has neither path_or_uri nor image_bytes")
    if bgr is None:
        raise OSError(f"could not decode image for sample {sample.image_id!r} ({sample.path_or_uri})")
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
