"""Canonical bounding-box geometry and conversions.

The databridge in-memory representation stores every box in one canonical form:
**absolute-pixel ``xywh``** -- ``(left, top, width, height)`` with the origin at
the top-left of the image. ``BBox`` is defined here (the single source) and
re-exported by :mod:`databridge.model`. All format
specifics (Pascal VOC ``xyxy``, YOLO normalized ``cxcywh``, ...) are converted
to/from this canonical form **only at the format boundary** (loaders convert in,
writers convert out); the IR never holds a format-native box.

This module is the single home for those conversions so each format loader/writer
does not re-derive them (a common source of off-by-one and normalization bugs).
Functions are pure and operate on plain tuples; there is no box *class* -- the
canonical box is the ``(float, float, float, float)`` tuple used across the model.
"""

from __future__ import annotations

import math

# Canonical box: (left, top, width, height) in absolute pixels, origin top-left.
BBox = tuple[float, float, float, float]
# Corner box: (x1, y1, x2, y2) absolute pixels (Pascal VOC / KITTI style).
XYXY = tuple[float, float, float, float]
# Center box: (cx, cy, w, h) absolute pixels.
CXCYWH = tuple[float, float, float, float]


def to_xyxy(box: BBox) -> XYXY:
    """Canonical ``xywh`` -> corner ``(x1, y1, x2, y2)`` (half-open: x2 = left + width)."""
    left, top, width, height = box
    return (left, top, left + width, top + height)


def from_xyxy(x1: float, y1: float, x2: float, y2: float) -> BBox:
    """Corner ``(x1, y1, x2, y2)`` -> canonical ``xywh``.

    Use this for the common 0-indexed half-open corner convention. For Pascal
    VOC's 1-indexed *inclusive* corners (where ``width = xmax - xmin + 1``), use
    :func:`from_xyxy_inclusive` instead so the width/height are exact.
    """
    return (x1, y1, x2 - x1, y2 - y1)


def from_xyxy_inclusive(xmin: float, ymin: float, xmax: float, ymax: float) -> BBox:
    """Pascal-VOC-style *inclusive* corners -> canonical ``xywh``.

    VOC corners are inclusive (a 1px box has ``xmax == xmin``), so the pixel
    extent is ``xmax - xmin + 1``. Callers that loaded VOC integer corners and
    want a byte-exact round trip must record the original ``index_origin`` /
    ``inclusive`` flags (see the OD record) and pair this with
    :func:`to_xyxy_inclusive` on write.
    """
    return (xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)


def to_xyxy_inclusive(box: BBox) -> XYXY:
    """Canonical ``xywh`` -> Pascal-VOC-style *inclusive* corners."""
    left, top, width, height = box
    return (left, top, left + width - 1, top + height - 1)


def to_cxcywh(box: BBox) -> CXCYWH:
    """Canonical ``xywh`` -> center ``(cx, cy, w, h)`` in absolute pixels."""
    left, top, width, height = box
    return (left + width / 2.0, top + height / 2.0, width, height)


def from_cxcywh(cx: float, cy: float, width: float, height: float) -> BBox:
    """Center ``(cx, cy, w, h)`` (absolute pixels) -> canonical ``xywh``."""
    return (cx - width / 2.0, cy - height / 2.0, width, height)


def to_normalized(box: BBox, image_width: float, image_height: float) -> BBox:
    """Canonical absolute ``xywh`` -> normalized ``xywh`` (each component / image size)."""
    _check_dims(image_width, image_height)
    left, top, width, height = box
    return (left / image_width, top / image_height, width / image_width, height / image_height)


def from_normalized(box: BBox, image_width: float, image_height: float) -> BBox:
    """Normalized ``xywh`` -> canonical absolute ``xywh``."""
    _check_dims(image_width, image_height)
    left, top, width, height = box
    return (left * image_width, top * image_height, width * image_width, height * image_height)


def to_yolo(box: BBox, image_width: float, image_height: float) -> CXCYWH:
    """Canonical absolute ``xywh`` -> YOLO normalized center box ``(cx, cy, w, h)`` in [0, 1]."""
    _check_dims(image_width, image_height)
    cx, cy, width, height = to_cxcywh(box)
    return (cx / image_width, cy / image_height, width / image_width, height / image_height)


def from_yolo(cx: float, cy: float, width: float, height: float, image_width: float, image_height: float) -> BBox:
    """YOLO normalized center box -> canonical absolute ``xywh``.

    Requires the image pixel dimensions: YOLO label files embed no size, so the
    image must be read to materialize an absolute box. Callers that cannot read
    the image should keep the native normalized values rather than calling this.
    """
    _check_dims(image_width, image_height)
    return from_cxcywh(cx * image_width, cy * image_height, width * image_width, height * image_height)


def is_finite(box: BBox) -> bool:
    """True if all four components are finite (no NaN/inf)."""
    return all(math.isfinite(v) for v in box)


def has_positive_area(box: BBox) -> bool:
    """True if width and height are both strictly positive (and finite)."""
    return is_finite(box) and box[2] > 0 and box[3] > 0


def area(box: BBox) -> float:
    """Area ``width * height`` of the canonical box (may be 0 for a degenerate box)."""
    return box[2] * box[3]


def clamp_to_image(box: BBox, image_width: float, image_height: float) -> BBox:
    """Clamp a canonical box so it lies within ``[0, image_*]`` on each axis.

    Clamps the corners, not the width/height, so a box partly outside the image
    is trimmed to the visible region. A box fully outside collapses to zero
    width/height (detectable with :func:`has_positive_area`).
    """
    _check_dims(image_width, image_height)
    x1, y1, x2, y2 = to_xyxy(box)
    cx1 = min(max(x1, 0.0), image_width)
    cy1 = min(max(y1, 0.0), image_height)
    cx2 = min(max(x2, 0.0), image_width)
    cy2 = min(max(y2, 0.0), image_height)
    return from_xyxy(cx1, cy1, cx2, cy2)


def _check_dims(image_width: float, image_height: float) -> None:
    if not (math.isfinite(image_width) and math.isfinite(image_height)) or image_width <= 0 or image_height <= 0:
        raise ValueError(f"image dimensions must be finite and positive, got ({image_width}, {image_height})")
