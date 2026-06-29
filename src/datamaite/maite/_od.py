"""MAITE object-detection surface for :class:`ObjectDetectionDataset`.

The OD analogue of :mod:`datamaite.maite._mot`: turns one
:class:`~datamaite.records.ImageObjectDetectionSample` into the MAITE
object-detection item ``(image, ObjectDetectionTarget, DatumMetadata)``.

* image -- decoded ``(C, H, W)`` ``uint8`` array (MAITE input shape semantics).
* target -- boxes in ``(x0, y0, x1, y1)`` (MAITE OD convention), integer labels
  (the source ``category_id``, matching ``DatasetMetadata.index2label``), and
  scores (``1.0`` for ground-truth annotations, or the stored confidence).
* metadata -- at least ``id``; plus ``height``/``width`` when known.

``numpy`` is a module-level dependency here (like :mod:`datamaite.maite._mot`);
the image decoder (OpenCV) is imported lazily per decode. The module itself is
only imported from ``ObjectDetectionDataset.__getitem__``, so ``import
datamaite`` / loading / validating never pulls either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from datamaite.geometry import to_xyxy
from datamaite.maite._common import EMPTY_BOXES, EMPTY_LABELS, EMPTY_SCORES
from datamaite.maite._image import decode_image
from datamaite.records import ImageObjectDetectionSample


@dataclass(frozen=True)
class ObjectDetectionTarget:
    """MAITE-conformant OD target: boxes (N,4) xyxy, labels (N,), scores (N,).

    ``labels`` carries the source ``category_id`` (matching
    ``DatasetMetadata.index2label``); ``-1`` is the sentinel for annotations
    with no usable integer category and is deliberately absent from
    ``index2label``.
    """

    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


def _label(category_id: int | str | None) -> int:
    # bool is an int subclass; a stray True must not collapse into label 1.
    if isinstance(category_id, int) and not isinstance(category_id, bool):
        return category_id
    return -1


def _target(sample: ImageObjectDetectionSample) -> ObjectDetectionTarget:
    dets = sample.detections
    if not dets:
        return ObjectDetectionTarget(EMPTY_BOXES, EMPTY_LABELS, EMPTY_SCORES)
    boxes = np.array([to_xyxy(d.bbox) for d in dets], dtype=np.float32)
    labels = np.array([_label(d.category_id) for d in dets], dtype=np.int64)
    scores = np.array([1.0 if d.score is None else d.score for d in dets], dtype=np.float32)
    return ObjectDetectionTarget(boxes, labels, scores)


def build_od_item(sample: ImageObjectDetectionSample) -> tuple[np.ndarray, ObjectDetectionTarget, dict[str, Any]]:
    """Build one MAITE OD item ``(image, target, datum_metadata)`` for ``sample``."""
    image = decode_image(sample, task_name="ObjectDetectionDataset", extra="od")
    meta: dict[str, Any] = {"id": sample.image_id}
    height = sample.height if sample.height is not None else int(image.shape[1])
    width = sample.width if sample.width is not None else int(image.shape[2])
    meta["height"] = height
    meta["width"] = width
    return image, _target(sample), meta
