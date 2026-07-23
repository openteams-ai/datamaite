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


def od_input(sample: ImageObjectDetectionSample) -> np.ndarray:
    """Decode one OD sample to its MAITE input array (``(C, H, W)`` ``uint8``)."""
    return decode_image(sample, task_name="ObjectDetectionDataset", extra="od")


def od_target(sample: ImageObjectDetectionSample) -> ObjectDetectionTarget:
    """Build one OD sample's MAITE target (no image decode required)."""
    return _target(sample)


def od_metadata(sample: ImageObjectDetectionSample, image: np.ndarray | None = None) -> dict[str, Any]:
    """Build one OD sample's MAITE datum metadata (``id``/``height``/``width``).

    The image is decoded only when ``height``/``width`` are not stored on the
    sample. When ``build_od_item`` has already decoded the image it is passed in
    via ``image`` to avoid a re-decode.
    """
    meta: dict[str, Any] = {"id": sample.image_id}
    height, width = sample.height, sample.width
    if height is None or width is None:
        if image is None:
            image = od_input(sample)
        height = sample.height if sample.height is not None else int(image.shape[1])
        width = sample.width if sample.width is not None else int(image.shape[2])
    meta["height"] = height
    meta["width"] = width
    return meta


def build_od_item(sample: ImageObjectDetectionSample) -> tuple[np.ndarray, ObjectDetectionTarget, dict[str, Any]]:
    """Build one MAITE OD item ``(image, target, datum_metadata)`` for ``sample``."""
    image = od_input(sample)
    return image, od_target(sample), od_metadata(sample, image)
