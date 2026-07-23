"""MAITE image-classification surface for :class:`ImageClassificationDataset`."""

from __future__ import annotations

from typing import Any

import numpy as np

from datamaite.maite._image import decode_image
from datamaite.records import ClassificationLabel, ImageClassificationSample
from datamaite.taxonomy import Taxonomy


def _candidate_source_id(label: ClassificationLabel) -> int | str | None:
    return label.source_category_id if label.source_category_id is not None else label.category_id


def _dense_index(label: ClassificationLabel, taxonomy: Taxonomy | None) -> int | None:
    source_id = _candidate_source_id(label)
    if taxonomy is None:
        if isinstance(source_id, int) and not isinstance(source_id, bool):
            return source_id
        return None

    # Prefer the taxonomy's explicit dense projection when source ids are unique.
    try:
        dense_ids = taxonomy.dense_ids()
    except ValueError:
        # Merged (multi-source) taxonomy with duplicate source ids: first
        # occurrence wins, to match the writer's ``by_source_id()`` lookup so the
        # MAITE target and the on-disk class for a duplicate id never disagree.
        dense_ids = {}
        for idx, entry in enumerate(taxonomy.entries):
            dense_ids.setdefault(entry.source_id, idx)
    if source_id in dense_ids:
        return dense_ids[source_id]
    if label.category_name is not None:
        for idx, entry in enumerate(taxonomy.entries):
            if entry.name == label.category_name:
                return idx
    return None


def _num_classes(sample: ImageClassificationSample, taxonomy: Taxonomy | None) -> int:
    if taxonomy is not None:
        return len(taxonomy.entries)
    indexes = [idx for label in sample.labels if (idx := _dense_index(label, None)) is not None]
    return max(indexes, default=-1) + 1


def _target(sample: ImageClassificationSample, taxonomy: Taxonomy | None) -> np.ndarray:
    target = np.zeros((_num_classes(sample, taxonomy),), dtype=np.float32)
    for label in sample.labels:
        idx = _dense_index(label, taxonomy)
        if idx is None or idx < 0 or idx >= len(target):
            continue
        target[idx] = 1.0 if label.score is None else float(label.score)
    return target


def ic_input(sample: ImageClassificationSample) -> np.ndarray:
    """Decode one IC sample to its MAITE input array (``(C, H, W)`` ``uint8``)."""
    return decode_image(sample, task_name="ImageClassificationDataset", extra="ic")


def ic_target(sample: ImageClassificationSample, taxonomy: Taxonomy | None) -> np.ndarray:
    """Build one IC sample's MAITE target vector (no image decode required).

    A dataset-level ``taxonomy`` is required for a stable target width across
    samples (every loader in this package supplies one). Without it the width is
    inferred per-sample from integer source ids, so a taxonomy-less dataset can
    yield ragged targets -- usable for a quick look, not for batched evaluation.
    """
    return _target(sample, taxonomy)


def ic_metadata(sample: ImageClassificationSample, image: np.ndarray | None = None) -> dict[str, Any]:
    """Build one IC sample's MAITE datum metadata (``id``/``split``/``height``/``width``).

    The image is decoded only when the true dimensions cannot be known without
    it: a region crop (whose clamped size is known only from the decoded array)
    or a sample missing stored ``height``/``width``. When ``build_ic_item`` has
    already decoded the image it is passed in via ``image`` to avoid a re-decode.
    """
    meta: dict[str, Any] = {"id": sample.image_id}
    if sample.split is not None:
        meta["split"] = sample.split
    has_region = getattr(sample, "region", None) is not None
    if has_region or sample.height is None or sample.width is None:
        if image is None:
            image = ic_input(sample)
        if has_region:
            # A region crop is clamped to image bounds at decode time, so its true
            # size is known only from the decoded array -- the stored dims are the
            # nominal (possibly over-edge) box size and would disagree for
            # edge-straddling crops.
            height, width = int(image.shape[1]), int(image.shape[2])
        else:
            height = sample.height if sample.height is not None else int(image.shape[1])
            width = sample.width if sample.width is not None else int(image.shape[2])
    else:
        height, width = sample.height, sample.width
    meta["height"] = height
    meta["width"] = width
    return meta


def build_ic_item(
    sample: ImageClassificationSample,
    taxonomy: Taxonomy | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build one MAITE IC item ``(image, target, datum_metadata)`` for ``sample``."""
    image = ic_input(sample)
    return image, ic_target(sample, taxonomy), ic_metadata(sample, image)
