"""Source-preserving records for still-image vision tasks.

The in-memory homes that make IC/OD format conversion as lossless as possible:
every field a real image dataset format carries has a typed slot here (or a
documented place in ``attributes`` / dataset-level :class:`DatasetMetadata`), so
a loader never has to pretend an image dataset is a one-frame video.

Records are frozen dataclasses, mirroring :class:`datamaite.model.BoxAnnotation`.
``attributes`` / ``metadata`` are the open-ended round-trip channels for anything
not promoted to a typed field. Bounding boxes use canonical absolute-pixel
``xywh`` (see :mod:`datamaite.geometry`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from datamaite.geometry import BBox
from datamaite.taxonomy import Taxonomy


@dataclass(frozen=True)
class ImageRecord:
    """Shared image/media source fields used by still-image tasks.

    ``split`` is first-class because image datasets commonly store train/val/test
    as part of the layout (YOLO/ImageFolder/Hugging Face folders, split COCO
    annotation files, etc.). Writers may use it to recreate split directories;
    loaders leave it ``None`` when the source layout has no split concept.
    """

    image_id: int | str
    path_or_uri: str | None = None  # None for byte-backed sources
    image_bytes: bytes | None = None
    file_name: str | None = None
    width: int | None = None
    height: int | None = None
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)  # per-image writer-consumed passthrough
    # Optional crop rectangle (canonical xywh). When set, MAITE decoding returns
    # this sub-region of the image (e.g. VisDrone IC object crops); None = full image.
    #
    # Keyword-only: a base-class field otherwise slots ahead of subclass fields
    # (``labels``/``detections``) in positional constructors, so an existing
    # positional call would bind its label/detection tuple into ``region``.
    # ``kw_only=True`` places it after the subclass fields, preserving the
    # positional signatures that predate this field.
    region: BBox | None = field(default=None, kw_only=True)


@dataclass(frozen=True)
class ClassificationLabel:
    """One image-level classification label or score.

    A tuple of these labels supports single-label classification today and gives
    multi-label/probabilistic formats a lossless place to land later.
    """

    category_id: int | str | None = None
    category_name: str | None = None
    source_category_id: int | str | None = None
    score: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageClassificationSample(ImageRecord):
    """One image plus image-level classification labels."""

    labels: tuple[ClassificationLabel, ...] = ()


@dataclass(frozen=True)
class ObjectDetectionAnnotation:
    """One detection on one image, preserving source provenance for faithful export."""

    bbox: BBox  # canonical absolute-pixel xywh
    category_id: int | str | None = None
    category_name: str | None = None
    source_category_id: int | str | None = None
    # COCO annotation ids are globally unique and consumers depend on them; keep
    # them. Order-only formats leave this None rather than synthesizing identity.
    source_annotation_id: int | str | None = None
    score: float | None = None  # confidence ONLY; ground-truth has none (writers default 1.0)
    area: float | None = None  # SOURCE area verbatim; never recompute as w*h
    segmentation: Any | None = None  # polygon list | RLE {counts,size}; None when absent
    iscrowd: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageObjectDetectionSample(ImageRecord):
    """One image plus its detections (the OD per-sample record)."""

    detections: tuple[ObjectDetectionAnnotation, ...] = ()

    def __post_init__(self) -> None:
        # Backwards compatibility for !23 positional construction where the
        # seventh positional argument was ``detections``. ``ImageRecord`` added
        # ``split`` before task labels; do not silently bind an annotation tuple
        # to that string field.
        if (
            self.detections == ()
            and isinstance(self.split, (list, tuple))
            and all(isinstance(item, ObjectDetectionAnnotation) for item in self.split)
        ):
            object.__setattr__(self, "detections", tuple(self.split))
            object.__setattr__(self, "split", None)


@dataclass(frozen=True)
class DatasetMetadata:
    """Dataset-level baggage with no per-sample home.

    ``taxonomy`` is the semantically load-bearing category set. ``splits`` is a
    normalized list of known split names in preferred writer order. The rest is
    round-trip provenance that a writer may re-emit verbatim.
    """

    taxonomy: Taxonomy | None = None
    source_dataset: str = "datamaite"
    splits: tuple[str, ...] = ()
    info: dict[str, Any] = field(default_factory=dict)
    licenses: tuple[dict[str, Any], ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)
