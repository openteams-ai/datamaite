"""Still-image object-detection dataset -- a native MAITE OD dataset.

The OD sibling of :class:`datamaite.model.BoxTrackDataset` (MOT). It is the
neutral hub every OD converter consumes **and** it implements the MAITE
object-detection protocol directly: ``len(ds)`` is the image count and ``ds[i]``
yields ``(image, ObjectDetectionTarget, DatumMetadata)`` for image ``i`` (see
:func:`datamaite.maite._od.build_od_item`). The MAITE surface is computed
lazily; indexing requires an image decoder (``pip install datamaite[od]``),
but ``import datamaite`` / ``load`` / ``validate`` never touch it.

Unlike the MOT model this is a *separate class* (not a degenerate
``BoxTrackDataset``): a still image is not a one-frame video, and MAITE OD is a
distinct protocol from MAITE MOT. Source-preserving records
(:mod:`datamaite.records`) stay on the object so converters can export
faithfully.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datamaite._types import DatasetFormat, Task
from datamaite.records import DatasetMetadata, ImageObjectDetectionSample


@dataclass(frozen=True)
class ObjectDetectionDataset:
    """A loaded still-image OD dataset that *is* a MAITE object-detection dataset.

    ``samples`` are the source-preserving per-image records every converter
    consumes. ``dataset_metadata`` carries the category :class:`~datamaite.taxonomy.Taxonomy`
    plus dataset-level provenance (COCO ``info``/``licenses``). ``dataset_id`` is
    the MAITE ``DatasetMetadata['id']``.
    """

    samples: tuple[ImageObjectDetectionSample, ...]
    dataset_metadata: DatasetMetadata = field(default_factory=DatasetMetadata)
    dataset_id: str = "datamaite"
    # Task marker for parity with BoxTrackDataset (Task.MOT) and
    # VideoClassificationDataset (Task.VC); task-aware writer/convert dispatch
    # keys on it.
    task: Task = Task.OD

    def __post_init__(self) -> None:
        if not isinstance(self.samples, tuple):
            object.__setattr__(self, "samples", tuple(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        sample = self.samples[index]  # IndexError past the end -> stops iteration
        try:
            from datamaite.maite._od import build_od_item
        except ImportError as exc:
            if "build_od_item" in str(exc):
                raise
            raise ImportError(
                "Indexing a datamaite OD dataset as a MAITE dataset requires the optional "
                "image stack. Install it with: pip install datamaite[od]"
            ) from exc
        return build_od_item(sample)

    def get_input(self, index: int, /) -> Any:
        """MAITE ``FieldwiseDataset.get_input``: the decoded image for ``index``."""
        from datamaite.maite._od import od_input

        return od_input(self.samples[index])

    def get_target(self, index: int, /) -> Any:
        """MAITE ``FieldwiseDataset.get_target``: the OD target for ``index`` (no image decode)."""
        from datamaite.maite._od import od_target

        return od_target(self.samples[index])

    def get_metadata(self, index: int, /) -> dict[str, Any]:
        """MAITE ``FieldwiseDataset.get_metadata``: datum metadata for ``index`` (decodes only if dims unknown)."""
        from datamaite.maite._od import od_metadata

        return od_metadata(self.samples[index])

    @property
    def metadata(self) -> dict[str, Any]:
        """MAITE ``DatasetMetadata``: dataset id + ``index2label`` map."""
        return {"id": self.dataset_id, "index2label": self.index2label()}

    def index2label(self) -> dict[int, str]:
        """Map integer ``category_id`` to label name (from the taxonomy; empty if none)."""
        taxonomy = self.dataset_metadata.taxonomy
        return taxonomy.index2label() if taxonomy is not None else {}

    @property
    def sample_count(self) -> int:
        """Number of image samples (alias of ``len(self)``, for parity with the MOT model)."""
        return len(self.samples)

    def iter_samples(self) -> Iterator[ImageObjectDetectionSample]:
        """Iterate the typed source records (not the decoded MAITE items)."""
        return iter(self.samples)

    @property
    def num_detections(self) -> int:
        """Total detections across all images."""
        return sum(len(s.detections) for s in self.samples)


def load_od(
    root: str | Path,
    *,
    dataset_format: DatasetFormat | str = DatasetFormat.COCO,
    registry_variant: str = "default",
    **options: Any,
) -> ObjectDetectionDataset:
    """Load a still-image object-detection dataset (task-first entry point).

    The OD analogue of :func:`datamaite.loaders.load_mot`: pins the return
    type to :class:`ObjectDetectionDataset` (a native MAITE object-detection
    dataset) and dispatches through the shared loader registry by wire
    ``dataset_format`` (COCO or YOLO today). Asking for a non-OD format raises
    ``TypeError``. ``**options`` are forwarded to the format loader (e.g.
    COCO's ``annotation_file`` / ``images_dir``).
    """
    # Imported here, not at module level: loaders -> model -> this module is
    # the import chain that builds the VisionDataset union, so a module-level
    # import back into loaders would be circular.
    from datamaite.loaders import _require_dataset_root, _warn_if_empty, get_loader

    _require_dataset_root(root)
    try:
        loader = get_loader(dataset_format, task=Task.OD, variant=registry_variant)
    except ValueError as task_error:
        try:
            loader = get_loader(dataset_format, variant=registry_variant)
        except ValueError:
            raise task_error from None
    dataset = loader.load(root, **options)
    if not isinstance(dataset, ObjectDetectionDataset):
        raise TypeError(f"load_od expected an ObjectDetectionDataset, got {type(dataset).__name__}")
    _warn_if_empty(dataset, root, dataset_format)
    return dataset
