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

from collections.abc import Iterator, Mapping
from dataclasses import InitVar, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, ClassVar

from datamaite._io import probe_image_dimensions
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
    # Runtime-only configuration for reopening remote media while public
    # records remain strings. InitVar prevents credentials from appearing in
    # repr/equality/asdict and the explicit pickle state excludes them too.
    _storage_options: InitVar[Mapping[str, Any] | None] = field(default=None, kw_only=True)
    _runtime_storage_options: ClassVar[Mapping[str, Any]]

    def __post_init__(self, _storage_options: Mapping[str, Any] | None) -> None:
        if not isinstance(self.samples, tuple):
            object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "_runtime_storage_options", dict(_storage_options or {}))

    def __getstate__(self) -> dict[str, Any]:
        """Serialize public dataset state, never credentials or decoded media."""
        return {item.name: getattr(self, item.name) for item in fields(self)}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Restore public state with process-local storage options intentionally unbound."""
        for name, value in state.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_runtime_storage_options", {})

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
        return build_od_item(sample, storage_options=self._runtime_storage_options)

    def get_input(self, index: int, /) -> Any:
        """MAITE ``FieldwiseDataset.get_input``: a freshly decoded image for ``index``."""
        from datamaite.maite._od import od_input

        return od_input(self.samples[index], storage_options=self._runtime_storage_options)

    def get_target(self, index: int, /) -> Any:
        """MAITE ``FieldwiseDataset.get_target``: the OD target for ``index`` (no image decode)."""
        from datamaite.maite._od import od_target

        return od_target(self.samples[index])

    def get_metadata(self, index: int, /) -> dict[str, Any]:
        """MAITE ``FieldwiseDataset.get_metadata``: datum metadata for ``index`` (decodes only if dims unknown)."""
        from datamaite.maite._od import od_metadata

        sample = self.samples[index]
        dimensions: tuple[int, int] | None = None
        needs_dimensions = sample.height is None or sample.width is None
        if needs_dimensions and sample.region is None:
            source = sample.image_bytes if sample.image_bytes is not None else sample.path_or_uri
            if source is not None:
                try:
                    dimensions = probe_image_dimensions(source, self._runtime_storage_options)
                except OSError:
                    dimensions = None
        image = self.get_input(index) if needs_dimensions and dimensions is None else None
        return od_metadata(
            sample,
            image,
            dimensions=dimensions,
            storage_options=self._runtime_storage_options,
        )

    def with_storage_options(self, storage_options: Mapping[str, Any] | None) -> ObjectDetectionDataset:
        """Return a copy bound to explicit process-local storage options."""
        return replace(self, _storage_options=storage_options)

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
    from datamaite._upath import to_dataset_path
    from datamaite.loaders import _reject_unsupported_remote, _require_dataset_root, _warn_if_empty, get_loader

    try:
        loader = get_loader(dataset_format, task=Task.OD, variant=registry_variant)
    except ValueError as task_error:
        try:
            loader = get_loader(dataset_format, variant=registry_variant)
        except ValueError:
            raise task_error from None
    resolved_root = to_dataset_path(root, options.get("storage_options"))
    _reject_unsupported_remote(resolved_root, loader)
    _require_dataset_root(resolved_root)
    dataset = loader.load(resolved_root, **options)
    if not isinstance(dataset, ObjectDetectionDataset):
        raise TypeError(f"load_od expected an ObjectDetectionDataset, got {type(dataset).__name__}")
    _warn_if_empty(dataset, root, dataset_format)
    return dataset
