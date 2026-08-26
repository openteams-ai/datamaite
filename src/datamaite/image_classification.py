"""Still-image classification dataset -- a native MAITE IC dataset.

The IC sibling of :class:`datamaite.object_detection.ObjectDetectionDataset` and
:class:`datamaite.model.BoxTrackDataset`. It is a task-specific model, not a
one-frame FMV/MOT surrogate: source records preserve image media, split, and
image-level labels while ``__getitem__`` exposes the MAITE image-classification
protocol lazily.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import InitVar, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, ClassVar

from datamaite._io import probe_image_dimensions
from datamaite._types import DatasetFormat, Task
from datamaite.records import DatasetMetadata, ImageClassificationSample


@dataclass(frozen=True)
class ImageClassificationDataset:
    """A loaded still-image IC dataset that structurally satisfies MAITE IC.

    ``samples`` are source-preserving image-level records. ``dataset_metadata``
    carries a :class:`datamaite.taxonomy.Taxonomy`; its dense projection is used
    for MAITE's one-hot/probability target vector and ``index2label`` metadata.
    """

    samples: tuple[ImageClassificationSample, ...]
    dataset_metadata: DatasetMetadata = field(default_factory=DatasetMetadata)
    dataset_id: str = "datamaite"
    task: Task = Task.IC
    # Runtime-only credentials/backend configuration used to reopen string
    # ``path_or_uri`` values lazily. InitVar keeps secrets out of dataclass
    # fields, so repr/equality/asdict remain source-record based.
    _storage_options: InitVar[Mapping[str, Any] | None] = field(default=None, kw_only=True)
    _runtime_storage_options: ClassVar[Mapping[str, Any]]
    _base_image_cache: ClassVar[dict[str, Any]]

    def __post_init__(self, _storage_options: Mapping[str, Any] | None) -> None:
        if not isinstance(self.samples, tuple):
            object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "_runtime_storage_options", dict(_storage_options or {}))
        object.__setattr__(self, "_base_image_cache", {})

    def __getstate__(self) -> dict[str, Any]:
        """Serialize public dataset state, never credentials or decoded media."""
        return {item.name: getattr(self, item.name) for item in fields(self)}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Restore public state with process-local storage options intentionally unbound."""
        for name, value in state.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_runtime_storage_options", {})
        object.__setattr__(self, "_base_image_cache", {})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        sample = self.samples[index]  # IndexError past the end -> stops iteration
        try:
            from datamaite.maite._ic import build_ic_item
        except ImportError as exc:
            if "build_ic_item" in str(exc):
                raise
            raise ImportError(
                "Indexing a datamaite IC dataset as a MAITE dataset requires the optional "
                "image stack. Install it with: pip install datamaite[ic]"
            ) from exc
        return build_ic_item(
            sample,
            self.dataset_metadata.taxonomy,
            storage_options=self._runtime_storage_options,
            base_cache=self._base_image_cache,
        )

    def get_input(self, index: int, /) -> Any:
        """MAITE ``FieldwiseDataset.get_input``: a fresh image array for ``index``."""
        from datamaite.maite._ic import ic_input

        return ic_input(
            self.samples[index],
            storage_options=self._runtime_storage_options,
            base_cache=self._base_image_cache,
        )

    def get_target(self, index: int, /) -> Any:
        """MAITE ``FieldwiseDataset.get_target``: the class vector for ``index`` (no image decode)."""
        from datamaite.maite._ic import ic_target

        return ic_target(self.samples[index], self.dataset_metadata.taxonomy)

    def get_metadata(self, index: int, /) -> dict[str, Any]:
        """MAITE ``FieldwiseDataset.get_metadata``: datum metadata for ``index`` (decodes only if dims unknown)."""
        from datamaite.maite._ic import ic_metadata

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
        needs_image = sample.region is not None or (needs_dimensions and dimensions is None)
        image = self.get_input(index) if needs_image else None
        return ic_metadata(
            sample,
            image,
            dimensions=dimensions,
            storage_options=self._runtime_storage_options,
            base_cache=self._base_image_cache,
        )

    def with_storage_options(self, storage_options: Mapping[str, Any] | None) -> ImageClassificationDataset:
        """Return a copy bound to explicit process-local storage options."""
        return replace(self, _storage_options=storage_options)

    @property
    def metadata(self) -> dict[str, Any]:
        """MAITE ``DatasetMetadata``: dataset id + dense ``index2label`` map."""
        return {"id": self.dataset_id, "index2label": self.index2label()}

    def index2label(self) -> dict[int, str]:
        """Dense class index to label name map (empty if no taxonomy)."""
        taxonomy = self.dataset_metadata.taxonomy
        return taxonomy.dense_index2label() if taxonomy is not None else {}

    @property
    def sample_count(self) -> int:
        """Number of image samples (alias of ``len(self)``)."""
        return len(self.samples)

    def iter_samples(self) -> Iterator[ImageClassificationSample]:
        """Iterate the typed source records (not decoded MAITE items)."""
        return iter(self.samples)


def load_ic(
    root: str | Path,
    *,
    dataset_format: DatasetFormat | str = DatasetFormat.YOLO,
    registry_variant: str = "default",
    **options: Any,
) -> ImageClassificationDataset:
    """Load a still-image classification dataset (task-first entry point).

    Concrete IC readers include the YOLO/Ultralytics classification folder
    layout and VisDrone still images (object crops derived from the DET
    annotations). Additional IC formats should return this same dataset model.
    """
    from datamaite._upath import to_dataset_path
    from datamaite.loaders import _reject_unsupported_remote, _require_dataset_root, _warn_if_empty, get_loader

    try:
        loader = get_loader(dataset_format, task=Task.IC, variant=registry_variant)
    except ValueError as task_error:
        try:
            loader = get_loader(dataset_format, variant=registry_variant)
        except ValueError:
            raise task_error from None
    resolved_root = to_dataset_path(root, options.get("storage_options"))
    _reject_unsupported_remote(resolved_root, loader)
    _require_dataset_root(resolved_root)
    dataset = loader.load(resolved_root, **options)
    if not isinstance(dataset, ImageClassificationDataset):
        raise TypeError(f"load_ic expected an ImageClassificationDataset, got {type(dataset).__name__}")
    _warn_if_empty(dataset, root, dataset_format)
    return dataset
