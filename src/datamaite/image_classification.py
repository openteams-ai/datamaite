"""Still-image classification dataset -- a native MAITE IC dataset.

The IC sibling of :class:`datamaite.object_detection.ObjectDetectionDataset` and
:class:`datamaite.model.BoxTrackDataset`. It is a task-specific model, not a
one-frame FMV/MOT surrogate: source records preserve image media, split, and
image-level labels while ``__getitem__`` exposes the MAITE image-classification
protocol lazily.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

    def __post_init__(self) -> None:
        if not isinstance(self.samples, tuple):
            object.__setattr__(self, "samples", tuple(self.samples))

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
        return build_ic_item(sample, self.dataset_metadata.taxonomy)

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

    Currently the concrete IC reader is YOLO/Ultralytics classification
    folder layout. Additional IC formats should return this same dataset model.
    """
    from datamaite.loaders import _require_dataset_root, _warn_if_empty, get_loader

    _require_dataset_root(root)
    try:
        loader = get_loader(dataset_format, task=Task.IC, variant=registry_variant)
    except ValueError as task_error:
        try:
            loader = get_loader(dataset_format, variant=registry_variant)
        except ValueError:
            raise task_error from None
    dataset = loader.load(root, **options)
    if not isinstance(dataset, ImageClassificationDataset):
        raise TypeError(f"load_ic expected an ImageClassificationDataset, got {type(dataset).__name__}")
    _warn_if_empty(dataset, root, dataset_format)
    return dataset
