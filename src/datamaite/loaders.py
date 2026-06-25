"""Loader architecture: the contract every dataset loader implements.

datamaite is an N-to-M bridge. A *loader* reads a dataset of one input
format from disk and produces a task-appropriate in-memory dataset (currently
:class:`datamaite.model.BoxTrackDataset` for MOT or
:class:`datamaite.model.VideoClassificationDataset` for video classification).
A *converter* writes supported in-memory datasets out to an output format. This
module defines the input-side architecture:

* :class:`Loader` -- the base class every loader subclasses;
* :func:`register_loader` -- the extension point that adds a format;
* :func:`load` -- the entry point that dispatches across registered formats.

Adding an input format means writing a ``Loader`` subclass and registering
it; nothing else in the package changes. See ``docs/architecture.md`` ->
"Adding a new loader".

Conventions every loader follows
--------------------------------
* **Return, don't raise, on bad data.** Loading is best-effort: an item that
  cannot be parsed is skipped and logged at WARNING, never fatal. A loader
  returns a (possibly empty) task-appropriate dataset rather than aborting the
  load. The authoritative "*why* is this dataset bad" answer comes from
  :func:`datamaite.validate`, which is deliberately a separate pass.
* **Options are keyword-only.** ``load(root, **options)`` -- each loader
  documents its own options (e.g. HMIE's ``require_video``). Where an option
  is shared across loaders (``require_video`` for any FMV format), keep the
  name and meaning consistent.
* **Task-appropriate models.** MOT loaders produce :class:`BoxTrackDataset`,
  which is MAITE-MOT-capable by default; video classification produces its own
  records because MAITE 0.9.5 has no video-classification protocol. See
  :mod:`datamaite.maite` for MOT details.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from datamaite._types import DatasetFormat
from datamaite.model import BoxTrackDataset, VisionDataset

logger = logging.getLogger(__name__)


class Loader(ABC):
    """Contract for reading one input format into the neutral model.

    A loader subclass sets :attr:`format` to the :class:`DatasetFormat` it
    handles and implements :meth:`load`. Registering it with
    :func:`register_loader` lets :func:`load` dispatch to it by format.
    """

    #: Input format this loader handles. Every concrete subclass sets this.
    format: ClassVar[DatasetFormat]

    @abstractmethod
    def load(self, root: str | Path, **options: Any) -> VisionDataset:
        """Read the dataset under ``root`` into an in-memory dataset.

        Best-effort by contract: unparseable items are skipped and logged,
        not raised; an empty dataset is returned when nothing loadable is
        found. ``options`` are loader-specific keyword arguments.
        """
        raise NotImplementedError

    @classmethod
    def sniff(cls, root: str | Path) -> bool:  # noqa: ARG003 - root is part of the hook contract; the default ignores it
        """Return True if ``root`` looks like this loader's format.

        Autodetection hook for ``load(root, dataset_format=None)``. The
        default returns ``False`` (makes no claim); a loader overrides it to
        opt into detection. No loader implements detection rules yet, so in
        practice callers pass an explicit ``dataset_format``.
        """
        return False


# Format -> loader-class registry. Populated by register_loader at import
# time (each format module decorates its Loader subclass). Built-in loaders are
# imported lazily so validation-only imports do not pull reader code into memory.
_LoaderT = TypeVar("_LoaderT", bound=Loader)
_LOADERS: dict[DatasetFormat, type[Loader]] = {}
_BUILTIN_LOADER_MODULES = (
    "datamaite._formats.flat_mp4.loader",
    "datamaite._formats.hmie.loader",
    "datamaite._formats.huggingface_video_classification.loader",
    "datamaite._formats.motchallenge.loader",
    "datamaite._formats.tao.loader",
    "datamaite._formats.visdrone.loader",
)
_BUILTIN_LOADERS_IMPORTED = False


def _ensure_builtin_loaders() -> None:
    """Import built-in loader modules once so their decorators register."""
    global _BUILTIN_LOADERS_IMPORTED
    if _BUILTIN_LOADERS_IMPORTED:
        return
    for module_name in _BUILTIN_LOADER_MODULES:
        import_module(module_name)
    _BUILTIN_LOADERS_IMPORTED = True


def register_loader(loader_cls: type[_LoaderT]) -> type[_LoaderT]:
    """Register ``loader_cls`` under its :attr:`Loader.format`.

    Intended as a decorator on a :class:`Loader` subclass. Re-registering a
    format replaces the previous loader (last registration wins). Raises
    ``TypeError`` if the class does not set ``format`` to a
    :class:`DatasetFormat`.
    """
    if loader_cls.__module__ not in _BUILTIN_LOADER_MODULES:
        _ensure_builtin_loaders()
    fmt = getattr(loader_cls, "format", None)
    if not isinstance(fmt, DatasetFormat):
        raise TypeError(f"{loader_cls.__name__} must set `format` to a DatasetFormat to be registered")
    _LOADERS[fmt] = loader_cls
    return loader_cls


def available_formats() -> list[DatasetFormat]:
    """Formats that currently have a registered loader, sorted by value."""
    _ensure_builtin_loaders()
    return sorted(_LOADERS, key=lambda f: f.value)


def get_loader(dataset_format: DatasetFormat | str) -> Loader:
    """Return a loader instance for ``dataset_format``.

    Accepts a :class:`DatasetFormat` or its string value (case-insensitive).
    Raises ``ValueError`` for an unknown format string, or when no loader is
    registered for an otherwise-valid format.
    """
    _ensure_builtin_loaders()
    fmt = dataset_format if isinstance(dataset_format, DatasetFormat) else DatasetFormat(str(dataset_format).lower())
    try:
        loader_cls = _LOADERS[fmt]
    except KeyError:
        known = ", ".join(f.value for f in available_formats()) or "(none)"
        raise ValueError(f"No loader registered for format {fmt.value!r}; available: {known}") from None
    return loader_cls()


def load(
    root: str | Path,
    *,
    dataset_format: DatasetFormat | str | None = DatasetFormat.HMIE,
    **options: Any,
) -> VisionDataset:
    """Load a dataset of any registered format into an in-memory model.

    Parameters
    ----------
    root
        Dataset root directory.
    dataset_format
        Which input format ``root`` is, as a :class:`DatasetFormat` or its
        string value. Pass ``None`` to autodetect via each loader's
        :meth:`Loader.sniff` (no format implements detection rules yet, so an
        explicit format is required in practice).
    **options
        Forwarded to the selected loader's :meth:`Loader.load` (e.g. HMIE's
        ``annotation_dir`` / ``video_dir`` / ``require_video``).

    Raises
    ------
    FileNotFoundError
        If ``root`` does not exist.
    NotADirectoryError
        If ``root`` exists but is not a directory.

    A missing or non-directory ``root`` is a caller error (e.g. a typo'd reload
    path), distinct from a valid-but-empty dataset: loaders are best-effort
    about *data* (skip unparseable items, warn, return empty), but a bad root
    fails loudly here rather than silently yielding an empty dataset.
    """
    _require_dataset_root(root)
    resolved = _detect_format(root) if dataset_format is None else dataset_format
    dataset = get_loader(resolved).load(root, **options)
    if _is_empty(dataset):
        fmt_value = resolved.value if isinstance(resolved, DatasetFormat) else resolved
        logger.warning(
            "Loaded an empty dataset from %s (format=%s): the root exists but no loadable items were found "
            "(wrong format, wrong subdirectory, or no matching data)",
            root,
            fmt_value,
        )
    return dataset


def _require_dataset_root(root: str | Path) -> None:
    """Raise if ``root`` is not an existing directory (a caller error)."""
    path = Path(root)
    if not path.exists():
        raise FileNotFoundError(f"dataset root does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"dataset root is not a directory: {path}")


def _is_empty(dataset: VisionDataset) -> bool:
    """Whether ``dataset`` carries no loadable content.

    Outcome-based, not ``len(dataset)``: ``BoxTrackDataset.__len__`` counts only
    *video-bearing* sequences, so an annotation-only dataset has ``len == 0``
    yet is not empty. A box-track dataset is empty only when it has no sequences
    at all; other task datasets fall back to ``len``.
    """
    if isinstance(dataset, BoxTrackDataset):
        return dataset.sequence_count == 0
    return len(dataset) == 0


def load_mot(
    root: str | Path,
    *,
    dataset_format: DatasetFormat | str = DatasetFormat.HMIE,
    **options: Any,
) -> BoxTrackDataset:
    """Load a video multi-object-tracking dataset (task-first entry point).

    The MOT analogue of :func:`datamaite.object_detection.load_od`: pins the
    return type to :class:`BoxTrackDataset` (a native MAITE multi-object-tracking
    dataset) and dispatches by wire ``dataset_format`` (HMIE, MOTChallenge, TAO,
    VisDrone video, flat MP4). ``**options`` are forwarded to the format loader
    (e.g. HMIE's ``require_video``). This is the public MOT loader; per-format
    helpers are internal to ``datamaite._formats.<format>.loader``.
    """
    dataset = load(root, dataset_format=dataset_format, **options)
    if not isinstance(dataset, BoxTrackDataset):
        raise TypeError(f"load_mot expected a BoxTrackDataset, got {type(dataset).__name__}")
    return dataset


def _detect_format(root: str | Path) -> DatasetFormat:
    """Pick a format by asking each registered loader to sniff ``root``."""
    for fmt in available_formats():
        if _LOADERS[fmt].sniff(root):
            return fmt
    known = ", ".join(f.value for f in available_formats()) or "(none)"
    raise ValueError(
        f"Could not autodetect dataset format for {root!r}; pass dataset_format explicitly (available: {known})"
    )
