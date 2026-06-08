"""Loader architecture: the contract every dataset loader implements.

databridge is an N-to-M bridge. A *loader* reads a dataset of one input
format from disk and produces the neutral
:class:`databridge.model.BoxTrackDataset` model; a *converter* writes that
model out to an output format. This module defines the input-side
architecture:

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
  returns a (possibly empty) :class:`BoxTrackDataset` rather than aborting the
  load. The authoritative "*why* is this dataset bad" answer comes from
  :func:`databridge.validate`, which is deliberately a separate pass.
* **Options are keyword-only.** ``load(root, **options)`` -- each loader
  documents its own options (e.g. HMIE's ``require_video``). Where an option
  is shared across loaders (``require_video`` for any FMV format), keep the
  name and meaning consistent.
* **One model out.** Every loader produces the same :class:`BoxTrackDataset`,
  so any converter can consume the result regardless of which format it came
  from. :class:`BoxTrackDataset` is MAITE-MOT-capable by default; see
  :mod:`databridge.maite` for details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar

from databridge._types import DatasetFormat
from databridge.model import BoxTrackDataset


class Loader(ABC):
    """Contract for reading one input format into the neutral model.

    A loader subclass sets :attr:`format` to the :class:`DatasetFormat` it
    handles and implements :meth:`load`. Registering it with
    :func:`register_loader` lets :func:`load` dispatch to it by format.
    """

    #: Input format this loader handles. Every concrete subclass sets this.
    format: ClassVar[DatasetFormat]

    @abstractmethod
    def load(self, root: str | Path, **options: Any) -> BoxTrackDataset:
        """Read the dataset under ``root`` into a :class:`BoxTrackDataset`.

        Best-effort by contract: unparseable items are skipped and logged,
        not raised; an empty :class:`BoxTrackDataset` is returned when nothing
        loadable is found. ``options`` are loader-specific keyword arguments.
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
_LOADERS: dict[DatasetFormat, type[Loader]] = {}
_BUILTIN_LOADER_MODULES = (
    "databridge._formats.flat_mp4.loader",
    "databridge._formats.hmie.loader",
    "databridge._formats.motchallenge.loader",
    "databridge._formats.tao.loader",
    "databridge._formats.visdrone.loader",
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


def register_loader(loader_cls: type[Loader]) -> type[Loader]:
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
) -> BoxTrackDataset:
    """Load a dataset of any registered format into the neutral model.

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
    """
    resolved = _detect_format(root) if dataset_format is None else dataset_format
    return get_loader(resolved).load(root, **options)


def _detect_format(root: str | Path) -> DatasetFormat:
    """Pick a format by asking each registered loader to sniff ``root``."""
    for fmt in available_formats():
        if _LOADERS[fmt].sniff(root):
            return fmt
    known = ", ".join(f.value for f in available_formats()) or "(none)"
    raise ValueError(
        f"Could not autodetect dataset format for {root!r}; pass dataset_format explicitly (available: {known})"
    )
