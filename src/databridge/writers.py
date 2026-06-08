"""Writer architecture: the contract every dataset writer implements.

The output-side mirror of :mod:`databridge.loaders`. A *writer* takes the
neutral :class:`databridge.model.BoxTrackDataset` and serialises it to one
on-disk output format; a *loader* does the inverse. This module defines:

* :class:`Writer` -- the base class every writer subclasses;
* :func:`register_writer` -- the extension point that adds an output format;
* :func:`write` -- the entry point that dispatches across registered formats.

Adding an output format means writing a ``Writer`` subclass and registering
it; nothing else in the package changes. See ``docs/architecture.md`` ->
"Adding a new writer".

The common output contract
--------------------------
Every writer's *input* is the neutral ``BoxTrackDataset`` (so any loader's
output can feed any writer) and its *output* is the list of files it created
(so callers and the conversion layer can act on what was written). The on-disk
shape is format-specific; the ``BoxTrackDataset``-in / ``list[Path]``-out
contract is common to every writer.

Conventions every writer follows
---------------------------------
* **Consume the neutral model, never a loader or a raw format.** A writer's
  only inputs are a :class:`BoxTrackDataset` and a destination directory.
* **Map best-effort; drop with a warning, don't crash.** Data the target
  format cannot represent (e.g. tracks for a track-less format, unlabeled
  boxes for a class-required format) is dropped and logged at WARNING.
  *Destination / IO* failures (unwritable path, full disk) do raise.
* **Options are keyword-only**, documented per writer. Variant selection for a
  format with multiple flavours (e.g. MOT16 vs MOT20 column conventions) is a
  writer option, not a separate :class:`~databridge._types.DatasetFormat`.

The end-to-end orchestration that pairs a loader and a writer (read format A
from disk, write format B to disk) is :func:`databridge.conversion.convert`;
a writer itself only consumes an in-memory ``BoxTrackDataset``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar

from databridge._types import DatasetFormat
from databridge.model import BoxTrackDataset


class Writer(ABC):
    """Contract for serialising the neutral model to one output format.

    A writer subclass sets :attr:`format` to the :class:`DatasetFormat` it
    emits and implements :meth:`write`. Registering it with
    :func:`register_writer` lets :func:`write` dispatch to it by format.
    """

    #: Output format this writer emits. Every concrete subclass sets this.
    format: ClassVar[DatasetFormat]

    @abstractmethod
    def write(self, dataset: BoxTrackDataset, dest: str | Path, **options: Any) -> list[Path]:
        """Serialise ``dataset`` under ``dest`` and return the files written.

        Best-effort by contract: data the target format cannot represent is
        dropped and logged at WARNING, not raised; destination/IO failures do
        raise. ``options`` are writer-specific keyword arguments. ``dest`` is
        created if missing.
        """
        raise NotImplementedError


# Format -> writer-class registry. Populated by register_writer at import time
# (each format module decorates its Writer subclass). Built-in writers are
# imported lazily so validation-only imports do not pull writer code into memory.
_WRITERS: dict[DatasetFormat, type[Writer]] = {}
_BUILTIN_WRITER_MODULES = (
    "databridge._formats.hmie.writer",
    "databridge._formats.motchallenge.writer",
    "databridge._formats.tao.writer",
)
_BUILTIN_WRITERS_IMPORTED = False


def _ensure_builtin_writers() -> None:
    """Import built-in writer modules once so their decorators register."""
    global _BUILTIN_WRITERS_IMPORTED
    if _BUILTIN_WRITERS_IMPORTED:
        return
    for module_name in _BUILTIN_WRITER_MODULES:
        import_module(module_name)
    _BUILTIN_WRITERS_IMPORTED = True


def register_writer(writer_cls: type[Writer]) -> type[Writer]:
    """Register ``writer_cls`` under its :attr:`Writer.format`.

    Intended as a decorator on a :class:`Writer` subclass. Re-registering a
    format replaces the previous writer (last registration wins). Raises
    ``TypeError`` if the class does not set ``format`` to a
    :class:`DatasetFormat`.
    """
    if writer_cls.__module__ not in _BUILTIN_WRITER_MODULES:
        _ensure_builtin_writers()
    fmt = getattr(writer_cls, "format", None)
    if not isinstance(fmt, DatasetFormat):
        raise TypeError(f"{writer_cls.__name__} must set `format` to a DatasetFormat to be registered")
    _WRITERS[fmt] = writer_cls
    return writer_cls


def available_output_formats() -> list[DatasetFormat]:
    """Formats that currently have a registered writer, sorted by value."""
    _ensure_builtin_writers()
    return sorted(_WRITERS, key=lambda f: f.value)


def get_writer(output_format: DatasetFormat | str) -> Writer:
    """Return a writer instance for ``output_format``.

    Accepts a :class:`DatasetFormat` or its string value (case-insensitive).
    Raises ``ValueError`` for an unknown format string, or when no writer is
    registered for an otherwise-valid format.
    """
    _ensure_builtin_writers()
    fmt = output_format if isinstance(output_format, DatasetFormat) else DatasetFormat(str(output_format).lower())
    try:
        writer_cls = _WRITERS[fmt]
    except KeyError:
        known = ", ".join(f.value for f in available_output_formats()) or "(none)"
        raise ValueError(f"No writer registered for format {fmt.value!r}; available: {known}") from None
    return writer_cls()


def write(
    dataset: BoxTrackDataset,
    dest: str | Path,
    *,
    output_format: DatasetFormat | str,
    **options: Any,
) -> list[Path]:
    """Write ``dataset`` to ``dest`` in ``output_format`` and return the files written.

    Parameters
    ----------
    dataset
        The neutral :class:`~databridge.model.BoxTrackDataset` to serialise.
    dest
        Destination directory (created if missing).
    output_format
        Which output format to emit, as a :class:`DatasetFormat` or its string
        value.
    **options
        Forwarded to the selected writer's :meth:`Writer.write`.
    """
    return get_writer(output_format).write(dataset, dest, **options)
