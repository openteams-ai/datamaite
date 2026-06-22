"""On-disk dataset conversion: read format A, write format B.

The end-to-end orchestration of the N-to-M bridge. :func:`convert` pairs a
box-track *loader* (:mod:`databridge.loaders`) with a *writer*
(:mod:`databridge.writers`) via the neutral
:class:`databridge.model.BoxTrackDataset` hub: it reads a dataset from disk in
one format and writes it back out in another, without the caller wiring the
in-memory model by hand.

Because both halves bind to ``BoxTrackDataset`` and not to each other, any
registered box-track input format can be converted to any registered box-track
output format -- adding a MOT loader or writer extends the matrix without
touching this module. Task siblings such as ``VideoClassificationDataset`` need
their own writer surface before conversion is enabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from databridge._types import DatasetFormat
from databridge.loaders import load
from databridge.model import BoxTrackDataset
from databridge.writers import write


def convert(
    src: str | Path,
    dest: str | Path,
    *,
    input_format: DatasetFormat | str,
    output_format: DatasetFormat | str,
    read_options: dict[str, Any] | None = None,
    **write_options: Any,
) -> list[Path] | None:
    """Convert the dataset at ``src`` (``input_format``) to ``dest`` (``output_format``).

    Reads ``src`` with the registered loader for ``input_format``. The loaded
    dataset must be a :class:`~databridge.model.BoxTrackDataset`; task siblings
    without writer support raise ``TypeError``.

    Parameters
    ----------
    src, dest
        Source dataset root and destination directory.
    input_format, output_format
        Registered input/output formats (a :class:`DatasetFormat` or its string
        value). Both are required keyword arguments -- for a format-neutral A->B
        bridge, defaulting the source would be an asymmetric, surprising contract.
    read_options
        Keyword options forwarded to the loader (e.g. HMIE's ``require_video``,
        ``annotation_dir``).
    **write_options
        Keyword options forwarded to the writer, including ``verbose``: when
        ``True`` the files written are returned, when ``False`` (default) the
        conversion runs for side effects and returns ``None``.

    Returns
    -------
    list[Path] | None
        The files written when ``verbose=True`` is passed; otherwise ``None``.
    """
    dataset = load(src, dataset_format=input_format, **(read_options or {}))
    if not isinstance(dataset, BoxTrackDataset):
        raise TypeError(
            "convert currently supports box-track datasets only; "
            f"{type(dataset).__name__} has no registered writer surface"
        )
    return write(dataset, dest, output_format=output_format, **write_options)
