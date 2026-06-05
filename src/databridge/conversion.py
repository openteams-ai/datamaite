"""On-disk dataset conversion: read format A, write format B.

The end-to-end orchestration of the N-to-M bridge. :func:`convert` pairs a
*loader* (:mod:`databridge.loaders`) with a *writer* (:mod:`databridge.writers`)
via the neutral :class:`databridge.model.BoxTrackDataset` hub: it reads a
dataset from disk in one format and writes it back out in another, without the
caller wiring the in-memory model by hand.

Because both halves bind to ``BoxTrackDataset`` and not to each other, any
registered input format can be converted to any registered output format --
adding a loader or a writer extends the matrix without touching this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from databridge._types import DatasetFormat
from databridge.loaders import load
from databridge.writers import write


def convert(
    src: str | Path,
    dest: str | Path,
    *,
    input_format: DatasetFormat | str,
    output_format: DatasetFormat | str,
    read_options: dict[str, Any] | None = None,
    **write_options: Any,
) -> list[Path]:
    """Convert the dataset at ``src`` (``input_format``) to ``dest`` (``output_format``).

    Reads ``src`` with the registered loader for ``input_format`` into a
    :class:`~databridge.model.BoxTrackDataset`, then writes it to ``dest`` with
    the registered writer for ``output_format``. Returns the files written.

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
        Keyword options forwarded to the writer.
    """
    dataset = load(src, dataset_format=input_format, **(read_options or {}))
    return write(dataset, dest, output_format=output_format, **write_options)
