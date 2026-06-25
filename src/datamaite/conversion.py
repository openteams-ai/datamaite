"""On-disk dataset conversion: read format A, write format B.

The end-to-end orchestration of the N-to-M bridge. :func:`convert` pairs a
*loader* (:mod:`datamaite.loaders`) with a compatible *writer*
(:mod:`datamaite.writers`): it reads a dataset from disk in one format and
writes it back out in another, without the caller wiring the in-memory model by
hand.

Writers declare the task dataset type they consume (for example
``BoxTrackDataset`` for MOT writers or ``VideoClassificationDataset`` for the
Hugging Face video-classification writer). ``convert`` deliberately delegates
that compatibility check to :func:`datamaite.write`, so task-compatible A→B
pairs work and cross-task conversions fail before a writer fabricates data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datamaite._types import DatasetFormat
from datamaite.loaders import load
from datamaite.writers import write


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

    Reads ``src`` with the registered loader for ``input_format`` and writes the
    loaded task dataset with the registered writer for ``output_format``. The
    writer's declared dataset type decides compatibility; incompatible task
    pairs raise ``TypeError``.

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
    return write(dataset, dest, output_format=output_format, **write_options)
