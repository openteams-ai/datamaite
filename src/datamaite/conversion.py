"""On-disk dataset conversion: read format A, write format B.

:func:`convert` pairs a task-aware loader with a task-aware writer through the
neutral per-task model hub. Conversion is task-closed: the selected writer must
consume the dataset produced by the selected loader, otherwise ``write`` raises
``TypeError`` instead of fabricating cross-task data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datamaite._types import DatasetFormat, Task
from datamaite.loaders import load
from datamaite.writers import write


def convert(
    src: str | Path,
    dest: str | Path,
    *,
    input_format: DatasetFormat | str | None,
    output_format: DatasetFormat | str,
    task: Task | str | None = None,
    input_variant: str = "default",
    output_variant: str = "default",
    read_options: dict[str, Any] | None = None,
    load_options: dict[str, Any] | None = None,
    write_options: dict[str, Any] | None = None,
    verbose: bool = False,
    **writer_options: Any,
) -> list[Path] | None:
    """Convert ``src`` from one on-disk format into another.

    Parameters
    ----------
    src, dest
        Source dataset root and destination directory.
    input_format, output_format
        Registered input/output formats. Both are required keyword arguments;
        pass ``input_format=None`` to opt into loader sniffing. Defaulting only
        the source format would be asymmetric and surprising.
    task
        Optional task discriminator for shared format families (for example
        ``task="ic"`` with ``input_format="yolo"``).
    input_variant, output_variant
        Registry variants within a task/format pair.
    read_options
        Backwards-compatible loader options dict.
    load_options
        Alias for ``read_options``. Supplying both is an error.
    write_options, **writer_options
        Writer options as a dict and/or direct keyword arguments. Direct keyword
        options preserve the pre-existing public API.
    verbose
        When ``True``, return the list of files written; when ``False`` (default)
        run for side effects and return ``None`` (the file list can be one path
        per frame image, so it is opt-in to keep notebooks/REPLs quiet).
    """
    loader_options = _merge_loader_options(read_options=read_options, load_options=load_options)
    merged_writer_options = _merge_writer_options(write_options=write_options, writer_options=writer_options)
    dataset = load(
        src,
        dataset_format=input_format,
        task=task,
        registry_variant=input_variant,
        **loader_options,
    )
    return write(
        dataset,
        dest,
        output_format=output_format,
        output_variant=output_variant,
        verbose=verbose,
        **merged_writer_options,
    )


def _merge_loader_options(
    *,
    read_options: dict[str, Any] | None,
    load_options: dict[str, Any] | None,
) -> dict[str, Any]:
    if read_options is not None and load_options is not None:
        raise ValueError("pass either read_options or load_options, not both")
    return dict(read_options if read_options is not None else load_options or {})


def _merge_writer_options(
    *,
    write_options: dict[str, Any] | None,
    writer_options: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(write_options or {})
    overlap = set(merged).intersection(writer_options)
    if overlap:
        names = ", ".join(sorted(overlap))
        raise ValueError(f"writer option(s) supplied twice: {names}")
    merged.update(writer_options)
    return merged
