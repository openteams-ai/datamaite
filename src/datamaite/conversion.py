"""On-disk dataset conversion: read format A, write format B.

:func:`convert` pairs a task-aware loader with a task-aware writer through the
neutral per-task model hub. Conversion is task-closed: the selected writer must
consume the dataset produced by the selected loader, otherwise ``write`` raises
``TypeError`` instead of fabricating cross-task data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datamaite._types import DatasetFormat, Task, WriteMode
from datamaite.loaders import load
from datamaite.writers import _check_destination, _validate_mode, write


def convert(
    src: str | Path,
    dest: str | Path,
    *,
    input_format: DatasetFormat | str | None,
    output_format: DatasetFormat | str,
    task: Task | str | None = None,
    input_variant: str = "default",
    output_variant: str = "default",
    mode: WriteMode | str = WriteMode.ERROR,
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
    mode
        Destination policy. Accepts a :class:`~datamaite._types.WriteMode`
        member or the equivalent string. ``"error"`` (default) raises ``FileExistsError``
        if destination is non-empty; ``"replace"`` clears the destination
        before anything is written (refusing destinations that resolve to
        the filesystem root, the home directory, or the current working
        directory); ``"append"`` writes into the existing destination, which
        may leave stale files behind that a reload of the destination would
        pick up. The destination policy is validated before ``src`` is
        loaded, so an invalid ``mode`` or a rejected ``dest`` raises before
        paying the cost of loading a large source dataset; the actual
        deletion for ``mode="replace"`` still happens only after a
        successful load, inside the writer.
    read_options
        Backwards-compatible loader options dict.
    load_options
        Alias for ``read_options``. Supplying both is an error.
    write_options, **writer_options
        Writer options as a dict and/or direct keyword arguments. Direct keyword
        options preserve the pre-existing public API. Passing ``mode`` in these
        dicts raises ``ValueError``.
    verbose
        When ``True``, return the list of files written; when ``False`` (default)
        run for side effects and return ``None`` (the file list can be one path
        per frame image, so it is opt-in to keep notebooks/REPLs quiet).
    """
    loader_options = _merge_loader_options(read_options=read_options, load_options=load_options)
    merged_writer_options = _merge_writer_options(write_options=write_options, writer_options=writer_options)
    if "mode" in merged_writer_options:
        raise ValueError("pass mode as a top-level convert() argument, not a writer option")
    resolved_mode = _validate_mode(mode)
    # Enforce the destination guardrail before loading `src`: a mistaken
    # convert(big_src, non_empty_dest) should fail fast rather than paying the
    # full load cost first. This check is non-destructive (no deletion) -- the
    # actual clearing for mode="replace" happens later, inside write(), only
    # after `src` has loaded successfully. That ordering matters: if we deleted
    # here and the load then failed, dest would be wiped for nothing.
    _check_destination(Path(dest), resolved_mode)
    if resolved_mode == "replace":
        src_resolved = Path(src).resolve()
        dest_resolved = Path(dest).resolve()
        # A mode="replace" clear of dest happens after the source loads but
        # before the writer reads the source's (lazy) media files. If dest is
        # the source, or an ancestor of it, that clear destroys the source
        # mid-conversion. resolve() collapses symlinks so an aliased dest is
        # caught too. (A dest *inside* src is safe: clearing a subdir does not
        # remove the source's own files.)
        if src_resolved.is_relative_to(dest_resolved):
            raise ValueError(
                f"Refusing to convert with mode='replace': the destination {dest} "
                f"(resolves to {dest_resolved}) is the source dataset or contains it "
                f"(source resolves to {src_resolved}); clearing it would destroy the source."
            )
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
        mode=mode,
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
