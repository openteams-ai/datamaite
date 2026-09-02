"""Writer architecture: the contract every dataset writer implements.

The output-side mirror of :mod:`datamaite.loaders`. A *writer* takes a neutral
source-preserving dataset of one task and serialises it to one on-disk
format/variant. Registrations are keyed by ``(Task, DatasetFormat, variant)``
while dispatch is object-driven: ``write(dataset, output_format=...)`` infers
``Task`` from ``dataset.task`` and then type-checks the dataset against the
selected writer's ``consumes`` class. Cross-task conversions therefore raise
rather than fabricating data.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar
from uuid import uuid4

from datamaite._io import (
    copy_resource,
    is_within,
    remove_resource,
    remove_tree,
    runtime_storage_options,
    source_storage_context,
    write_mode_context,
)
from datamaite._types import DatasetFormat, Task, WriteMode
from datamaite._upath import is_remote_path, to_dataset_path
from datamaite.model import BoxTrackDataset, VisionDataset

_DatasetT = TypeVar("_DatasetT", bound=VisionDataset)


@dataclass(frozen=True)
class WriterKey:
    """Registry key for one task/format/layout variant."""

    task: Task
    format: DatasetFormat
    variant: str = "default"


@dataclass(frozen=True)
class WriterCapabilities:
    """A writer's declared re-emit contract (see architecture.md)."""

    required_fields: frozenset[str] = frozenset()
    lossy_without: dict[str, str] = field(default_factory=dict)
    forbids_dense_remap: bool = False
    emits_empty_label_files: bool = False


class Writer(ABC, Generic[_DatasetT]):
    """Contract for serialising one task's neutral model to one output format."""

    format: ClassVar[DatasetFormat]
    task: ClassVar[Task] = Task.MOT
    variant: ClassVar[str] = "default"
    consumes: ClassVar[type] = BoxTrackDataset
    capabilities: ClassVar[WriterCapabilities] = WriterCapabilities()

    @abstractmethod
    def write(self, dataset: _DatasetT, dest: str | Path, **options: Any) -> list[Path]:
        """Serialise ``dataset`` under ``dest`` and return the files written."""
        raise NotImplementedError

    def validate_options(self, **options: Any) -> None:  # noqa: ARG002 - options is part of the hook contract; default ignores it
        """Validate writer ``options`` before the destination policy runs.

        Called by :func:`write` BEFORE ``_prepare_destination`` so an invalid
        option raises without a ``mode="replace"`` clear having already deleted
        the destination. Default: no-op; writers whose options can raise should
        override this.
        """
        return

    def validate_dataset(self, dataset: VisionDataset) -> None:
        """Reject datasets this writer cannot serialise without silent corruption.

        Concrete on the base class (not an overridable no-op): SafeTensors-backed
        samples share a container ``path_or_uri``, so copy-by-file writers would
        duplicate the whole file. ``write()`` / ``convert()`` call this before
        the destination is touched. Direct ``Writer.write()`` bypasses it unless
        the caller invokes this method first.
        """
        samples = getattr(dataset, "samples", None)
        if samples is None:
            return
        for sample in samples:
            metadata = getattr(sample, "metadata", None) or {}
            if isinstance(metadata.get("safetensors_key"), str):
                location = getattr(sample, "path_or_uri", None) or getattr(sample, "file_name", None) or ""
                raise ValueError(
                    "Writing or converting SafeTensors-backed image samples is not supported. "
                    "Loading and MAITE indexing are supported. "
                    f"First offending sample: {getattr(sample, 'image_id', None)!r} ({location})"
                )


_WRITERS: dict[WriterKey, type[Writer[Any]]] = {}
_BUILTIN_WRITER_MODULES = (
    "datamaite._formats.coco.writer",
    "datamaite._formats.flat_images.writer",
    "datamaite._formats.flat_mp4.writer",
    "datamaite._formats.hmie.writer",
    "datamaite._formats.huggingface_video_classification.writer",
    "datamaite._formats.huggingface_vision.writer",
    "datamaite._formats.motchallenge.writer",
    "datamaite._formats.tao.writer",
    "datamaite._formats.visdrone.static_writer",
    "datamaite._formats.visdrone.writer",
    "datamaite._formats.yolo.writer",
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


def _coerce_task(task: Task | str | None) -> Task | None:
    if task is None or isinstance(task, Task):
        return task
    return Task(str(task).lower())


def _coerce_format(dataset_format: DatasetFormat | str) -> DatasetFormat:
    return dataset_format if isinstance(dataset_format, DatasetFormat) else DatasetFormat(str(dataset_format).lower())


_VALID_WRITE_MODES = ("error", "replace", "append")


def _validate_mode(mode: WriteMode | str) -> str:
    """Normalize ``mode`` to its lowercase string value.

    Accepts both :class:`~datamaite._types.WriteMode` members and plain
    strings (case-insensitive) so existing ``mode == "append"``-style
    comparisons downstream keep working regardless of which form a caller
    passes.
    """
    normalized = str(mode.value if isinstance(mode, WriteMode) else mode).strip().lower()
    if normalized not in _VALID_WRITE_MODES:
        raise ValueError(f"mode must be one of {sorted(_VALID_WRITE_MODES)!r}; got {mode!r}")
    return normalized


def _check_destination(dest: Path, mode: str) -> list[Path]:  # noqa: C901 - local/object safety branches
    """Validate the destination policy without touching anything on disk.

    This is the non-destructive half of :func:`_prepare_destination`: it raises
    the same errors (``NotADirectoryError`` for a non-directory ``dest``,
    ``FileExistsError`` for a non-empty ``dest`` under ``mode="error"``,
    ``ValueError`` for a ``mode="replace"`` destination that resolves to the
    filesystem root, or that contains (or equals) the current working
    directory or the home directory, or that is itself a symlink) but never
    deletes anything. It exists so callers such as :func:`datamaite.
    conversion.convert` can enforce the guardrail *before* paying the cost of
    loading the source dataset, without risking a deletion ahead of a load
    that might fail.
    """
    exists = dest.exists()
    is_directory = dest.is_dir() if exists else False
    if exists and not is_directory:
        raise NotADirectoryError(f"Destination {dest} exists and is not a directory")
    if not is_directory:
        return []
    # For mode="replace" the safety checks (protected paths + symlink alias)
    # must run BEFORE the empty-directory short-circuit below: an *empty*
    # symlinked destination is still an alias we refuse to write/clear through,
    # and refusing the fs root / cwd / home must not depend on the dir being
    # non-empty.
    if mode == "replace":
        if is_remote_path(dest):
            # Refuse clearing an entire bucket/container; object-store replace
            # is a non-atomic recursive prefix delete.
            remote_parts = tuple(part for part in dest.parts if part not in {"/", ""})
            if len(remote_parts) <= 1:
                raise ValueError(f"Refusing to replace an object-store root: {dest}")
        else:
            resolved = dest.resolve()
            cwd = Path.cwd().resolve()
            home = Path.home().resolve()
            # Refuse the filesystem root, and any directory that CONTAINS the cwd or home.
            if resolved == Path(resolved.anchor) or cwd.is_relative_to(resolved) or home.is_relative_to(resolved):
                raise ValueError(
                    f"Refusing to replace the contents of {dest} (resolves to {resolved}); it is the "
                    "filesystem root or contains the current working directory or home directory"
                )
            if dest.is_symlink():
                raise ValueError(f"Refusing to replace through a symlinked destination: {dest}")
    entries = list(dest.iterdir())
    if not entries:
        return []
    if mode == "append":
        return entries
    if mode == "error":
        raise FileExistsError(
            f"Destination {dest} already exists and is not empty. "
            "Pass mode='replace' to clear it first, or mode='append' to write into it "
            "(append may leave stale files that a reload of the destination would pick up)."
        )
    return entries


def _prepare_destination(dest: Path, mode: str) -> None:
    """Enforce the destination policy before any writer touches ``dest``.

    ``error`` refuses a non-empty destination directory, ``replace`` deletes
    its contents (never ``dest`` itself), and ``append`` writes into whatever
    is there. Replace refuses destinations that resolve to the filesystem
    root, the home directory, or the current working directory.

    Validation runs first via :func:`_check_destination` (see that function's
    docstring for why it is split out), then, for ``mode="replace"``, the
    actual clearing happens here.
    """
    entries = _check_destination(dest, mode)
    if not entries:
        return
    if mode != "replace":
        return
    for entry in entries:
        if entry.is_dir() and (is_remote_path(entry) or not entry.is_symlink()):
            remove_tree(entry)
        else:
            remove_resource(entry)


def _key_for(writer_cls: type[Writer[Any]]) -> WriterKey:
    fmt = getattr(writer_cls, "format", None)
    if not isinstance(fmt, DatasetFormat):
        raise TypeError(f"{writer_cls.__name__} must set `format` to a DatasetFormat to be registered")
    task = getattr(writer_cls, "task", Task.MOT)
    if not isinstance(task, Task):
        raise TypeError(f"{writer_cls.__name__} must set `task` to a Task to be registered")
    variant = str(getattr(writer_cls, "variant", "default") or "default")
    return WriterKey(task=task, format=fmt, variant=variant)


def register_writer(writer_cls: type[Writer[Any]]) -> type[Writer[Any]]:
    """Register ``writer_cls`` under ``(task, format, variant)``.

    Raises ``ValueError`` if a *different* class is already registered under the
    same key, so a duplicate ``(task, format, variant)`` fails loudly instead of
    silently shadowing the existing writer. Re-registering the same class (e.g.
    a module re-import) stays idempotent.
    """
    if writer_cls.__module__ not in _BUILTIN_WRITER_MODULES:
        _ensure_builtin_writers()
    key = _key_for(writer_cls)
    existing = _WRITERS.get(key)
    if existing is not None and existing is not writer_cls:
        raise ValueError(f"A writer is already registered for {key}: {existing.__name__}")
    _WRITERS[key] = writer_cls
    return writer_cls


def available_output_formats(*, task: Task | str | None = None) -> list[DatasetFormat]:
    """Formats that currently have a registered writer, sorted by value."""
    _ensure_builtin_writers()
    resolved_task = _coerce_task(task)
    formats = {key.format for key in _WRITERS if resolved_task is None or key.task == resolved_task}
    return sorted(formats, key=lambda f: f.value)


def available_writer_keys() -> list[WriterKey]:
    """Registered writer keys, sorted for diagnostics/tests."""
    _ensure_builtin_writers()
    return sorted(
        _WRITERS,
        key=lambda key: (key.task.value, key.format.value, key.variant),
    )


def get_writer(
    output_format: DatasetFormat | str,
    *,
    task: Task | str | None = None,
    variant: str = "default",
) -> Writer[Any]:
    """Return a writer instance for ``output_format``/``task``/``variant``.

    Calling the returned :class:`Writer` instance's ``.write()`` directly
    bypasses the destination policy (``mode``) enforced by the module-level
    :func:`write`/:func:`datamaite.conversion.convert`; use those functions
    instead when the destination policy should apply.
    """
    _ensure_builtin_writers()
    fmt = _coerce_format(output_format)
    resolved_task = _coerce_task(task)
    resolved_variant = str(variant or "default")

    if resolved_task is not None:
        key = WriterKey(task=resolved_task, format=fmt, variant=resolved_variant)
        try:
            return _WRITERS[key]()
        except KeyError:
            if resolved_variant == "default":
                same_task = [
                    candidate
                    for candidate in available_writer_keys()
                    if candidate.task == resolved_task and candidate.format == fmt
                ]
                if len(same_task) == 1:
                    return _WRITERS[same_task[0]]()
            known = ", ".join(f"{k.task.value}:{k.format.value}:{k.variant}" for k in available_writer_keys())
            raise ValueError(f"No writer registered for {key}; available: {known or '(none)'}") from None

    candidates = [
        key
        for key in available_writer_keys()
        if key.format == fmt and (resolved_variant == "default" or key.variant == resolved_variant)
    ]
    if len(candidates) == 1:
        return _WRITERS[candidates[0]]()
    if not candidates:
        known = ", ".join(f.value for f in available_output_formats()) or "(none)"
        raise ValueError(f"No writer registered for format {fmt.value!r}; available: {known}")
    choices = ", ".join(f"task={key.task.value!r}, variant={key.variant!r}" for key in candidates)
    raise ValueError(f"Multiple writers registered for format {fmt.value!r}; specify task/variant ({choices})")


def _present_attrs(obj: Any, attrs: tuple[str, ...]) -> list[str]:
    """Truthy values of ``attrs`` on ``obj`` (missing/empty attrs skipped)."""
    return [value for attr in attrs if (value := getattr(obj, attr, None))]


def _sequence_media_paths(sequence: Any) -> list[str]:
    """Media/annotation paths a MOT ``VideoSequence`` references."""
    paths = _present_attrs(sequence, ("video_path", "frame_dir", "annotation_path"))
    paths.extend(frame_file for frame_file in (getattr(sequence, "frame_files", ()) or ()) if frame_file)
    return paths


def _sample_media_paths(sample: Any) -> list[str]:
    """Media/annotation paths an image/video sample record references."""
    return _present_attrs(sample, ("path_or_uri", "video_path", "metadata_path"))


def _resolve_quietly(value: str, storage_options: Any = None) -> Path | None:
    try:
        path = to_dataset_path(value, storage_options)
        return path if is_remote_path(path) else path.resolve()
    except (OSError, ValueError):
        return None


def _dataset_source_paths(dataset: VisionDataset) -> list[Path]:
    """Resolved local paths a dataset's lazy media/annotations point at.

    Writers read these *after* ``_prepare_destination`` runs, so a
    ``mode="replace"`` clear of a destination that contains any of them would
    delete the writer's own inputs mid-write. Collected by duck-typing across
    the task models -- MOT ``VideoSequence`` media/frames/annotations and
    image/video sample ``path_or_uri`` / ``video_path`` / ``metadata_path`` --
    so new dataset types are covered as long as they reuse those field names.
    Byte-backed records (no path) contribute nothing and are safe.
    """
    raw: list[str] = []
    sequences = getattr(dataset, "sequences", None)
    samples = getattr(dataset, "samples", None)
    if sequences is not None:
        for seq in sequences:
            raw.extend(_sequence_media_paths(seq))
    elif samples is not None:
        for sample in samples:
            raw.extend(_sample_media_paths(sample))
    options = runtime_storage_options(dataset)
    return [resolved for resolved in (_resolve_quietly(value, options) for value in raw) if resolved is not None]


def _reject_source_under_destination(dataset: VisionDataset, dest: Path) -> None:
    """Refuse a replace-mode write whose dest contains the dataset's own inputs.

    ``write()`` clears the destination before the writer reads the dataset's
    lazy media, so if any referenced path lives inside (or equals) ``dest`` the
    clear destroys the writer's inputs -- e.g. ``write(load(p), p,
    mode="replace")`` round-tripping a loaded dataset onto its own directory.
    Enforced here (not only in :func:`datamaite.conversion.convert`) so the
    module-level ``write()`` API is protected too.
    """
    dest_display = dest if is_remote_path(dest) else dest.resolve()
    for source in _dataset_source_paths(dataset):
        if is_within(source, dest):
            raise ValueError(
                f"Refusing to replace {dest} (resolves to {dest_display}): the dataset's "
                f"media/annotations live inside it (e.g. {source}); clearing it would destroy "
                "the writer's own inputs mid-write. Write to a different directory."
            )


def write(
    dataset: VisionDataset,
    dest: str | Path,
    *,
    output_format: DatasetFormat | str,
    output_variant: str = "default",
    mode: WriteMode | str = WriteMode.ERROR,
    verbose: bool = False,
    storage_options: dict[str, Any] | None = None,
    source_storage_options: dict[str, Any] | None = None,
    **options: Any,
) -> list[Path] | None:
    """Write ``dataset`` to ``dest`` in ``output_format``.

    ``output_variant`` selects the writer registry variant. A plain
    ``variant=...`` keyword remains a writer option for formats such as
    VisDrone, preserving the pre-existing API.

    ``mode`` controls the destination policy. Accepts either a
    :class:`~datamaite._types.WriteMode` member or the equivalent string.
    ``"error"`` (default) raises ``FileExistsError`` when ``dest`` exists and
    is not empty; ``"replace"`` deletes the contents of ``dest`` *before* this
    call writes anything (refusing the filesystem root and any destination
    that contains or equals the home directory or the current working
    directory, and refusing a ``dest`` that is itself a symlink) -- so if the
    write fails or produces an empty dataset, ``dest`` is left emptied rather
    than restored; ``"append"`` writes into the existing destination, which
    may leave stale files behind that a reload of the destination would pick
    up. Calling a ``Writer`` instance's ``.write()`` directly bypasses this
    policy and :meth:`Writer.validate_dataset` (SafeTensors rejection) unless
    the caller invokes ``writer.validate_dataset(dataset)`` first. Writer-option
    validation (e.g. an invalid ``class_map`` or ``split``) runs before the
    destination is touched, so an invalid option raises without a
    ``mode="replace"`` clear having already deleted ``dest``.

    ``verbose``: when ``True``, return the list of files written; when ``False``
    (default) write for side effects and return ``None``. The full file list can
    be large (one path per frame image), so it is opt-in to keep
    interactive/REPL output quiet.
    """
    resolved_mode = _validate_mode(mode)
    task = getattr(dataset, "task", None)
    if not isinstance(task, Task):
        raise TypeError(f"Cannot infer writer task from {type(dataset).__name__}: missing Task-valued `task`")
    try:
        writer = get_writer(output_format, task=task, variant=output_variant)
    except ValueError as task_error:
        # Keep cross-task failures actionable when the requested output format
        # has a single writer for another task (e.g. MOT dataset -> COCO OD):
        # select that writer and let the consumes check raise TypeError. If the
        # format is itself multi-task (e.g. YOLO IC + OD), there is no single
        # cross-task writer to type-check against, so preserve the clearer
        # "no writer for this task" registry error.
        try:
            writer = get_writer(output_format, variant=output_variant)
        except ValueError:
            raise task_error from None
    if not isinstance(dataset, writer.consumes):
        raise TypeError(
            f"{writer.format.value!r} writer consumes {writer.consumes.__name__}; "
            f"got {type(dataset).__name__} (conversion is task-closed)"
        )
    # Validate writer options BEFORE the destination policy runs: a
    # mode="replace" clear must never happen ahead of an option error (#55
    # Fix A1). Direct Writer.write() calls still re-validate inline (cheap),
    # which also covers callers who bypass this module-level write().
    writer.validate_options(**options)
    writer.validate_dataset(dataset)
    resolved_dest = to_dataset_path(dest, storage_options)
    if resolved_mode == "replace":
        _reject_source_under_destination(dataset, resolved_dest)
    writer_options = dict(options)
    if storage_options is not None:
        writer_options["storage_options"] = storage_options
    with source_storage_context(dataset, source_storage_options), write_mode_context(resolved_mode):
        if is_remote_path(resolved_dest) and resolved_mode == "replace":
            files = _write_remote_staged_replace(
                writer,
                dataset,
                resolved_dest,
                writer_options=writer_options,
            )
        elif is_remote_path(resolved_dest) and resolved_mode == "error":
            files = _write_remote_error(writer, dataset, resolved_dest, writer_options=writer_options)
        else:
            _prepare_destination(resolved_dest, resolved_mode)
            files = writer.write(dataset, resolved_dest, **writer_options)
    return files if verbose else None


def _write_remote_error(
    writer: Writer[Any],
    dataset: VisionDataset,
    dest: Path,
    *,
    writer_options: dict[str, Any],
) -> list[Path]:
    """Stage an error-mode write without deleting unrelated final objects."""
    _check_destination(dest, "error")
    stage = dest.parent / f".{dest.name}.datamaite-stage-{uuid4().hex}"
    promoted: list[Path] = []
    try:
        staged_files = writer.write(dataset, stage, **writer_options)
        _check_destination(dest, "error")
        final_files = [dest.joinpath(*staged.relative_to(stage).parts) for staged in staged_files]
        promoted.extend(final_files)
        _copy_resource_pairs(list(zip(staged_files, final_files, strict=True)))
        return final_files
    except Exception:
        for path in promoted:
            with contextlib.suppress(OSError):
                remove_resource(path, missing_ok=True)
        raise
    finally:
        with contextlib.suppress(OSError):
            if stage.exists():
                remove_tree(stage)


def _remote_files(root: Path) -> list[Path]:
    """Inventory files below a remote prefix with one backend find call."""
    try:
        names = root.fs.find(root.path, withdirs=False)  # type: ignore[attr-defined]
    except (AttributeError, NotImplementedError, OSError):
        return [path for path in root.rglob("*") if path.is_file()]
    return [root.with_segments(name) for name in names]  # type: ignore[attr-defined]


def _copy_resource_pairs(pairs: list[tuple[Path, Path]]) -> None:
    """Copy independent objects with bounded concurrency."""
    if not pairs:
        return

    def copy_pair(pair: tuple[Path, Path]) -> None:
        copy_resource(pair[0], pair[1])

    with ThreadPoolExecutor(max_workers=min(8, len(pairs))) as executor:
        list(executor.map(copy_pair, pairs))


def _write_remote_staged_replace(
    writer: Writer[Any],
    dataset: VisionDataset,
    dest: Path,
    *,
    writer_options: dict[str, Any],
) -> list[Path]:
    """Stage replacement and retain a rollback copy until promotion succeeds."""
    _check_destination(dest, "replace")
    token = uuid4().hex
    stage = dest.parent / f".{dest.name}.datamaite-stage-{token}"
    backup = dest.parent / f".{dest.name}.datamaite-backup-{token}"
    backup_files: list[tuple[Path, Path]] = []
    destination_mutated = False
    promotion_complete = False
    rollback_complete = False
    try:
        stage_files = writer.write(dataset, stage, **writer_options)
        for original in _remote_files(dest) if dest.exists() else []:
            archived = backup.joinpath(*original.relative_to(dest).parts)
            backup_files.append((original, archived))
        _copy_resource_pairs([(original, archived) for original, archived in backup_files])
        destination_mutated = True
        _prepare_destination(dest, "replace")
        final_files = [dest.joinpath(*staged.relative_to(stage).parts) for staged in stage_files]
        _copy_resource_pairs(list(zip(stage_files, final_files, strict=True)))
        promotion_complete = True
        return final_files
    except Exception as original_error:
        if destination_mutated:
            try:
                if dest.exists():
                    remove_tree(dest)
                _copy_resource_pairs([(archived, original) for original, archived in backup_files])
                rollback_complete = True
            except Exception:
                raise RuntimeError(
                    f"remote replacement failed and rollback was incomplete; backup retained at {backup}"
                ) from original_error
        raise
    finally:
        with contextlib.suppress(OSError):
            if stage.exists():
                remove_tree(stage)
        if not destination_mutated or promotion_complete or rollback_complete:
            with contextlib.suppress(OSError):
                if backup.exists():
                    remove_tree(backup)
