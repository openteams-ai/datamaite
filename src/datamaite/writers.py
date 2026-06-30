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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

from datamaite._types import DatasetFormat, Task
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


_WRITERS: dict[WriterKey, type[Writer[Any]]] = {}
_BUILTIN_WRITER_MODULES = (
    "datamaite._formats.coco.writer",
    "datamaite._formats.hmie.writer",
    "datamaite._formats.huggingface_video_classification.writer",
    "datamaite._formats.motchallenge.writer",
    "datamaite._formats.tao.writer",
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
    """Return a writer instance for ``output_format``/``task``/``variant``."""
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


def write(
    dataset: VisionDataset,
    dest: str | Path,
    *,
    output_format: DatasetFormat | str,
    output_variant: str = "default",
    verbose: bool = False,
    **options: Any,
) -> list[Path] | None:
    """Write ``dataset`` to ``dest`` in ``output_format``.

    ``output_variant`` selects the writer registry variant. A plain
    ``variant=...`` keyword remains a writer option for formats such as
    VisDrone, preserving the pre-existing API.

    ``verbose``: when ``True``, return the list of files written; when ``False``
    (default) write for side effects and return ``None``. The full file list can
    be large (one path per frame image), so it is opt-in to keep
    interactive/REPL output quiet.
    """
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
    files = writer.write(dataset, dest, **options)
    return files if verbose else None
