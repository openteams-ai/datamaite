"""Loader architecture: the contract every dataset loader implements.

datamaite is an N-to-M bridge. A *loader* reads one on-disk format/variant
for one task and produces that task's source-preserving in-memory dataset
(``BoxTrackDataset`` for MOT, ``ObjectDetectionDataset`` for OD,
``ImageClassificationDataset`` for IC, ``VideoClassificationDataset`` for VC).
A writer does the inverse. This module defines the input side:

* :class:`Loader` -- the base class every loader subclasses;
* :func:`register_loader` -- the extension point that adds a task/format/variant;
* :func:`load` -- the entry point that dispatches across registered loaders.

Registrations are keyed by ``(Task, DatasetFormat, variant)``. The task and
variant axes matter because a single wire format family can serve multiple tasks
(e.g. YOLO classification vs future YOLO object detection; Hugging Face video
classification vs future image classification) without those loaders clobbering
each other. Adding an input format means writing a ``Loader`` subclass and
registering it; nothing else in the package changes. See ``docs/architecture.md``
-> "Adding a new loader".

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
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from datamaite._types import DatasetFormat, Task
from datamaite.model import BoxTrackDataset, VisionDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoaderKey:
    """Registry key for one task/format/layout variant."""

    task: Task
    format: DatasetFormat
    variant: str = "default"


class Loader(ABC):
    """Contract for reading one task/format/variant into an in-memory dataset."""

    #: Vision task this loader produces. Existing FMV loaders default to MOT.
    task: ClassVar[Task] = Task.MOT
    #: Input wire format this loader handles. Every concrete subclass sets this.
    format: ClassVar[DatasetFormat]
    #: Layout/model-family discriminator within ``(task, format)``.
    variant: ClassVar[str] = "default"

    @abstractmethod
    def load(self, root: str | Path, **options: Any) -> VisionDataset:
        """Read the dataset under ``root`` into an in-memory dataset.

        Best-effort by contract: unparseable items are skipped and logged,
        not raised; an empty dataset is returned when nothing loadable is
        found. ``options`` are loader-specific keyword arguments.
        """
        raise NotImplementedError

    @classmethod
    def sniff(cls, root: str | Path) -> bool:  # noqa: ARG003 - root is part of the hook contract; default ignores it
        """Return True if ``root`` looks like this loader's format/variant."""
        return False


_LoaderT = TypeVar("_LoaderT", bound=Loader)
_LOADERS: dict[LoaderKey, type[Loader]] = {}
_BUILTIN_LOADER_MODULES = (
    "datamaite._formats.coco.loader",
    "datamaite._formats.flat_mp4.loader",
    "datamaite._formats.hmie.loader",
    "datamaite._formats.huggingface_video_classification.loader",
    "datamaite._formats.motchallenge.loader",
    "datamaite._formats.tao.loader",
    "datamaite._formats.visdrone.loader",
    "datamaite._formats.yolo.classification",
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


def _coerce_task(task: Task | str | None) -> Task | None:
    if task is None or isinstance(task, Task):
        return task
    return Task(str(task).lower())


def _coerce_format(dataset_format: DatasetFormat | str) -> DatasetFormat:
    return dataset_format if isinstance(dataset_format, DatasetFormat) else DatasetFormat(str(dataset_format).lower())


def _key_for(loader_cls: type[Loader]) -> LoaderKey:
    fmt = getattr(loader_cls, "format", None)
    if not isinstance(fmt, DatasetFormat):
        raise TypeError(f"{loader_cls.__name__} must set `format` to a DatasetFormat to be registered")
    task = getattr(loader_cls, "task", Task.MOT)
    if not isinstance(task, Task):
        raise TypeError(f"{loader_cls.__name__} must set `task` to a Task to be registered")
    variant = str(getattr(loader_cls, "variant", "default") or "default")
    return LoaderKey(task=task, format=fmt, variant=variant)


def register_loader(loader_cls: type[_LoaderT]) -> type[_LoaderT]:
    """Register ``loader_cls`` under ``(task, format, variant)``.

    Raises ``ValueError`` if a *different* class is already registered under the
    same key, so a duplicate ``(task, format, variant)`` fails loudly instead of
    silently shadowing the existing loader. Re-registering the same class (e.g.
    a module re-import) stays idempotent.
    """
    if loader_cls.__module__ not in _BUILTIN_LOADER_MODULES:
        _ensure_builtin_loaders()
    key = _key_for(loader_cls)
    existing = _LOADERS.get(key)
    if existing is not None and existing is not loader_cls:
        raise ValueError(f"A loader is already registered for {key}: {existing.__name__}")
    _LOADERS[key] = loader_cls
    return loader_cls


def available_formats(*, task: Task | str | None = None) -> list[DatasetFormat]:
    """Formats that currently have a registered loader, sorted by value.

    Pass ``task=...`` to list formats for one task only. Without a task, each
    format appears once even when multiple task variants are registered.
    """
    _ensure_builtin_loaders()
    resolved_task = _coerce_task(task)
    formats = {key.format for key in _LOADERS if resolved_task is None or key.task == resolved_task}
    return sorted(formats, key=lambda f: f.value)


def available_loader_keys() -> list[LoaderKey]:
    """Registered loader keys, sorted for diagnostics/tests."""
    _ensure_builtin_loaders()
    return sorted(
        _LOADERS,
        key=lambda key: (key.task.value, key.format.value, key.variant),
    )


def get_loader(
    dataset_format: DatasetFormat | str,
    *,
    task: Task | str | None = None,
    variant: str = "default",
) -> Loader:
    """Return a loader instance for ``dataset_format``/``task``/``variant``.

    ``task`` is optional for backwards compatibility. If omitted, the format
    must have exactly one registered loader; otherwise a ``ValueError`` asks the
    caller to use the task-first wrappers (``load_mot`` / ``load_od`` /
    ``load_ic``) or pass ``task=...`` explicitly.
    """
    _ensure_builtin_loaders()
    fmt = _coerce_format(dataset_format)
    resolved_task = _coerce_task(task)
    resolved_variant = str(variant or "default")

    if resolved_task is not None:
        key = LoaderKey(task=resolved_task, format=fmt, variant=resolved_variant)
        try:
            return _LOADERS[key]()
        except KeyError:
            if resolved_variant == "default":
                same_task = [
                    candidate
                    for candidate in available_loader_keys()
                    if candidate.task == resolved_task and candidate.format == fmt
                ]
                if len(same_task) == 1:
                    return _LOADERS[same_task[0]]()
            known = ", ".join(f"{k.task.value}:{k.format.value}:{k.variant}" for k in available_loader_keys())
            raise ValueError(f"No loader registered for {key}; available: {known or '(none)'}") from None

    candidates = [
        key
        for key in available_loader_keys()
        if key.format == fmt and (resolved_variant == "default" or key.variant == resolved_variant)
    ]
    if len(candidates) == 1:
        return _LOADERS[candidates[0]]()
    if not candidates:
        known = ", ".join(f.value for f in available_formats()) or "(none)"
        raise ValueError(f"No loader registered for format {fmt.value!r}; available: {known}")
    choices = ", ".join(f"task={key.task.value!r}, variant={key.variant!r}" for key in candidates)
    raise ValueError(f"Multiple loaders registered for format {fmt.value!r}; specify task/variant ({choices})")


def load(
    root: str | Path,
    *,
    dataset_format: DatasetFormat | str | None = DatasetFormat.HMIE,
    task: Task | str | None = None,
    registry_variant: str = "default",
    **options: Any,
) -> VisionDataset:
    """Load a dataset of any registered format into an in-memory model.

    ``registry_variant`` selects the loader registry variant. A plain
    ``variant=...`` keyword remains a loader option for formats such as
    VisDrone, preserving the pre-existing API.

    A missing or non-directory ``root`` is a caller error (e.g. a typo'd reload
    path), distinct from a valid-but-empty dataset: loaders are best-effort
    about *data* (skip unparseable items, warn, return empty), but a bad root
    fails loudly here rather than silently yielding an empty dataset.
    """
    _require_dataset_root(root)
    if dataset_format is None:
        resolved_format, resolved_task, resolved_variant = _detect_format(root, task=task, variant=registry_variant)
    else:
        resolved_format, resolved_task, resolved_variant = dataset_format, task, registry_variant
    dataset = get_loader(resolved_format, task=resolved_task, variant=resolved_variant).load(root, **options)
    _warn_if_empty(dataset, root, resolved_format)
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


def _warn_if_empty(dataset: VisionDataset, root: str | Path, fmt: DatasetFormat | str | None) -> None:
    """Warn when a valid root yielded no loadable items (shared by the loaders).

    Keeps the "bad root raises, empty-but-valid warns" contract identical across
    the generic ``load`` and the task-first ``load_mot``/``load_od``/``load_ic``
    helpers.
    """
    if _is_empty(dataset):
        fmt_value = fmt.value if isinstance(fmt, DatasetFormat) else fmt
        logger.warning(
            "Loaded an empty dataset from %s (format=%s): the root exists but no loadable items were found "
            "(wrong format, wrong subdirectory, or no matching data)",
            root,
            fmt_value,
        )


def load_mot(
    root: str | Path,
    *,
    dataset_format: DatasetFormat | str = DatasetFormat.HMIE,
    registry_variant: str = "default",
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
    _require_dataset_root(root)
    try:
        loader = get_loader(dataset_format, task=Task.MOT, variant=registry_variant)
    except ValueError:
        # Fall back to the format's sole loader so a wrong-task format (e.g. COCO,
        # which is OD) loads and then fails the BoxTrackDataset check below with a
        # clear "load_mot expected a BoxTrackDataset" message.
        loader = get_loader(dataset_format, variant=registry_variant)
    dataset = loader.load(root, **options)
    if not isinstance(dataset, BoxTrackDataset):
        raise TypeError(f"load_mot expected a BoxTrackDataset, got {type(dataset).__name__}")
    _warn_if_empty(dataset, root, dataset_format)
    return dataset


def _detect_format(
    root: str | Path,
    *,
    task: Task | str | None = None,
    variant: str = "default",
) -> tuple[DatasetFormat, Task, str]:
    """Pick a format by asking each registered loader to sniff ``root``."""
    _ensure_builtin_loaders()
    resolved_task = _coerce_task(task)
    requested_variant = str(variant or "default")
    for key in available_loader_keys():
        if resolved_task is not None and key.task != resolved_task:
            continue
        if requested_variant != "default" and key.variant != requested_variant:
            continue
        if _LOADERS[key].sniff(root):
            return key.format, key.task, key.variant
    known = ", ".join(f"{k.task.value}:{k.format.value}:{k.variant}" for k in available_loader_keys()) or "(none)"
    raise ValueError(
        f"Could not autodetect dataset format for {root!r}; pass dataset_format explicitly (available: {known})"
    )
