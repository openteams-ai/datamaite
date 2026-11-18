"""Checks that an operation leaves its inputs alone: in-memory datasets, and files outside its destination."""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Iterator
from pathlib import Path

from datamaite.object_detection import ObjectDetectionDataset

from .compare import compare
from .output import tree_digest
from .snapshot import DatasetSnapshot, snapshot


def input_snapshots(**datasets: ObjectDetectionDataset) -> dict[str, tuple[ObjectDetectionDataset, DatasetSnapshot]]:
    """Each in-memory input with its snapshot now, before any operation runs."""
    return {name: (dataset, snapshot(dataset)) for name, dataset in datasets.items()}


def changed_inputs(inputs: dict[str, tuple[ObjectDetectionDataset, DatasetSnapshot]]) -> list[str]:
    """How each input now differs from its snapshot before the operation: no operation may change its input."""
    return [
        f"{name} dataset changed: {difference}"
        for name, (dataset, before) in inputs.items()
        # Only the data: the input's own problems are the scenario's, not the operation's.
        for difference in compare(before, dataclasses.replace(snapshot(dataset), problems=()))
    ]


@contextlib.contextmanager
def only_writes_to(root: Path, destination: str) -> Iterator[None]:
    """Fail if the block changes, adds or removes any file under ``root`` outside ``root / destination``.

    ``root`` holds the source folder (for a conversion) and a neighbouring
    folder, so a writer that writes beside its destination, or into its source,
    is caught.
    """
    neighbour = root / "neighbour"
    neighbour.mkdir(exist_ok=True)
    (neighbour / "keep.txt").write_text("not the writer's", encoding="utf-8")

    def outside() -> dict[str, str]:
        return {path: digest for path, digest in tree_digest(root).items() if not path.startswith(f"{destination}/")}

    before = outside()
    yield
    after = outside()
    changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
    if changed:
        raise AssertionError(f"the operation changed files outside {destination}/: {changed}")
