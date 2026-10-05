"""IR-1-H-3: MAITE components are advertised via package entry points."""

from __future__ import annotations

from importlib.metadata import distribution, entry_points

import pytest

from datamaite.image_classification import ImageClassificationDataset
from datamaite.model import BoxTrackDataset
from datamaite.object_detection import ObjectDetectionDataset

EXPECTED = [
    ("maite.protocols.multiobject_tracking.Dataset", "datamaite_BoxTrackDataset", BoxTrackDataset),
    ("maite.protocols.object_detection.Dataset", "datamaite_ObjectDetectionDataset", ObjectDetectionDataset),
    (
        "maite.protocols.image_classification.Dataset",
        "datamaite_ImageClassificationDataset",
        ImageClassificationDataset,
    ),
]


@pytest.mark.parametrize(("group", "name", "cls"), EXPECTED, ids=[name for _, name, _ in EXPECTED])
def test_entry_point_resolves_to_dataset_class(group: str, name: str, cls: type) -> None:
    (ep,) = entry_points(group=group, name=name)
    assert ep.load() is cls


def test_all_datamaite_entry_points_are_prefixed() -> None:
    # Every maite.protocols.* entry this distribution declares must be
    # namespaced, and must be one we expect (no stale or typo'd entries).
    # distribution().entry_points is a flat EntryPoint list on every supported
    # Python; bare entry_points() is a group-keyed mapping on 3.10/3.11.
    ours = [ep for ep in distribution("datamaite").entry_points if ep.group.startswith("maite.protocols.")]
    assert {(ep.group, ep.name) for ep in ours} == {(g, n) for g, n, _ in EXPECTED}
    assert all(ep.name.startswith("datamaite_") for ep in ours)
