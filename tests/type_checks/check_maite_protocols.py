# Missing imports must fail here: with the project-wide "warning" level, a
# broken env (e.g. no maite extra) resolves the protocols as Unknown and the
# run passes having checked nothing.
# pyright: reportMissingImports=error
"""Static conformance of the dataset models to MAITE's dataset protocols (#124).

Not collected by pytest: pyright checks this file in CI (``pyright src/
tests/type_checks/``). MAITE's protocols are ``runtime_checkable``, but a runtime
``isinstance`` only checks member *presence*; these assignments check member
*types*, which is what downstream type-checked code relies on.
"""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from typing import Any

import numpy as np
from maite.protocols import image_classification as ic
from maite.protocols import multiobject_tracking as mot
from maite.protocols import object_detection as od
from typing_extensions import assert_type

from datamaite.image_classification import ImageClassificationDataset
from datamaite.maite import IcDatumMetadata, OdDatumMetadata
from datamaite.maite._decode import DecodedFrame
from datamaite.maite._mot import MotDatumMetadata, _MotTarget
from datamaite.maite._od import ObjectDetectionTarget
from datamaite.model import BoxTrackDataset
from datamaite.object_detection import ObjectDetectionDataset

od_dataset: od.Dataset = ObjectDetectionDataset(())
od_fieldwise: od.FieldwiseDataset = ObjectDetectionDataset(())
ic_dataset: ic.Dataset = ImageClassificationDataset(())
ic_fieldwise: ic.FieldwiseDataset = ImageClassificationDataset(())
mot_dataset: mot.Dataset = BoxTrackDataset((), {})


# Item types are concrete, not ``Any``: a regression to ``Any`` fails these.
def _item_types(od_ds: ObjectDetectionDataset, ic_ds: ImageClassificationDataset, mot_ds: BoxTrackDataset) -> None:
    image, od_target, od_meta = od_ds[0]
    assert_type(image, np.ndarray)
    assert_type(od_target, ObjectDetectionTarget)
    assert_type(od_meta, OdDatumMetadata)
    assert_type(od_meta["height"], int)
    assert_type(od_ds.get_input(0), np.ndarray)
    assert_type(od_ds.get_target(0), ObjectDetectionTarget)
    assert_type(od_ds.get_metadata(0), OdDatumMetadata)
    assert_type(od_ds.get_metadata(0).get("file_name"), str | None)
    # Undeclared, format-specific keys read as Any (PEP 728 extra_items), not an error.
    # pyright only: mypy does not implement extra_items yet and still flags this read.
    assert_type(od_meta["coco_url"], Any)

    ic_image, ic_target, ic_meta = ic_ds[0]
    assert_type(ic_image, np.ndarray)
    assert_type(ic_target, np.ndarray)
    assert_type(ic_meta, IcDatumMetadata)
    assert_type(ic_meta.get("split"), str | None)
    assert_type(ic_ds.get_input(0), np.ndarray)
    assert_type(ic_ds.get_target(0), np.ndarray)
    assert_type(ic_ds.get_metadata(0), IcDatumMetadata)

    stream, mot_target, mot_meta = mot_ds[0]
    assert_type(stream, Iterable[DecodedFrame])
    assert_type(mot_target, _MotTarget)
    assert_type(mot_meta, MotDatumMetadata)
    assert_type(mot_meta["time_base"], Fraction)
