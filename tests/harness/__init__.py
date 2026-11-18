"""Conversion test harness: one definition of "the same dataset" for every format.

Use it to check that loading, writing or converting a dataset keeps
everything reachable from it, unless a path is excluded or the format declares
the difference lost::

    from tests.harness import assert_same, loss_stages, snapshot

    original = snapshot(load_od(src, dataset_format="coco"))
    converted = snapshot(load_od(out, dataset_format="yolo"))
    assert_same(original, converted, allowed=loss_stages(["coco", "yolo"]))

See ``tests/README.md`` ("Conversion harness") for how it works and how a
format joins it.
"""

from .compare import Loss, assert_same, compare
from .formats import FORMATS, FormatSpec, harness_formats, loss_stages
from .output import output_problems
from .paths import assert_no_source_paths, find_source_paths
from .snapshot import BoxSnapshot, DatasetSnapshot, SampleSnapshot, canonical, problems, snapshot

__all__ = [
    "FORMATS",
    "BoxSnapshot",
    "DatasetSnapshot",
    "FormatSpec",
    "Loss",
    "SampleSnapshot",
    "assert_no_source_paths",
    "assert_same",
    "canonical",
    "compare",
    "find_source_paths",
    "harness_formats",
    "loss_stages",
    "output_problems",
    "problems",
    "snapshot",
]
