"""Cross-format input/output conversion matrix against the example-data repo.

These are **opt-in integration tests**. They run datamaite end-to-end against
a real checkout of the shared example-data repo
(``jatic/orchestration-interoperability/datamaite-example-datasets``) rather than
the synthetic ``tmp_path`` fixtures the unit suite uses.

To run locally, point ``DATAMAITE_DATASETS_ROOT`` at the ``datasets/``
directory of an example-data checkout and select the marker::

    git clone https://gitlab.jatic.net/jatic/orchestration-interoperability/datamaite-example-datasets.git
    export DATAMAITE_DATASETS_ROOT=$PWD/datamaite-example-datasets/datasets
    poetry run pytest -m integration -s     # -s shows the summary report

Skipped by default: the marker is deselected in ``pyproject.toml`` *and* every
test skips when the env var is unset, so the hermetic unit suite and the 90%
coverage gate are unaffected.

What each matrix cell validates
-------------------------------
For every ``input -> output`` conversion the test asserts, after a
``convert() -> reload`` round-trip:

* **counts** -- ``sequence_count`` and ``num_boxes`` match the values pinned in
  the example-data ``MANIFEST.md``; and
* **geometry** -- the *multiset of bounding-box coordinates* ``(left, top,
  width, height)`` is identical before and after. This is the strong,
  format-independent invariant: every box must survive, in the same place,
  with none dropped, moved, or duplicated.

What it deliberately does **not** check: JSON byte/hash equality, frame-index
numbering, or track-id values -- those are legitimately format-specific (e.g.
HMIE remaps frame numbers by fps/afr, and track ids are reassigned per format),
so asserting them would produce false failures, not real ones.

A compact summary table is printed once at the end of the run (visible in CI,
which passes ``-s``).
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest

from datamaite import load, load_ic, load_od, write
from datamaite.conversion import convert
from datamaite.image_classification import ImageClassificationDataset
from datamaite.loaders import load_mot
from datamaite.model import BoxTrackDataset, VideoClassificationDataset
from datamaite.object_detection import ObjectDetectionDataset

pytestmark = pytest.mark.integration

# input id -> (dataset dir relative to DATAMAITE_DATASETS_ROOT, dataset_format,
#              expected sequence_count, expected num_boxes)
# The id is decoupled from dataset_format so one format can appear as more than
# one fixture -- e.g. the synthetic `hmie/valid` and the real-MEVA
# `hmie/real-meva` set both load with dataset_format="hmie".
# Counts are pinned to the example-data repo's MANIFEST.md.
MOT_INPUTS: dict[str, tuple[str, str, int, int]] = {
    "hmie": ("hmie/valid", "hmie", 3, 30),
    "hmie_real_meva": ("hmie/real-meva", "hmie", 1, 120),
    "motchallenge": ("motchallenge/valid", "motchallenge", 1, 5),
    "visdrone_video": ("visdrone/valid", "visdrone_video", 1, 3),
    "tao": ("tao/valid", "tao", 1, 2),
    "flat_mp4": ("flat_mp4/valid", "flat_mp4", 2, 0),
}

# Output formats that produce a reloadable BoxTrackDataset.
BOX_OUTPUTS = ["motchallenge", "tao", "visdrone_video", "hmie"]

# Inputs that actually carry boxes (flat_mp4 has none -> excluded from the
# conversion matrix; still covered by the load test above).
BOX_INPUTS = [name for name, (_, _, _, boxes) in MOT_INPUTS.items() if boxes > 0]

BBox = tuple[float, float, float, float]

# (in_fmt, out_fmt, seqs, boxes, geometry_preserved) rows for the end-of-run
# report. Module-level so the session fixture can render one table at teardown.
_REPORT: list[tuple[str, str, int, int, bool]] = []


def _bbox_multiset(dataset: BoxTrackDataset) -> Counter[BBox]:
    """Multiset of rounded ``(left, top, width, height)`` over every box.

    The conversion-stable invariant: box geometry must survive a round-trip
    even when frame numbering or track ids are reassigned by the target format.
    """
    counts: Counter[BBox] = Counter()
    for seq in dataset.sequences:
        for box in seq.boxes:
            left, top, width, height = box.bbox
            counts[(round(left, 1), round(top, 1), round(width, 1), round(height, 1))] += 1
    return counts


@pytest.fixture(scope="session")
def datasets_root() -> Path:
    """Path to the example-data ``datasets/`` dir, or skip when unavailable."""
    root = os.environ.get("DATAMAITE_DATASETS_ROOT")
    if not root:
        pytest.skip("DATAMAITE_DATASETS_ROOT not set")
    path = Path(root)
    if not path.is_dir():
        pytest.skip(f"DATAMAITE_DATASETS_ROOT does not exist: {path}")
    return path


@pytest.fixture(scope="session", autouse=True)
def _matrix_report() -> Iterator[None]:
    """Print one compact summary table after the matrix runs (CI passes -s)."""
    yield
    if not _REPORT:
        return
    cells = {(i, o): ok for i, o, _, _, ok in _REPORT}
    meta = {i: (s, b) for i, o, s, b, _ in _REPORT}
    line = "=" * 78
    print(f"\n{line}\nexample-data conversion matrix: counts + box-geometry preservation\n{line}")
    print(f"{'input (seqs/boxes)':24s}" + "".join(f"{'-> ' + o:>18s}" for o in BOX_OUTPUTS))
    for in_fmt in BOX_INPUTS:
        if in_fmt not in meta:
            continue
        seqs, boxes = meta[in_fmt]
        row = "".join(f"{('geom OK' if cells.get((in_fmt, o)) else 'FAIL'):>18s}" for o in BOX_OUTPUTS)
        print(f"{f'{in_fmt} ({seqs}/{boxes})':24s}{row}")
    passed = sum(1 for v in cells.values() if v)
    print("-" * 78)
    print(f"{passed}/{len(cells)} conversions preserved counts + bounding-box geometry (L,T,W,H).")
    print("not asserted: JSON hash, frame-index numbering, track-id values (format-specific).")
    print(line)


@pytest.mark.parametrize("fmt", list(MOT_INPUTS), ids=list(MOT_INPUTS))
def test_input_loads_with_expected_counts(datasets_root: Path, fmt: str) -> None:
    """Each input format loads from the dataset data with its pinned counts."""
    rel, dataset_format, exp_seqs, exp_boxes = MOT_INPUTS[fmt]
    dataset = load_mot(datasets_root / rel, dataset_format=dataset_format)
    assert dataset.sequence_count == exp_seqs, f"{fmt}: sequence_count {dataset.sequence_count} != {exp_seqs}"
    assert dataset.num_boxes == exp_boxes, f"{fmt}: num_boxes {dataset.num_boxes} != {exp_boxes}"


@pytest.mark.parametrize("out_fmt", BOX_OUTPUTS, ids=lambda v: f"to-{v}")
@pytest.mark.parametrize("in_fmt", BOX_INPUTS, ids=lambda v: f"from-{v}")
def test_conversion_preserves_counts_and_geometry(
    datasets_root: Path, tmp_path: Path, in_fmt: str, out_fmt: str
) -> None:
    """input -> convert -> reload preserves counts AND bounding-box geometry."""
    rel, dataset_format, exp_seqs, exp_boxes = MOT_INPUTS[in_fmt]
    source_geometry = _bbox_multiset(load_mot(datasets_root / rel, dataset_format=dataset_format))

    dest = tmp_path / f"{in_fmt}__to__{out_fmt}"
    convert(datasets_root / rel, dest, input_format=dataset_format, output_format=out_fmt)
    reloaded = load_mot(dest, dataset_format=out_fmt)

    geometry_ok = _bbox_multiset(reloaded) == source_geometry
    _REPORT.append((in_fmt, out_fmt, reloaded.sequence_count, reloaded.num_boxes, geometry_ok))

    assert reloaded.num_boxes == exp_boxes, f"{in_fmt}->{out_fmt}: box count {reloaded.num_boxes} != {exp_boxes}"
    assert reloaded.sequence_count == exp_seqs, (
        f"{in_fmt}->{out_fmt}: seq count {reloaded.sequence_count} != {exp_seqs}"
    )
    assert geometry_ok, f"{in_fmt}->{out_fmt}: bounding-box geometry (L,T,W,H) not preserved through round-trip"


# Still-image OD / IC and video-classification coverage (issue #56). Each format
# with a same-format writer gets a load -> write -> reload round-trip asserting
# the task's invariants; #56 explicitly keeps cross-task conversions out of this
# matrix, so these are *same-format* round-trips (coco->coco, yolo->yolo,
# hf->hf). Formats with a loader but no same-format writer (visdrone_static's
# OD/IC layout) stay load-only -- real-data load coverage without a round-trip.
# Counts are pinned to the example-data repo's MANIFEST.md.

# id -> (dataset dir, dataset_format, expected images, expected detections)
OD_ROUNDTRIP: dict[str, tuple[str, str, int, int]] = {
    "coco": ("coco/valid", "coco", 7, 11),
}
OD_LOAD_ONLY: dict[str, tuple[str, str, int, int]] = {
    "visdrone_static": ("visdrone_static/valid", "visdrone", 7, 11),
}
# id -> (dataset dir, dataset_format, expected samples)
IC_ROUNDTRIP: dict[str, tuple[str, str, int]] = {
    "yolo": ("yolo/valid", "yolo", 11),
}
IC_LOAD_ONLY: dict[str, tuple[str, str, int]] = {
    "visdrone_static": ("visdrone_static/valid", "visdrone", 11),
}
# id -> (dataset dir, dataset_format, expected samples)
VC_ROUNDTRIP: dict[str, tuple[str, str, int]] = {
    "hf_video_cls": ("hf_video_cls/valid", "huggingface_video_classification", 8),
}


def _od_boxes(dataset: ObjectDetectionDataset) -> Counter[BBox]:
    """Multiset of rounded ``(left, top, width, height)`` over every detection."""
    counts: Counter[BBox] = Counter()
    for sample in dataset.iter_samples():
        for det in sample.detections:
            left, top, width, height = det.bbox
            counts[(round(left, 1), round(top, 1), round(width, 1), round(height, 1))] += 1
    return counts


def _od_categories(dataset: ObjectDetectionDataset) -> Counter[str]:
    return Counter(det.category_name for s in dataset.iter_samples() for det in s.detections)


def _od_taxonomy(dataset: ObjectDetectionDataset) -> list[tuple[int | None, str]]:
    """Ordered ``(source_id, name)`` for every taxonomy category, used or not.

    Compared directly (not via the categories referenced by detections) so a
    writer dropping unused categories is caught -- the fixture defines 12 but
    only 3 are annotated.
    """
    taxonomy = dataset.dataset_metadata.taxonomy
    return [(entry.source_id, entry.name) for entry in taxonomy.entries] if taxonomy else []


def _ic_associations(dataset: ImageClassificationDataset) -> list[tuple[str, str, str | None]]:
    """Per-sample ``(image basename, class, split)`` -- binds each identity to its
    label and split, so a reassignment among samples is caught (aggregate counts
    alone would not). Basename is the stable identity; the writer rewrites the
    directory prefix on round-trip."""
    return sorted(
        (Path(s.file_name).name, s.labels[0].category_name, s.split)
        for s in dataset.iter_samples()
        if s.labels and s.file_name
    )


def _vc_associations(dataset: VideoClassificationDataset) -> list[tuple[str, str, str | None]]:
    """Per-sample ``(video basename, label, split)`` -- binds each identity to its
    label and split (aggregate label/split counters would miss a reassignment)."""
    return sorted((Path(s.file_name).name, s.label, s.split) for s in dataset.iter_samples() if s.file_name)


@pytest.mark.parametrize("fmt", list(OD_LOAD_ONLY), ids=list(OD_LOAD_ONLY))
def test_od_loads_with_expected_counts(datasets_root: Path, fmt: str) -> None:
    """OD formats with no same-format writer: load with pinned image + detection counts."""
    rel, dataset_format, exp_images, exp_detections = OD_LOAD_ONLY[fmt]
    dataset = load_od(datasets_root / rel, dataset_format=dataset_format)
    assert dataset.sample_count == exp_images, f"{fmt}: images {dataset.sample_count} != {exp_images}"
    assert dataset.num_detections == exp_detections, f"{fmt}: detections {dataset.num_detections} != {exp_detections}"


@pytest.mark.parametrize("fmt", list(OD_ROUNDTRIP), ids=list(OD_ROUNDTRIP))
def test_od_roundtrip_preserves_counts_and_geometry(datasets_root: Path, tmp_path: Path, fmt: str) -> None:
    """OD: load -> write -> reload preserves image/detection/category counts + bbox geometry (#56)."""
    rel, dataset_format, exp_images, exp_detections = OD_ROUNDTRIP[fmt]
    source = load_od(datasets_root / rel, dataset_format=dataset_format)
    assert source.sample_count == exp_images
    assert source.num_detections == exp_detections

    dest = tmp_path / f"{fmt}__roundtrip"
    write(source, dest, output_format=dataset_format)
    reloaded = load_od(dest, dataset_format=dataset_format)

    assert reloaded.sample_count == exp_images, f"{fmt}: images {reloaded.sample_count} != {exp_images}"
    assert reloaded.num_detections == exp_detections, f"{fmt}: detections {reloaded.num_detections} != {exp_detections}"
    assert _od_categories(reloaded) == _od_categories(source), f"{fmt}: per-category detection counts changed"
    assert _od_taxonomy(reloaded) == _od_taxonomy(source), f"{fmt}: taxonomy categories changed (order/id/name)"
    assert _od_boxes(reloaded) == _od_boxes(source), f"{fmt}: bbox geometry (L,T,W,H) not preserved through round-trip"


@pytest.mark.parametrize("fmt", list(IC_LOAD_ONLY), ids=list(IC_LOAD_ONLY))
def test_ic_loads_with_expected_counts(datasets_root: Path, fmt: str) -> None:
    """IC formats with no same-format writer: load with pinned sample count."""
    rel, dataset_format, exp_samples = IC_LOAD_ONLY[fmt]
    dataset = load_ic(datasets_root / rel, dataset_format=dataset_format)
    assert dataset.sample_count == exp_samples, f"{fmt}: samples {dataset.sample_count} != {exp_samples}"


@pytest.mark.parametrize("fmt", list(IC_ROUNDTRIP), ids=list(IC_ROUNDTRIP))
def test_ic_roundtrip_preserves_counts_classes_splits(datasets_root: Path, tmp_path: Path, fmt: str) -> None:
    """IC: load -> write -> reload preserves sample count, class names/order, and splits (#56)."""
    rel, dataset_format, exp_samples = IC_ROUNDTRIP[fmt]
    source = load_ic(datasets_root / rel, dataset_format=dataset_format)
    assert source.sample_count == exp_samples

    dest = tmp_path / f"{fmt}__roundtrip"
    write(source, dest, output_format=dataset_format)
    reloaded = load_ic(dest, dataset_format=dataset_format)

    assert reloaded.sample_count == exp_samples, f"{fmt}: samples {reloaded.sample_count} != {exp_samples}"
    assert reloaded.index2label() == source.index2label(), f"{fmt}: class names/order changed"
    assert _ic_associations(reloaded) == _ic_associations(source), (
        f"{fmt}: per-sample (image, class, split) associations changed"
    )


@pytest.mark.parametrize("fmt", list(VC_ROUNDTRIP), ids=list(VC_ROUNDTRIP))
def test_vc_roundtrip_preserves_counts_labels_splits(datasets_root: Path, tmp_path: Path, fmt: str) -> None:
    """VC: load -> write -> reload preserves sample count, labels, and splits (#56)."""
    rel, dataset_format, exp_samples = VC_ROUNDTRIP[fmt]
    source = load(datasets_root / rel, dataset_format=dataset_format)
    assert source.sample_count == exp_samples

    dest = tmp_path / f"{fmt}__roundtrip"
    write(source, dest, output_format=dataset_format)
    reloaded = load(dest, dataset_format=dataset_format)

    assert reloaded.sample_count == exp_samples, f"{fmt}: samples {reloaded.sample_count} != {exp_samples}"
    assert _vc_associations(reloaded) == _vc_associations(source), (
        f"{fmt}: per-sample (video, label, split) associations changed"
    )
