"""Conversion harness: YOLO and COCO load, write, convert and append (#150).

Each cell runs one operation on one synthetic scenario and compares the result
with the scenario's expected dataset using :mod:`tests.harness`:

* ``memory``    -- the input dataset as built in memory (catches bad input types).
* ``load``      -- a hand-written original on disk, loaded.
* ``roundtrip`` -- the input written with datamaite, then loaded.
* ``convert``   -- a hand-written original in format A, converted to B, then loaded.

Appends run separately, as the explicit cases in ``APPEND_CASES``.

Every difference is a failure unless a format in the cell declares it as a
loss it cannot avoid (``tests/harness/formats.py``). Known bugs are listed in
``KNOWN_BUGS`` with the differences they cause: a cell is an expected failure
only if its known bugs explain every difference, and fixing a bug fails its
cells until they are removed from the entry.
"""

from __future__ import annotations

import dataclasses
import fnmatch
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from datamaite import convert, load_od, write
from datamaite._types import DatasetFormat
from datamaite.object_detection import ObjectDetectionDataset

from .harness import (
    DatasetSnapshot,
    Loss,
    assert_no_source_paths,
    compare,
    loss_stages,
    snapshot,
)
from .harness.compare import expected_after_losses, without_classes
from .harness.formats import FORMATS, NOT_YET_COVERED, convertible_od_formats, harness_formats
from .harness.guards import changed_inputs, input_snapshots, only_writes_to
from .harness.known_bugs import APPENDED_IMAGES, CROSS_CHECKS, IGNORING_CLASSES, KnownBug, check_differences
from .harness.output import output_problems, relocate, tree_digest
from .harness.scenarios import APPEND_CASES, CONVERT, LOAD, MEMORY, ROUNDTRIP, SCENARIOS, AppendCase, Scenario
from .harness.snapshot import SampleSnapshot, edit

FORMATS_UNDER_TEST = harness_formats()


def _missing(*samples: str) -> str:
    """A pattern for exactly these samples (``"<key> [split <split>]"``) reported missing."""
    return rf"^({'|'.join(re.escape(sample) for sample in samples)}): missing \(\d+ box\(es\)\)$"


# Scenarios whose boxes the COCO writer drops altogether (#152, #153 and #160).
_NO_BOXES_WRITTEN_BY_COCO = ("numpy_ids", "numpy_taxonomy_ids", "no_taxonomy")

COCO_AREA_BUG = KnownBug(
    "#163",
    "the COCO writer leaves out area when a box has none, though COCO evaluation (pycocotools) needs it",
    (
        *(
            f"roundtrip-{scenario.name}-coco"
            for scenario in SCENARIOS
            if ROUNDTRIP in scenario.operations and scenario.name not in _NO_BOXES_WRITTEN_BY_COCO
        ),
        *(f"convert-{scenario.name}-yolo-coco" for scenario in SCENARIOS if CONVERT in scenario.operations),
        *(f"append-{case.name}-coco" for case in APPEND_CASES),
    ),
    (r": box '\w+' \(id '?\w+'?\) at \[[\d., ]+\]( \(crowd\))?: area expected \d+(\.\d+)?, got None$",),
)

# Cell ids are "<operation>-<scenario>-<format>" and "convert-<scenario>-<from>-<to>".
KNOWN_BUGS: tuple[KnownBug, ...] = (
    KnownBug(
        "#159",
        "NumPy integer taxonomy ids are left out of the MAITE index2label",
        ("memory-numpy_taxonomy_ids",),
        (
            r"^problem: .*: MAITE index2label maps \d+ to None$",
            r"^problem: taxonomy: class '(person|car|bus)' \(id \d\) is not in the MAITE index2label as '\w+'$",
        ),
    ),
    KnownBug(
        "#160",
        "the COCO writer drops categories whose taxonomy id is a NumPy integer",
        ("roundtrip-numpy_taxonomy_ids-coco",),
        (
            r"^labels \{id: name\}: expected \{0: 'person', 1: 'car', 2: 'bus'\}, got \{\}$",
            r": missing box '(person|car|bus)'",
            r": unexpected box None \(id [012]\)",
            r"^output: annotations/instances_train\.json: annotation \d refers to a missing category$",
        ),
    ),
    KnownBug(
        "#161",
        "the COCO writer writes a.png and ./a.png to one file, so the second image overwrites the first",
        ("roundtrip-filename_alias-coco",),
        (
            r"^a\.png \[split train\]: image_sha256 expected '\w+', got '\w+'$",
            # The two records now share one image, so they pair up crosswise.
            r"^a\.png \[split train\]: image_id expected [12], got [12]$",
            r"^a\.png \[split train\]: (missing|unexpected) box '(person|car)'",
        ),
    ),
    KnownBug(
        "#162",
        "the COCO loader drops category attributes, which the COCO writer emits",
        ("load-category_details-coco", "roundtrip-category_details-coco", "convert-category_details-coco-coco"),
        (r"^class 'car' \(id 1\): attributes\['keypoints'\] expected \['front', 'back'\], got <missing>$",),
    ),
    KnownBug(
        "#164",
        "COCO drops class synsets: the writer leaves them out and the loader does not read LVIS's synset key",
        ("load-record_details-coco", "roundtrip-record_details-coco", "convert-record_details-coco-coco"),
        (r"^class '(person|car)' \(id [01]\): synset expected '(person|car)\.n\.01', got None$",),
    ),
    COCO_AREA_BUG,
    KnownBug(
        "no issue yet",
        "the COCO writer numbers boxes from 1 in order, so a box with no id takes an id another box has",
        ("roundtrip-annotation_ids-coco",),
        (
            r"^a\.png \[split train\]: box 'car' \(id 1\) at \[20, 10, 10, 8\]"
            r": source_annotation_id expected 1, got 2$",
            r"^b\.png \[split train\]: box 'car' \(id 1\) at \[4, 4, 10, 8\]"
            r": source_annotation_id expected 2, got 3$",
        ),
    ),
    KnownBug(
        "#151",
        "COCO append replaces an existing annotation file, dropping the earlier images",
        tuple(f"append-{case.name}-coco" for case in APPEND_CASES if case.name != "nothing"),
        (
            r"^labels \{id: name\}: expected ",
            # a.png and b.png are the images written before the append.
            r"^[ab]\.png \[split train\]: missing \(\d box\(es\)\)$",
            r"^[ab]\.png \[split train\]: box classes expected \[.*\], got None$",
        ),
    ),
    KnownBug(
        "#152",
        "the COCO writer drops class names when the dataset has no taxonomy",
        ("roundtrip-no_taxonomy-coco",),
        (
            r"^labels \{id: name\}: expected \{2: 'two', 10: 'ten'\}, got \{\}$",
            r": missing box '(two|ten)'",
            r": unexpected box None \(id (2|10)\)",
            r"^output: annotations/instances_train\.json: annotation \d refers to a missing category$",
        ),
    ),
    KnownBug(
        "#153",
        "the COCO writer drops boxes whose class id is a NumPy integer",
        ("roundtrip-numpy_ids-coco",),
        (r": missing box '(person|car|bus)'",),
    ),
)


def _cells() -> list[tuple[str, str, Scenario, tuple[DatasetFormat, ...]]]:
    cells = []
    for scenario in SCENARIOS:
        for operation in scenario.operations:
            if operation == MEMORY:
                combos: list[tuple[DatasetFormat, ...]] = [()]
            elif operation == CONVERT:
                combos = [(a, b) for a in FORMATS_UNDER_TEST for b in FORMATS_UNDER_TEST]
            else:
                combos = [(fmt,) for fmt in FORMATS_UNDER_TEST]
            for formats in combos:
                cell_id = "-".join([operation, scenario.name, *(fmt.value for fmt in formats)])
                cells.append((cell_id, operation, scenario, formats))
    return cells


CELLS = _cells()


def _params() -> list:
    params = []
    for cell_id, operation, scenario, formats in CELLS:
        marks = [pytest.mark.optional] if scenario.needs_image_stack else []
        params.append(pytest.param(cell_id, operation, scenario, formats, id=cell_id, marks=marks))
    return params


def run_cell(operation: str, scenario: Scenario, formats: tuple[DatasetFormat, ...], tmp_path: Path) -> list[str]:
    """Run one cell and return every difference from the scenario's expected dataset."""
    source, expected = scenario.build()
    decode = scenario.needs_image_stack
    # Snapshot before anything runs: most scenarios' input and expected are one object, so a
    # writer that edits a list or dict of its input in place would otherwise edit the answer too.
    want = snapshot(expected, decode=decode)
    inputs = input_snapshots(input=source, expected=expected)
    written_problems: list[str] = []
    if operation == MEMORY:
        actual = source
    elif operation == LOAD:
        (fmt,) = formats
        actual = load_od(FORMATS[fmt].author(expected, tmp_path / "src"), dataset_format=fmt)
    elif operation == ROUNDTRIP:
        (fmt,) = formats
        with only_writes_to(tmp_path, "out"):
            write(source, tmp_path / "out", output_format=fmt)
        assert_no_source_paths(tmp_path / "out", tmp_path)
        out = relocate(tmp_path)
        written_problems = output_problems(out, fmt)
        actual = load_od(out, dataset_format=fmt)
    else:
        assert operation == CONVERT
        a, b = formats
        FORMATS[a].author(expected, tmp_path / "src")
        with only_writes_to(tmp_path, "out"):
            convert(tmp_path / "src", tmp_path / "out", input_format=a, output_format=b, task="od")
        assert_no_source_paths(tmp_path / "out", tmp_path)
        out = relocate(tmp_path)
        written_problems = output_problems(out, b)
        actual = load_od(out, dataset_format=b)
    # Each format's losses apply in turn: a conversion gives back what B makes of A's output.
    allowed = loss_stages(formats)
    differences = compare(want, snapshot(actual, decode=decode), allowed=allowed)
    return written_problems + changed_inputs(inputs) + differences


@pytest.mark.parametrize(("cell_id", "operation", "scenario", "formats"), _params())
def test_harness(
    cell_id: str, operation: str, scenario: Scenario, formats: tuple[DatasetFormat, ...], tmp_path: Path
) -> None:
    differences = run_cell(operation, scenario, formats, tmp_path)
    check_differences(differences, [bug for bug in KNOWN_BUGS if bug.applies_to(cell_id)])


# ---------------------------------------------------------------------------
# Appends
# ---------------------------------------------------------------------------

APPEND_CELLS = [(f"append-{case.name}-{fmt.value}", case, fmt) for case in APPEND_CASES for fmt in FORMATS_UNDER_TEST]


def run_append(case: AppendCase, fmt: DatasetFormat, tmp_path: Path, *, class_blind: bool = False) -> list[str]:
    """Write ``case``'s first dataset, append its second, and return every difference from its expected one.

    With ``class_blind``, also compare with every class left out, for a cell whose known bug
    only relabels boxes: each box must still be there, on its image and in its place.
    """
    first, second, expected = case.build()
    want = snapshot(expected)
    inputs = input_snapshots(first=first, second=second, expected=expected)
    out = tmp_path / "out"
    with only_writes_to(tmp_path, "out"):
        write(first, out, output_format=fmt)
    before = tree_digest(out)
    refused = False
    with only_writes_to(tmp_path, "out"):
        try:
            write(second, out, output_format=fmt, mode="append")
        except ValueError:
            if not case.names_only:
                raise
            refused = True
    if refused:
        # A writer may refuse an append whose class lists disagree, if it leaves the dataset as it was.
        after = tree_digest(out)
        changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
        return changed_inputs(inputs) + ([f"the refused append changed {changed}"] if changed else [])
    assert_no_source_paths(out, tmp_path)
    out = relocate(tmp_path)
    written_problems = output_problems(out, fmt)
    got = snapshot(load_od(out, dataset_format=fmt))
    allowed = FORMATS[fmt].losses
    if case.names_only:
        # Only the class ids may differ, so compare with every id replaced by its class name, after
        # the format's losses: the values a format derives from the ids, such as id_density, follow them.
        want, got, allowed = _ids_as_names(expected_after_losses(want, allowed)), _ids_as_names(got), set()
    differences = written_problems + changed_inputs(inputs) + compare(want, got, allowed=allowed)
    if class_blind:
        blind_want, blind_got = without_classes(expected_after_losses(want, allowed)), without_classes(got)
        differences += [f"{IGNORING_CLASSES}{d}" for d in compare(blind_want, blind_got)]
        # A bug that relabels the earlier boxes does not excuse a wrong class on an appended box.
        appended = {sample.key for sample in snapshot(second).samples}
        new_want = _class_names_on(expected_after_losses(want, allowed), appended)
        differences += [f"{APPENDED_IMAGES}{d}" for d in compare(new_want, _class_names_on(got, appended))]
    return differences


def _class_names_on(snap: DatasetSnapshot, keys: set[str]) -> DatasetSnapshot:
    """Only the samples with these keys, each box with its class name but no class id."""
    tree = snap.tree
    for field in ("category_id", "source_category_id"):
        edit(tree, f"samples[*].detections[*].{field}", lambda _value: None)
    samples = zip(snap.samples, tree["samples"], strict=True)
    return DatasetSnapshot(
        fields={}, samples=tuple(SampleSnapshot(s.key, fields) for s, fields in samples if s.key in keys)
    )


def _ids_as_names(snap: DatasetSnapshot) -> DatasetSnapshot:
    """``snap`` with each class id replaced by its class name, in the classes and the boxes."""
    labels = snap.labels
    tree = snap.tree
    edit(tree, "dataset_metadata.taxonomy.entries[*].source_id", lambda label_id: labels.get(label_id, label_id))
    for box in (box for sample in tree["samples"] for box in sample["detections"]):
        source_id = box["source_category_id"]
        # A source class id that is the box's own class id names the box's class, whatever the labels say.
        if source_id is not None and source_id == box["category_id"]:
            box["source_category_id"] = box["category_name"]
        else:
            box["source_category_id"] = labels.get(source_id, source_id)
        box["category_id"] = box["category_name"]
    return snap.with_tree(tree)


@pytest.mark.parametrize(("cell_id", "case", "fmt"), [pytest.param(*cell, id=cell[0]) for cell in APPEND_CELLS])
def test_append(cell_id: str, case: AppendCase, fmt: DatasetFormat, tmp_path: Path) -> None:
    bugs = [bug for bug in KNOWN_BUGS if bug.applies_to(cell_id)]
    class_blind = any(bug.relabels_only for bug in bugs)
    check_differences(run_append(case, fmt, tmp_path, class_blind=class_blind), bugs)


# ---------------------------------------------------------------------------
# Cases outside the matrix
# ---------------------------------------------------------------------------


def test_coco_writer_keeps_images_that_share_an_id(tmp_path: Path) -> None:
    """Two images with the same id (#134, case 5). COCO needs unique ids, so new ids are allowed."""
    from .harness.scenarios import box, dataset, image

    original = dataset(image(1, "a.png", box(1, "car", 2, 2)), image(1, "b.png", box(1, "car", 4, 4)))
    want = snapshot(original)
    write(original, tmp_path / "out", output_format="coco")
    reloaded = load_od(tmp_path / "out", dataset_format="coco")
    allowed = FORMATS[DatasetFormat.COCO].losses | {Loss.IMAGE_IDS}
    differences = compare(want, snapshot(original)) + compare(want, snapshot(reloaded), allowed=allowed)
    check_differences(differences, [COCO_AREA_BUG])
    assert sorted(sample.image_id for sample in reloaded.samples) == [1, 2]


def test_yolo_classes_txt_without_data_yaml(tmp_path: Path) -> None:
    """Darknet-style YOLO (Label Studio, LabelImg): class names in classes.txt, no data.yaml (#154)."""
    import yaml

    from .harness.scenarios import rich

    expected = rich()
    root = FORMATS[DatasetFormat.YOLO].author(expected, tmp_path / "src")
    names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
    (root / "data.yaml").unlink()
    (root / "classes.txt").write_text("".join(f"{names[index]}\n" for index in sorted(names)), encoding="utf-8")
    assert (root / "classes.txt").read_text(encoding="utf-8") == "person\ncar\nbus\ntruck\n"

    want, got, allowed = snapshot(expected), snapshot(load_od(root, dataset_format="yolo")), loss_stages(["yolo"])
    # Every box must still be there and in its place: the bug only loses class names.
    blind = compare(without_classes(expected_after_losses(want, allowed)), without_classes(got))
    bug = KnownBug(
        "#154",
        "the YOLO loader ignores classes.txt, naming classes after their ids and dropping unused ones",
        (),
        (
            r"^labels \{id: name\}: expected \{0: 'person', 1: 'car', 2: 'bus', 3: 'truck'\}, "
            r"got \{0: '0', 1: '1', 2: '2'\}$",
            r"^class '(person|car|bus)' \(id [012]\): name expected '(person|car|bus)', got '[012]'$",
            r"^[abd]\.png \[split train\]: missing box '(person|car|bus)' \(id [012]\)",
            r"^[abd]\.png \[split train\]: unexpected box '[012]' \(id [012]\)",
        ),
        relabels_only=True,
    )
    check_differences(compare(want, got, allowed=allowed) + [f"{IGNORING_CLASSES}{d}" for d in blind], [bug])


@pytest.mark.optional
def test_coco_exif_photo_sizes_match_pixels(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A COCO file from a tool that ignores EXIF: sizes and boxes in the stored frame (#133, case 1).

    ``a.jpg`` is stored 64x32 and displayed 32x64 (orientation 6). Its marker is
    at (2, 40, 20, 16) in the displayed frame, which is (40, 10, 16, 20) in the
    stored one. The loader turns the record to the displayed frame and warns.
    """
    from .harness.scenarios import exif_photo

    photo = exif_photo().samples[0]
    (tmp_path / "annotations").mkdir()
    (tmp_path / "a.jpg").write_bytes(photo.image_bytes or b"")
    polygon = [40, 10, 56, 10, 56, 30, 40, 30]
    (tmp_path / "annotations" / "instances.json").write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 32}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [40, 10, 16, 20], "segmentation": [polygon]}
                ],
                "categories": [{"id": 1, "name": "car"}],
            }
        )
    )
    with caplog.at_level("WARNING", logger="datamaite._formats.coco.loader"):
        loaded = load_od(tmp_path, dataset_format="coco")
    sample = loaded.samples[0]
    assert (sample.width, sample.height) == (32, 64)
    assert sample.detections[0].bbox == (2, 40, 20, 16)
    assert sample.detections[0].segmentation == [[22, 40, 22, 56, 2, 56, 2, 40]]
    assert "EXIF-rotated" in caplog.text
    assert snapshot(loaded, decode=True).problems == ()
    assert _marker_misses(loaded) == []


def test_hand_written_yolo_keeps_class_names_that_yaml_would_retype(tmp_path: Path) -> None:
    """Unquoted, YAML reads the class name ``yes`` as ``True`` and ``007`` as a number."""
    from .harness.scenarios import box, dataset, image

    expected = dataset(image(1, "a.png", box(0, "yes", 2, 3), box(1, "007", 20, 10)))
    loaded = load_od(FORMATS[DatasetFormat.YOLO].author(expected, tmp_path), dataset_format="yolo")
    assert compare(snapshot(expected), snapshot(loaded), allowed=FORMATS[DatasetFormat.YOLO].losses) == []


def _marker_misses(dataset: ObjectDetectionDataset) -> list[str]:
    """Each EXIF photo box whose decoded pixels are not the fixture's marker colour."""
    from .harness.scenarios import MARKER

    misses = []
    for index, sample in enumerate(dataset.samples):
        if not (sample.file_name or "").endswith(".jpg"):
            continue
        pixels = dataset.get_input(index)
        for detection in sample.detections:
            x, y, w, h = (round(float(value)) for value in detection.bbox)
            region = pixels[:, y : y + h, x : x + w].reshape(pixels.shape[0], -1)
            colour = [round(float(c)) for c in region.mean(axis=1)] if region.size else []
            if not colour or any(abs(a - b) > 30 for a, b in zip(colour, MARKER, strict=True)):
                misses.append(f"{sample.file_name}: box {[x, y, w, h]} covers colour {colour}, not the marker {MARKER}")
    return misses


@pytest.mark.optional
@pytest.mark.parametrize("path", ["memory", "load-coco", "load-yolo", "roundtrip-coco", "roundtrip-yolo"])
def test_exif_photo_boxes_cover_their_pixels(path: str, tmp_path: Path) -> None:
    """Each box sits on its marker in the decoded image, including after a 180-degree EXIF turn.

    Orientation 3 keeps the image size, so only the pixels show whether a box
    and its image still line up.
    """
    from .harness.scenarios import exif_photo

    source = exif_photo()
    operation, _, fmt = path.partition("-")
    if operation == "memory":
        loaded = source
    elif operation == "load":
        loaded = load_od(FORMATS[DatasetFormat(fmt)].author(source, tmp_path / "src"), dataset_format=fmt)
    else:
        write(source, tmp_path / "out", output_format=fmt)
        loaded = load_od(tmp_path / "out", dataset_format=fmt)
    assert _marker_misses(loaded) == []


def test_yolo_loader_keeps_a_sparse_names_mapping(tmp_path: Path) -> None:
    """A data.yaml ``names`` mapping with gaps is read as given; only the writer must make ids dense."""
    from .harness.scenarios import box, dataset, image, png

    data = png()
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train" / "a.png").write_bytes(data)
    (tmp_path / "labels" / "train" / "a.txt").write_text("3 0.25 0.25 0.15625 0.166667\n10 0.5 0.5 0.15625 0.166667\n")
    (tmp_path / "data.yaml").write_text("path: .\ntrain: images/train\nnames:\n  3: car\n  10: bus\n")
    expected = dataset(
        image(1, "a.png", box(3, "car", 11, 8), box(10, "bus", 27, 20), data=data), labels={3: "car", 10: "bus"}
    )

    allowed = FORMATS[DatasetFormat.YOLO].losses - {Loss.LABEL_IDS}
    assert compare(snapshot(expected), snapshot(load_od(tmp_path, dataset_format="yolo")), allowed=allowed) == []


def test_relocation_catches_output_that_points_at_its_source(tmp_path: Path) -> None:
    """Output whose images are links to the source loads in place but not once the source is gone."""
    from .harness.scenarios import rich

    expected = rich()
    src = FORMATS[DatasetFormat.COCO].author(expected, tmp_path / "src")
    out = tmp_path / "out"
    (out / "annotations").mkdir(parents=True)
    annotations = "annotations/instances_train.json"
    (out / annotations).write_bytes((src / annotations).read_bytes())
    for image in src.glob("*.png"):
        (out / image.name).symlink_to(Path("..") / "src" / image.name)
    allowed = FORMATS[DatasetFormat.COCO].losses

    assert compare(snapshot(expected), snapshot(load_od(out, dataset_format="coco")), allowed=allowed) == []
    differences = compare(
        snapshot(expected), snapshot(load_od(relocate(tmp_path), dataset_format="coco")), allowed=allowed
    )
    assert len(differences) == 5
    assert all("image_sha256 expected" in d and d.endswith(", got None") for d in differences)


def _corrupted_coco_differences(tmp_path: Path, corrupt: Callable[[dict[str, Any]], None]) -> list[str]:
    """Write ``record_details`` to COCO, change the box with annotation id 42 in the JSON, reload, compare.

    Returns only the differences the change adds to those of the unchanged
    output, so the known bugs of the plain round trip don't hide or add any.
    """
    from .harness.scenarios import record_details

    expected = snapshot(record_details())
    write(record_details(), tmp_path / "out", output_format="coco")
    allowed = loss_stages(["coco"])
    before = compare(expected, snapshot(load_od(tmp_path / "out", dataset_format="coco")), allowed=allowed)
    path = tmp_path / "out" / "annotations" / "instances_train.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    corrupt(next(annotation for annotation in document["annotations"] if annotation["id"] == 42))
    path.write_text(json.dumps(document), encoding="utf-8")
    after = compare(expected, snapshot(load_od(tmp_path / "out", dataset_format="coco")), allowed=allowed)
    return [difference for difference in after if difference not in before]


def test_a_corrupted_coco_polygon_is_reported(tmp_path: Path) -> None:
    """A writer that moves one polygon point by a pixel must fail the comparison."""

    def move_a_point(annotation: dict[str, Any]) -> None:
        annotation["segmentation"][0][1] += 1.0

    assert _corrupted_coco_differences(tmp_path, move_a_point) == [
        "a.png [split train]: box 'car' (id 1) at [20, 10, 10, 8]: segmentation expected "
        "[[22.0, 11.0, 29.5, 11.0, 29.5, 17.25, 22.0, 17.25]], got [[22.0, 12.0, 29.5, 11.0, 29.5, 17.25, 22.0, 17.25]]"
    ]


def test_a_deleted_coco_box_attribute_is_reported(tmp_path: Path) -> None:
    def drop_occluded(annotation: dict[str, Any]) -> None:
        del annotation["occluded"]

    assert _corrupted_coco_differences(tmp_path, drop_occluded) == [
        "a.png [split train]: box 'car' (id 1) at [20, 10, 10, 8]: attributes['occluded'] expected True, got <missing>"
    ]


_Breakage = Callable[[ObjectDetectionDataset, dict[str, Any]], Any]


def _patch_write(monkeypatch: pytest.MonkeyPatch, change: _Breakage) -> None:
    """Make every ``write`` in this module call ``change(dataset, kwargs)`` first, as a broken writer would."""
    real_write = write

    def broken_write(dataset: ObjectDetectionDataset, path: Path, **kwargs: Any) -> Any:
        return real_write(change(dataset, kwargs) or dataset, path, **kwargs)

    monkeypatch.setattr(sys.modules[__name__], "write", broken_write)


def test_a_writer_that_changes_its_input_is_caught(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A writer that deletes image metadata from its input in place must not change the expected answer too."""

    def drop_capture(dataset: ObjectDetectionDataset, _kwargs: dict[str, Any]) -> None:
        del dataset.samples[0].metadata["capture"]

    _patch_write(monkeypatch, drop_capture)
    scenario = next(s for s in SCENARIOS if s.name == "record_details")
    differences = run_cell(ROUNDTRIP, scenario, (DatasetFormat.COCO,), tmp_path)
    lost = "metadata['capture'] expected {'camera': 'cam1', 'exposure': 0.008}, got <missing>"
    # Reported twice: the input changed, and the written dataset lacks it (the answer was taken first).
    assert f"input dataset changed: a.png [split train]: {lost}" in differences
    assert f"a.png [split train]: {lost}" in differences


#: Prefixes of the differences a class-blind append adds to its plain comparison.


def test_a_known_relabelling_bug_does_not_hide_lost_boxes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A bug that only relabels boxes (as #130 did) must not explain an append that also drops boxes."""

    def drop_appended_boxes(dataset: ObjectDetectionDataset, kwargs: dict[str, Any]) -> Any:
        if kwargs.get("mode") != "append":
            return None
        return dataclasses.replace(
            dataset, samples=tuple(dataclasses.replace(s, detections=()) for s in dataset.samples)
        )

    _patch_write(monkeypatch, drop_appended_boxes)
    case = next(c for c in APPEND_CASES if c.name == "subset_of_classes")
    pattern = r"^c\.png \[split train\]: missing box "
    relabelling = KnownBug("#130", "relabels boxes", (), (pattern,), relabels_only=True)
    differences = run_append(case, DatasetFormat.YOLO, tmp_path, class_blind=True)
    # Without the class-blind comparison, the relabelling bug would explain the lost box.
    assert all(relabelling.explains(d) for d in differences if not d.startswith(CROSS_CHECKS))
    with pytest.raises(AssertionError, match=r"ignoring classes: c\.png \[split train\]: missing box"):
        check_differences(differences, [relabelling])


def test_a_known_relabelling_bug_does_not_hide_a_wrong_class_on_an_appended_box(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bug that relabels the earlier boxes (as #130 did) must not explain a wrong class on an appended box."""

    def bus_as_car(dataset: ObjectDetectionDataset, kwargs: dict[str, Any]) -> Any:
        if kwargs.get("mode") != "append":
            return None
        car = next(box for s in dataset.samples for box in s.detections if box.category_name == "car")
        relabelled = {"category_id": car.category_id, "category_name": "car"}
        return dataclasses.replace(
            dataset,
            samples=tuple(
                dataclasses.replace(
                    s,
                    detections=tuple(
                        dataclasses.replace(box, **relabelled) if box.category_name == "bus" else box
                        for box in s.detections
                    ),
                )
                for s in dataset.samples
            ),
        )

    _patch_write(monkeypatch, bus_as_car)
    case = next(c for c in APPEND_CASES if c.name == "new_class")
    pattern = r"^c\.png \[split train\]: (missing|unexpected) box "
    relabelling = KnownBug("#130", "relabels boxes", (), (pattern,), relabels_only=True)
    differences = run_append(case, DatasetFormat.YOLO, tmp_path, class_blind=True)
    # Without the appended images' own check, the relabelling bug would explain the wrong class.
    assert all(relabelling.explains(d) for d in differences if not d.startswith(CROSS_CHECKS))
    with pytest.raises(AssertionError, match=r"appended images: c\.png \[split train\]: missing box 'bus'"):
        check_differences(differences, [relabelling])


def test_an_append_that_writes_outside_the_destination_is_caught(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def also_overwrite_the_neighbour(_dataset: ObjectDetectionDataset, kwargs: dict[str, Any]) -> Any:
        if kwargs.get("mode") == "append":
            (tmp_path / "neighbour" / "keep.txt").write_text("written by the writer", encoding="utf-8")
        return None

    _patch_write(monkeypatch, also_overwrite_the_neighbour)
    case = next(c for c in APPEND_CASES if c.name == "same_classes")
    with pytest.raises(AssertionError, match=r"changed files outside out/: \['neighbour/keep\.txt'\]"):
        run_append(case, DatasetFormat.YOLO, tmp_path)


def test_a_write_outside_the_destination_is_caught(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.txt").write_text("source")
    with only_writes_to(tmp_path, "out"):
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "fine.txt").write_text("output")
    for stray in (tmp_path / "stray.txt", tmp_path / "neighbour" / "keep.txt", tmp_path / "src" / "a.txt"):
        with pytest.raises(AssertionError, match="changed files outside out/"), only_writes_to(tmp_path, "out"):
            stray.write_text("written by the writer")


def test_tree_digest_sees_a_changed_source_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one")
    before = tree_digest(tmp_path)
    (tmp_path / "a.txt").write_text("two")
    assert tree_digest(tmp_path) != before


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


def test_harness_covers_every_od_format() -> None:
    """A new read+write OD format must join the harness or be listed as not yet covered."""
    registered = set(convertible_od_formats())
    assert registered == set(FORMATS) | NOT_YET_COVERED
    assert not set(FORMATS) & NOT_YET_COVERED
    assert [fmt.value for fmt in FORMATS_UNDER_TEST] == ["coco", "yolo"]


#: Record fields that no scenario sets, each with the reason. Every other field
#: has a value other than its default in some scenario, so a format that loses it
#: is caught; the snapshot compares every field, but only the ones a scenario sets.
NOT_EXERCISED = {
    "ObjectDetectionDataset.task": "an object-detection dataset's task is always od",
    "ObjectDetectionDataset.dataset_id": "excluded from every snapshot",
    "ObjectDetectionDataset.image_color_mode": "excluded from every snapshot",
    "ImageObjectDetectionSample.path_or_uri": "excluded from every snapshot; the image content is compared",
    "DatasetMetadata.source_dataset": "excluded from every snapshot",
    "Taxonomy.source_dataset": "excluded from every snapshot",
    "ImageObjectDetectionSample.region": "an image-classification crop (VisDrone object crops); no OD format has one",
}


def _record_fields_never_set(value: Any, unset: dict[str, set[str]]) -> None:
    """Remove from ``unset`` (record name -> field names) each field ``value`` sets, at any depth."""
    from .harness.snapshot import canonical, field_defaults

    if isinstance(value, (list, tuple)):
        for item in value:
            _record_fields_never_set(item, unset)
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        return
    defaults = field_defaults(type(value))
    fields = unset.setdefault(type(value).__name__, {f.name for f in dataclasses.fields(value)})
    for f in dataclasses.fields(value):
        field_value = getattr(value, f.name)
        if f.name not in defaults or canonical(field_value) != defaults[f.name]:
            fields.discard(f.name)
        _record_fields_never_set(field_value, unset)


def test_every_record_field_is_set_by_some_scenario() -> None:
    """Each field of each record is set to a value other than its default by a scenario or an append case.

    Scenarios that need OpenCV to build are left out, so this runs without it; they set no other field.
    """
    unset: dict[str, set[str]] = {}
    scenarios = [s.build() for s in SCENARIOS if not s.needs_image_stack]
    _record_fields_never_set(scenarios + [c.build() for c in APPEND_CASES], unset)
    never_set = sorted(f"{record}.{name}" for record, names in unset.items() for name in names)
    assert never_set == sorted(NOT_EXERCISED)


def test_every_known_bug_matches_a_cell() -> None:
    ids = [cell_id for cell_id, *_ in CELLS] + [cell_id for cell_id, *_ in APPEND_CELLS]
    stale = [pattern for bug in KNOWN_BUGS for pattern in bug.cells if not fnmatch.filter(ids, pattern)]
    assert not stale, f"KNOWN_BUGS patterns that match no cell: {stale}"


def test_source_path_check_finds_a_planted_absolute_path(tmp_path: Path) -> None:
    """The #157 check catches an absolute source path and names the file and field."""
    source = tmp_path / "src"
    source.mkdir()
    out = tmp_path / "out"
    (out / "annotations").mkdir(parents=True)
    (out / "annotations" / "instances.json").write_text(
        json.dumps({"images": [{"file_name": "a.png", "metadata_file": f"{source}/train/metadata.jsonl"}]})
    )
    (out / "data.yaml").write_text(f"path: {source.resolve()}\nnames:\n  0: car\n")
    (out / "escaped.json").write_text('{"dir": "' + source.as_posix().replace("/", "\\/") + '"}')
    (out / "image.png").write_bytes(str(source).encode())  # not a text file: skipped

    with pytest.raises(AssertionError, match="absolute source path") as failure:
        assert_no_source_paths(out, source)
    message = str(failure.value)
    assert "annotations/instances.json: images[0].metadata_file = " in message
    assert "data.yaml: line 1: path: " in message
    assert "escaped.json: dir = " in message
    assert "image.png" not in message


def test_source_path_check_allows_relative_paths(tmp_path: Path) -> None:
    """Relative paths, and a sibling folder that shares the source's name as a prefix, pass."""
    source = tmp_path / "src"
    out = tmp_path / "out"
    out.mkdir()
    (out / "instances.json").write_text(
        json.dumps({"metadata_file": "train/metadata.jsonl", "other": f"{tmp_path}/src2/a.png", "name": "src"})
    )
    (out / "data.yaml").write_text("path: .\ntrain: images/train\n")
    assert_no_source_paths(out, source)


def test_source_path_check_covers_both_spellings_of_a_linked_folder(tmp_path: Path) -> None:
    """On macOS /var links to /private/var; a path stored in either spelling is found."""
    from .harness.paths import source_path_forms

    forms = source_path_forms(tmp_path)
    assert tmp_path.resolve().as_posix() in forms
    if tmp_path.resolve().as_posix().startswith("/private/var/"):
        assert any(form.startswith("/var/") for form in forms)
        out = tmp_path / "out"
        out.mkdir()
        unresolved = tmp_path.resolve().as_posix()[len("/private") :]
        (out / "labels.txt").write_text(f"{unresolved}/a.png\n")
        with pytest.raises(AssertionError, match=r"labels\.txt: line 1"):
            assert_no_source_paths(out, tmp_path)


def test_a_relabelling_bug_never_explains_its_cross_checks() -> None:
    bug = KnownBug("#155", "names boxes wrongly", (), (r"missing box None \(id \d+\)",), relabels_only=True)
    other = KnownBug("#163", "no area", (), (r"missing box None \(id \d+\)",))
    difference = "a.jpg: missing box None (id 21) at [0, 0, 1, 1]"

    assert bug.explains(difference)
    assert not bug.explains(f"{IGNORING_CLASSES}{difference}")
    assert not bug.explains(f"{APPENDED_IMAGES}{difference}")
    assert other.explains(f"{IGNORING_CLASSES}{difference}")
    # Matched without the prefix, so an anchored pattern applies to a cross-check too.
    anchored = KnownBug("#156", "no size", (), (r"^a\.jpg: width expected \d+, got None$",))
    assert anchored.explains(f"{IGNORING_CLASSES}a.jpg: width expected 64, got None")
