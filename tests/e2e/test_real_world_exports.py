"""Real-world YOLO, COCO and Hugging Face exports, checked with the conversion harness.

These are opt-in integration tests (``-m integration``). They run against the
real-world exports in the example-data repo
(``jatic/orchestration-interoperability/datamaite-example-datasets``): files
written by CVAT, Label Studio, FiftyOne, Roboflow, Ultralytics and Hugging Face
``datasets``, kept with their quirks. ``CASES`` lists one loader call per
export, and ``real_world_expected.json`` holds what each export contains, read
from its files without datamaite (see ``real_world_expected.py``).

Each case's cells check three separate things:

* ``load`` -- the loader: labels and class details, images, splits, sizes,
  image metadata, boxes, crowd flags, polygons and masks, areas, annotation
  ids and attributes, and the dataset's info and licenses match
  ``real_world_expected.json``. Fields the export's files don't hold are
  expected at their defaults, except values the loader derives (see
  ``LOADER_DERIVES``).
* ``self-check`` -- the loaded dataset agrees with itself, its MAITE targets
  and its pixels (``snapshot(..., decode=True).problems``).
* ``roundtrip``, ``to-coco``, ``to-yolo`` -- the writers: the dataset is
  written in its own format or converted, then reloaded and compared with the
  loaded source, allowing only the losses the formats declare. The source is a
  copy in the test folder, snapshotted before writing and checked to be
  unchanged after. The output must not contain the test folder's absolute
  path (#157), must pass direct file checks, and is moved away from the source
  before it is reloaded.

``test_paired_exports`` loads two exports of the same images and boxes, from
different tools, and checks they agree on those (each tool has its own
metadata, so other fields are not compared).

Known bugs are ``KnownBug`` entries listing the differences they cause, as in
``tests/test_conversion_harness.py``: any other difference in their cells
still fails. A case whose export is missing is skipped locally; set
``DATAMAITE_REQUIRE_REAL_EXPORTS=1`` (as the ``integration`` job does) to make
it a failure.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import logging
import os
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from datamaite import convert, load_od, write
from datamaite._types import DatasetFormat
from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import DatasetMetadata, ImageObjectDetectionSample, ObjectDetectionAnnotation
from datamaite.taxonomy import CategoryEntry, Taxonomy

from ..harness import (
    FORMATS,
    DatasetSnapshot,
    Loss,
    SampleSnapshot,
    compare,
    find_source_paths,
    output_problems,
    snapshot,
)
from ..harness.compare import Allowed, expected_after_losses, without_class_names
from ..harness.guards import changed_inputs, input_snapshots, only_writes_to
from ..harness.known_bugs import IGNORING_CLASSES, KnownBug, check_differences
from ..harness.output import relocate
from ..harness.snapshot import BOX_TOLERANCE, Record
from ._harness import disk_snapshot, sample_key, split_folders, split_folders_hold_file_names

pytestmark = [pytest.mark.optional, pytest.mark.integration]

LOAD, SELF_CHECK, ROUNDTRIP = "load", "self-check", "roundtrip"
EXPECTED = json.loads(Path(__file__).with_name("real_world_expected.json").read_text(encoding="utf-8"))
REQUIRE_ENV = "DATAMAITE_REQUIRE_REAL_EXPORTS"


@dataclass(frozen=True)
class Dropped:
    """A box in the files that the loader drops on purpose, with a warning containing ``warning``."""

    key: str
    bbox: tuple[float, float, float, float]
    warning: str


@dataclass(frozen=True)
class Case:
    """One loader call on one export. What the export holds is ``EXPECTED[id]``."""

    id: str
    path: str
    format: str
    options: dict[str, Any] = field(default_factory=dict)
    # Checks to run; the default is load, self-check, a round trip and both conversions.
    cells: tuple[str, ...] = ()
    dropped: tuple[Dropped, ...] = ()
    # A module the loader needs for this layout.
    needs: str | None = None

    def cell_names(self) -> tuple[str, ...]:
        if self.cells:
            return self.cells
        targets = [f"to-{fmt}" for fmt in ("coco", "yolo") if fmt != self.format]
        return (LOAD, SELF_CHECK, ROUNDTRIP, *targets)


_ROBOFLOW_ZERO_WIDTH = "img_2042_jpg.rf.494e0524877176c2503c42580709ec18.jpg"

CASES = [
    # CVAT COCO 1.0: one instances_<subset>.json per subset, images in images/<Subset>/.
    Case("coco-cvat", "coco/cvat", "coco"),
    Case(
        "coco-cvat-train",
        "coco/cvat",
        "coco",
        options={"annotation_file": "annotations/instances_Train.json", "images_dir": "images/Train"},
    ),
    Case(
        "coco-cvat-validation",
        "coco/cvat",
        "coco",
        options={"annotation_file": "annotations/instances_Validation.json", "images_dir": "images/Validation"},
    ),
    # CVAT's data.yaml has Train:/Validation: keys, which Ultralytics rejects.
    # datamaite finds the folders itself and names the splits train and val.
    Case("yolo-cvat-ultralytics", "yolo/cvat-ultralytics", "yolo"),
    # Label Studio writes its server's media path as file_name
    # (../../label-studio/data/media/upload/...). No COCO reader can reach the
    # images from that, and datamaite refuses names that leave the dataset, so
    # the expected result is every annotation with no image. Nothing to write.
    Case("coco-label-studio", "coco/label-studio", "coco", cells=(LOAD, SELF_CHECK)),
    # No data.yaml: class names are only in classes.txt.
    Case("yolo-label-studio", "yolo/label-studio", "yolo"),
    Case("yolo-ultralytics-od", "yolo/ultralytics-od", "yolo"),
    # FiftyOne puts the images in data/, beside labels.json.
    Case("coco-fiftyone", "coco/fiftyone", "coco", options={"images_dir": "data"}),
    # metadata.jsonl has integer categories and no names, so the ids are the names.
    Case("hf-imagefolder-od", "huggingface_vision/imagefolder-od", "huggingface_vision"),
    Case("hf-parquet-od", "huggingface_vision/parquet-od", "huggingface_vision", needs="pyarrow"),
    # Roboflow: one _annotations.coco.json per split folder, loaded one split at
    # a time, so no split name. train has one zero-width box, which the loaders
    # drop with a warning.
    Case(
        "coco-roboflow-train",
        "coco/roboflow/train",
        "coco",
        dropped=(
            Dropped(_ROBOFLOW_ZERO_WIDTH, (53.0, 436.0, 0.0, 6.0), "Skipping COCO annotation with missing/invalid"),
        ),
    ),
    Case("coco-roboflow-valid", "coco/roboflow/valid", "coco"),
    Case("coco-roboflow-test", "coco/roboflow/test", "coco"),
    # data.yaml points at ../train/images etc.; the folder is valid/, the key val.
    Case(
        "yolo-roboflow-od",
        "yolo/roboflow-od",
        "yolo",
        dropped=(
            Dropped(
                _ROBOFLOW_ZERO_WIDTH,
                (52.998, 436.002, 0.0, 6.0),
                "img_2042_jpg.rf.494e0524877176c2503c42580709ec18.txt:14",
            ),
        ),
    ),
    # Sizes and boxes in the displayed (EXIF-rotated) frame, as tools that honour EXIF write them.
    Case("coco-exif-orientation", "coco/exif-orientation", "coco"),
    Case("yolo-exif-orientation", "yolo/exif-orientation", "yolo"),
]
CASES_BY_ID = {case.id: case for case in CASES}


@dataclass(frozen=True)
class Pair:
    """Exports of the same images and boxes from two tools, which must load the same.

    Class ids, image ids and split names are not compared. ``crowd`` says what
    to do with the first side's crowd boxes when the second side's format has
    no crowd flag: ``"flag"`` compares them as ordinary boxes, ``"drop"``
    leaves them out (the second tool did not export them).
    """

    name: str
    first: tuple[str, ...]
    second: tuple[str, ...]
    crowd: str = "flag"


PAIRS = [
    Pair("fiftyone-coco-hf-parquet", ("coco-fiftyone",), ("hf-parquet-od",)),
    Pair("exif-coco-yolo", ("coco-exif-orientation",), ("yolo-exif-orientation",)),
    Pair(
        "roboflow-coco-yolo",
        ("coco-roboflow-train", "coco-roboflow-valid", "coco-roboflow-test"),
        ("yolo-roboflow-od",),
    ),
    # CVAT's YOLO export leaves out the four mask-only (crowd) annotations.
    Pair("cvat-coco-yolo", ("coco-cvat-train", "coco-cvat-validation"), ("yolo-cvat-ultralytics",), crowd="drop"),
]

# Formats whose written files ``output_problems`` can check.
_CHECKED_OUTPUT_FORMATS = ("coco", "yolo")

# Cell ids are "<case>-<cell>" and "pair-<pair>".
KNOWN_BUGS: tuple[KnownBug, ...] = (
    KnownBug(
        "#154",
        "the YOLO loader ignores classes.txt, so classes are named 0..n",
        ("yolo-label-studio-load",),
        (
            r"^labels \{id: name\}: expected \{0: 'Apple', ",
            r"^class '[^']+' \(id \d+\): name expected '[^']+', got '\d+'$",
            r": missing box '[^']+' \(id \d+\)",
            r": unexpected box '\d+' \(id \d+\)",
        ),
        relabels_only=True,
    ),
    KnownBug(
        "#155",
        "Hugging Face integer classes are named in the taxonomy but not on the boxes",
        ("hf-imagefolder-od-*",),
        (
            r": (missing|unexpected) box (None|'\d+') \(id \d+\)",
            r"^problem: \S+: box None \(id \d+\) does not match the dataset labels ",
            r"^problem: \S+: box None \(id \d+\): MAITE index2label maps \d+ to '\d+'$",
        ),
        relabels_only=True,
    ),
    KnownBug(
        "#156",
        "the Hugging Face loader leaves image width and height unset",
        ("hf-*-load", "hf-*-self-check", "hf-*-roundtrip"),
        (
            r"^problem: \S+\.jpg: stored size NonexNone but the image decodes to \d+x\d+$",
            r"^\S+\.jpg \[split \w+\]: (width|height) expected \d+, got None$",
        ),
    ),
    KnownBug(
        "#157",
        "the COCO writer copies the absolute Hugging Face metadata_file path",
        ("hf-*-to-coco",),
        (r"^source path: annotations/instances(_\w+)?\.json: images\[\d+\]\.metadata_file = '",),
    ),
    KnownBug(
        "#163",
        "the COCO writer leaves out area when a box has none, though COCO evaluation (pycocotools) needs it",
        # YOLO and Hugging Face boxes have no area. hf-imagefolder-od's boxes have no class (#155),
        # so they match only once classes are left out.
        ("yolo-*-to-coco", "hf-*-to-coco"),
        (
            r": box ('[^']+'|None) \(id (\d+|None)\) at \[[-\d.e, ]+\]: "
            r"area expected \d+(\.\d+)?(e[-+]?\d+)?, got None$",
        ),
    ),
    KnownBug(
        "#165",
        "the COCO writer renumbers annotation ids that repeat in another split, though each split has its own file",
        # CVAT numbers each instances_<Subset>.json's annotations from 1; Validation's are written as 47..57.
        ("coco-cvat-roundtrip",),
        (
            r"^\S+\.jpg \[split Validation\]: box '[^']+' \(id \d+\) at \[[\d., ]+\]( \(crowd\))?: "
            r"source_annotation_id expected \d+, got \d+$",
        ),
    ),
)

# Losses for source formats the harness does not cover yet (#149). Hugging Face
# image ids are file paths, its boxes store class names (so ids are rebuilt
# from them), and classes no box uses are not written.
_EXTRA_LOSSES: dict[str, frozenset[Loss]] = {
    "huggingface_vision": frozenset(
        {
            Loss.IMAGE_IDS,
            Loss.LABEL_IDS_FROM_NAMES,
            Loss.UNUSED_LABELS,
            Loss.SOURCE_CATEGORY_IDS,
            Loss.SPLIT_LIST_FROM_IMAGES,
            Loss.HUGGINGFACE_SOURCE_DETAILS,
            # The loader marks the taxonomy dense when its class ids are 0..n-1.
            Loss.ID_DENSITY_FROM_IDS,
        }
    )
}

# What each loader makes up rather than reads from the files, so a load is
# compared with real_world_expected.json after these. Image ids are compared
# nowhere: the expected file uses the file name.
LOADER_DERIVES: dict[str, frozenset[Loss]] = {
    # The source class id is the category id, the split list is the annotation files' splits,
    # and the taxonomy is marked sparse.
    "coco": frozenset({Loss.IMAGE_IDS, Loss.SOURCE_CATEGORY_IDS, Loss.SPLIT_LIST_BY_NAME, Loss.ID_DENSITY}),
    # The loader also records each image's label file and each box's label row.
    "yolo": frozenset(
        {
            Loss.IMAGE_IDS,
            Loss.SOURCE_CATEGORY_IDS,
            Loss.SPLIT_LIST_FROM_IMAGES,
            Loss.YOLO_SOURCE_DETAILS,
            Loss.ID_DENSITY_FROM_IDS,
        }
    ),
    "huggingface_vision": frozenset(
        {
            Loss.IMAGE_IDS,
            Loss.SOURCE_CATEGORY_IDS,
            Loss.SPLIT_LIST_FROM_IMAGES,
            Loss.HUGGINGFACE_SOURCE_DETAILS,
            Loss.ID_DENSITY_FROM_IDS,
        }
    ),
}


def _losses(*formats: str) -> tuple[frozenset[Loss], ...]:
    """Each format's losses in pipeline order, so ``compare`` applies the source's before the target's."""
    stages = []
    for fmt in formats:
        spec = FORMATS.get(DatasetFormat(fmt))
        stages.append(spec.losses if spec is not None else _EXTRA_LOSSES[fmt])
    return tuple(stages)


def _bugs(cell_id: str) -> list[KnownBug]:
    return [bug for bug in KNOWN_BUGS if bug.applies_to(cell_id)]


def _params() -> list[Any]:
    return [pytest.param(case, cell, id=f"{case.id}-{cell}") for case in CASES for cell in case.cell_names()]


def _require() -> bool:
    return os.environ.get(REQUIRE_ENV, "").lower() in {"1", "true", "yes"}


def _unavailable(reason: str) -> None:
    if _require():
        pytest.fail(f"{reason}, and {REQUIRE_ENV} is set")
    pytest.skip(reason)


@pytest.fixture(scope="module")
def datasets_root() -> Path:
    """The example-data ``datasets/`` dir, or skip when unavailable."""
    root = os.environ.get("DATAMAITE_DATASETS_ROOT")
    if not root:
        _unavailable("DATAMAITE_DATASETS_ROOT is not set")
    path = Path(str(root))
    if not path.is_dir():
        _unavailable(f"DATAMAITE_DATASETS_ROOT does not exist: {path}")
    return path


def _source(datasets_root: Path, case: Case) -> Path:
    src = datasets_root / case.path
    if not src.is_dir():
        _unavailable(f"{case.path} is not in this example-data checkout")
    if case.needs and importlib.util.find_spec(case.needs) is None:
        _unavailable(f"{case.id} needs {case.needs}")
    return src


def _load(src: Path, case: Case) -> ObjectDetectionDataset:
    return load_od(src, dataset_format=case.format, **case.options)


# ---------------------------------------------------------------------------
# Loading, against real_world_expected.json
# ---------------------------------------------------------------------------


def _is_dropped(case: Case, key: str, box: list[Any]) -> bool:
    return any(
        d.key == key and all(abs(a - b) <= BOX_TOLERANCE for a, b in zip(d.bbox, box[2:6], strict=True))
        for d in case.dropped
    )


def _box(case: Case, key: str, row: list[Any]) -> ObjectDetectionAnnotation | None:
    """One expected box, ``None`` if the loader drops it on purpose. A field the files don't hold is its default."""
    label_id, name, x, y, w, h, crowd, *rest = row
    if _is_dropped(case, key, row):
        return None
    details: dict[str, Any] = {}
    if rest:  # COCO: the rest of the annotation.
        (annotation,) = rest
        details = {
            "source_annotation_id": annotation["id"],
            "area": annotation["area"],
            "segmentation": annotation["segmentation"],
            "attributes": annotation["attributes"],
        }
    return ObjectDetectionAnnotation(
        bbox=(x, y, w, h), category_id=label_id, category_name=name, iscrowd=crowd, **details
    )


def expected_snapshot(case: Case) -> DatasetSnapshot:
    """``EXPECTED[case.id]`` as a snapshot, without the boxes the loader drops on purpose.

    It is the snapshot of a dataset built from the expected file, so a field
    the file does not record (it is not in the export's files) is at its
    default: no score, no crop region, no synset, no split list, a sparse
    taxonomy. ``LOADER_DERIVES`` turns the values a loader works out itself
    into what it must give. Each image's content is the file's hash.
    """
    entry = EXPECTED[case.id]
    samples = [
        ImageObjectDetectionSample(
            image_id=image["key"],
            file_name=_file_name(case, image),
            split=image["split"],
            width=image["width"],
            height=image["height"],
            metadata=image["metadata"],
            detections=tuple(box for row in image["boxes"] if (box := _box(case, image["key"], row)) is not None),
        )
        for image in entry["images"]
    ]
    details = entry["label_details"]
    entries = tuple(
        CategoryEntry(source_id=int(label_id), name=name, **details.get(label_id, {}))
        for label_id, name in entry["labels"].items()
    )
    metadata = DatasetMetadata(
        taxonomy=Taxonomy(entries=entries) if entries else None,
        info=entry.get("info", {}),
        licenses=tuple(entry.get("licenses", ())),
    )
    snap = snapshot(ObjectDetectionDataset(samples=tuple(samples), dataset_metadata=metadata), key=_key)
    tree = snap.tree
    for fields, image in zip(tree["samples"], entry["images"], strict=True):
        fields["image_sha256"] = image["sha256"]
    return replace(snap.with_tree(tree), problems=())


def _file_name(case: Case, image: dict[str, Any]) -> str:
    """The image's file name as the loader gives it: Hugging Face names are under the split folder."""
    if split_folders_hold_file_names(case.format) and image["split"]:
        return f"{image['split']}/{image['key']}"
    return str(image["key"])


def _key(sample: ImageObjectDetectionSample) -> str:
    return str(sample.image_id)


def _class_blind(want: DatasetSnapshot, got: DatasetSnapshot, allowed: Allowed) -> list[str]:
    """The differences once class names are left out, for a cell whose known bug only names boxes wrongly.

    A box that is lost, moved or given another class id shows up here even
    though the naming bug explains the full comparison's differences, and that
    bug can't explain these (``KnownBug.explains``). The ids stay: #154 and #155
    keep them. ``got``'s problems are left out: the full comparison reports them.
    """
    blind = compare(
        without_class_names(expected_after_losses(want, allowed)), without_class_names(replace(got, problems=()))
    )
    return [f"{IGNORING_CLASSES}{d}" for d in blind]


def load_differences(
    case: Case, dataset: ObjectDetectionDataset, warnings: list[str], *, bugs: list[KnownBug]
) -> list[str]:
    """Every way ``dataset`` differs from what the export's files hold."""
    folders = split_folders(dataset, case.format)
    actual = replace(snapshot(dataset, key=lambda sample: sample_key(sample, folders)), problems=())
    expected, allowed = expected_snapshot(case), LOADER_DERIVES[case.format]
    differences = compare(expected, actual, allowed=allowed)
    if any(bug.relabels_only for bug in bugs):
        differences += _class_blind(expected, actual, allowed)
    differences += [
        f"{d.key}: no warning containing {d.warning!r} for the dropped box at {list(d.bbox)}"
        for d in case.dropped
        if not any(d.warning in message for message in warnings)
    ]
    return differences


# ---------------------------------------------------------------------------
# Writing and converting
# ---------------------------------------------------------------------------


def write_differences(case: Case, cell: str, export: Path, tmp_path: Path, *, bugs: list[KnownBug]) -> list[str]:
    """Write or convert a copy of the export, reload it and return every difference from the loaded source."""
    src = Path(shutil.copytree(export, tmp_path / "src"))
    source = _load(src, case)
    # Snapshot (reading and hashing every image) before writing, so a writer
    # that changes the source images can't make them match its output.
    folders = split_folders(source, case.format)
    before = disk_snapshot(source, folders)
    inputs = input_snapshots(input=source)
    out = tmp_path / "out"
    target = case.format if cell == ROUNDTRIP else cell.removeprefix("to-")
    # Nothing outside out/ may change: not the source folder, nor a file beside out/.
    with only_writes_to(tmp_path, "out"):
        if cell == ROUNDTRIP:
            write(source, out, output_format=target)
        else:
            convert(src, out, input_format=case.format, output_format=target, task="od", load_options=case.options)
    differences = changed_inputs(inputs)
    differences += [f"source path: {place}" for place in find_source_paths(out, tmp_path)]
    out = relocate(tmp_path)
    if target in _CHECKED_OUTPUT_FORMATS:
        differences += output_problems(out, target)
    reloaded = load_od(out, dataset_format=target)
    after = disk_snapshot(reloaded, folders)
    # ``before`` is the loaded export, which no datamaite writer wrote, so the
    # source writer's losses do not apply: only what the source loader makes up
    # (such as image ids from file paths), then the target's losses.
    allowed = (LOADER_DERIVES[case.format], *_losses(target))
    differences += compare(before, after, allowed=allowed)
    if any(bug.relabels_only for bug in bugs):
        differences += _class_blind(before, after, allowed)
    return differences


@pytest.mark.parametrize(("case", "cell"), _params())
def test_real_world_export(
    datasets_root: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture, case: Case, cell: str
) -> None:
    export = _source(datasets_root, case)
    bugs = _bugs(f"{case.id}-{cell}")
    if cell == LOAD:
        with caplog.at_level(logging.WARNING):
            dataset = _load(export, case)
        warnings = [record.getMessage() for record in caplog.records]
        differences = load_differences(case, dataset, warnings, bugs=bugs)
    elif cell == SELF_CHECK:
        dataset = _load(export, case)
        if all(sample.path_or_uri or sample.image_bytes for sample in dataset.samples):
            problems = disk_snapshot(dataset).problems
        else:  # Label Studio's images are unreachable (see CASES), so only the records are checked.
            problems = snapshot(dataset).problems
        differences = [f"problem: {problem}" for problem in problems]
    else:
        differences = write_differences(case, cell, export, tmp_path, bugs=bugs)
    check_differences(differences, bugs)


# ---------------------------------------------------------------------------
# Paired exports
# ---------------------------------------------------------------------------


def _pair_snapshot(datasets_root: Path, case_ids: tuple[str, ...], crowd: str | None) -> DatasetSnapshot:
    samples: list[SampleSnapshot] = []
    for case_id in case_ids:
        case = CASES_BY_ID[case_id]
        dataset = _load(_source(datasets_root, case), case)
        snap = disk_snapshot(dataset, split_folders(dataset, case.format))
        for s in snap.samples:
            boxes = [b for b in s.boxes if not (crowd == "drop" and b.iscrowd)]
            # Each tool keeps its own metadata, ids, splits and attributes: compare the images and boxes only.
            detections = [
                Record(
                    category_name=b.label,
                    category_id=None,
                    bbox=list(b.bbox),
                    iscrowd=0 if crowd == "flag" else b.iscrowd,
                )
                for b in boxes
            ]
            fields = Record(
                image_sha256=s.image_sha256, width=s.width, height=s.height, split=None, detections=detections
            )
            samples.append(SampleSnapshot(s.key, fields))
    return DatasetSnapshot(fields={}, samples=tuple(sorted(samples, key=lambda s: s.key)))


@pytest.mark.parametrize("pair", PAIRS, ids=[f"pair-{pair.name}" for pair in PAIRS])
def test_paired_exports(datasets_root: Path, pair: Pair) -> None:
    first = _pair_snapshot(datasets_root, pair.first, pair.crowd)
    second = _pair_snapshot(datasets_root, pair.second, "flag" if pair.crowd == "flag" else None)
    check_differences(compare(first, second), _bugs(f"pair-{pair.name}"))


def test_a_writer_that_changes_a_class_id_fails_a_relabelling_cell(
    datasets_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writer bug in a cell whose known bug (#155) only names boxes wrongly is not excused by it."""
    from datamaite._formats.huggingface_vision import writer

    case = CASES_BY_ID["hf-imagefolder-od"]
    export = _source(datasets_root, case)
    category_value = writer._category_value
    shifted_boxes: list[ObjectDetectionAnnotation] = []

    def shifted(detection: ObjectDetectionAnnotation, taxonomy: Taxonomy | None) -> int | str | None:
        # Only the first box of class 21, so 21 is still used and the dataset's classes don't change.
        value = category_value(detection, taxonomy)
        if value == 21 and not shifted_boxes:
            shifted_boxes.append(detection)
            return 11
        return value

    monkeypatch.setattr(writer, "_category_value", shifted)
    bugs = _bugs(f"{case.id}-{ROUNDTRIP}")
    assert any(bug.relabels_only for bug in bugs)

    differences = write_differences(case, ROUNDTRIP, export, tmp_path, bugs=bugs)
    assert shifted_boxes

    with pytest.raises(AssertionError, match=r"ignoring classes: .*box None \(id 21\)"):
        check_differences(differences, bugs)


def test_every_known_bug_matches_a_cell() -> None:
    ids = [f"{case.id}-{cell}" for case in CASES for cell in case.cell_names()] + [f"pair-{p.name}" for p in PAIRS]
    stale = [pattern for bug in KNOWN_BUGS for pattern in bug.cells if not fnmatch.filter(ids, pattern)]
    assert not stale, f"KNOWN_BUGS patterns that match no cell: {stale}"


def test_expected_data_covers_every_case() -> None:
    assert sorted(EXPECTED) == sorted(CASES_BY_ID)


def test_sample_key_strips_only_split_folders() -> None:
    from datamaite.records import ImageObjectDetectionSample

    def key(file_name: str, folders: frozenset[str]) -> str:
        return sample_key(ImageObjectDetectionSample(image_id=1, file_name=file_name, detections=()), folders)

    assert key("train/a.png", frozenset({"train"})) == "a.png"
    assert key("cam1/a.png", frozenset({"train"})) != key("cam2/a.png", frozenset({"train"}))
    assert key("train/a.png", frozenset()) == "train/a.png"


def test_disk_snapshot_fills_each_missing_size_from_the_pixels() -> None:
    from ..harness.scenarios import box, dataset, image

    sample = replace(image(1, "a.png", box(0, "car", 1, 1)), width=64, height=None)
    snap = disk_snapshot(dataset(sample))
    assert (snap.samples[0].width, snap.samples[0].height) == (64, 48)
    assert snap.problems == ("a.png: stored size 64xNone but the image decodes to 64x48",)
