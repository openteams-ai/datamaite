"""Tests of the conversion harness's comparison itself (#150).

Each test builds two snapshots by hand and checks that ``compare`` reports a
deliberate change, or accepts exactly the change an allowed loss covers.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from typing import Any

import pytest

from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import DatasetMetadata, ImageObjectDetectionSample, ObjectDetectionAnnotation
from datamaite.taxonomy import CategoryEntry, Taxonomy

from .harness import BoxSnapshot, DatasetSnapshot, Loss, SampleSnapshot, compare
from .harness.snapshot import EXCLUDED_PATHS, Record, canonical, field_defaults, remove


def _box(label: str, label_id: int, x: float = 1.0, **fields: object) -> BoxSnapshot:
    record = canonical(ObjectDetectionAnnotation(bbox=(x, 2.0, 3.0, 4.0), category_id=label_id, category_name=label))
    record.update(fields)
    return BoxSnapshot(record)


def _sample(
    key: str, *boxes: BoxSnapshot, split: str | None = "train", sha: str | None = None, **fields: object
) -> SampleSnapshot:
    record = canonical(ImageObjectDetectionSample(image_id=key, file_name=key, width=64, height=48, split=split))
    remove(record, "path_or_uri")
    record.update(image_sha256=sha or f"sha-{key}", detections=[box.fields for box in boxes], **fields)
    return SampleSnapshot(key, record)


def _with(sample: SampleSnapshot, **fields: object) -> SampleSnapshot:
    """``sample`` with these fields changed."""
    return replace(sample, fields=Record({**sample.fields, **fields}))


def _snap(
    *samples: SampleSnapshot,
    labels: dict[int | str, str],
    entries: tuple[CategoryEntry, ...] | None = None,
    **metadata: Any,
) -> DatasetSnapshot:
    if entries is None:
        entries = tuple(CategoryEntry(source_id=i, name=n) for i, n in labels.items())
    dataset = ObjectDetectionDataset(
        samples=(), dataset_metadata=DatasetMetadata(taxonomy=Taxonomy(entries=entries), **metadata)
    )
    tree = canonical(dataset)
    for pattern in EXCLUDED_PATHS:
        remove(tree, pattern)
    tree.pop("samples")
    return DatasetSnapshot(fields=tree, samples=samples)


def test_renumbered_class_ids_follow_taxonomy_order() -> None:
    expected = _snap(_sample("a.png", _box("car", 2), _box("bus", 10, x=5)), labels={10: "bus", 2: "car"})
    in_taxonomy_order = _snap(_sample("a.png", _box("car", 1), _box("bus", 0, x=5)), labels={0: "bus", 1: "car"})
    in_numeric_order = _snap(_sample("a.png", _box("car", 0), _box("bus", 1, x=5)), labels={0: "car", 1: "bus"})

    assert compare(expected, in_taxonomy_order, allowed={Loss.LABEL_IDS}) == []
    assert compare(expected, in_numeric_order, allowed={Loss.LABEL_IDS})


def test_class_ids_missing_from_the_labels_are_renumbered_in_numeric_order() -> None:
    expected = _snap(_sample("a.png", _box("ten", 10), _box("two", 2, x=5)), labels={})
    actual = _snap(_sample("a.png", _box("ten", 1), _box("two", 0, x=5)), labels={})

    assert compare(expected, actual, allowed={Loss.LABEL_IDS}) == []
    assert compare(
        expected,
        replace(actual, samples=(_sample("a.png", _box("ten", 0), _box("two", 1, x=5)),)),
        allowed={Loss.LABEL_IDS},
    )


def test_single_split_name_unnames_a_single_split_dataset_but_does_not_allow_repartitioning() -> None:
    expected = _snap(_sample("a.png"), _sample("b.png"), _sample("c.png"), labels={})
    unnamed = _snap(*(_with(s, split=None) for s in expected.samples), labels={})
    repartitioned = _snap(
        _sample("a.png", split="val"), _sample("b.png", split="test"), _sample("c.png", split="test"), labels={}
    )
    partly_unnamed = _snap(_sample("a.png", split=None), _sample("b.png"), _sample("c.png"), labels={})

    assert compare(expected, unnamed, allowed={Loss.SINGLE_SPLIT_NAME}) == []
    assert len(compare(expected, repartitioned, allowed={Loss.SINGLE_SPLIT_NAME})) == 3
    assert compare(expected, partly_unnamed, allowed={Loss.SINGLE_SPLIT_NAME}) == [
        "b.png: split expected None, got 'train'",
        "c.png: split expected None, got 'train'",
    ]
    two_splits = _snap(_sample("a.png"), _sample("b.png", split="val"), labels={})
    assert compare(two_splits, two_splits, allowed={Loss.SINGLE_SPLIT_NAME}) == []


def test_losses_of_each_format_apply_in_conversion_order() -> None:
    """A single ``val`` split through COCO (unnamed) then YOLO (unnamed goes to train) comes back as ``train``."""
    expected = _snap(_sample("a.png", split="val"), _sample("b.png", split="val"), labels={})
    in_train = _snap(_sample("a.png"), _sample("b.png"), labels={})
    coco, yolo = {Loss.SINGLE_SPLIT_NAME}, {Loss.DEFAULT_SPLIT}

    assert compare(expected, in_train, allowed=[coco, yolo]) == []
    assert len(compare(expected, in_train, allowed=[yolo, coco])) == 2
    assert len(compare(expected, in_train, allowed=coco | yolo)) == 2
    with pytest.raises(TypeError, match="not a mix"):
        compare(expected, in_train, allowed=[Loss.DEFAULT_SPLIT, coco])  # type: ignore[list-item]


def test_renumbered_class_ids_must_be_dense() -> None:
    expected = _snap(_sample("a.png", _box("car", 3), _box("bus", 10, x=5)), labels={3: "car", 10: "bus"})
    shifted = _snap(_sample("a.png", _box("car", 103), _box("bus", 110, x=5)), labels={103: "car", 110: "bus"})

    differences = compare(expected, shifted, allowed={Loss.LABEL_IDS})

    assert "labels {id: name}: expected {0: 'car', 1: 'bus'}, got {103: 'car', 110: 'bus'}" in differences
    assert any("missing box 'car' (id 0)" in d for d in differences)


def test_allowed_losses_do_not_change_the_actual_dataset() -> None:
    expected = _snap(_sample("a.png", split="validation"), labels={})
    renamed_back = _snap(_sample("a.png", split="validation"), labels={})

    # The format writes "val"; giving back "validation" is not what the loss allows.
    assert compare(expected, renamed_back, allowed={Loss.SPLIT_ALIASES}) == [
        "a.png [split val]: split expected 'val', got 'validation'"
    ]


def test_snapshot_flags_a_maite_target_that_disagrees_with_the_records(monkeypatch) -> None:
    import numpy as np

    from .harness import snapshot
    from .harness.scenarios import multi_class

    real_get_target = ObjectDetectionDataset.get_target

    def zero_boxes(self, index):
        target = real_get_target(self, index)
        return replace(target, boxes=np.zeros_like(target.boxes))

    assert snapshot(multi_class()).problems == ()
    monkeypatch.setattr(ObjectDetectionDataset, "get_target", zero_boxes)
    problems = snapshot(multi_class()).problems
    assert len(problems) == 6
    assert all("MAITE target box [0.0, 0.0, 0.0, 0.0]" in problem for problem in problems)


@pytest.mark.parametrize("score", [float("nan"), float("inf")])
def test_snapshot_flags_a_maite_target_score_that_is_not_finite(monkeypatch, score: float) -> None:
    import numpy as np

    from .harness import snapshot
    from .harness.scenarios import multi_class

    real_get_target = ObjectDetectionDataset.get_target

    def bad_scores(self, index):
        target = real_get_target(self, index)
        return replace(target, scores=np.full_like(target.scores, score))

    monkeypatch.setattr(ObjectDetectionDataset, "get_target", bad_scores)
    problems = snapshot(multi_class()).problems
    assert len(problems) == 6
    assert all(f"MAITE target score {score} but the record's is 1.0" in problem for problem in problems)


def test_snapshot_flags_labels_missing_from_the_public_index2label(monkeypatch) -> None:
    from .harness import snapshot
    from .harness.scenarios import multi_class

    monkeypatch.setattr(ObjectDetectionDataset, "index2label", lambda _self: {0: "person", 1: "car"})
    problems = snapshot(multi_class()).problems
    assert problems == (
        "taxonomy: class 'bus' (id 2) is not in the MAITE index2label as 'bus'",
        "b.png: box 'bus' (id 2): MAITE index2label maps 2 to None",
        "d.png: box 'bus' (id 2): MAITE index2label maps 2 to None",
    )


@pytest.mark.optional
def test_snapshot_with_decode_compares_the_decoded_pixels(monkeypatch) -> None:
    """The same file bytes decoded to other pixels (all zeros) are reported, not only a changed size."""
    import numpy as np

    from .harness import snapshot
    from .harness.scenarios import multi_class

    dataset = multi_class()
    expected = snapshot(dataset, decode=True)
    real_get_input = ObjectDetectionDataset.get_input
    monkeypatch.setattr(ObjectDetectionDataset, "get_input", lambda self, i: np.zeros_like(real_get_input(self, i)))
    differences = compare(expected, snapshot(dataset, decode=True))
    assert len(differences) == 4
    assert all(": decoded_sha256 expected '" in difference for difference in differences)


@pytest.mark.optional
def test_snapshot_with_decode_holds_one_decoded_image_at_a_time(monkeypatch) -> None:
    """A real dataset's decoded images would not fit in memory together."""
    import weakref

    from .harness import snapshot
    from .harness.scenarios import multi_class

    dataset = multi_class()
    real_get_input = ObjectDetectionDataset.get_input
    decoded: list[weakref.ref[Any]] = []
    alive_at_each_decode = []

    def tracked_get_input(self: ObjectDetectionDataset, index: int) -> Any:
        alive_at_each_decode.append(sum(ref() is not None for ref in decoded))
        image = real_get_input(self, index)
        decoded.append(weakref.ref(image))
        return image

    monkeypatch.setattr(ObjectDetectionDataset, "get_input", tracked_get_input)
    snapshot(dataset, decode=True)
    assert len(alive_at_each_decode) == len(dataset.samples) > 1
    assert max(alive_at_each_decode) <= 1


def test_snapshot_flags_an_unused_class_missing_from_the_public_index2label(monkeypatch) -> None:
    """No box uses rich()'s 'truck', but MAITE's index2label must still list it, and nothing else."""
    from .harness import snapshot
    from .harness.scenarios import rich

    monkeypatch.setattr(ObjectDetectionDataset, "index2label", lambda _self: {0: "person", 1: "car", 2: "bus", 9: "x"})
    assert snapshot(rich()).problems == (
        "taxonomy: class 'truck' (id 3) is not in the MAITE index2label as 'truck'",
        "taxonomy: MAITE index2label maps 9 to 'x', which is no class",
    )


def test_problems_flags_a_taxonomy_with_a_repeated_or_missing_class_id() -> None:
    """Entries with no id or a repeated id were dropped or overwritten silently; now each is a problem."""
    from .harness import problems, snapshot
    from .harness.scenarios import box, image

    entries = (
        CategoryEntry(source_id=1, name="car"),
        CategoryEntry(source_id=1, name="auto"),
        CategoryEntry(source_id=None, name="bus"),  # type: ignore[arg-type]
    )
    dataset = ObjectDetectionDataset(
        samples=(image(1, "a.png", box(1, "auto", 2, 3), data=b"pixels"),),
        dataset_metadata=DatasetMetadata(taxonomy=Taxonomy(entries=entries)),
    )

    found = problems(dataset)
    assert found == [
        "taxonomy: class 'bus' has no id",
        "taxonomy: class id 1 is used by 2 entries ['car', 'auto']",
    ]
    snap = snapshot(dataset)
    assert snap.problems == tuple(found)
    # Every entry is still in the snapshot, the one without an id included.
    assert [entry["name"] for entry in snap.entries] == ["car", "auto", "bus"]
    assert compare(snap, snap)[:2] == [f"problem: {problem}" for problem in found]


def test_problems_flags_ordered_names_that_are_not_the_entries_names() -> None:
    """``ordered_names`` is excluded as derived from the entries; a taxonomy where it is not must be flagged."""
    from .harness import problems

    taxonomy = Taxonomy(entries=(CategoryEntry(source_id=0, name="car"),), ordered_names=("bus",))
    dataset = ObjectDetectionDataset(samples=(), dataset_metadata=DatasetMetadata(taxonomy=taxonomy))
    assert problems(dataset) == ["taxonomy: ordered_names ['bus'] are not the entries' names ['car']"]


def test_box_matching_does_not_depend_on_box_order() -> None:
    # x=10.005 is within tolerance of both expected boxes, x=9.995 only of the first. Matching
    # the first expected box greedily to whichever comes first fails for one of the two orders.
    expected_boxes = (_box("car", 1, x=10.0), _box("car", 1, x=10.01))
    actual_boxes = (_box("car", 1, x=10.005), _box("car", 1, x=9.995))
    expected = _snap(_sample("a.png", *expected_boxes), labels={1: "car"})

    for boxes in (actual_boxes, actual_boxes[::-1]):
        assert compare(expected, _snap(_sample("a.png", *boxes), labels={1: "car"})) == []


def test_a_rename_is_allowed_only_for_colliding_names() -> None:
    first, second = _sample("a.png", _box("car", 1), sha="one"), _sample("a.png", _box("bus", 2), sha="two")
    expected = _snap(first, second, labels={1: "car", 2: "bus"})
    renamed = _snap(first, replace(_with(second, file_name="a_1.png"), key="a_1.png"), labels={1: "car", 2: "bus"})
    relabelled_second = _sample("a_1.png", _box("car", 1), sha="two")
    relabelled = _snap(first, relabelled_second, labels={1: "car", 2: "bus"})

    assert compare(expected, renamed) == []
    assert compare(expected, relabelled) == [
        "a.png [split train]: image_id expected 'a.png', got 'a_1.png'",
        "a.png [split train]: missing box 'bus' (id 2) at [1, 2, 3, 4]",
        "a.png [split train]: unexpected box 'car' (id 1) at [1, 2, 3, 4]",
    ]
    lone = _snap(_sample("b.png"), labels={})
    assert compare(lone, _snap(replace(_sample("b.png"), key="c.png"), labels={})) == [
        "b.png [split train]: image was renamed to c.png [split train]"
    ]


def test_label_file_stems_allow_a_rename_only_where_stems_collide_in_a_split() -> None:
    """YOLO names label files after image stems: a.png and a.jpg collide, b.png and b.jpg in two splits don't."""
    samples = (
        _sample("a.png", _box("car", 1), sha="a1"),
        _sample("a.jpg", _box("car", 1, x=5.0), sha="a2"),
        _sample("b.png", sha="b1"),
        _sample("b.jpg", split="val", sha="b2"),
    )
    expected = _snap(*samples, labels={1: "car"})

    def renamed(index: int, name: str) -> DatasetSnapshot:
        moved = replace(_with(samples[index], file_name=name, image_id=name), key=name)
        return _snap(*samples[:index], moved, *samples[index + 1 :], labels={1: "car"})

    allowed = {Loss.LABEL_FILE_STEMS, Loss.IMAGE_IDS}
    assert compare(expected, renamed(1, "a_2.jpg"), allowed=allowed) == []
    assert compare(expected, renamed(3, "b_2.jpg"), allowed=allowed) == [
        "b.jpg [split val]: image was renamed to b_2.jpg [split val]"
    ]
    # The renamed image keeps its boxes.
    lost_box = replace(_with(samples[1], file_name="a_2.jpg", detections=[]), key="a_2.jpg")
    assert compare(expected, _snap(samples[0], lost_box, *samples[2:], labels={1: "car"}), allowed=allowed) == [
        "a.jpg [split train]: missing box 'car' (id 1) at [5, 2, 3, 4]"
    ]


def test_renamed_images_with_the_same_pixels_pair_with_their_own_boxes() -> None:
    """Three copies of one image named a.png, written as a.png, a_2.png and a_3.png in another order."""
    records = [
        _with(_sample("a.png", _box("car", 1, x=x), sha="same"), image_id=image_id)
        for image_id, x in ((1, 1.0), (2, 5.0), (3, 9.0))
    ]

    def renamed(sample: SampleSnapshot, key: str) -> SampleSnapshot:
        return replace(_with(sample, file_name=key), key=key)

    labels: dict[int | str, str] = {1: "car"}
    expected = _snap(*records, labels=labels)
    written = _snap(records[2], renamed(records[0], "a_2.png"), renamed(records[1], "a_3.png"), labels=labels)
    moved_box = _snap(
        records[2],
        renamed(records[0], "a_2.png"),
        renamed(_with(records[1], detections=[_box("car", 1, x=20.0).fields]), "a_3.png"),
        labels=labels,
    )

    assert compare(expected, written) == []
    assert compare(expected, moved_box) == [
        "a.png [split train]: missing box 'car' (id 1) at [5, 2, 3, 4]",
        "a.png [split train]: unexpected box 'car' (id 1) at [20, 2, 3, 4]",
    ]


def test_a_renamed_image_must_keep_its_file_name_unless_its_name_collided() -> None:
    """Pairing by key normalises the file name; the file name itself is still compared."""
    expected = _snap(_sample("a.png"), labels={})
    assert compare(expected, _snap(_with(_sample("a.png"), file_name="./a.png"), labels={})) == [
        "a.png [split train]: file_name expected 'a.png', got './a.png'"
    ]


def test_compare_reports_a_lost_sample_changed_pixels_and_a_relabelled_box() -> None:
    expected = _snap(_sample("a.png", _box("car", 1)), _sample("b.png"), labels={1: "car", 2: "bus"})

    lost = _snap(_sample("a.png", _box("car", 1)), labels={1: "car", 2: "bus"})
    repainted = _snap(_sample("a.png", _box("car", 1), sha="other"), _sample("b.png"), labels={1: "car", 2: "bus"})
    relabelled = _snap(_sample("a.png", _box("bus", 2)), _sample("b.png"), labels={1: "car", 2: "bus"})

    assert compare(expected, lost) == ["b.png [split train]: missing (0 box(es))"]
    assert compare(expected, repainted) == ["a.png [split train]: image_sha256 expected 'sha-a.png', got 'other'"]
    assert compare(expected, relabelled) == [
        "a.png [split train]: missing box 'car' (id 1) at [1, 2, 3, 4]",
        "a.png [split train]: unexpected box 'bus' (id 2) at [1, 2, 3, 4]",
    ]


def test_compare_reports_a_changed_crowd_flag_and_class_details() -> None:
    detailed = (
        CategoryEntry(source_id=1, name="car", supercategory="vehicle", synset="car.n.01", attributes={"a": 1}),
    )
    expected = _snap(_sample("a.png", _box("car", 1, iscrowd=1)), labels={}, entries=detailed)
    not_crowd = replace(expected, samples=(_sample("a.png", _box("car", 1)),))
    no_details = _snap(
        *expected.samples, labels={}, entries=(CategoryEntry(source_id=1, name="car", supercategory="vehicle"),)
    )

    assert compare(expected, not_crowd) == [
        "a.png [split train]: missing box 'car' (id 1) at [1, 2, 3, 4] (crowd)",
        "a.png [split train]: unexpected box 'car' (id 1) at [1, 2, 3, 4]",
    ]
    assert compare(expected, no_details) == [
        "class 'car' (id 1): attributes['a'] expected 1, got <missing>",
        "class 'car' (id 1): synset expected 'car.n.01', got None",
    ]
    plain = _snap(*not_crowd.samples, labels={1: "car"})
    assert compare(expected, plain, allowed={Loss.CROWD, Loss.LABEL_DETAILS}) == []
    assert compare(expected, plain, allowed={Loss.CROWD}) == [
        "class 'car' (id 1): attributes['a'] expected 1, got <missing>",
        "class 'car' (id 1): supercategory expected 'vehicle', got None",
        "class 'car' (id 1): synset expected 'car.n.01', got None",
    ]


def test_compare_reports_every_class_field_and_the_class_order() -> None:
    expected = _snap(labels={10: "bus", 2: "car"})

    assert compare(expected, _snap(labels={10: "bus", 2: "auto"})) == [
        "labels {id: name}: expected {2: 'car', 10: 'bus'}, got {2: 'auto', 10: 'bus'}",
        "class 'car' (id 2): name expected 'car', got 'auto'",
    ]
    assert compare(expected, _snap(labels={2: "car", 10: "bus"})) == ["class order: expected ids [10, 2], got [2, 10]"]
    excluded = (CategoryEntry(source_id=10, name="bus", eval_excluded=True), CategoryEntry(source_id=2, name="car"))
    assert compare(expected, _snap(labels={}, entries=excluded)) == [
        "class 'bus' (id 10): eval_excluded expected False, got True"
    ]


def test_compare_reports_the_taxonomys_own_fields() -> None:
    """The taxonomy's own fields were never compared before; ``id_density`` is now, and its losses model it."""
    sparse = _snap(labels={0: "car", 1: "bus"})
    dense_tree = sparse.tree
    dense_tree["dataset_metadata"]["taxonomy"]["id_density"] = "dense"
    dense = sparse.with_tree(dense_tree)

    assert compare(sparse, dense) == ["dataset: dataset_metadata.taxonomy.id_density expected 'sparse', got 'dense'"]
    assert compare(sparse, dense, allowed={Loss.ID_DENSITY_FROM_IDS}) == []
    assert compare(dense, sparse, allowed={Loss.ID_DENSITY}) == []
    assert compare(sparse, dense, allowed={Loss.ID_DENSITY}) == [
        "dataset: dataset_metadata.taxonomy.id_density expected 'sparse', got 'dense'"
    ]
    gappy = _snap(labels={0: "car", 2: "bus"})
    assert compare(gappy, gappy, allowed={Loss.ID_DENSITY_FROM_IDS}) == []
    # Renumbered 0..n-1, the ids are dense.
    renumbered_tree = _snap(labels={0: "car", 1: "bus"}).tree
    renumbered_tree["dataset_metadata"]["taxonomy"]["id_density"] = "dense"
    renumbered = sparse.with_tree(renumbered_tree)
    assert compare(gappy, renumbered, allowed=[{Loss.LABEL_IDS, Loss.ID_DENSITY_FROM_IDS}]) == []


def test_output_problems_reads_coco_files_directly(tmp_path) -> None:
    import json

    from .harness.output import output_problems

    (tmp_path / "annotations").mkdir()
    (tmp_path / "a.png").write_bytes(b"png")
    document = {
        "images": [{"id": 1, "file_name": "a.png"}, {"id": 1, "file_name": "gone.png"}],
        "annotations": [
            {"id": 7, "image_id": 1, "category_id": 3},
            {"id": 7, "image_id": 2, "category_id": 9},
        ],
        "categories": [{"id": 3, "name": "car"}],
    }
    (tmp_path / "annotations" / "instances.json").write_text(json.dumps(document))

    assert output_problems(tmp_path, "coco") == [
        "output: annotations/instances.json: repeated image ids ['1']",
        "output: annotations/instances.json: repeated annotation ids ['7']",
        "output: annotations/instances.json: annotation 7 refers to a missing image",
        "output: annotations/instances.json: annotation 7 refers to a missing category",
        "output: annotations/instances.json: image 'gone.png' is not on disk",
    ]


def test_output_problems_reads_yolo_files_directly(tmp_path) -> None:
    from .harness.output import output_problems

    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train" / "a.png").write_bytes(b"png")
    (tmp_path / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n2 0.5 0.5 0.1 0.1\n1 0.5 1.5 0.1\n")
    (tmp_path / "labels" / "train" / "orphan.txt").write_text("")
    (tmp_path / "data.yaml").write_text("train: images/train\nnames: [car, bus]\n")

    assert output_problems(tmp_path, "yolo") == [
        "output: labels/train/a.txt: line 2: class 2 is outside names (0..1)",
        "output: labels/train/a.txt: line 3: expected 4 coordinates in [0, 1], got [0.5, 1.5, 0.1]",
        "output: labels/train/orphan.txt: no image for this label file",
    ]
    (tmp_path / "data.yaml").write_text("names:\n  0: car\n  2: bus\n")
    assert output_problems(tmp_path, "yolo") == ["output: data.yaml: names keys [0, 2] are not 0..1"]


def test_output_problems_does_not_take_a_folder_for_a_yolo_image(tmp_path) -> None:
    from .harness.output import output_problems

    (tmp_path / "images" / "train" / "a").mkdir(parents=True)
    (tmp_path / "labels" / "train").mkdir(parents=True)
    (tmp_path / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (tmp_path / "data.yaml").write_text("names: [car]\n")

    assert output_problems(tmp_path, "yolo") == ["output: labels/train/a.txt: no image for this label file"]


def test_output_problems_checks_the_yolo_split_paths(tmp_path) -> None:
    """datamaite's loader finds the folders anyway, so only this check sees a wrong split path."""
    from .harness.output import output_problems

    (tmp_path / "data" / "images" / "train").mkdir(parents=True)
    (tmp_path / "data.yaml").write_text(
        "path: data\ntrain: images/train\nval: [images/val, images/extra]\ntest: 3\nnc: 999\nnames: [car]\n"
    )

    assert output_problems(tmp_path, "yolo") == [
        "output: data.yaml: val path 'images/val' is not on disk",
        "output: data.yaml: val path 'images/extra' is not on disk",
        "output: data.yaml: test is 3, not a path",
        "output: data.yaml: nc is 999 but names has 1 classes",
    ]


@pytest.mark.parametrize(
    ("train", "problems"),
    [
        ("data.yaml", ["output: data.yaml: train path 'data.yaml' is not a folder or a .txt list"]),
        ("empty_images", []),
    ],
)
def test_output_problems_checks_that_the_yolo_split_paths_reach_the_images(tmp_path, train, problems) -> None:
    """A split path that exists but is not where the images are reads back fine in datamaite."""
    from .harness.output import output_problems

    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train" / "a.png").write_bytes(b"png")
    (tmp_path / "empty_images").mkdir()
    (tmp_path / "data.yaml").write_text(f"train: {train}\nnames: [car]\n")

    unreached = "output: data.yaml: no split path reaches images/train/a.png"
    assert output_problems(tmp_path, "yolo") == [*problems, unreached]


def test_output_problems_reads_a_yolo_image_list(tmp_path) -> None:
    from .harness.output import output_problems

    (tmp_path / "images").mkdir()
    for name in ("a.png", "b.png"):
        (tmp_path / "images" / name).write_bytes(b"png")
    (tmp_path / "train.txt").write_text("images/a.png\n")
    (tmp_path / "data.yaml").write_text("train: train.txt\nnames: [car]\n")

    assert output_problems(tmp_path, "yolo") == ["output: data.yaml: no split path reaches images/b.png"]


def test_output_problems_needs_a_yolo_split_path_for_written_images(tmp_path) -> None:
    from .harness.output import output_problems

    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "train" / "a.png").write_bytes(b"png")
    (tmp_path / "data.yaml").write_text("names: [car]\n")

    assert output_problems(tmp_path, "yolo") == ["output: data.yaml: no split path, though images/ holds 1 image(s)"]


def test_output_problems_checks_each_image_in_a_yolo_image_list(tmp_path) -> None:
    from .harness.output import output_problems

    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.png").write_bytes(b"png")
    (tmp_path / "train.txt").write_text("images/a.png\nimages/gone.png\n")
    (tmp_path / "data.yaml").write_text("train: train.txt\nnames: [car]\n")

    assert output_problems(tmp_path, "yolo") == [
        "output: data.yaml: train list 'train.txt' names 'images/gone.png', not on disk"
    ]


def test_default_split_fills_only_missing_splits_with_train() -> None:
    expected = _snap(_sample("a.png", split=None), _sample("b.png", split="val"), labels={})

    assert (
        compare(
            expected, _snap(_sample("a.png"), _sample("b.png", split="val"), labels={}), allowed={Loss.DEFAULT_SPLIT}
        )
        == []
    )
    assert compare(
        expected,
        _snap(_sample("a.png", split="test"), _sample("b.png", split="val"), labels={}),
        allowed={Loss.DEFAULT_SPLIT},
    ) == ["a.png [split train]: split expected 'train', got 'test'"]


def test_image_ids_loss_ignores_only_image_ids() -> None:
    expected = _snap(_sample("a.png", _box("car", 1)), labels={1: "car"})
    new_id = replace(expected, samples=(_with(expected.samples[0], image_id="images/train/a.png"),))
    new_id_and_size = replace(new_id, samples=(_with(new_id.samples[0], width=10),))

    assert compare(expected, new_id, allowed={Loss.IMAGE_IDS}) == []
    assert compare(expected, new_id, allowed=set()) == [
        "a.png [split train]: image_id expected 'a.png', got 'images/train/a.png'"
    ]
    assert compare(expected, new_id_and_size, allowed={Loss.IMAGE_IDS}) == [
        "a.png [split train]: width expected 64, got 10"
    ]


# ---------------------------------------------------------------------------
# Every other record field
# ---------------------------------------------------------------------------


def _with_box(box: BoxSnapshot, *, width: int = 64, height: int = 48) -> DatasetSnapshot:
    return _snap(_sample("a.png", box, width=width, height=height), labels={1: "car"})


def test_compare_reports_each_other_box_field_of_a_matched_box() -> None:
    polygon = [[1.0, 2.0, 4.0, 2.0, 4.0, 6.0]]
    expected = _with_box(_box("car", 1, segmentation=polygon, attributes={"occluded": True, "colour": "red"}))
    moved_point = _with_box(
        _box("car", 1, segmentation=[[1.0, 2.0, 4.0, 2.0, 4.0, 9.0]], attributes={"occluded": True, "colour": "red"})
    )
    lost_attribute = _with_box(_box("car", 1, segmentation=polygon, attributes={"colour": "red"}))

    assert compare(expected, moved_point) == [
        "a.png [split train]: box 'car' (id 1) at [1, 2, 3, 4]: segmentation expected "
        "[[1.0, 2.0, 4.0, 2.0, 4.0, 6.0]], got [[1.0, 2.0, 4.0, 2.0, 4.0, 9.0]]"
    ]
    assert compare(expected, lost_attribute) == [
        "a.png [split train]: box 'car' (id 1) at [1, 2, 3, 4]: attributes['occluded'] expected True, got <missing>"
    ]


def test_box_geometry_is_compared_in_pixels_and_every_other_number_closely() -> None:
    polygon = [[1.0, 2.0, 4.0, 2.0]]
    expected = _with_box(_box("car", 1, segmentation=polygon, area=12.0, score=0.5))

    # Box and polygon coordinates are pixels: equal within 0.01.
    within = _box("car", 1, x=1.004, segmentation=[[1.004, 2.0, 4.0, 1.996]], area=12.0, score=0.5)
    assert compare(expected, _with_box(within)) == []
    moved = _box("car", 1, segmentation=[[1.02, 2.0, 4.0, 2.0]], area=12.0, score=0.5)
    assert compare(expected, _with_box(moved)) == [
        "a.png [split train]: box 'car' (id 1) at [1, 2, 3, 4]: segmentation expected "
        "[[1.0, 2.0, 4.0, 2.0]], got [[1.02, 2.0, 4.0, 2.0]]"
    ]
    # Any other number is equal only within a relative 1e-6: a score of 0.5 is not 0.509.
    assert compare(expected, _with_box(_box("car", 1, segmentation=polygon, area=12.000001, score=0.5000001))) == []
    assert compare(expected, _with_box(_box("car", 1, segmentation=polygon, area=12.009, score=0.509))) == [
        "a.png [split train]: box 'car' (id 1) at [1, 2, 3, 4]: area expected 12.0, got 12.009",
        "a.png [split train]: box 'car' (id 1) at [1, 2, 3, 4]: score expected 0.5, got 0.509",
    ]
    # Two integers are exact: a large id one off is a different id, though within 1e-6 of it.
    big_ids = _with_box(_box("car", 1, source_annotation_id=10_000_000))
    assert compare(big_ids, _with_box(_box("car", 1, source_annotation_id=10_000_001))) == [
        "a.png [split train]: box 'car' (id 1) at [1, 2, 3, 4]: source_annotation_id expected 10000000, got 10000001"
    ]
    assert compare(
        _snap(_sample("a.png", image_id=10_000_000), labels={}), _snap(_sample("a.png", image_id=10_000_001), labels={})
    ) == ["a.png [split train]: image_id expected 10000000, got 10000001"]
    # NaN is NaN, and a flag is not a number: True is not 1.
    nan = _with_box(_box("car", 1, attributes={"weight": float("nan")}))
    assert compare(nan, _with_box(_box("car", 1, attributes={"weight": float("nan")}))) == []
    flagged = _with_box(_box("car", 1, attributes={"occluded": True}))
    assert compare(flagged, _with_box(_box("car", 1, attributes={"occluded": 1}))) == [
        "a.png [split train]: box 'car' (id 1) at [1, 2, 3, 4]: attributes['occluded'] expected True, got 1"
    ]


def test_boxes_that_differ_only_in_other_fields_pair_with_their_own() -> None:
    first, second = _box("car", 1, segmentation=[[1.0]]), _box("car", 1, segmentation=[[2.0]])
    expected = _snap(_sample("a.png", first, second), labels={1: "car"})

    for boxes in ((first, second), (second, first)):
        assert compare(expected, _snap(_sample("a.png", *boxes), labels={1: "car"})) == []
    changed = _snap(_sample("a.png", first, _box("car", 1, segmentation=[[3.0]])), labels={1: "car"})
    assert compare(expected, changed) == [
        "a.png [split train]: box 'car' (id 1) at [1, 2, 3, 4]: segmentation expected [[2.0]], got [[3.0]]"
    ]


def test_box_matching_still_pairs_as_many_boxes_as_possible() -> None:
    """Pairing boxes with equal fields first must not leave a box unmatched that a plain match would pair."""
    near, far = _box("car", 1, x=10.0, area=1.0), _box("car", 1, x=10.01, area=2.0)
    expected = _snap(_sample("a.png", near, far), labels={1: "car"})
    # The first actual box has near's area but is within tolerance of both; the second only of near.
    actual = _snap(
        _sample("a.png", _box("car", 1, x=10.005, area=1.0), _box("car", 1, x=9.995, area=3.0)), labels={1: "car"}
    )

    differences = compare(expected, actual)
    assert not any("missing box" in d or "unexpected box" in d for d in differences)
    assert len(differences) == 2


def test_a_small_number_outside_box_geometry_is_not_rounded_away() -> None:
    """An exposure of 0.008 written as 0.0 is a lost value, though it is within the box tolerance."""
    expected = _snap(_sample("a.png", metadata={"capture": {"exposure": 0.008}}), labels={})
    actual = _snap(_sample("a.png", metadata={"capture": {"exposure": 0.0}}), labels={})

    assert compare(expected, actual) == ["a.png [split train]: metadata['capture']['exposure'] expected 0.008, got 0.0"]


def test_compare_reports_image_metadata_key_by_key_and_dataset_fields() -> None:
    metadata = {"license": 1, "capture": {"camera": "cam1"}}
    expected = _snap(_sample("a.png", metadata=metadata), labels={}, info={"year": 2024}, licenses=({"id": 1},))
    actual = _snap(
        _sample("a.png", metadata={"capture": {"camera": "cam2"}}), labels={}, info={"year": 2024.0}, licenses=()
    )

    assert compare(expected, actual) == [
        "dataset: dataset_metadata.licenses expected [{'id': 1}], got []",
        "a.png [split train]: metadata['capture']['camera'] expected 'cam1', got 'cam2'",
        "a.png [split train]: metadata['license'] expected 1, got <missing>",
    ]


def test_box_field_losses_each_clear_only_their_field() -> None:
    full = {
        "score": 0.5,
        "segmentation": [[1.0, 2.0, 3.0, 4.0]],
        "area": 9.0,
        "attributes": {"occluded": True},
        "source_annotation_id": 7,
    }
    defaults = field_defaults(ObjectDetectionAnnotation)
    expected = _with_box(_box("car", 1, **full))
    losses = {
        "score": Loss.SCORE,
        "segmentation": Loss.SEGMENTATION,
        "area": Loss.AREA,
        "attributes": Loss.BOX_ATTRIBUTES,
        "source_annotation_id": Loss.ANNOTATION_IDS,
    }
    for field, loss in losses.items():
        cleared = _with_box(_box("car", 1, **{**full, field: defaults[field]}))
        assert compare(expected, cleared, allowed={loss}) == [], loss
        assert len(compare(expected, cleared)) == 1, loss
        # Every other loss together still leaves this field as it was.
        all_cleared = _with_box(_box("car", 1))
        assert len(compare(expected, all_cleared, allowed=set(losses.values()) - {loss})) == 1, loss


def test_area_from_box_fills_only_a_missing_area() -> None:
    no_area = _with_box(_box("car", 1))
    with_area = _with_box(_box("car", 1, area=5.0))

    assert compare(no_area, _with_box(_box("car", 1, area=12.0)), allowed={Loss.AREA_FROM_BOX}) == []
    # The 3x4 box's width and height may each come back 0.01 off, so its area up to 0.01 * (3 + 4) + 0.01².
    assert compare(no_area, _with_box(_box("car", 1, area=3.01 * 4.01)), allowed={Loss.AREA_FROM_BOX}) == []
    assert compare(no_area, _with_box(_box("car", 1, area=12.08)), allowed={Loss.AREA_FROM_BOX}) == [
        "a.png [split train]: box 'car' (id 1) at [1, 2, 3, 4]: area expected 12.0, got 12.08"
    ]
    assert compare(no_area, no_area, allowed={Loss.AREA_FROM_BOX}) == [
        "a.png [split train]: box 'car' (id 1) at [1, 2, 3, 4]: area expected 12.0, got None"
    ]
    assert compare(with_area, _with_box(_box("car", 1, area=12.0)), allowed={Loss.AREA_FROM_BOX}) == [
        "a.png [split train]: box 'car' (id 1) at [1, 2, 3, 4]: area expected 5.0, got 12.0"
    ]


def test_new_annotation_ids_allow_any_integer_only_where_there_was_none() -> None:
    no_id, with_id = _with_box(_box("car", 1)), _with_box(_box("car", 1, source_annotation_id=42))

    assert compare(no_id, _with_box(_box("car", 1, source_annotation_id=3)), allowed={Loss.NEW_ANNOTATION_IDS}) == []
    assert compare(no_id, _with_box(_box("car", 1, source_annotation_id="3")), allowed={Loss.NEW_ANNOTATION_IDS}) == [
        "a.png [split train]: box 'car' (id 1) at [1, 2, 3, 4]: source_annotation_id expected <any int>, got '3'"
    ]
    assert compare(with_id, _with_box(_box("car", 1, source_annotation_id=3)), allowed={Loss.NEW_ANNOTATION_IDS}) == [
        "a.png [split train]: box 'car' (id 1) at [1, 2, 3, 4]: source_annotation_id expected 42, got 3"
    ]


def test_source_category_ids_come_back_as_the_class_id_after_renumbering() -> None:
    expected = _snap(_sample("a.png", _box("car", 3, source_category_id="n02958343")), labels={3: "car"})
    renumbered = _snap(_sample("a.png", _box("car", 0, source_category_id=0)), labels={0: "car"})

    assert compare(expected, renumbered, allowed=[{Loss.LABEL_IDS, Loss.SOURCE_CATEGORY_IDS}]) == []
    assert compare(expected, renumbered, allowed={Loss.LABEL_IDS}) == [
        "a.png [split train]: box 'car' (id 0) at [1, 2, 3, 4]: source_category_id expected 'n02958343', got 0"
    ]


def test_dataset_and_image_field_losses_each_clear_only_their_field() -> None:
    expected = _snap(
        _sample("a.png", metadata={"license": 1}),
        labels={},
        splits=("train",),
        info={"year": 2024},
        licenses=({"id": 1},),
        extras={"note": "x"},
    )
    plain = _snap(_sample("a.png"), labels={})
    every_loss = {Loss.IMAGE_METADATA, Loss.SPLIT_LIST, Loss.INFO, Loss.LICENSES, Loss.DATASET_EXTRAS}

    assert compare(expected, plain, allowed=every_loss) == []
    for loss in every_loss:
        assert len(compare(expected, plain, allowed=every_loss - {loss})) == 1, loss


def test_class_field_losses_each_clear_only_their_field() -> None:
    entries = (CategoryEntry(source_id=1, name="car", eval_excluded=True, source_dataset="coco"),)
    expected = _snap(labels={1: "car"}, entries=entries)
    plain = _snap(labels={1: "car"})
    every_loss = {Loss.CLASS_EVAL_EXCLUDED, Loss.CLASS_SOURCE_DATASET}

    assert compare(expected, plain, allowed=every_loss) == []
    for loss in every_loss:
        assert len(compare(expected, plain, allowed=every_loss - {loss})) == 1, loss


def test_split_list_from_images_lists_the_splits_in_yolo_order() -> None:
    samples = (
        _sample("a.png", split="holdout"),
        _sample("b.png", split="test"),
        _sample("c.png"),
        _sample("d.png", split="extra"),
    )
    expected = _snap(*samples, labels={})

    listed = _snap(*samples, labels={}, splits=("train", "test", "extra", "holdout"))
    assert compare(expected, listed, allowed={Loss.SPLIT_LIST_FROM_IMAGES}) == []
    reordered = _snap(*samples, labels={}, splits=("extra", "holdout", "test", "train"))
    assert len(compare(expected, reordered, allowed={Loss.SPLIT_LIST_FROM_IMAGES})) == 1


def test_yolo_source_details_replace_nothing_but_add_the_loaders_provenance() -> None:
    box = _box("car", 1, attributes={"occluded": True})
    expected = _snap(_sample("a.png", box, metadata={"license": 1}), labels={1: "car"})
    provenance = {
        "source_format": "yolo",
        "variant": "default",
        "source_file_name": "images/train/a.png",
        "label_file": "labels/train/a.txt",
    }
    # Six decimals, as a label row stores them.
    row = {"yolo_bbox": [0.039062, 0.083333, 0.046875, 0.083333], "source_line": 1}
    loaded = _snap(_sample("a.png", _box("car", 1, attributes=row), metadata=provenance), labels={1: "car"})
    yolo = {Loss.IMAGE_METADATA, Loss.BOX_ATTRIBUTES, Loss.YOLO_SOURCE_DETAILS}

    assert compare(expected, loaded, allowed=yolo) == []
    # Without the metadata and attribute losses, the source's own values must still be there.
    assert compare(expected, loaded, allowed={Loss.YOLO_SOURCE_DETAILS}) == [
        "a.png [split train]: metadata['license'] expected 1, got <missing>",
        "a.png [split train]: box 'car' (id 1) at [1, 2, 3, 4]: attributes['occluded'] expected True, got <missing>",
    ]
    moved_row = _snap(
        _sample("a.png", _box("car", 1, attributes={**row, "yolo_bbox": [0.5, 0.5, 0.1, 0.1]}), metadata=provenance),
        labels={1: "car"},
    )
    assert len(compare(expected, moved_row, allowed=yolo)) == 1


# ---------------------------------------------------------------------------
# The canonical walk
# ---------------------------------------------------------------------------


def test_canonical_values_ignore_numpy_types_tuples_and_enums() -> None:
    import numpy as np

    from datamaite._types import Task

    assert canonical({np.int64(1): (np.float32(0.5), [np.int32(2)])}) == {1: [0.5, [2]]}
    assert type(canonical(np.int64(3))) is int
    assert type(canonical(np.bool_(True))) is bool
    assert canonical(np.array([[1, 2]])) == [[1, 2]]
    assert canonical({(1, 2): "pair"}) == {(1, 2): "pair"}
    assert canonical(Task.OD) == "od"
    assert canonical(b"abc").startswith("<3 bytes, sha256 ")


def test_canonical_refuses_a_type_it_does_not_know_and_names_where_it_is() -> None:
    from pathlib import Path

    from .harness.scenarios import box, image

    sample = replace(image(1, "a.png", box(1, "car", 2, 3), data=b"pixels"), metadata={"source": Path("a.png")})
    dataset = ObjectDetectionDataset(samples=(sample,))
    with pytest.raises(TypeError, match=r"^dataset\.samples\[0\]\.metadata\['source'\]: .* PosixPath$"):
        canonical(dataset)
    with pytest.raises(TypeError, match=r"^dataset\[0\]: .* object$"):
        canonical([object()])


def test_snapshot_hashes_the_image_wherever_it_is_stored(tmp_path) -> None:
    from .harness import snapshot
    from .harness.scenarios import image

    (tmp_path / "a.png").write_bytes(b"pixels")
    in_memory = image(1, "a.png", data=b"pixels")
    on_disk = replace(in_memory, image_bytes=None, path_or_uri=str(tmp_path / "a.png"))
    snaps = [snapshot(ObjectDetectionDataset(samples=(sample,))) for sample in (in_memory, on_disk)]

    assert snaps[0].samples[0].image_sha256 == snaps[1].samples[0].image_sha256 is not None
    assert "image_bytes" not in snaps[0].samples[0].fields
    assert compare(*snaps) == []


# ---------------------------------------------------------------------------
# Field coverage guards
# ---------------------------------------------------------------------------


def test_every_excluded_path_matches_a_real_snapshot(tmp_path) -> None:
    """Each excluded path names a value in a loaded dataset, so the list can't go stale, and has a reason."""
    from datamaite import load_od

    from .harness import snapshot
    from .harness.formats import author_coco, author_yolo
    from .harness.scenarios import record_details
    from .harness.snapshot import matches

    for name, author in (("coco", author_coco), ("yolo", author_yolo)):
        loaded = load_od(author(record_details(), tmp_path / name), dataset_format=name)
        tree = canonical(loaded)
        for pattern, reason in EXCLUDED_PATHS.items():
            assert reason.strip(), pattern
            assert matches(tree, pattern), (name, pattern)
            assert not matches(snapshot(loaded).tree, pattern), (name, pattern)


def test_a_field_added_to_a_nested_dataclass_is_compared() -> None:
    """A field added to any record at any depth shows up in the snapshot and in compare, with no harness change."""
    from .harness import snapshot
    from .harness.scenarios import image

    @dataclasses.dataclass(frozen=True)
    class TaggedBox(ObjectDetectionAnnotation):
        reviewer: str = "ann"

    @dataclasses.dataclass(frozen=True)
    class TaggedEntry(CategoryEntry):
        colour: str = "red"

    @dataclasses.dataclass(frozen=True)
    class VersionedTaxonomy(Taxonomy):
        version: int = 1

    def build(reviewer: str, colour: str, version: int) -> ObjectDetectionDataset:
        detection = TaggedBox(bbox=(2.0, 3.0, 10.0, 8.0), category_id=1, category_name="car", reviewer=reviewer)
        taxonomy = VersionedTaxonomy(entries=(TaggedEntry(source_id=1, name="car", colour=colour),), version=version)
        return ObjectDetectionDataset(
            samples=(image(1, "a.png", detection, data=b"pixels"),),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy),
        )

    base = snapshot(build("ann", "red", 1))
    assert base.samples[0].boxes[0].fields["reviewer"] == "ann"
    assert base.entries[0]["colour"] == "red"
    assert base.fields["dataset_metadata"]["taxonomy"]["version"] == 1
    assert compare(base, snapshot(build("bob", "blue", 2))) == [
        "class 'car' (id 1): colour expected 'red', got 'blue'",
        "dataset: dataset_metadata.taxonomy.version expected 1, got 2",
        "a.png [split train]: box 'car' (id 1) at [2, 3, 10, 8]: reviewer expected 'ann', got 'bob'",
    ]


def test_snapshot_carries_every_record_field() -> None:
    """Change each field of each record in turn: unless its path is excluded, compare must report it."""
    from datamaite._types import Task

    from .harness import snapshot
    from .harness.scenarios import box, image

    def build(changes: dict[type, dict[str, object]]) -> ObjectDetectionDataset:
        def changed(record: Any) -> Any:
            return replace(record, **changes.get(type(record), {}))

        detection = changed(box(1, "car", 2, 3))
        sample = changed(replace(image(1, "a.png", data=b"pixels"), detections=(detection,)))
        entry = changed(CategoryEntry(source_id=1, name="car"))
        taxonomy = changed(Taxonomy(entries=(entry,)))
        metadata = changed(DatasetMetadata(taxonomy=taxonomy))
        return changed(ObjectDetectionDataset(samples=(sample,), dataset_metadata=metadata))

    new_values: dict[str, object] = {
        "region": (1.0, 1.0, 2.0, 2.0),
        "splits": ("train",),
        "licenses": ({"id": 1},),
        "eval_excluded": True,
        "score": 0.5,
        "area": 3.0,
        "bbox": (20.0, 3.0, 10.0, 8.0),
        "category_id": 2,
        "iscrowd": 1,
        "image_id": 7,
        "width": 32,
        "height": 32,
        "image_bytes": b"other pixels",
        "id_density": "dense",
        "task": Task.IC,
        "source_id": 2,
        "segmentation": [[1.0, 2.0, 3.0]],
    }
    # Fields whose change shows up under another name: the image's hash, the pairing of boxes and images.
    reported_as = {
        "image_bytes": "image_sha256",
        "bbox": "box",
        "category_id": "box",
        "category_name": "box",
        "iscrowd": "box",
        "file_name": "renamed",
        "source_id": "labels",
    }
    paths = {
        ObjectDetectionDataset: "",
        DatasetMetadata: "dataset_metadata.",
        Taxonomy: "dataset_metadata.taxonomy.",
        CategoryEntry: "dataset_metadata.taxonomy.entries[*].",
        ImageObjectDetectionSample: "samples[*].",
        ObjectDetectionAnnotation: "samples[*].detections[*].",
    }
    containers = {"samples", "dataset_metadata", "taxonomy", "entries", "detections"}
    base = snapshot(build({}))
    checked = []
    for record_type, prefix in paths.items():
        for field in dataclasses.fields(record_type):
            if field.name in containers or f"{prefix}{field.name}" in EXCLUDED_PATHS:
                continue
            value = new_values.get(field.name, {"changed": 1} if "dict" in str(field.type) else "changed")
            differences = compare(base, snapshot(build({record_type: {field.name: value}})))
            word = reported_as.get(field.name, field.name)
            assert any(word in difference for difference in differences), (record_type.__name__, field.name)
            checked.append(f"{prefix}{field.name}")
    assert len(checked) > 30
