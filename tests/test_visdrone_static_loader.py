"""VisDrone static-images (OD + IC) loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import datamaite
from datamaite import DatasetFormat, ImageClassificationDataset, ObjectDetectionDataset, Task, load_ic, load_od
from datamaite._formats.visdrone.static_loader import (
    VISDRONE_STATIC_CLASSES,
    VisDroneImageClassificationLoader,
    VisDroneObjectDetectionLoader,
    build_taxonomy,
    infer_split,
    parse_annotation_file,
)
from datamaite.loaders import get_loader
from datamaite.records import ImageObjectDetectionSample


class TestRegistry:
    def test_format_value(self) -> None:
        assert DatasetFormat.VISDRONE.value == "visdrone"

    def test_loaders_are_task_aware(self) -> None:
        od = get_loader(DatasetFormat.VISDRONE, task=Task.OD, variant="default")
        ic = get_loader(DatasetFormat.VISDRONE, task=Task.IC, variant="default")
        assert isinstance(od, VisDroneObjectDetectionLoader)
        assert isinstance(ic, VisDroneImageClassificationLoader)
        assert od.task is Task.OD
        assert ic.task is Task.IC

    def test_ambiguous_format_without_task_raises(self) -> None:
        # OD and IC both register under VISDRONE/default, so a task-less lookup is
        # ambiguous and must demand a task rather than silently pick one.
        with pytest.raises(ValueError, match="Multiple loaders"):
            get_loader(DatasetFormat.VISDRONE)

    def test_missing_layout_returns_empty(self, tmp_path: Path) -> None:
        od = VisDroneObjectDetectionLoader().load(tmp_path)
        ic = VisDroneImageClassificationLoader().load(tmp_path)
        assert isinstance(od, ObjectDetectionDataset)
        assert od.sample_count == 0
        assert isinstance(ic, ImageClassificationDataset)
        assert ic.sample_count == 0


class TestParsing:
    def test_parses_valid_rows_and_discards_trailing_fields(self, tmp_path):
        ann = tmp_path / "a.txt"
        ann.write_text("10,20,30,40,1,4,0,1\n5,5,8,8,0,2,1,0,999,extra\n", encoding="utf-8")
        rows = parse_annotation_file(ann)
        assert len(rows) == 2
        assert (rows[0].left, rows[0].top, rows[0].width, rows[0].height) == (10.0, 20.0, 30.0, 40.0)
        assert rows[0].category == 4
        assert rows[0].truncation == 0
        assert rows[0].occlusion == 1
        assert rows[0].score == 1.0
        assert rows[0].line_number == 1

    def test_skips_malformed_rows(self, tmp_path, caplog):
        ann = tmp_path / "b.txt"
        # short row, non-numeric, zero-area, out-of-range category, then one good row
        ann.write_text("1,2,3\nx,2,3,4,1,1,0,0\n1,1,0,5,1,1,0,0\n1,1,4,4,1,99,0,0\n1,1,4,4,1,3,0,0\n", encoding="utf-8")
        with caplog.at_level("WARNING"):
            rows = parse_annotation_file(ann)
        assert len(rows) == 1
        assert rows[0].category == 3
        assert caplog.text.count("Skipping") == 4

    def test_taxonomy_od_is_dense_12(self):
        tax = build_taxonomy(include_ignored_regions=True)
        assert len(tax.entries) == 12
        assert tax.id_density == "dense"
        assert tax.index2label()[0] == "ignored regions"
        assert tax.by_source_id(0).eval_excluded
        assert tax.by_source_id(11).eval_excluded

    def test_taxonomy_ic_default_drops_class_zero(self):
        tax = build_taxonomy(include_ignored_regions=False)
        assert len(tax.entries) == 11
        assert tax.id_density == "sparse"
        assert tax.entries[0].source_id == 1  # pedestrian
        assert tax.dense_ids()[4] == 3  # car -> dense index 3

    def test_infer_split(self):
        assert infer_split("VisDrone2019-DET-train") == "train"
        assert infer_split("VisDrone2019-DET-val") == "val"
        assert infer_split("random-folder") is None

    def test_infer_split_keeps_test_dev_and_test_challenge_identity(self):
        # The official DET roots include test-dev and test-challenge; they
        # must not collapse onto plain "test".
        assert infer_split("VisDrone2019-DET-test-dev") == "test-dev"
        assert infer_split("VisDrone2019-DET-test-challenge") == "test-challenge"
        assert infer_split("VisDrone2019-DET-test_dev") == "test-dev"
        assert infer_split("VisDrone2019-DET-test") == "test"

    def test_class_list_length(self):
        assert len(VISDRONE_STATIC_CLASSES) == 12


def _make_od_root(root: Path, *, name: str = "VisDrone2019-DET-train") -> Path:
    base = root / name
    (base / "images").mkdir(parents=True)
    (base / "annotations").mkdir(parents=True)
    (base / "images" / "0001.jpg").write_bytes(b"fake")
    (base / "images" / "0002.jpg").write_bytes(b"fake")  # label-less image
    (base / "annotations" / "0001.txt").write_text("10,20,30,40,1,4,0,1\n0,0,5,5,0,0,0,0\n", encoding="utf-8")
    return base


class TestObjectDetection:
    def test_happy_path(self, tmp_path: Path) -> None:
        base = _make_od_root(tmp_path)
        ds = load_od(base, dataset_format="visdrone")
        assert isinstance(ds, ObjectDetectionDataset)
        assert ds.task is Task.OD
        assert ds.sample_count == 2  # both images, incl. label-less
        by_id = {s.image_id: s for s in ds.samples}
        assert isinstance(by_id["0001"], ImageObjectDetectionSample)
        assert len(by_id["0001"].detections) == 2
        assert by_id["0002"].detections == ()  # label-less -> empty
        det = by_id["0001"].detections[0]
        assert det.bbox == (10.0, 20.0, 30.0, 40.0)  # verbatim xywh
        assert det.category_id == 4
        assert det.category_name == "car"
        assert det.score is None
        assert det.attributes == {
            "visdrone_category_id": det.category_id,
            "visdrone_score": 1.0,
            "truncation": 0,
            "occlusion": 1,
            "source_line": 1,
        }
        assert ds.index2label()[9] == "bus"
        assert len(ds.dataset_metadata.taxonomy.entries) == 12
        assert ds.samples[0].split == "train"
        assert ds.dataset_metadata.splits == ("train",)

    def test_missing_dirs_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "images").mkdir()  # no annotations/
        ds = load_od(tmp_path, dataset_format="visdrone")
        assert ds.sample_count == 0

    def test_reads_tif_images_by_default(self, tmp_path: Path) -> None:
        # The static writer copies source images verbatim, so .tif sources
        # (e.g. arriving via flat_images) must reload rather than vanish.
        base = tmp_path / "VisDrone2019-DET-train"
        (base / "images").mkdir(parents=True)
        (base / "annotations").mkdir(parents=True)
        (base / "images" / "0001.tif").write_bytes(b"II*\x00 fake tif")
        (base / "annotations" / "0001.txt").write_text("10,20,30,40,1,4,0,1\n", encoding="utf-8")
        ds = load_od(base, dataset_format="visdrone")
        assert ds.sample_count == 1
        assert ds.samples[0].file_name == "0001.tif"
        assert len(ds.samples[0].detections) == 1

    def test_split_from_test_dev_root(self, tmp_path: Path) -> None:
        base = _make_od_root(tmp_path, name="VisDrone2019-DET-test-dev")
        ds = load_od(base, dataset_format="visdrone")
        assert ds.samples[0].split == "test-dev"
        assert ds.dataset_metadata.splits == ("test-dev",)

    def test_orphan_annotation_warns(self, tmp_path: Path, caplog) -> None:
        base = _make_od_root(tmp_path)
        (base / "annotations" / "9999.txt").write_text("1,1,2,2,1,1,0,0\n", encoding="utf-8")
        with caplog.at_level("WARNING"):
            load_od(base, dataset_format="visdrone")
        assert "9999.txt" in caplog.text
        assert "no matching image" in caplog.text


def _make_ic_root(root: Path) -> Path:
    base = root / "VisDrone2019-DET-val"
    (base / "images").mkdir(parents=True)
    (base / "annotations").mkdir(parents=True)
    (base / "images" / "img.jpg").write_bytes(b"fake")
    # class 0 (ignored regions), car(4), bus(9)
    (base / "annotations" / "img.txt").write_text(
        "0,0,10,10,1,0,0,0\n10,20,30,40,1,4,0,1\n5,5,8,8,0,9,1,0\n", encoding="utf-8"
    )
    return base


class TestImageClassification:
    def test_default_drops_ignored_regions(self, tmp_path: Path) -> None:
        base = _make_ic_root(tmp_path)
        ds = load_ic(base, dataset_format="visdrone")
        assert isinstance(ds, ImageClassificationDataset)
        assert ds.task is Task.IC
        assert ds.sample_count == 2  # class-0 dropped
        names = [s.labels[0].category_name for s in ds.samples]
        assert names == ["car", "bus"]
        car = ds.samples[0]
        assert car.region == (10.0, 20.0, 30.0, 40.0)
        assert car.width == 30
        assert car.height == 40  # crop dims
        assert car.image_id == "img#2"  # traceable to source line
        assert car.path_or_uri.endswith("img.jpg")
        assert car.labels[0].attributes == {
            "visdrone_category_id": car.labels[0].category_id,
            "visdrone_score": 1.0,
            "truncation": 0,
            "occlusion": 1,
        }
        assert ds.index2label() == {
            0: "pedestrian",
            1: "people",
            2: "bicycle",
            3: "car",
            4: "van",
            5: "truck",
            6: "tricycle",
            7: "awning-tricycle",
            8: "bus",
            9: "motor",
            10: "others",
        }

    def test_include_ignored_regions(self, tmp_path: Path) -> None:
        base = _make_ic_root(tmp_path)
        ds = load_ic(base, dataset_format="visdrone", include_ignored_regions=True)
        assert ds.sample_count == 3
        assert len(ds.dataset_metadata.taxonomy.entries) == 12

    def test_maite_crop_indexing(self, tmp_path: Path) -> None:
        pytest.importorskip("cv2")
        import cv2
        import numpy as np

        base = tmp_path / "VisDrone2019-DET-val"
        (base / "images").mkdir(parents=True)
        (base / "annotations").mkdir(parents=True)
        # Image must be large enough that the box (bottom-right at 40, 60) isn't
        # clamped by decode_image's image-bounds clamping (see test_image_region_crop.py).
        cv2.imwrite(str(base / "images" / "img.jpg"), np.zeros((70, 70, 3), dtype=np.uint8))
        (base / "annotations" / "img.txt").write_text("10,20,30,40,1,4,0,1\n", encoding="utf-8")
        ds = load_ic(base, dataset_format="visdrone")
        image, target, _meta = ds[0]
        assert image.shape == (3, 40, 30)  # crop H x W
        assert target.shape == (11,)
        assert int(target.argmax()) == 3  # car -> dense index 3

    def test_no_annotations_returns_empty(self, tmp_path: Path) -> None:
        base = tmp_path / "VisDrone2019-DET-val"
        (base / "images").mkdir(parents=True)
        (base / "annotations").mkdir(parents=True)
        (base / "images" / "img.jpg").write_bytes(b"fake")  # no annotation file
        ds = load_ic(base, dataset_format="visdrone")
        assert ds.sample_count == 0

    def test_class_zero_only_yields_no_ic_but_od(self, tmp_path: Path) -> None:
        # A file whose only rows are class 0 (ignored regions) contributes nothing
        # to IC by default, while OD keeps the row (all 12 classes).
        base = tmp_path / "VisDrone2019-DET-val"
        (base / "images").mkdir(parents=True)
        (base / "annotations").mkdir(parents=True)
        (base / "images" / "img.jpg").write_bytes(b"fake")
        (base / "annotations" / "img.txt").write_text("0,0,10,10,1,0,0,0\n", encoding="utf-8")
        assert load_ic(base, dataset_format="visdrone").sample_count == 0
        od = load_od(base, dataset_format="visdrone")
        assert od.sample_count == 1
        assert od.samples[0].detections[0].category_id == 0  # class 0 retained in OD

    def test_maite_crop_metadata_matches_clamped_crop(self, tmp_path: Path) -> None:
        pytest.importorskip("cv2")
        import cv2
        import numpy as np

        base = tmp_path / "VisDrone2019-DET-val"
        (base / "images").mkdir(parents=True)
        (base / "annotations").mkdir(parents=True)
        cv2.imwrite(str(base / "images" / "img.jpg"), np.zeros((20, 20, 3), dtype=np.uint8))
        # Box hangs off the right edge: nominal 10x10, clamps to 5 (cols 15..20) x 10.
        (base / "annotations" / "img.txt").write_text("15,5,10,10,1,4,0,1\n", encoding="utf-8")
        ds = load_ic(base, dataset_format="visdrone")
        assert (ds.samples[0].width, ds.samples[0].height) == (10, 10)  # nominal box dims on the model
        image, _target, meta = ds[0]
        assert image.shape == (3, 10, 5)  # decoded crop is clamped to image bounds
        assert (meta["width"], meta["height"]) == (5, 10)  # metadata matches the decoded crop


class TestImageExtensions:
    """`image_extensions` accepts a bare string or a collection, case-insensitively.

    A bare string like ".jpg" must be treated as one extension, not iterated into
    the character set {'.', 'j', 'p', 'g'} (which matched nothing and loaded zero
    images before this was normalized).
    """

    def test_od_string_extension_is_not_split_into_chars(self, tmp_path: Path) -> None:
        base = _make_od_root(tmp_path)
        ds = load_od(base, dataset_format="visdrone", image_extensions=".jpg")
        assert ds.sample_count == 2

    def test_od_string_without_dot_and_uppercase(self, tmp_path: Path) -> None:
        base = _make_od_root(tmp_path)
        ds = load_od(base, dataset_format="visdrone", image_extensions="JPG")
        assert ds.sample_count == 2

    def test_ic_string_extension_is_not_split_into_chars(self, tmp_path: Path) -> None:
        base = _make_ic_root(tmp_path)
        ds = load_ic(base, dataset_format="visdrone", image_extensions=".jpg")
        assert ds.sample_count == 2  # car + bus crops

    def test_ic_collection_extension(self, tmp_path: Path) -> None:
        base = _make_ic_root(tmp_path)
        ds = load_ic(base, dataset_format="visdrone", image_extensions={".JPG", "png"})
        assert ds.sample_count == 2

    def test_non_matching_extension_yields_no_samples(self, tmp_path: Path) -> None:
        base = _make_od_root(tmp_path)
        ds = load_od(base, dataset_format="visdrone", image_extensions=".png")
        assert ds.sample_count == 0


class TestSniff:
    def test_sniff_true_on_static_root(self, tmp_path: Path) -> None:
        base = _make_od_root(tmp_path)
        assert VisDroneObjectDetectionLoader.sniff(base) is True
        assert VisDroneImageClassificationLoader.sniff(base) is True

    def test_sniff_false_without_annotations_dir(self, tmp_path: Path) -> None:
        (tmp_path / "images").mkdir()
        assert VisDroneObjectDetectionLoader.sniff(tmp_path) is False

    def test_sniff_false_on_non_visdrone_text(self, tmp_path: Path) -> None:
        (tmp_path / "images").mkdir()
        (tmp_path / "annotations").mkdir()
        (tmp_path / "annotations" / "a.txt").write_text("hello world\n", encoding="utf-8")
        assert VisDroneObjectDetectionLoader.sniff(tmp_path) is False

    def test_sniff_false_on_video_layout(self, tmp_path: Path) -> None:
        # VisDrone-Video annotation rows also have >=8 numeric comma fields; only
        # the absence of a flat images/ dir (video uses sequences/) rejects them.
        base = tmp_path / "VisDrone2019-VID-train"
        (base / "sequences").mkdir(parents=True)
        (base / "annotations").mkdir(parents=True)
        (base / "annotations" / "seq1.txt").write_text("1,1,10,20,30,40,1,4,0,0\n", encoding="utf-8")
        assert VisDroneObjectDetectionLoader.sniff(base) is False
        assert VisDroneImageClassificationLoader.sniff(base) is False

    def test_lazy_exports_resolve(self) -> None:
        assert datamaite.VisDroneObjectDetectionLoader is VisDroneObjectDetectionLoader
        assert datamaite.VisDroneImageClassificationLoader is VisDroneImageClassificationLoader
