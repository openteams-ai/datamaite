"""Tests for the COCO object-detection loader."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from datamaite import DatasetFormat, ObjectDetectionDataset, available_formats, load, load_mot, load_od

_LOADER_LOGGER = "datamaite._formats.coco.loader"


def _coco() -> dict[str, Any]:
    return {
        "info": {"year": 2017, "description": "tiny"},
        "licenses": [{"id": 1, "name": "CC", "url": "http://x"}],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "person"},
            {"id": 3, "name": "car", "supercategory": "vehicle"},
        ],
        "images": [
            {"id": 100, "file_name": "a.jpg", "width": 640, "height": 480, "license": 1},
            {"id": 101, "file_name": "sub/b.jpg", "width": 320, "height": 240},
        ],
        "annotations": [
            {
                "id": 900,
                "image_id": 100,
                "category_id": 1,
                "bbox": [10, 20, 30, 40],
                "area": 1200,
                "iscrowd": 0,
                "segmentation": [[10, 20, 40, 20, 40, 60, 10, 60]],
            },
            {"id": 901, "image_id": 100, "category_id": 3, "bbox": [5, 5, 10, 10], "area": 100, "iscrowd": 0},
        ],
    }


def _write(root: Path, payload: dict[str, Any], name: str = "instances.json") -> Path:
    ann = root / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    path = ann / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestHappyPath:
    def test_load_directory(self, tmp_path: Path) -> None:
        _write(tmp_path, _coco())
        ds = load_od(tmp_path, dataset_format="coco")
        assert isinstance(ds, ObjectDetectionDataset)
        assert ds.sample_count == 2
        assert ds.num_detections == 2
        assert ds.index2label() == {1: "person", 3: "car"}

    def test_bbox_passthrough_and_provenance(self, tmp_path: Path) -> None:
        _write(tmp_path, _coco())
        ds = load_od(tmp_path, dataset_format="coco")
        s100 = next(s for s in ds.iter_samples() if s.image_id == 100)
        assert s100.width == 640
        assert s100.height == 480
        det = s100.detections[0]
        assert det.bbox == (10.0, 20.0, 30.0, 40.0)  # COCO xywh == canonical xywh
        assert det.category_id == 1
        assert det.category_name == "person"
        assert det.source_annotation_id == 900
        assert det.area == 1200.0  # source area preserved verbatim
        assert det.segmentation == [[10, 20, 40, 20, 40, 60, 10, 60]]
        # per-image non-core keys ride in metadata
        assert s100.metadata.get("license") == 1

    def test_iscrowd_boolean_is_preserved_not_dropped(self, tmp_path: Path) -> None:
        # A JSON boolean ``iscrowd: true`` must map to 1, not be silently flipped
        # to 0 (the int parser rejects bools, so it needs explicit handling).
        payload = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5], "iscrowd": True}],
            "categories": [{"id": 1, "name": "person"}],
        }
        _write(tmp_path, payload)
        ds = load_od(tmp_path, dataset_format="coco")
        assert ds.samples[0].detections[0].iscrowd == 1

    def test_taxonomy_and_dataset_metadata(self, tmp_path: Path) -> None:
        _write(tmp_path, _coco())
        ds = load_od(tmp_path, dataset_format="coco")
        tax = ds.dataset_metadata.taxonomy
        assert tax is not None
        assert tax.by_source_id(3).supercategory == "vehicle"
        assert ds.dataset_metadata.info == {"year": 2017, "description": "tiny"}
        assert ds.dataset_metadata.licenses[0]["name"] == "CC"

    def test_image_path_resolution_relative_to_root(self, tmp_path: Path) -> None:
        _write(tmp_path, _coco())
        ds = load_od(tmp_path, dataset_format="coco")
        s101 = next(s for s in ds.iter_samples() if s.image_id == 101)
        assert s101.path_or_uri == str(tmp_path / "sub" / "b.jpg")

    def test_load_explicit_annotation_file(self, tmp_path: Path) -> None:
        path = tmp_path / "custom.json"
        path.write_text(json.dumps(_coco()), encoding="utf-8")
        ds = load_od(tmp_path, dataset_format="coco", annotation_file=path)
        assert ds.sample_count == 2

    def test_relative_annotation_file_anchors_to_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Relative override paths resolve against root, never the process CWD
        # (matching the other loaders' override style).
        _write(tmp_path / "data", _coco(), name="custom.json")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        ds = load_od(tmp_path / "data", dataset_format="coco", annotation_file="annotations/custom.json")
        assert ds.sample_count == 2

    def test_relative_images_dir_anchors_to_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write(tmp_path / "data", _coco())
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        ds = load_od(tmp_path / "data", dataset_format="coco", images_dir="images")
        s100 = next(s for s in ds.iter_samples() if s.image_id == 100)
        assert s100.path_or_uri == str(tmp_path / "data" / "images" / "a.jpg")


class TestMalformed:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        ds = load_od(tmp_path, dataset_format="coco")
        assert ds.sample_count == 0

    def test_explicit_missing_annotation_file_raises(self, tmp_path: Path) -> None:
        # A wrong user argument is not a best-effort case; it must not
        # degrade into a silently empty dataset.
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_od(tmp_path, dataset_format="coco", annotation_file=tmp_path / "nope.json")

    def test_duplicate_image_id_keeps_first(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        payload = _coco()
        payload["images"].append({"id": 100, "file_name": "dup.jpg", "width": 1, "height": 1})
        _write(tmp_path, payload)
        with caplog.at_level(logging.WARNING, logger=_LOADER_LOGGER):
            ds = load_od(tmp_path, dataset_format="coco")
        # The duplicate must not re-receive image 100's detections.
        assert ds.sample_count == 2
        assert ds.num_detections == 2
        assert "duplicate COCO image id 100" in caplog.text

    def test_orphaned_annotations_warn(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        payload = _coco()
        payload["annotations"].append({"id": 902, "image_id": 999, "category_id": 1, "bbox": [1, 2, 3, 4]})
        _write(tmp_path, payload)
        with caplog.at_level(logging.WARNING, logger=_LOADER_LOGGER):
            ds = load_od(tmp_path, dataset_format="coco")
        assert ds.num_detections == 2  # orphan not attached anywhere
        assert "Dropping 1 COCO annotation(s)" in caplog.text

    def test_multiple_annotation_jsons_warn(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        _write(tmp_path, _coco(), name="instances_train2017.json")
        empty = dict(_coco(), images=[], annotations=[])
        _write(tmp_path, empty, name="instances_val2017.json")
        with caplog.at_level(logging.WARNING, logger=_LOADER_LOGGER):
            ds = load_od(tmp_path, dataset_format="coco")
        assert ds.sample_count == 2  # sorted() picks the train file first
        assert "Multiple COCO annotation JSONs" in caplog.text
        assert "instances_val2017.json" in caplog.text

    def test_missing_category_id_kept_with_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        payload = _coco()
        payload["annotations"].append({"id": 902, "image_id": 100, "bbox": [1, 2, 3, 4]})
        _write(tmp_path, payload)
        with caplog.at_level(logging.WARNING, logger=_LOADER_LOGGER):
            ds = load_od(tmp_path, dataset_format="coco")
        unlabeled = next(d for s in ds.iter_samples() for d in s.detections if d.source_annotation_id == 902)
        assert unlabeled.category_id is None
        assert unlabeled.category_name is None
        assert "missing/invalid category_id" in caplog.text

    def test_undefined_category_id_kept_with_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        payload = _coco()
        payload["annotations"].append({"id": 902, "image_id": 100, "category_id": 999, "bbox": [1, 2, 3, 4]})
        _write(tmp_path, payload)
        with caplog.at_level(logging.WARNING, logger=_LOADER_LOGGER):
            ds = load_od(tmp_path, dataset_format="coco")
        det = next(d for s in ds.iter_samples() for d in s.detections if d.source_annotation_id == 902)
        assert det.category_id == 999  # source id preserved for round-trip
        assert det.category_name is None
        assert 999 not in ds.index2label()
        assert "not defined in categories[]" in caplog.text

    def test_missing_categories_section(self, tmp_path: Path) -> None:
        payload = _coco()
        del payload["categories"]
        _write(tmp_path, payload)
        ds = load_od(tmp_path, dataset_format="coco")
        assert ds.sample_count == 2
        assert ds.num_detections == 2
        assert ds.index2label() == {}
        det = next(iter(ds.samples[0].detections))
        assert det.category_id == 1  # source id preserved even without a taxonomy entry
        assert det.category_name is None

    def test_video_dataset_keys_warn(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        payload = _coco()
        payload["videos"] = [{"id": 1, "name": "seq"}]
        _write(tmp_path, payload)
        with caplog.at_level(logging.WARNING, logger=_LOADER_LOGGER):
            load_od(tmp_path, dataset_format="coco")
        assert "TAO-style video" in caplog.text

    def test_invalid_width_preserved_in_metadata(self, tmp_path: Path) -> None:
        payload = _coco()
        payload["images"] = [{"id": 1, "file_name": "a.jpg", "width": -5, "height": 10}]
        payload["annotations"] = []
        _write(tmp_path, payload)
        ds = load_od(tmp_path, dataset_format="coco")
        sample = ds.samples[0]
        assert sample.width is None
        assert sample.metadata["width"] == -5  # bad source value stays visible
        assert sample.height == 10
        assert "height" not in sample.metadata

    def test_huge_int_id_kept_exact(self, tmp_path: Path) -> None:
        big = 2**53 + 1  # would round to 2**53 through a float
        payload = _coco()
        payload["images"] = [{"id": big, "file_name": "a.jpg", "width": 1, "height": 1}]
        payload["annotations"] = [{"id": 1, "image_id": big, "category_id": 1, "bbox": [1, 2, 3, 4]}]
        _write(tmp_path, payload)
        ds = load_od(tmp_path, dataset_format="coco")
        assert ds.samples[0].image_id == big
        assert len(ds.samples[0].detections) == 1

    def test_skips_bad_records(self, tmp_path: Path) -> None:
        payload = _coco()
        payload["annotations"].append({"id": 902, "image_id": 100, "category_id": 1, "bbox": [1, 2, -3, 4]})  # neg w
        payload["annotations"].append({"id": 903, "bbox": [0, 0, 1, 1]})  # no image_id
        payload["images"].append({"id": 102})  # no file_name
        payload["categories"].append({"id": 5})  # no name
        _write(tmp_path, payload)
        ds = load_od(tmp_path, dataset_format="coco")
        # bad image dropped (still 2 valid), bad annotations dropped (still 2), bad category dropped
        assert ds.sample_count == 2
        assert ds.num_detections == 2
        assert 5 not in ds.index2label()

    def test_unsafe_file_name_drops_path(self, tmp_path: Path) -> None:
        payload = _coco()
        payload["images"] = [{"id": 1, "file_name": "../escape.jpg", "width": 10, "height": 10}]
        payload["annotations"] = []
        _write(tmp_path, payload)
        ds = load_od(tmp_path, dataset_format="coco")
        assert ds.samples[0].path_or_uri is None  # traversal rejected, sample still kept

    @pytest.mark.parametrize(
        "file_name",
        ["/abs/escape.jpg", "http://host/escape.jpg", "a\\b.jpg", "C:evil.jpg", "sub/../../escape.jpg"],
    )
    def test_unsafe_file_name_variants_drop_path(self, tmp_path: Path, file_name: str) -> None:
        payload = _coco()
        payload["images"] = [{"id": 1, "file_name": file_name, "width": 10, "height": 10}]
        payload["annotations"] = []
        _write(tmp_path, payload)
        ds = load_od(tmp_path, dataset_format="coco")
        assert ds.samples[0].path_or_uri is None


class TestRegistryDispatch:
    def test_coco_is_a_registered_format(self) -> None:
        assert DatasetFormat.COCO in available_formats()

    def test_generic_load_dispatches_to_coco(self, tmp_path: Path) -> None:
        _write(tmp_path, _coco())
        ds = load(tmp_path, dataset_format="coco")
        assert isinstance(ds, ObjectDetectionDataset)
        assert ds.sample_count == 2

    def test_load_od_rejects_mot_formats(self, tmp_path: Path) -> None:
        # HMIE is registered, but it yields a BoxTrackDataset -- the task-first
        # wrapper must refuse it rather than mislabel the result.
        with pytest.raises(TypeError, match="load_od expected an ObjectDetectionDataset"):
            load_od(tmp_path, dataset_format="hmie")

    def test_load_mot_rejects_coco(self, tmp_path: Path) -> None:
        _write(tmp_path, _coco())
        with pytest.raises(TypeError, match="load_mot expected a BoxTrackDataset"):
            load_mot(tmp_path, dataset_format="coco")
