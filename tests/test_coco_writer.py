"""Tests for the COCO writer and the COCO load -> write -> load round trip."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from datamaite import (
    CocoWriter,
    DatasetFormat,
    ObjectDetectionDataset,
    convert,
    load_od,
    write,
)
from datamaite.model import BoxTrackDataset
from datamaite.records import ImageObjectDetectionSample, ObjectDetectionAnnotation
from datamaite.writers import available_output_formats, get_writer

from ._hmie_factory import SnippetSpec, single_video_dataset

_WRITER_LOGGER = "datamaite._formats.coco.writer"


def _coco_payload() -> dict[str, Any]:
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
                "custom_flag": True,
            },
            {"id": 901, "image_id": 100, "category_id": 3, "bbox": [5, 5, 10, 10]},
        ],
    }


def _coco_root(root: Path, payload: dict[str, Any] | None = None, *, with_images: bool = True) -> Path:
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "annotations" / "instances.json").write_text(
        json.dumps(payload if payload is not None else _coco_payload()), encoding="utf-8"
    )
    if with_images:
        (root / "a.jpg").write_bytes(b"img-a")
        (root / "sub").mkdir(exist_ok=True)
        (root / "sub" / "b.jpg").write_bytes(b"img-b")
    return root


def _fingerprint(ds: ObjectDetectionDataset) -> tuple[Any, ...]:
    samples = tuple(
        (
            s.image_id,
            s.file_name,
            s.width,
            s.height,
            tuple(sorted(s.metadata.items())),
            tuple(
                (
                    d.bbox,
                    d.category_id,
                    d.category_name,
                    d.source_annotation_id,
                    d.area,
                    repr(d.segmentation),
                    d.iscrowd,
                    tuple(sorted((k, repr(v)) for k, v in d.attributes.items())),
                )
                for d in s.detections
            ),
        )
        for s in ds.iter_samples()
    )
    meta = ds.dataset_metadata
    return (
        samples,
        meta.taxonomy.entries if meta.taxonomy is not None else (),
        tuple(sorted(meta.info.items())),
        meta.licenses,
    )


class TestCocoWriterRegistry:
    def test_registered_and_public_api(self) -> None:
        assert DatasetFormat.COCO in available_output_formats()
        assert isinstance(get_writer(DatasetFormat.COCO), CocoWriter)
        assert isinstance(get_writer("coco"), CocoWriter)

    def test_declared_contract(self) -> None:
        assert CocoWriter.consumes is ObjectDetectionDataset
        assert "score" in CocoWriter.capabilities.lossy_without


class TestCocoWriterHappyPath:
    def test_write_produces_reloadable_root(self, tmp_path: Path) -> None:
        src = _coco_root(tmp_path / "src")
        ds = load_od(src, dataset_format="coco")

        out = tmp_path / "out"
        files = write(ds, out, output_format="coco", verbose=True)

        assert out / "annotations" / "instances.json" in files
        assert (out / "a.jpg").read_bytes() == b"img-a"
        assert (out / "sub" / "b.jpg").read_bytes() == b"img-b"
        assert _fingerprint(load_od(out, dataset_format="coco")) == _fingerprint(ds)

    def test_convert_coco_to_coco_end_to_end(self, tmp_path: Path) -> None:
        src = _coco_root(tmp_path / "src")

        files = convert(tmp_path / "src", tmp_path / "out", input_format="coco", output_format="coco", verbose=True)

        assert files
        assert _fingerprint(load_od(tmp_path / "out", dataset_format="coco")) == _fingerprint(
            load_od(src, dataset_format="coco")
        )

    def test_include_images_false_writes_json_only(self, tmp_path: Path) -> None:
        src = _coco_root(tmp_path / "src")
        ds = load_od(src, dataset_format="coco")
        out = tmp_path / "out"

        files = write(ds, out, output_format="coco", include_images=False, verbose=True)

        assert files == [out / "annotations" / "instances.json"]
        assert not (out / "a.jpg").exists()

    def test_missing_image_source_keeps_json_entry(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        src = _coco_root(tmp_path / "src", with_images=False)
        ds = load_od(src, dataset_format="coco")
        out = tmp_path / "out"

        with caplog.at_level(logging.WARNING, logger=_WRITER_LOGGER):
            files = write(ds, out, output_format="coco", verbose=True)

        assert files == [out / "annotations" / "instances.json"]
        document = json.loads((out / "annotations" / "instances.json").read_text(encoding="utf-8"))
        assert {img["id"] for img in document["images"]} == {100, 101}
        # The samples name an image source; the files just aren't on disk.
        assert "does not exist" in caplog.text
        assert "No image source" not in caplog.text


class TestTaskClosedDispatch:
    def test_write_rejects_box_track_dataset(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="consumes ObjectDetectionDataset"):
            write(BoxTrackDataset(sequences=(), categories={}), tmp_path, output_format="coco")

    def test_convert_mot_to_coco_raises(self, tmp_path: Path) -> None:
        single_video_dataset(tmp_path / "src", [SnippetSpec(name="video_001_000001")])
        with pytest.raises(TypeError, match="consumes ObjectDetectionDataset"):
            convert(tmp_path / "src", tmp_path / "out", input_format="hmie", output_format="coco")

    def test_convert_coco_to_mot_raises(self, tmp_path: Path) -> None:
        _coco_root(tmp_path / "src")
        with pytest.raises(TypeError, match="consumes BoxTrackDataset"):
            convert(tmp_path / "src", tmp_path / "out", input_format="coco", output_format="motchallenge")


class TestCocoWriterDrops:
    def test_score_dropped_with_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id=1,
                    file_name="a.jpg",
                    detections=(ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=1, score=0.9),),
                ),
            ),
        )
        with caplog.at_level(logging.WARNING, logger=_WRITER_LOGGER):
            write(ds, tmp_path, output_format="coco", include_images=False)
        document = json.loads((tmp_path / "annotations" / "instances.json").read_text(encoding="utf-8"))
        assert "score" not in document["annotations"][0]
        assert "no score field" in caplog.text

    def test_non_integer_category_dropped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id=1,
                    file_name="a.jpg",
                    detections=(
                        ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id="widget"),
                        ObjectDetectionAnnotation(bbox=(0.0, 0.0, 1.0, 1.0), category_id=2),
                    ),
                ),
            ),
        )
        with caplog.at_level(logging.WARNING, logger=_WRITER_LOGGER):
            write(ds, tmp_path, output_format="coco", include_images=False)
        document = json.loads((tmp_path / "annotations" / "instances.json").read_text(encoding="utf-8"))
        assert [a["category_id"] for a in document["annotations"]] == [2]
        assert "require an integer category_id" in caplog.text

    def test_non_integer_image_id_dropped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(image_id="img-one", file_name="a.jpg"),
                ImageObjectDetectionSample(image_id=2, file_name="b.jpg"),
            ),
        )
        with caplog.at_level(logging.WARNING, logger=_WRITER_LOGGER):
            write(ds, tmp_path, output_format="coco", include_images=False)
        document = json.loads((tmp_path / "annotations" / "instances.json").read_text(encoding="utf-8"))
        assert [img["id"] for img in document["images"]] == [2]
        assert "image ids are integers" in caplog.text

    def test_duplicate_annotation_ids_reallocated(self, tmp_path: Path) -> None:
        detections = (
            ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=1, source_annotation_id=7),
            ObjectDetectionAnnotation(bbox=(0.0, 0.0, 1.0, 1.0), category_id=1, source_annotation_id=7),
        )
        ds = ObjectDetectionDataset(
            samples=(ImageObjectDetectionSample(image_id=1, file_name="a.jpg", detections=detections),),
        )
        write(ds, tmp_path, output_format="coco", include_images=False)
        document = json.loads((tmp_path / "annotations" / "instances.json").read_text(encoding="utf-8"))
        ids = [a["id"] for a in document["annotations"]]
        assert ids[0] == 7
        assert len(set(ids)) == 2

    def test_unsafe_file_name_not_written(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        ds = ObjectDetectionDataset(
            samples=(ImageObjectDetectionSample(image_id=1, file_name="../escape.jpg", image_bytes=b"x"),),
        )
        out = tmp_path / "deep" / "out"
        with caplog.at_level(logging.WARNING, logger=_WRITER_LOGGER):
            files = write(ds, out, output_format="coco", verbose=True)
        assert files == [out / "annotations" / "instances.json"]
        assert not (tmp_path / "deep" / "escape.jpg").exists()
        assert "unsafe file_name" in caplog.text

    def test_annotation_file_name_must_be_bare(self, tmp_path: Path) -> None:
        ds = ObjectDetectionDataset(samples=())
        with pytest.raises(ValueError, match="bare file name"):
            write(ds, tmp_path, output_format="coco", annotation_file_name="../evil.json")
