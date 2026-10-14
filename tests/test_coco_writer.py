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

    @pytest.mark.optional
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

    def test_non_integer_image_id_gets_unused_integer(self, tmp_path: Path) -> None:
        # YOLO/Hugging Face/VisDrone/flat-image records use string ids; dropping
        # them emptied every conversion into COCO. They get a fresh integer that
        # avoids the integer ids already present: taking 1 here would make the
        # later sample with id 1 a "duplicate" and drop it.
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id="img-one",
                    file_name="a.jpg",
                    detections=(ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=1),),
                ),
                ImageObjectDetectionSample(image_id=1, file_name="b.jpg"),
            ),
        )
        write(ds, tmp_path, output_format="coco", include_images=False)
        document = json.loads((tmp_path / "annotations" / "instances.json").read_text(encoding="utf-8"))
        assert [(img["id"], img["file_name"]) for img in document["images"]] == [(2, "a.jpg"), (1, "b.jpg")]
        assert [ann["image_id"] for ann in document["annotations"]] == [2]

    def test_duplicate_string_image_id_keeps_the_first(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        # Fresh integer ids must not hide a repeated source id as two images.
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(image_id="img", file_name="a.jpg"),
                ImageObjectDetectionSample(image_id="img", file_name="b.jpg"),
            ),
        )
        with caplog.at_level(logging.WARNING, logger="datamaite"):
            write(ds, tmp_path, output_format="coco", include_images=False)
        document = json.loads((tmp_path / "annotations" / "instances.json").read_text(encoding="utf-8"))
        assert [img["file_name"] for img in document["images"]] == ["a.jpg"]
        assert "Skipping duplicate image id img" in caplog.text

    def test_colliding_file_names_get_distinct_files(self, tmp_path: Path) -> None:
        # COCO keeps images in one flat directory. Same-named images from
        # different splits (YOLO images/train/img.jpg, images/val/img.jpg) used
        # to share one file, so one split's boxes pointed at the other's pixels.
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(image_id=1, file_name="img.jpg", split="train", image_bytes=b"train-pixels"),
                ImageObjectDetectionSample(image_id=2, file_name="img.jpg", split="val", image_bytes=b"val-pixels"),
            ),
        )
        write(ds, tmp_path, output_format="coco")
        document = json.loads((tmp_path / "annotations" / "instances.json").read_text(encoding="utf-8"))
        assert [img["file_name"] for img in document["images"]] == ["img.jpg", "img_2.jpg"]
        assert (tmp_path / "img.jpg").read_bytes() == b"train-pixels"
        assert (tmp_path / "img_2.jpg").read_bytes() == b"val-pixels"

    def test_file_names_differing_only_in_case_get_distinct_files(self, tmp_path: Path) -> None:
        # On case-insensitive filesystems (macOS, Windows) IMG.jpg and img.jpg
        # are one file, so the second image would overwrite the first.
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(image_id=1, file_name="IMG.jpg", split="train", image_bytes=b"train-pixels"),
                ImageObjectDetectionSample(image_id=2, file_name="img.jpg", split="val", image_bytes=b"val-pixels"),
            ),
        )
        write(ds, tmp_path, output_format="coco")
        document = json.loads((tmp_path / "annotations" / "instances.json").read_text(encoding="utf-8"))
        assert [img["file_name"] for img in document["images"]] == ["IMG.jpg", "img_2.jpg"]
        assert (tmp_path / "IMG.jpg").read_bytes() == b"train-pixels"
        assert (tmp_path / "img_2.jpg").read_bytes() == b"val-pixels"

    def test_without_images_file_names_are_relative_to_the_sources_common_directory(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Bare names from images/train/ and images/val/ could not all be found
        # under one images_dir; paths relative to images/ can.
        images = tmp_path / "src" / "images"
        for split in ("train", "val"):
            (images / split).mkdir(parents=True)
            (images / split / "img.jpg").write_bytes(b"pixels")
        ds = ObjectDetectionDataset(
            samples=tuple(
                ImageObjectDetectionSample(image_id=i, file_name="img.jpg", path_or_uri=str(images / split / "img.jpg"))
                for i, split in enumerate(["train", "val"], start=1)
            ),
        )
        out = tmp_path / "out"
        with caplog.at_level(logging.INFO, logger=_WRITER_LOGGER):
            write(ds, out, output_format="coco", include_images=False)
        document = json.loads((out / "annotations" / "instances.json").read_text(encoding="utf-8"))
        assert [img["file_name"] for img in document["images"]] == ["train/img.jpg", "val/img.jpg"]
        assert f"relative to {images.as_posix()}" in caplog.text
        reloaded = load_od(out, dataset_format="coco", images_dir=str(images))
        assert [Path(s.path_or_uri or "").is_file() for s in reloaded.samples] == [True, True]

    def test_without_images_sources_with_no_shared_directory_keep_bare_file_names(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A local path and an s3:// URI share no parent: the relative names would
        # be absolute or contain ":", which used to skip both samples and boxes.
        detection = ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=1, category_name="car")
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id=1,
                    file_name="x.jpg",
                    path_or_uri="/data/a/x.jpg",
                    width=8,
                    height=8,
                    detections=(detection,),
                ),
                ImageObjectDetectionSample(
                    image_id=2,
                    file_name="y.jpg",
                    path_or_uri="s3://b/y.jpg",
                    width=8,
                    height=8,
                    detections=(detection,),
                ),
                ImageObjectDetectionSample(
                    image_id=3, file_name="z.jpg", image_bytes=b"pixels", width=8, height=8, detections=(detection,)
                ),
            ),
        )
        with caplog.at_level(logging.WARNING, logger=_WRITER_LOGGER):
            write(ds, tmp_path, output_format="coco", include_images=False)
        document = json.loads((tmp_path / "annotations" / "instances.json").read_text(encoding="utf-8"))
        assert [img["file_name"] for img in document["images"]] == ["x.jpg", "y.jpg", "z.jpg"]
        assert len(document["annotations"]) == 3
        assert "unsafe file_name" not in caplog.text
        assert caplog.text.count("keep a bare COCO file_name") == 1
        assert "3 image(s)" in caplog.text

    def test_without_images_bytes_only_sample_among_relative_names_is_warned(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        images = tmp_path / "src" / "images"
        for split in ("train", "val"):
            (images / split).mkdir(parents=True)
            (images / split / "img.jpg").write_bytes(b"pixels")
        ds = ObjectDetectionDataset(
            samples=(
                *(
                    ImageObjectDetectionSample(
                        image_id=i, file_name="img.jpg", path_or_uri=str(images / split / "img.jpg")
                    )
                    for i, split in enumerate(["train", "val"], start=1)
                ),
                ImageObjectDetectionSample(image_id=3, file_name="bytes.jpg", image_bytes=b"pixels"),
            ),
        )
        with caplog.at_level(logging.WARNING, logger=_WRITER_LOGGER):
            write(ds, tmp_path / "out", output_format="coco", include_images=False)
        document = json.loads((tmp_path / "out" / "annotations" / "instances.json").read_text(encoding="utf-8"))
        assert [img["file_name"] for img in document["images"]] == ["train/img.jpg", "val/img.jpg", "bytes.jpg"]
        assert f"1 image(s) keep a bare COCO file_name because no safe path relative to {images.as_posix()}" in (
            caplog.text
        )

    def test_without_images_one_source_directory_keeps_bare_file_names(self, tmp_path: Path) -> None:
        images = tmp_path / "src"
        images.mkdir()
        ds = ObjectDetectionDataset(
            samples=tuple(
                ImageObjectDetectionSample(image_id=i, file_name=name, path_or_uri=str(images / name))
                for i, name in enumerate(["a.jpg", "b.jpg"], start=1)
            ),
        )
        write(ds, tmp_path / "out", output_format="coco", include_images=False)
        document = json.loads((tmp_path / "out" / "annotations" / "instances.json").read_text(encoding="utf-8"))
        assert [img["file_name"] for img in document["images"]] == ["a.jpg", "b.jpg"]

    def test_colliding_file_names_are_kept_and_warned_without_images(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Without copies, a renamed img_2.jpg would point at no file at all.
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(image_id=1, file_name="img.jpg", split="train"),
                ImageObjectDetectionSample(image_id=2, file_name="img.jpg", split="val"),
                ImageObjectDetectionSample(image_id=3, file_name="IMG.jpg", split="val"),
            ),
        )
        with caplog.at_level(logging.WARNING, logger=_WRITER_LOGGER):
            write(ds, tmp_path, output_format="coco", include_images=False)
        document = json.loads((tmp_path / "annotations" / "instances.json").read_text(encoding="utf-8"))
        assert [img["file_name"] for img in document["images"]] == ["img.jpg", "img.jpg", "IMG.jpg"]
        # Case-only differences are distinct files on the user's own filesystem.
        assert "1 file_name(s) are shared by several images (e.g. 'img.jpg')" in caplog.text

    def test_missing_dimensions_are_read_from_the_image(self, tmp_path: Path) -> None:
        # COCO images[] entries need width/height; VisDrone sources store none.
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            + (64).to_bytes(4, "big")
            + (48).to_bytes(4, "big")
            + b"\x08\x02\x00\x00\x00"
        )
        ds = ObjectDetectionDataset(
            samples=(ImageObjectDetectionSample(image_id=1, file_name="a.png", image_bytes=png),)
        )
        write(ds, tmp_path, output_format="coco")
        document = json.loads((tmp_path / "annotations" / "instances.json").read_text(encoding="utf-8"))
        assert (document["images"][0]["width"], document["images"][0]["height"]) == (64, 48)

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

    @pytest.mark.parametrize(
        "bad_name",
        [
            "../outside.json",
            r"..\outside.json",
            "/absolute/outside.json",
            r"C:\outside.json",
            r"C:relative.json",
            r"\\server\share\outside.json",
            "instances.json:stream",
            ".",
            "..",
            "",
        ],
    )
    def test_annotation_file_name_must_be_portable_and_bare(self, tmp_path: Path, bad_name: str) -> None:
        ds = ObjectDetectionDataset(samples=())
        with pytest.raises(ValueError, match="bare file name"):
            write(ds, tmp_path, output_format="coco", annotation_file_name=bad_name)
