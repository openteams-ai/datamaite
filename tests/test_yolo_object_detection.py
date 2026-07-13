"""YOLO object-detection reader/writer tests."""

from __future__ import annotations

import logging
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest

from datamaite import (
    DatasetFormat,
    DatasetMetadata,
    ObjectDetectionDataset,
    Task,
    YoloObjectDetectionLoader,
    YoloObjectDetectionWriter,
    convert,
    load,
    load_od,
    write,
)
from datamaite._formats.yolo import loader as yolo_loader
from datamaite.loaders import available_formats, get_loader
from datamaite.records import ImageObjectDetectionSample, ObjectDetectionAnnotation
from datamaite.taxonomy import CategoryEntry, Taxonomy
from datamaite.writers import available_output_formats, get_writer

_LOGGER = "datamaite._formats.yolo"


def _png_bytes(width: int = 100, height: int = 50) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + (b"\x00\x00\x00" * width) for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def _jpeg_bytes(width: int = 80, height: int = 40) -> bytes:
    sof0_payload = b"\x08" + struct.pack(">HH", height, width) + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00"
    return b"\xff\xd8" + b"\xff\xc0" + struct.pack(">H", len(sof0_payload) + 2) + sof0_payload + b"\xff\xd9"


def _jpeg_with_tem_bytes(width: int = 80, height: int = 40) -> bytes:
    # TEM (0x01) is a standalone marker with no length bytes; the parser must
    # skip it and continue scanning for SOF.
    return b"\xff\xd8" + b"\xff\x01" + _jpeg_bytes(width=width, height=height)[2:]


def _gif_bytes(width: int = 32, height: int = 16) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x00" * 8


def _bmp_bytes(width: int = 24, height: int = 12) -> bytes:
    header = bytearray(b"BM" + b"\x00" * 30)
    header[18:22] = struct.pack("<i", width)
    header[22:26] = struct.pack("<i", height)
    return bytes(header)


def _write_image(path: Path, *, width: int = 100, height: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png_bytes(width, height))


def _od_dataset(root: Path) -> None:
    _write_image(root / "images" / "train" / "a.png", width=100, height=50)
    _write_image(root / "images" / "val" / "b.png", width=80, height=40)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train" / "a.txt").write_text("1 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    (root / "labels" / "val" / "b.txt").write_text("", encoding="utf-8")
    (root / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames: ['cat', 'dog']\n",
        encoding="utf-8",
    )


def _fingerprint(ds: ObjectDetectionDataset) -> tuple[Any, ...]:
    return tuple(
        (
            sample.file_name,
            sample.split,
            sample.width,
            sample.height,
            tuple(
                (
                    tuple(round(value, 6) for value in det.bbox),
                    det.category_id,
                    det.category_name,
                    det.score,
                )
                for det in sample.detections
            ),
        )
        for sample in ds.samples
    )


class TestYoloObjectDetectionRegistry:
    def test_yolo_package_exports_both_task_variants(self) -> None:
        import datamaite._formats.yolo as yolo

        assert yolo.YoloImageClassificationLoader is not None
        assert yolo.YoloImageClassificationWriter is not None
        assert yolo.YoloObjectDetectionLoader is not None
        assert yolo.YoloObjectDetectionWriter is not None
        assert not hasattr(yolo, "load_yolo_image_classification")
        assert not hasattr(yolo, "load_yolo_object_detection")

    def test_loader_and_writer_are_task_aware(self) -> None:
        loader = get_loader(DatasetFormat.YOLO, task=Task.OD, variant="default")
        writer = get_writer("yolo", task="od", variant="default")

        assert isinstance(loader, YoloObjectDetectionLoader)
        assert isinstance(writer, YoloObjectDetectionWriter)
        assert loader.task is Task.OD
        assert writer.task is Task.OD
        assert DatasetFormat.YOLO in available_formats(task=Task.OD)
        assert DatasetFormat.YOLO in available_output_formats(task=Task.OD)

    def test_plain_yolo_lookup_is_ambiguous_now_that_yolo_has_two_tasks(self) -> None:
        with pytest.raises(ValueError, match="Multiple loaders registered"):
            get_loader("yolo")
        with pytest.raises(ValueError, match="Multiple writers registered"):
            get_writer("yolo")


class TestYoloObjectDetectionLoader:
    def test_direct_loader_bad_or_empty_roots_return_empty(self, tmp_path: Path) -> None:
        assert not YoloObjectDetectionLoader.sniff(tmp_path / "missing")
        assert YoloObjectDetectionLoader().load(tmp_path / "missing").sample_count == 0
        assert YoloObjectDetectionLoader().load(tmp_path).sample_count == 0

    def test_loads_images_labels_by_split_layout(self, tmp_path: Path) -> None:
        _od_dataset(tmp_path)

        assert YoloObjectDetectionLoader.sniff(tmp_path)
        ds = load_od(tmp_path, dataset_format="yolo")

        assert isinstance(ds, ObjectDetectionDataset)
        assert ds.task is Task.OD
        assert ds.sample_count == 2
        assert ds.num_detections == 1
        assert ds.index2label() == {0: "cat", 1: "dog"}
        assert ds.dataset_metadata.splits == ("train", "val")
        sample = ds.samples[0]
        assert sample.file_name == "a.png"
        assert sample.width == 100
        assert sample.height == 50
        det = sample.detections[0]
        assert det.bbox == pytest.approx((40.0, 15.0, 20.0, 20.0))
        assert det.category_id == 1
        assert det.category_name == "dog"

    def test_generic_load_can_disambiguate_with_task(self, tmp_path: Path) -> None:
        _od_dataset(tmp_path)

        ds = load(tmp_path, dataset_format="yolo", task="od")

        assert isinstance(ds, ObjectDetectionDataset)
        assert ds.sample_count == 2

    def test_autodetect_raises_when_yolo_od_also_matches_ic(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "images" / "a.png")
        (tmp_path / "labels").mkdir()
        (tmp_path / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Ambiguous autodetect"):
            load(tmp_path, dataset_format=None)

    def test_loads_split_images_labels_layout(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "train" / "images" / "nested" / "a.png", width=20, height=10)
        (tmp_path / "train" / "labels" / "nested").mkdir(parents=True)
        (tmp_path / "train" / "labels" / "nested" / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        (tmp_path / "data.yaml").write_text("names:\n  0: widget\n", encoding="utf-8")

        ds = load_od(tmp_path, dataset_format="yolo")

        assert ds.sample_count == 1
        assert ds.samples[0].split == "train"
        assert ds.samples[0].file_name == "nested/a.png"
        assert ds.samples[0].detections[0].bbox == pytest.approx((5.0, 2.5, 10.0, 5.0))
        assert ds.samples[0].detections[0].category_name == "widget"

    def test_loads_data_yaml_image_list(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "images" / "train" / "a.png", width=30, height=20)
        (tmp_path / "labels" / "train").mkdir(parents=True)
        (tmp_path / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
        (tmp_path / "train.txt").write_text("images/train/a.png\n", encoding="utf-8")
        (tmp_path / "data.yaml").write_text("path: .\ntrain: train.txt\nnames: [thing]\n", encoding="utf-8")

        ds = load_od(tmp_path, dataset_format="yolo")

        assert ds.sample_count == 1
        assert ds.samples[0].split == "train"
        assert ds.samples[0].file_name == "a.png"
        assert ds.samples[0].detections[0].bbox == pytest.approx((12.0, 6.0, 6.0, 8.0))

    def test_optional_confidence_column_is_loaded_as_score(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "images" / "a.png")
        (tmp_path / "labels").mkdir()
        (tmp_path / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2 0.875\n", encoding="utf-8")

        ds = load_od(tmp_path, dataset_format="yolo")

        assert ds.samples[0].detections[0].score == pytest.approx(0.875)

    def test_sparse_names_mapping_preserves_source_ids(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "images" / "a.png")
        (tmp_path / "labels").mkdir()
        (tmp_path / "labels" / "a.txt").write_text(
            "2 0.5 0.5 0.2 0.2\n5 0.25 0.25 0.1 0.1\n",
            encoding="utf-8",
        )
        (tmp_path / "data.yaml").write_text("names:\n  2: car\n  5: person\n", encoding="utf-8")

        ds = load_od(tmp_path, dataset_format="yolo")

        assert ds.index2label() == {2: "car", 5: "person"}
        assert [det.category_name for det in ds.samples[0].detections] == ["car", "person"]
        assert ds.dataset_metadata.taxonomy is not None
        assert ds.dataset_metadata.taxonomy.id_density == "sparse"

    def test_label_referencing_class_missing_from_names_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write_image(tmp_path / "images" / "a.png")
        (tmp_path / "labels").mkdir()
        (tmp_path / "labels" / "a.txt").write_text("2 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (tmp_path / "data.yaml").write_text("names: [only]\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            ds = load_od(tmp_path, dataset_format="yolo")

        assert ds.samples[0].detections[0].category_name is None
        assert "not defined in data.yaml names" in caplog.text

    @pytest.mark.parametrize("confidence", ["nope", "-0.1", "1.1"])
    def test_invalid_confidence_row_is_skipped(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        confidence: str,
    ) -> None:
        _write_image(tmp_path / "images" / "a.png")
        (tmp_path / "labels").mkdir()
        (tmp_path / "labels" / "a.txt").write_text(
            f"0 0.5 0.5 0.2 0.2 {confidence}\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            ds = load_od(tmp_path, dataset_format="yolo")

        assert ds.num_detections == 0
        assert "invalid confidence" in caplog.text

    def test_malformed_rows_are_skipped_best_effort(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        _write_image(tmp_path / "images" / "a.png")
        (tmp_path / "labels").mkdir()
        (tmp_path / "labels" / "a.txt").write_text(
            "0 0.5 0.5 0.2 0.2\nbad 0.5 0.5 0.2 0.2\n0 0.5 0.5 -0.2 0.2\n0 0.5 0.5 0.2\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            ds = load_od(tmp_path, dataset_format="yolo")

        assert ds.num_detections == 1
        assert "invalid class id" in caplog.text
        assert "out-of-range normalized bbox" in caplog.text
        assert "expected 5 or 6 fields" in caplog.text

    def test_reads_common_header_dimensions(self, tmp_path: Path) -> None:
        jpg = tmp_path / "a.jpg"
        tem_jpg = tmp_path / "tem.jpg"
        gif = tmp_path / "a.gif"
        bmp = tmp_path / "a.bmp"
        jpg.write_bytes(_jpeg_bytes(width=80, height=40))
        tem_jpg.write_bytes(_jpeg_with_tem_bytes(width=81, height=41))
        gif.write_bytes(_gif_bytes(width=32, height=16))
        bmp.write_bytes(_bmp_bytes(width=24, height=12))

        assert yolo_loader._read_image_size(jpg) == (80, 40)
        assert yolo_loader._read_image_size(tem_jpg) == (81, 41)
        assert yolo_loader._read_image_size(gif) == (32, 16)
        assert yolo_loader._read_image_size(bmp) == (24, 12)

    def test_fallback_simple_yaml_parser(self) -> None:
        parsed = yolo_loader._parse_simple_yaml(
            "# comment\n"
            "path: '.'\n"
            "train:\n"
            "  - images/train\n"
            "names:\n"
            "  0: cat # inline comment\n"
            "  1: 'dog # not comment'\n"
        )

        assert parsed["path"] == "."
        assert parsed["train"] == ["images/train"]
        assert yolo_loader._names_from_yaml(parsed["names"]) == ((0, "cat"), (1, "dog # not comment"))

    def test_unreadable_image_dimensions_keep_sample_but_drop_labels(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()
        (tmp_path / "images" / "a.jpg").write_bytes(b"not a real jpeg")
        (tmp_path / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            ds = load_od(tmp_path, dataset_format="yolo")

        assert ds.sample_count == 1
        assert ds.num_detections == 0
        assert "image dimensions could not be determined" in caplog.text


class TestYoloObjectDetectionWriter:
    def test_write_and_reload_round_trip(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        out = tmp_path / "out"
        _od_dataset(src)
        ds = load_od(src, dataset_format="yolo")

        files = write(ds, out, output_format="yolo", verbose=True)

        assert out / "images" / "train" / "a.png" in files
        assert out / "labels" / "train" / "a.txt" in files
        assert out / "labels" / "val" / "b.txt" in files
        assert out / "data.yaml" in files
        assert (out / "images" / "train" / "a.png").read_bytes() == (src / "images" / "train" / "a.png").read_bytes()
        assert (out / "labels" / "train" / "a.txt").read_text(encoding="utf-8") == "1 0.5 0.5 0.2 0.4\n"
        assert _fingerprint(load_od(out, dataset_format="yolo")) == _fingerprint(ds)

    def test_convert_yolo_od_to_yolo_od(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        out = tmp_path / "out"
        _od_dataset(src)

        files = convert(src, out, input_format="yolo", output_format="yolo", task="od", verbose=True)

        assert files
        assert load_od(out, dataset_format="yolo").num_detections == 1

    def test_include_images_false_keeps_labels_aligned_with_existing_images(self, tmp_path: Path) -> None:
        (tmp_path / "images" / "train").mkdir(parents=True)
        (tmp_path / "images" / "train" / "x.png").write_bytes(b"existing")
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id="x",
                    file_name="x.png",
                    width=100,
                    height=50,
                    detections=(ObjectDetectionAnnotation(bbox=(40.0, 15.0, 20.0, 20.0), category_id=0),),
                ),
            ),
            dataset_metadata=DatasetMetadata(
                taxonomy=Taxonomy(entries=(CategoryEntry(source_id=0, name="thing"),), id_density="dense")
            ),
        )

        files = write(
            ds, tmp_path, output_format="yolo", include_images=False, write_data_yaml=False, mode="append", verbose=True
        )

        assert files == [tmp_path / "labels" / "train" / "x.txt"]
        assert (tmp_path / "images" / "train" / "x.png").read_bytes() == b"existing"

    def test_sparse_taxonomy_is_projected_to_dense_yolo_ids(self, tmp_path: Path) -> None:
        taxonomy = Taxonomy(
            entries=(CategoryEntry(source_id=2, name="car"), CategoryEntry(source_id=7, name="person")),
            source_dataset="coco",
            id_density="sparse",
        )
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id=1,
                    image_bytes=_png_bytes(),
                    file_name="a.png",
                    width=100,
                    height=50,
                    detections=(ObjectDetectionAnnotation(bbox=(40.0, 15.0, 20.0, 20.0), category_id=7),),
                ),
            ),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy),
        )

        write(ds, tmp_path, output_format="yolo")

        assert (tmp_path / "labels" / "train" / "a.txt").read_text(encoding="utf-8") == "1 0.5 0.5 0.2 0.4\n"
        data_yaml_lines = (tmp_path / "data.yaml").read_text(encoding="utf-8").splitlines()
        names_line = next(line for line in data_yaml_lines if line.startswith("names:"))
        assert names_line == 'names: ["car", "person"]'

    def test_edge_crossing_box_is_clamped_before_yolo_projection(self, tmp_path: Path) -> None:
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id="edge",
                    image_bytes=_png_bytes(width=100, height=100),
                    file_name="edge.png",
                    width=100,
                    height=100,
                    detections=(ObjectDetectionAnnotation(bbox=(95.0, 10.0, 20.0, 20.0), category_id=0),),
                ),
            ),
            dataset_metadata=DatasetMetadata(
                taxonomy=Taxonomy(entries=(CategoryEntry(source_id=0, name="thing"),), id_density="dense")
            ),
        )

        write(ds, tmp_path, output_format="yolo")

        assert (tmp_path / "labels" / "train" / "edge.txt").read_text(encoding="utf-8") == ("0 0.975 0.2 0.05 0.2\n")

    def test_fully_outside_box_is_dropped_after_clipping(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id="outside",
                    image_bytes=_png_bytes(width=100, height=100),
                    file_name="outside.png",
                    width=100,
                    height=100,
                    detections=(ObjectDetectionAnnotation(bbox=(120.0, 10.0, 5.0, 5.0), category_id=0),),
                ),
            ),
            dataset_metadata=DatasetMetadata(
                taxonomy=Taxonomy(entries=(CategoryEntry(source_id=0, name="thing"),), id_density="dense")
            ),
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            write(ds, tmp_path, output_format="yolo")

        assert (tmp_path / "labels" / "train" / "outside.txt").read_text(encoding="utf-8") == ""
        assert "outside image after clipping" in caplog.text

    def test_precision_must_keep_fractional_yolo_coordinates(self, tmp_path: Path) -> None:
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id="x",
                    image_bytes=_png_bytes(),
                    file_name="x.png",
                    width=100,
                    height=50,
                ),
            ),
        )

        with pytest.raises(ValueError, match="precision must be"):
            write(ds, tmp_path, output_format="yolo", precision=0)

    def test_unknown_source_id_is_not_treated_as_dense_position(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        taxonomy = Taxonomy(
            entries=(CategoryEntry(source_id=10, name="car"), CategoryEntry(source_id=20, name="plane")),
            id_density="sparse",
        )
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id="x",
                    image_bytes=_png_bytes(),
                    file_name="x.png",
                    width=100,
                    height=50,
                    detections=(
                        ObjectDetectionAnnotation(
                            bbox=(40.0, 15.0, 20.0, 20.0),
                            category_id=1,
                            source_category_id=777,
                        ),
                    ),
                ),
            ),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy),
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            write(ds, tmp_path, output_format="yolo")

        assert (tmp_path / "labels" / "train" / "x.txt").read_text(encoding="utf-8") == ""
        assert "unresolved category" in caplog.text

    def test_unknown_source_id_can_still_resolve_by_category_name(self, tmp_path: Path) -> None:
        taxonomy = Taxonomy(
            entries=(CategoryEntry(source_id=10, name="car"), CategoryEntry(source_id=20, name="plane")),
            id_density="sparse",
        )
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id="x",
                    image_bytes=_png_bytes(),
                    file_name="x.png",
                    width=100,
                    height=50,
                    detections=(
                        ObjectDetectionAnnotation(
                            bbox=(40.0, 15.0, 20.0, 20.0),
                            category_id=1,
                            source_category_id=777,
                            category_name="plane",
                        ),
                    ),
                ),
            ),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy),
        )

        write(ds, tmp_path, output_format="yolo")

        assert (tmp_path / "labels" / "train" / "x.txt").read_text(encoding="utf-8") == "1 0.5 0.5 0.2 0.4\n"

    def test_invalid_bbox_is_skipped_with_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id="x",
                    image_bytes=_png_bytes(),
                    file_name="x.png",
                    width=100,
                    height=50,
                    detections=(ObjectDetectionAnnotation(bbox=(1.0, 2.0, 0.0, 4.0), category_id=0),),
                ),
            ),
            dataset_metadata=DatasetMetadata(
                taxonomy=Taxonomy(entries=(CategoryEntry(source_id=0, name="thing"),), id_density="dense")
            ),
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            write(ds, tmp_path, output_format="yolo")

        assert (tmp_path / "labels" / "train" / "x.txt").read_text(encoding="utf-8") == ""
        assert "invalid bbox" in caplog.text

    def test_scores_are_omitted_by_default_and_optionally_written(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id="x",
                    image_bytes=_png_bytes(),
                    file_name="x.png",
                    width=100,
                    height=50,
                    detections=(ObjectDetectionAnnotation(bbox=(40.0, 15.0, 20.0, 20.0), category_id=0, score=0.875),),
                ),
            ),
            dataset_metadata=DatasetMetadata(
                taxonomy=Taxonomy(entries=(CategoryEntry(source_id=0, name="thing"),), id_density="dense")
            ),
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            write(ds, tmp_path / "default", output_format="yolo")
        write(ds, tmp_path / "with_scores", output_format="yolo", include_scores=True)

        assert (tmp_path / "default" / "labels" / "train" / "x.txt").read_text(encoding="utf-8") == (
            "0 0.5 0.5 0.2 0.4\n"
        )
        assert (tmp_path / "with_scores" / "labels" / "train" / "x.txt").read_text(encoding="utf-8") == (
            "0 0.5 0.5 0.2 0.4 0.875\n"
        )
        assert "Dropped score" in caplog.text

    def test_missing_dimensions_emit_empty_label_file_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id="x",
                    image_bytes=_png_bytes(),
                    file_name="x.png",
                    detections=(ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=0),),
                ),
            ),
            dataset_metadata=DatasetMetadata(
                taxonomy=Taxonomy(entries=(CategoryEntry(source_id=0, name="thing"),), id_density="dense")
            ),
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            files = write(ds, tmp_path, output_format="yolo", verbose=True)

        assert tmp_path / "labels" / "train" / "x.txt" in files
        assert (tmp_path / "labels" / "train" / "x.txt").read_text(encoding="utf-8") == ""
        assert "missing/invalid image width/height" in caplog.text

    def test_missing_image_source_skips_sample_without_stray_label(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id="x",
                    path_or_uri=str(tmp_path / "missing.png"),
                    file_name="x.png",
                    width=10,
                    height=10,
                    detections=(ObjectDetectionAnnotation(bbox=(1.0, 1.0, 2.0, 2.0), category_id=0),),
                ),
            ),
        )
        out = tmp_path / "out"

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            files = write(ds, out, output_format="yolo", write_data_yaml=False, verbose=True)

        assert files == []
        assert not (out / "labels").exists()
        assert "missing image file" in caplog.text

    def test_unsafe_file_name_is_skipped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id="x",
                    image_bytes=_png_bytes(),
                    file_name="../escape.png",
                    width=10,
                    height=10,
                ),
            ),
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            files = write(ds, tmp_path, output_format="yolo", write_data_yaml=False, verbose=True)

        assert files == []
        assert "unsafe file name" in caplog.text
        assert not (tmp_path.parent / "escape.png").exists()
