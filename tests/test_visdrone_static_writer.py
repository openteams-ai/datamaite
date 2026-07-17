"""Tests for the VisDrone Static-Images writers (IR-3.2-S-7)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from datamaite import (
    DatasetFormat,
    Task,
    VisDroneImageClassificationWriter,
    VisDroneObjectDetectionWriter,
    load_ic,
    load_od,
    write,
)
from datamaite.image_classification import ImageClassificationDataset
from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import (
    ClassificationLabel,
    ImageClassificationSample,
    ImageObjectDetectionSample,
    ObjectDetectionAnnotation,
)
from datamaite.writers import available_output_formats, get_writer

_JPEG = b"\xff\xd8\xff\xe0 fake jpeg payload"

_LOGGER = "datamaite._formats.visdrone.static_writer"


def _visdrone_source_root(tmp_path: Path) -> Path:
    """An official-style VisDrone-DET split root with two images."""
    root = tmp_path / "VisDrone2019-DET-train"
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir()
    (root / "images" / "0000001.jpg").write_bytes(_JPEG)
    (root / "images" / "0000002.jpg").write_bytes(_JPEG)
    (root / "annotations" / "0000001.txt").write_text(
        "684,8,273,116,0,0,0,0\n406,119,265,70,1,4,0,1\n255,22,119,128,1,5,1,2\n",
        encoding="utf-8",
    )
    (root / "annotations" / "0000002.txt").write_text("10,20,30,40,1,9,0,0\n", encoding="utf-8")
    return root


def _od_fingerprint(dataset: ObjectDetectionDataset) -> list[tuple]:
    return sorted(
        (
            detection.bbox,
            detection.category_id,
            detection.attributes["visdrone_score"],
            detection.attributes["truncation"],
            detection.attributes["occlusion"],
        )
        for sample in dataset.samples
        for detection in sample.detections
    )


class TestVisDroneStaticWriterRegistry:
    def test_registered_for_both_tasks(self) -> None:
        assert DatasetFormat.VISDRONE in available_output_formats(task=Task.OD)
        assert DatasetFormat.VISDRONE in available_output_formats(task=Task.IC)
        assert isinstance(get_writer("visdrone", task=Task.OD), VisDroneObjectDetectionWriter)
        assert isinstance(get_writer("visdrone", task=Task.IC), VisDroneImageClassificationWriter)

    def test_format_selection_is_task_configurable_via_write(self, tmp_path: Path) -> None:
        # IR-3.2-S-7: detection vs classification selection must be configurable.
        source = _visdrone_source_root(tmp_path)
        od_dest = tmp_path / "od"
        ic_dest = tmp_path / "ic"

        write(load_od(source, dataset_format="visdrone"), od_dest, output_format="visdrone")
        write(load_ic(source, dataset_format="visdrone"), ic_dest, output_format="visdrone")

        assert (od_dest / "VisDrone2019-DET-train" / "annotations" / "0000001.txt").is_file()
        assert (ic_dest / "VisDrone2019-DET-train" / "annotations" / "0000001.txt").is_file()


class TestVisDroneObjectDetectionWriter:
    def test_roundtrip_preserves_rows_exactly(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        source = _visdrone_source_root(tmp_path)
        dataset = load_od(source, dataset_format="visdrone")
        dest = tmp_path / "dest"

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            write(dataset, dest, output_format="visdrone")

        # A VisDrone -> VisDrone round-trip resolves every class from the
        # preserved visdrone_category_id attribute: no fallback warnings.
        assert "fell back" not in caplog.text
        emitted = (dest / "VisDrone2019-DET-train" / "annotations" / "0000001.txt").read_text(encoding="utf-8")
        assert emitted == "684,8,273,116,0,0,0,0\n406,119,265,70,1,4,0,1\n255,22,119,128,1,5,1,2\n"

        reloaded = load_od(dest / "VisDrone2019-DET-train", dataset_format="visdrone")
        assert _od_fingerprint(reloaded) == _od_fingerprint(dataset)
        assert len(reloaded) == len(dataset)

    def test_sample_without_detections_gets_empty_annotation_file(self, tmp_path: Path) -> None:
        source = tmp_path / "img.jpg"
        source.write_bytes(_JPEG)
        dataset = ObjectDetectionDataset(
            samples=(ImageObjectDetectionSample(image_id="img", path_or_uri=str(source), file_name="img.jpg"),)
        )
        dest = tmp_path / "dest"

        files = get_writer("visdrone", task=Task.OD).write(dataset, dest)

        ann = dest / "VisDrone2019-DET-train" / "annotations" / "img.txt"
        assert ann.read_text(encoding="utf-8") == ""
        assert sorted(f.name for f in files) == ["img.jpg", "img.txt"]
        # The emitted root loads back with the same (empty-detection) sample count.
        reloaded = load_od(dest / "VisDrone2019-DET-train", dataset_format="visdrone")
        assert len(reloaded) == 1
        assert reloaded.samples[0].detections == ()

    def test_split_handling_and_aliases(self, tmp_path: Path) -> None:
        source = tmp_path / "img.jpg"
        source.write_bytes(_JPEG)
        detection = ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=4)
        dataset = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id="a", path_or_uri=str(source), file_name="a.jpg", split="val", detections=(detection,)
                ),
                ImageObjectDetectionSample(
                    image_id="b", path_or_uri=str(source), file_name="b.jpg", detections=(detection,)
                ),
            )
        )
        dest = tmp_path / "dest"

        get_writer("visdrone", task=Task.OD).write(dataset, dest, split="validation")

        # "validation" fallback normalises to VisDrone's "val"; the preserved
        # sample split also lands there.
        assert (dest / "VisDrone2019-DET-val" / "images" / "a.jpg").is_file()
        assert (dest / "VisDrone2019-DET-val" / "images" / "b.jpg").is_file()

    def test_class_map_overrides_and_drops_unmapped(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        source = tmp_path / "img.jpg"
        source.write_bytes(_JPEG)
        dataset = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id="img",
                    path_or_uri=str(source),
                    file_name="img.jpg",
                    detections=(
                        ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=7, category_name="person"),
                        ObjectDetectionAnnotation(bbox=(5.0, 6.0, 7.0, 8.0), category_id=8, category_name="ghost"),
                    ),
                ),
            )
        )
        dest = tmp_path / "dest"

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            get_writer("visdrone", task=Task.OD).write(dataset, dest, class_map={"person": 1})

        rows = (dest / "VisDrone2019-DET-train" / "annotations" / "img.txt").read_text(encoding="utf-8")
        assert rows == "1,2,3,4,1,1,0,0\n"  # person -> VisDrone pedestrian (1); ghost dropped
        assert "not present in class_map" in caplog.text

    def test_generic_category_fallback_warns_once(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        source = tmp_path / "img.jpg"
        source.write_bytes(_JPEG)
        detections = tuple(
            ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=4, category_name="car") for _ in range(5)
        )
        dataset = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id="img", path_or_uri=str(source), file_name="img.jpg", detections=detections
                ),
            )
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            get_writer("visdrone", task=Task.OD).write(dataset, tmp_path / "dest")

        assert caplog.text.count("fell back to") == 1  # aggregated, not per-box

    def test_out_of_range_category_is_dropped(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        source = tmp_path / "img.jpg"
        source.write_bytes(_JPEG)
        dataset = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id="img",
                    path_or_uri=str(source),
                    file_name="img.jpg",
                    detections=(ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=99),),
                ),
            )
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            get_writer("visdrone", task=Task.OD).write(dataset, tmp_path / "dest")

        rows = (tmp_path / "dest" / "VisDrone2019-DET-train" / "annotations" / "img.txt").read_text(encoding="utf-8")
        assert rows == ""
        assert "outside the VisDrone" in caplog.text

    def test_skips_missing_source_and_writes_bytes(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        dataset = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(image_id="ghost", path_or_uri=str(tmp_path / "missing.jpg")),
                ImageObjectDetectionSample(image_id="bytes", image_bytes=_JPEG, file_name="bytes.jpg"),
            )
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            files = get_writer("visdrone", task=Task.OD).write(dataset, tmp_path / "dest")

        assert "missing image file" in caplog.text
        image = tmp_path / "dest" / "VisDrone2019-DET-train" / "images" / "bytes.jpg"
        assert image.read_bytes() == _JPEG
        assert image in files

    def test_validate_options(self) -> None:
        writer = get_writer("visdrone", task=Task.OD)
        with pytest.raises(ValueError, match="split"):
            writer.validate_options(split="bogus")
        with pytest.raises(ValueError, match="class ids must be"):
            writer.validate_options(class_map={"x": -1})

    def test_tif_source_roundtrips(self, tmp_path: Path) -> None:
        # Images are copied verbatim, so a .tif arriving via flat_images must
        # reload from the emitted root instead of reloading as zero samples.
        dataset = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id="scan",
                    image_bytes=b"II*\x00 fake tif",
                    file_name="scan.tif",
                    detections=(
                        ObjectDetectionAnnotation(
                            bbox=(1.0, 2.0, 3.0, 4.0),
                            category_id=4,
                            attributes={"visdrone_category_id": 4},
                        ),
                    ),
                ),
            )
        )
        dest = tmp_path / "dest"

        get_writer("visdrone", task=Task.OD).write(dataset, dest)

        assert (dest / "VisDrone2019-DET-train" / "images" / "scan.tif").is_file()
        reloaded = load_od(dest / "VisDrone2019-DET-train", dataset_format="visdrone")
        assert len(reloaded) == 1
        assert len(reloaded.samples[0].detections) == 1

    def test_suffix_the_loader_cannot_read_is_skipped(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        # No transcoding happens, so an unreadable suffix would silently
        # vanish on reload; the writer skips it loudly instead.
        dataset = ObjectDetectionDataset(
            samples=(ImageObjectDetectionSample(image_id="anim", image_bytes=b"RIFF", file_name="anim.webp"),)
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            files = get_writer("visdrone", task=Task.OD).write(dataset, tmp_path / "dest")

        assert files == []
        assert "not readable by the VisDrone static loader" in caplog.text

    def test_test_dev_split_roundtrips(self, tmp_path: Path) -> None:
        # A writer-emitted test-dev root must reload as split="test-dev",
        # not collapse onto plain "test".
        dataset = ObjectDetectionDataset(
            samples=(ImageObjectDetectionSample(image_id="img", image_bytes=_JPEG, file_name="img.jpg"),)
        )
        dest = tmp_path / "dest"

        get_writer("visdrone", task=Task.OD).write(dataset, dest, split="test-dev")

        root = dest / "VisDrone2019-DET-test-dev"
        assert (root / "images" / "img.jpg").is_file()
        reloaded = load_od(root, dataset_format="visdrone")
        assert reloaded.samples[0].split == "test-dev"
        # And writing the reloaded dataset preserves the split root name.
        second = tmp_path / "second"
        get_writer("visdrone", task=Task.OD).write(reloaded, second)
        assert (second / "VisDrone2019-DET-test-dev" / "images" / "img.jpg").is_file()

    def test_generic_category_zero_is_not_written_as_ignored_region(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        # A COCO-style dense taxonomy where 0 is a real class ("person") must
        # not silently become VisDrone category 0 = "ignored regions"/score 0.
        dataset = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id="img",
                    image_bytes=_JPEG,
                    file_name="img.jpg",
                    detections=(
                        ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=0, category_name="person"),
                    ),
                ),
            )
        )
        dest = tmp_path / "dest"

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            get_writer("visdrone", task=Task.OD).write(dataset, dest)

        rows = (dest / "VisDrone2019-DET-train" / "annotations" / "img.txt").read_text(encoding="utf-8")
        assert rows == ""  # dropped, not reinterpreted
        assert "ignored regions" in caplog.text
        assert "class_map" in caplog.text

    def test_genuine_ignored_region_still_writes_category_zero(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        # Category 0 from a real source (attribute or class_map) is a genuine
        # ignored region: written with the GT score-0 evaluation flag.
        from_attribute = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id="a",
                    image_bytes=_JPEG,
                    file_name="a.jpg",
                    detections=(
                        ObjectDetectionAnnotation(
                            bbox=(1.0, 2.0, 3.0, 4.0),
                            category_id=0,
                            category_name="ignored regions",
                            attributes={"visdrone_category_id": 0},
                        ),
                    ),
                ),
            )
        )
        from_class_map = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id="b",
                    image_bytes=_JPEG,
                    file_name="b.jpg",
                    detections=(
                        ObjectDetectionAnnotation(bbox=(5.0, 6.0, 7.0, 8.0), category_id=3, category_name="mask"),
                    ),
                ),
            )
        )
        writer = get_writer("visdrone", task=Task.OD)

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            writer.write(from_attribute, tmp_path / "via-attribute")
            writer.write(from_class_map, tmp_path / "via-class-map", class_map={"mask": 0})

        via_attribute = tmp_path / "via-attribute" / "VisDrone2019-DET-train" / "annotations" / "a.txt"
        via_class_map = tmp_path / "via-class-map" / "VisDrone2019-DET-train" / "annotations" / "b.txt"
        assert via_attribute.read_text(encoding="utf-8") == "1,2,3,4,0,0,0,0\n"
        assert via_class_map.read_text(encoding="utf-8") == "5,6,7,8,0,0,0,0\n"
        assert "would have been reinterpreted" not in caplog.text


class TestVisDroneImageClassificationWriter:
    def test_roundtrip_preserves_crops(self, tmp_path: Path) -> None:
        source = _visdrone_source_root(tmp_path)
        dataset = load_ic(source, dataset_format="visdrone")
        dest = tmp_path / "dest"

        write(dataset, dest, output_format="visdrone")

        reloaded = load_ic(dest / "VisDrone2019-DET-train", dataset_format="visdrone")
        original = sorted((s.region, s.labels[0].category_id) for s in dataset.samples)
        roundtrip = sorted((s.region, s.labels[0].category_id) for s in reloaded.samples)
        assert roundtrip == original
        # Crops from the same source image share one written image file.
        images = list((dest / "VisDrone2019-DET-train" / "images").iterdir())
        assert len(images) == 2

    def test_regionless_sample_with_dims_becomes_full_image_box(self, tmp_path: Path) -> None:
        dataset = ImageClassificationDataset(
            samples=(
                ImageClassificationSample(
                    image_id="whole",
                    image_bytes=_JPEG,
                    file_name="whole.jpg",
                    width=64,
                    height=48,
                    labels=(ClassificationLabel(category_id=4, source_category_id=4, category_name="car"),),
                ),
            )
        )
        dest = tmp_path / "dest"

        get_writer("visdrone", task=Task.IC).write(dataset, dest)

        rows = (dest / "VisDrone2019-DET-train" / "annotations" / "whole.txt").read_text(encoding="utf-8")
        assert rows == "0,0,64,48,1,4,0,0\n"

    def test_regionless_sample_without_dims_is_skipped(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        dataset = ImageClassificationDataset(
            samples=(
                ImageClassificationSample(
                    image_id="nodims",
                    image_bytes=_JPEG,
                    labels=(ClassificationLabel(category_id=4),),
                ),
            )
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            files = get_writer("visdrone", task=Task.IC).write(dataset, tmp_path / "dest")

        assert files == []
        assert "without a crop region or image dimensions" in caplog.text

    def test_unlabeled_sample_is_skipped(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        dataset = ImageClassificationDataset(
            samples=(ImageClassificationSample(image_id="bare", image_bytes=_JPEG, width=8, height=8),)
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            files = get_writer("visdrone", task=Task.IC).write(dataset, tmp_path / "dest")

        assert files == []
        assert "no labels" in caplog.text

    def test_validate_options(self) -> None:
        writer = get_writer("visdrone", task=Task.IC)
        with pytest.raises(ValueError, match="split"):
            writer.validate_options(split="bogus")
        with pytest.raises(ValueError, match="class ids must be"):
            writer.validate_options(class_map={"x": -1})

    def test_image_whose_rows_all_drop_is_not_copied(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        # If the only annotation row for an image drops (unmapped under
        # class_map), copying the image would emit an image-only root that
        # reloads rejected (sniff) or with the image lost (IC load).
        source = tmp_path / "img.jpg"
        source.write_bytes(_JPEG)
        dataset = ImageClassificationDataset(
            samples=(
                ImageClassificationSample(
                    image_id="crop",
                    path_or_uri=str(source),
                    file_name="img.jpg",
                    region=(1.0, 2.0, 3.0, 4.0),
                    labels=(ClassificationLabel(category_id=8, category_name="ghost"),),
                ),
            )
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            files = get_writer("visdrone", task=Task.IC).write(dataset, tmp_path / "dest", class_map={"person": 1})

        assert files == []
        assert not (tmp_path / "dest" / "VisDrone2019-DET-train" / "images").exists()
        assert "not present in class_map" in caplog.text

    def test_shared_image_with_one_surviving_row_is_copied_once(self, tmp_path: Path) -> None:
        source = tmp_path / "img.jpg"
        source.write_bytes(_JPEG)
        samples = tuple(
            ImageClassificationSample(
                image_id=f"crop{i}",
                path_or_uri=str(source),
                file_name="img.jpg",
                region=(1.0, 2.0, 3.0, 4.0),
                labels=(ClassificationLabel(category_id=i, category_name=name),),
            )
            for i, name in ((8, "ghost"), (7, "person"))
        )
        dataset = ImageClassificationDataset(samples=samples)
        dest = tmp_path / "dest"

        get_writer("visdrone", task=Task.IC).write(dataset, dest, class_map={"person": 1})

        root = dest / "VisDrone2019-DET-train"
        assert [f.name for f in (root / "images").iterdir()] == ["img.jpg"]
        assert (root / "annotations" / "img.txt").read_text(encoding="utf-8") == "1,2,3,4,1,1,0,0\n"
        # Every written image has an annotation file, so the root reloads.
        reloaded = load_ic(root, dataset_format="visdrone")
        assert len(reloaded.samples) == 1

    def test_generic_zero_label_is_not_written_as_ignored_region(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        dataset = ImageClassificationDataset(
            samples=(
                ImageClassificationSample(
                    image_id="crop",
                    image_bytes=_JPEG,
                    file_name="img.jpg",
                    region=(1.0, 2.0, 3.0, 4.0),
                    labels=(ClassificationLabel(category_id=0, category_name="person"),),
                ),
            )
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            files = get_writer("visdrone", task=Task.IC).write(dataset, tmp_path / "dest")

        assert files == []  # row dropped, so the image is not copied either
        assert "ignored regions" in caplog.text
        assert "class_map" in caplog.text
