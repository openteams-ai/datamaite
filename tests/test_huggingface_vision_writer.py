"""Tests for the Hugging Face Vision still-image writers (IR-3.2-S-6)."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import pytest

from datamaite import (
    DatasetFormat,
    HuggingFaceVisionImageClassificationWriter,
    HuggingFaceVisionObjectDetectionWriter,
    Task,
    load_ic,
    load_od,
    write,
)
from datamaite.image_classification import ImageClassificationDataset
from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import (
    ClassificationLabel,
    DatasetMetadata,
    ImageClassificationSample,
    ImageObjectDetectionSample,
    ObjectDetectionAnnotation,
)
from datamaite.taxonomy import CategoryEntry, Taxonomy
from datamaite.writers import available_output_formats, get_writer

_JPEG = b"\xff\xd8\xff\xe0 fake jpeg payload"

_LOGGER = "datamaite._formats.huggingface_vision.writer"


def _ic_dataset(tmp_path: Path) -> ImageClassificationDataset:
    source = tmp_path / "source"
    source.mkdir(parents=True, exist_ok=True)
    (source / "a.jpg").write_bytes(_JPEG)
    taxonomy = Taxonomy(
        entries=(CategoryEntry(source_id=0, name="cat"), CategoryEntry(source_id=1, name="dog")),
        source_dataset="huggingface_vision",
        id_density="dense",
    )
    samples = (
        ImageClassificationSample(
            image_id="a.jpg",
            path_or_uri=str(source / "a.jpg"),
            file_name="a.jpg",
            split="train",
            labels=(ClassificationLabel(category_id=0, source_category_id=0, category_name="cat"),),
        ),
        ImageClassificationSample(
            image_id="b.jpg",
            image_bytes=_JPEG,
            file_name="b.jpg",
            split="validation",
            labels=(ClassificationLabel(category_id=1, source_category_id=1, category_name="dog"),),
        ),
    )
    return ImageClassificationDataset(
        samples=samples, dataset_metadata=DatasetMetadata(taxonomy=taxonomy, splits=("train", "validation"))
    )


def _od_dataset(tmp_path: Path) -> ObjectDetectionDataset:
    source = tmp_path / "od-source"
    source.mkdir(parents=True, exist_ok=True)
    (source / "img1.jpg").write_bytes(_JPEG)
    (source / "img2.jpg").write_bytes(_JPEG)
    samples = (
        ImageObjectDetectionSample(
            image_id="img1.jpg",
            path_or_uri=str(source / "img1.jpg"),
            file_name="img1.jpg",
            width=640,
            height=480,
            split="train",
            detections=(
                ObjectDetectionAnnotation(
                    bbox=(10.0, 20.0, 30.0, 40.0),
                    category_id=0,
                    source_category_id=0,
                    source_annotation_id=101,
                    area=1200.0,
                ),
                ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=2, source_category_id=2),
            ),
        ),
        ImageObjectDetectionSample(
            image_id="img2.jpg",
            path_or_uri=str(source / "img2.jpg"),
            file_name="img2.jpg",
            width=320,
            height=240,
            detections=(),
        ),
    )
    return ObjectDetectionDataset(samples=samples)


class TestHuggingFaceVisionWriterRegistry:
    def test_registered_for_both_tasks(self) -> None:
        assert DatasetFormat.HUGGINGFACE_VISION in available_output_formats(task=Task.IC)
        assert DatasetFormat.HUGGINGFACE_VISION in available_output_formats(task=Task.OD)
        assert isinstance(get_writer("huggingface_vision", task=Task.IC), HuggingFaceVisionImageClassificationWriter)
        assert isinstance(get_writer("huggingface_vision", task=Task.OD), HuggingFaceVisionObjectDetectionWriter)

    def test_format_selection_is_task_configurable_via_write(self, tmp_path: Path) -> None:
        # IR-3.2-S-6: detection vs classification selection must be configurable.
        # write() keys the choice on dataset.task within the one output_format.
        ic_dest = tmp_path / "ic"
        od_dest = tmp_path / "od"

        write(_ic_dataset(tmp_path), ic_dest, output_format="huggingface_vision")
        write(_od_dataset(tmp_path), od_dest, output_format="huggingface_vision")

        assert (ic_dest / "train" / "cat" / "a.jpg").is_file()
        assert (od_dest / "train" / "metadata.jsonl").is_file()
        assert (od_dest / "data" / "metadata.jsonl").is_file()


class TestHuggingFaceVisionImageClassificationWriter:
    def test_writes_split_class_layout_and_roundtrips(self, tmp_path: Path) -> None:
        dataset = _ic_dataset(tmp_path)
        dest = tmp_path / "dest"

        files = get_writer("huggingface_vision", task=Task.IC).write(dataset, dest)

        assert (dest / "train" / "cat" / "a.jpg").is_file()
        assert (dest / "validation" / "dog" / "b.jpg").is_file()
        assert len(files) == 2

        reloaded = load_ic(dest, dataset_format="huggingface_vision")
        assert [(s.split, s.labels[0].category_name) for s in reloaded.samples] == [
            ("train", "cat"),
            ("validation", "dog"),
        ]

    def test_skips_region_multi_label_and_missing_source(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        label = ClassificationLabel(category_id=0, category_name="cat")
        samples = (
            ImageClassificationSample(image_id="crop", image_bytes=_JPEG, labels=(label,), region=(1, 1, 2, 2)),
            ImageClassificationSample(image_id="none-source", labels=(label,)),
            ImageClassificationSample(image_id="unlabeled", image_bytes=_JPEG),
            ImageClassificationSample(
                image_id="multi",
                image_bytes=_JPEG,
                file_name="multi.jpg",
                labels=(label, ClassificationLabel(category_id=1, category_name="dog")),
            ),
        )
        dataset = ImageClassificationDataset(samples=samples)
        dest = tmp_path / "dest"

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            files = get_writer("huggingface_vision", task=Task.IC).write(dataset, dest)

        assert [f.relative_to(dest).as_posix() for f in files] == ["train/cat/multi.jpg"]
        assert "crop region" in caplog.text
        assert "no image source" in caplog.text
        assert "no labels" in caplog.text
        assert "using first label" in caplog.text

    def test_validate_options_rejects_unsafe_default_split(self) -> None:
        with pytest.raises(ValueError, match="default_split"):
            get_writer("huggingface_vision", task=Task.IC).validate_options(default_split="../evil")

    def test_default_split_rejects_unknown_and_normalizes_aliases(self, tmp_path: Path) -> None:
        # A custom split dir like holdout/ would reload as a CLASS folder, so
        # only ImageFolder-recognized splits (and their aliases) are accepted.
        writer = get_writer("huggingface_vision", task=Task.IC)
        with pytest.raises(ValueError, match="default_split"):
            writer.validate_options(default_split="holdout")

        sample = ImageClassificationSample(
            image_id="a.jpg",
            image_bytes=_JPEG,
            file_name="a.jpg",
            labels=(ClassificationLabel(category_id=0, category_name="cat"),),
        )
        dest = tmp_path / "dest"
        writer.write(ImageClassificationDataset(samples=(sample,)), dest, default_split="dev")
        assert (dest / "validation" / "cat" / "a.jpg").is_file()

    def test_unknown_sample_split_falls_back_and_roundtrips(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        # Round-trip safety: writing dev-style custom splits verbatim would make
        # the loader read `holdout/cat/a.jpg` as class "holdout"; the writer
        # instead falls back to default_split with a warning.
        sample = ImageClassificationSample(
            image_id="a.jpg",
            image_bytes=_JPEG,
            file_name="a.jpg",
            split="holdout",
            labels=(ClassificationLabel(category_id=0, category_name="cat"),),
        )
        dest = tmp_path / "dest"

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            get_writer("huggingface_vision", task=Task.IC).write(ImageClassificationDataset(samples=(sample,)), dest)

        assert (dest / "train" / "cat" / "a.jpg").is_file()
        assert "not a Hugging Face ImageFolder split" in caplog.text
        reloaded = load_ic(dest, dataset_format="huggingface_vision")
        assert [(s.split, s.labels[0].category_name) for s in reloaded.samples] == [("train", "cat")]

    def test_empty_dataset_warns_and_writes_nothing(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        dest = tmp_path / "dest"
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            files = get_writer("huggingface_vision", task=Task.IC).write(ImageClassificationDataset(samples=()), dest)
        assert files == []
        assert "No Hugging Face vision IC images were written" in caplog.text

    def test_skips_missing_source_file_and_unresolved_label(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        samples = (
            ImageClassificationSample(
                image_id="missing",
                path_or_uri=str(tmp_path / "does-not-exist.jpg"),
                file_name="missing.jpg",
                labels=(ClassificationLabel(category_id=0, category_name="cat"),),
            ),
            ImageClassificationSample(
                image_id="no-label-resolvable",
                image_bytes=_JPEG,
                file_name="x.jpg",
                labels=(ClassificationLabel(category_id=None, source_category_id=None, category_name=None),),
            ),
        )
        dest = tmp_path / "dest"
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            files = get_writer("huggingface_vision", task=Task.IC).write(
                ImageClassificationDataset(samples=samples), dest
            )
        assert files == []
        assert "missing image file" in caplog.text
        assert "unresolved label" in caplog.text

    def test_resolves_class_name_via_taxonomy_fallbacks(self, tmp_path: Path) -> None:
        # by_source_id misses -> dense positional index; and by_name fallback.
        dense_taxonomy = Taxonomy(
            entries=(CategoryEntry(source_id="a", name="cat"), CategoryEntry(source_id="b", name="dog")),
            source_dataset="huggingface_vision",
            id_density="dense",
        )
        by_name_taxonomy = Taxonomy(
            entries=(CategoryEntry(source_id=99, name="bird"),),
            source_dataset="huggingface_vision",
            id_density="sparse",
        )
        dense_sample = ImageClassificationSample(
            image_id="dense",
            image_bytes=_JPEG,
            file_name="dense.jpg",
            split="train",
            labels=(ClassificationLabel(category_id=1, source_category_id=1, category_name=None),),
        )
        name_sample = ImageClassificationSample(
            image_id="named",
            image_bytes=_JPEG,
            file_name="named.jpg",
            split="train",
            labels=(ClassificationLabel(category_id=5, source_category_id=5, category_name="bird"),),
        )
        dest_dense = tmp_path / "dense"
        dest_name = tmp_path / "name"
        get_writer("huggingface_vision", task=Task.IC).write(
            ImageClassificationDataset(
                samples=(dense_sample,), dataset_metadata=DatasetMetadata(taxonomy=dense_taxonomy)
            ),
            dest_dense,
        )
        get_writer("huggingface_vision", task=Task.IC).write(
            ImageClassificationDataset(
                samples=(name_sample,), dataset_metadata=DatasetMetadata(taxonomy=by_name_taxonomy)
            ),
            dest_name,
        )
        assert (dest_dense / "train" / "dog" / "dense.jpg").is_file()  # positional index 1 -> "dog"
        assert (dest_name / "train" / "bird" / "named.jpg").is_file()  # by_name fallback

    def test_preserves_non_ascii_class_names_roundtrip(self, tmp_path: Path) -> None:
        # Class names with spaces/unicode/punctuation are legitimate ImageFolder
        # labels; they must not be slugged (which drops write->reload identity)
        # nor skipped (which silently loses every sample of that class).
        class_names = ["traffic light", "café", "hot dog"]
        taxonomy = Taxonomy(
            entries=tuple(CategoryEntry(source_id=index, name=name) for index, name in enumerate(class_names)),
            source_dataset="huggingface_vision",
            id_density="dense",
        )
        samples = tuple(
            ImageClassificationSample(
                image_id=f"{index}.jpg",
                image_bytes=_JPEG,
                file_name=f"{index}.jpg",
                split="train",
                labels=(ClassificationLabel(category_id=index, source_category_id=index, category_name=name),),
            )
            for index, name in enumerate(class_names)
        )
        dataset = ImageClassificationDataset(
            samples=samples, dataset_metadata=DatasetMetadata(taxonomy=taxonomy, splits=("train",))
        )
        dest = tmp_path / "dest"

        files = get_writer("huggingface_vision", task=Task.IC).write(dataset, dest)

        assert len(files) == 3
        assert (dest / "train" / "traffic light" / "0.jpg").is_file()
        assert (dest / "train" / "café" / "1.jpg").is_file()
        assert (dest / "train" / "hot dog" / "2.jpg").is_file()

        reloaded = load_ic(dest, dataset_format="huggingface_vision")
        assert len(reloaded.samples) == 3
        assert sorted(s.labels[0].category_name for s in reloaded.samples) == sorted(class_names)


class TestHuggingFaceVisionObjectDetectionWriter:
    def test_writes_jsonl_metadata_and_roundtrips(self, tmp_path: Path) -> None:
        dataset = _od_dataset(tmp_path)
        dest = tmp_path / "dest"

        files = get_writer("huggingface_vision", task=Task.OD).write(dataset, dest)

        assert (dest / "train" / "img1.jpg").is_file()
        assert (dest / "data" / "img2.jpg").is_file()  # unsplit sample lands under data/
        # Metadata lives INSIDE each image directory with directory-relative
        # file_name: HF's ImageFolder only associates metadata within a split's
        # directory tree, so a root metadata file would lose objects there.
        train_rows = [json.loads(line) for line in (dest / "train" / "metadata.jsonl").read_text().splitlines()]
        data_rows = [json.loads(line) for line in (dest / "data" / "metadata.jsonl").read_text().splitlines()]
        assert train_rows[0]["file_name"] == "img1.jpg"
        assert train_rows[0]["objects"]["bbox"] == [[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]]
        assert train_rows[0]["objects"]["categories"] == [0, 2]
        assert train_rows[0]["objects"]["id"] == [101, None]
        assert train_rows[0]["objects"]["area"] == [1200.0, None]
        assert (train_rows[0]["width"], train_rows[0]["height"]) == (640, 480)
        assert data_rows[0]["objects"] == {"bbox": [], "categories": []}
        assert set(files[-2:]) == {dest / "data" / "metadata.jsonl", dest / "train" / "metadata.jsonl"}

        reloaded = load_od(dest, dataset_format="huggingface_vision")
        first = next(s for s in reloaded.samples if s.split == "train")
        assert [(d.bbox, d.category_id, d.source_annotation_id, d.area) for d in first.detections] == [
            ((10.0, 20.0, 30.0, 40.0), 0, 101, 1200.0),
            ((1.0, 2.0, 3.0, 4.0), 2, None, None),
        ]
        assert (first.width, first.height) == (640, 480)

    def test_csv_metadata_roundtrips(self, tmp_path: Path) -> None:
        dataset = _od_dataset(tmp_path)
        dest = tmp_path / "dest"

        get_writer("huggingface_vision", task=Task.OD).write(dataset, dest, metadata_format="csv")

        assert (dest / "train" / "metadata.csv").is_file()
        assert (dest / "data" / "metadata.csv").is_file()
        reloaded = load_od(dest, dataset_format="huggingface_vision")
        first = next(s for s in reloaded.samples if s.split == "train")
        assert [(d.bbox, d.category_id) for d in first.detections] == [
            ((10.0, 20.0, 30.0, 40.0), 0),
            ((1.0, 2.0, 3.0, 4.0), 2),
        ]

    def test_split_fallback_option(self, tmp_path: Path) -> None:
        dataset = _od_dataset(tmp_path)
        dest = tmp_path / "dest"

        get_writer("huggingface_vision", task=Task.OD).write(dataset, dest, split="val")

        # "val" normalises to "validation"; the unsplit sample uses the fallback.
        assert (dest / "validation" / "img2.jpg").is_file()
        assert (dest / "train" / "img1.jpg").is_file()  # preserved split wins

    def test_unknown_sample_split_falls_back(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        # A custom split dir would be dropped by the loader (and by Hugging Face
        # split inference), so unknown sample splits fall back with a warning.
        sample = ImageObjectDetectionSample(image_id="img.jpg", image_bytes=_JPEG, file_name="img.jpg", split="holdout")
        dest = tmp_path / "dest"

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            get_writer("huggingface_vision", task=Task.OD).write(ObjectDetectionDataset(samples=(sample,)), dest)

        assert (dest / "data" / "img.jpg").is_file()
        assert "not a Hugging Face ImageFolder split" in caplog.text
        rows = [json.loads(line) for line in (dest / "data" / "metadata.jsonl").read_text().splitlines()]
        assert rows[0]["file_name"] == "img.jpg"

    def test_preserves_category_names_roundtrip(self, tmp_path: Path) -> None:
        # The objects convention has one categories slot per box and no side
        # channel for a ClassLabel name table, so detections carrying both an id
        # and a name keep the name: reloads see "person", not "0".
        sample = ImageObjectDetectionSample(
            image_id="img.jpg",
            image_bytes=_JPEG,
            file_name="img.jpg",
            split="train",
            detections=(
                ObjectDetectionAnnotation(
                    bbox=(10.0, 20.0, 30.0, 40.0), category_id=0, source_category_id=1, category_name="person"
                ),
                ObjectDetectionAnnotation(
                    bbox=(1.0, 2.0, 3.0, 4.0), category_id=1, source_category_id=3, category_name="traffic light"
                ),
            ),
        )
        dest = tmp_path / "dest"

        get_writer("huggingface_vision", task=Task.OD).write(ObjectDetectionDataset(samples=(sample,)), dest)

        rows = [json.loads(line) for line in (dest / "train" / "metadata.jsonl").read_text().splitlines()]
        assert rows[0]["objects"]["categories"] == ["person", "traffic light"]

        reloaded = load_od(dest, dataset_format="huggingface_vision")
        assert [d.category_name for d in reloaded.samples[0].detections] == ["person", "traffic light"]
        taxonomy = reloaded.dataset_metadata.taxonomy
        assert taxonomy is not None
        assert sorted(e.name for e in taxonomy.entries) == ["person", "traffic light"]

    def test_drops_scores_and_segmentation_with_warnings(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        sample = ImageObjectDetectionSample(
            image_id="img.jpg",
            image_bytes=_JPEG,
            file_name="img.jpg",
            detections=(
                ObjectDetectionAnnotation(
                    bbox=(1.0, 2.0, 3.0, 4.0),
                    category_id=0,
                    score=0.9,
                    segmentation=[[1, 2, 3, 4, 5, 6]],
                    iscrowd=1,
                ),
            ),
        )
        dataset = ObjectDetectionDataset(samples=(sample,))
        dest = tmp_path / "dest"

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            get_writer("huggingface_vision", task=Task.OD).write(dataset, dest)

        rows = [json.loads(line) for line in (dest / "data" / "metadata.jsonl").read_text().splitlines()]
        assert "score" not in json.dumps(rows)
        assert "Dropped score" in caplog.text
        assert "segmentation/iscrowd" in caplog.text

    def test_skips_samples_without_source(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        dataset = ObjectDetectionDataset(samples=(ImageObjectDetectionSample(image_id="ghost"),))
        dest = tmp_path / "dest"

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            files = get_writer("huggingface_vision", task=Task.OD).write(dataset, dest)

        assert [f.name for f in files] == ["metadata.jsonl"]
        assert "no image source" in caplog.text

    def test_deduplicates_colliding_file_names_and_skips_unsafe(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        samples = (
            ImageObjectDetectionSample(image_id="one", image_bytes=_JPEG, file_name="img.jpg", split="train"),
            ImageObjectDetectionSample(image_id="two", image_bytes=_JPEG, file_name="img.jpg", split="train"),
            ImageObjectDetectionSample(image_id="bad", image_bytes=_JPEG, file_name="a\\b.jpg", split="train"),
        )
        dest = tmp_path / "dest"
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            files = get_writer("huggingface_vision", task=Task.OD).write(ObjectDetectionDataset(samples=samples), dest)
        written = {f.relative_to(dest).as_posix() for f in files if f.name != "metadata.jsonl"}
        assert written == {"train/img.jpg", "train/img-1.jpg"}
        assert "unsafe file name" in caplog.text

    def test_validate_options(self) -> None:
        writer = get_writer("huggingface_vision", task=Task.OD)
        with pytest.raises(ValueError, match="metadata_format"):
            writer.validate_options(metadata_format="parquet")
        with pytest.raises(ValueError, match="split"):
            writer.validate_options(split="../evil")
        with pytest.raises(ValueError, match="split"):
            writer.validate_options(split="holdout")

    def test_write_mode_error_refuses_non_empty_dest(self, tmp_path: Path) -> None:
        dataset = _od_dataset(tmp_path)
        dest = tmp_path / "dest"
        dest.mkdir()
        (dest / "existing.txt").write_text("data", encoding="utf-8")

        with pytest.raises(FileExistsError):
            write(dataset, dest, output_format="huggingface_vision")


class TestHuggingFaceVisionODWriterTaxonomyNames:
    """Id-only detections resolve their names through the dataset taxonomy (like the IC writer)."""

    def test_taxonomy_resolves_names_for_id_only_detections(self, tmp_path: Path) -> None:
        taxonomy = Taxonomy(
            entries=(CategoryEntry(source_id=0, name="person"), CategoryEntry(source_id=1, name="dog")),
            source_dataset="test",
            id_density="dense",
        )
        sample = ImageObjectDetectionSample(
            image_id="img.jpg",
            image_bytes=_JPEG,
            file_name="img.jpg",
            split="train",
            detections=(
                ObjectDetectionAnnotation(bbox=(10.0, 20.0, 30.0, 40.0), category_id=0),
                ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=1),
            ),
        )
        dest = tmp_path / "dest"

        get_writer("huggingface_vision", task=Task.OD).write(
            ObjectDetectionDataset(samples=(sample,), dataset_metadata=DatasetMetadata(taxonomy=taxonomy)), dest
        )

        rows = [json.loads(line) for line in (dest / "train" / "metadata.jsonl").read_text().splitlines()]
        assert rows[0]["objects"]["categories"] == ["person", "dog"]

    def test_dense_positional_fallback_and_unresolved_id_stays_int(self, tmp_path: Path) -> None:
        # source ids are strings, so by_source_id misses int 1; a dense taxonomy
        # falls back to positional lookup (mirrors the IC writer). An id with no
        # taxonomy entry at all keeps the bare int.
        taxonomy = Taxonomy(
            entries=(CategoryEntry(source_id="a", name="cat"), CategoryEntry(source_id="b", name="dog")),
            source_dataset="test",
            id_density="dense",
        )
        sample = ImageObjectDetectionSample(
            image_id="img.jpg",
            image_bytes=_JPEG,
            file_name="img.jpg",
            split="train",
            detections=(
                ObjectDetectionAnnotation(bbox=(10.0, 20.0, 30.0, 40.0), category_id=1),
                ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=7),
            ),
        )
        dest = tmp_path / "dest"

        get_writer("huggingface_vision", task=Task.OD).write(
            ObjectDetectionDataset(samples=(sample,), dataset_metadata=DatasetMetadata(taxonomy=taxonomy)), dest
        )

        rows = [json.loads(line) for line in (dest / "train" / "metadata.jsonl").read_text().splitlines()]
        assert rows[0]["objects"]["categories"] == ["dog", 7]

    def test_without_taxonomy_int_ids_are_written_as_before(self, tmp_path: Path) -> None:
        sample = ImageObjectDetectionSample(
            image_id="img.jpg",
            image_bytes=_JPEG,
            file_name="img.jpg",
            split="train",
            detections=(ObjectDetectionAnnotation(bbox=(10.0, 20.0, 30.0, 40.0), category_id=3),),
        )
        dest = tmp_path / "dest"

        get_writer("huggingface_vision", task=Task.OD).write(ObjectDetectionDataset(samples=(sample,)), dest)

        rows = [json.loads(line) for line in (dest / "train" / "metadata.jsonl").read_text().splitlines()]
        assert rows[0]["objects"]["categories"] == [3]

    def test_placeholder_names_for_integer_ids_round_trip_as_ids(self, tmp_path: Path) -> None:
        """Integer-only categories are loaded with names "2", "10"...; writing those names
        as strings made the reload renumber every box (#129)."""
        src = tmp_path / "src"
        (src / "train").mkdir(parents=True)
        (src / "train" / "a.jpg").write_bytes(_JPEG)
        (src / "train" / "metadata.jsonl").write_text(
            json.dumps({"file_name": "a.jpg", "objects": {"bbox": [[1, 1, 5, 5]] * 3, "categories": [2, 10, 5]}}),
            encoding="utf-8",
        )
        dest = tmp_path / "dest"

        write(load_od(src, dataset_format="huggingface_vision"), dest, output_format="huggingface_vision")

        rows = [json.loads(line) for line in (dest / "train" / "metadata.jsonl").read_text().splitlines()]
        assert rows[0]["objects"]["categories"] == [2, 10, 5]
        reloaded = load_od(dest, dataset_format="huggingface_vision")
        assert [d.category_id for d in reloaded.samples[0].detections] == [2, 10, 5]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "no issue yet: a box named as its id as text (2: '2') is written as the name, so a reload renumbers "
        "the classes in text order ('10' before '2'). The natural #155 fix names such boxes, so fix this with it."
    ),
)
def test_od_classes_named_as_their_ids_keep_their_ids(tmp_path: Path) -> None:
    (tmp_path / "a.jpg").write_bytes(_JPEG)
    sample = ImageObjectDetectionSample(
        image_id="a.jpg",
        path_or_uri=str(tmp_path / "a.jpg"),
        file_name="a.jpg",
        split="train",
        detections=(
            ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=2, category_name="2"),
            ObjectDetectionAnnotation(bbox=(5.0, 6.0, 7.0, 8.0), category_id=10, category_name="10"),
        ),
    )
    taxonomy = Taxonomy(entries=(CategoryEntry(source_id=2, name="2"), CategoryEntry(source_id=10, name="10")))
    dataset = ObjectDetectionDataset(samples=(sample,), dataset_metadata=DatasetMetadata(taxonomy=taxonomy))

    write(dataset, tmp_path / "out", output_format="huggingface_vision")
    reloaded = load_od(tmp_path / "out", dataset_format="huggingface_vision")

    assert [(d.category_id, d.category_name) for d in reloaded.samples[0].detections] == [(2, "2"), (10, "10")]


class TestHuggingFaceVisionOdAppend:
    """#130 case 2: appending keeps the rows already in the metadata files."""

    @staticmethod
    def _one(name: str, *, split: str | None = None, category: str = "car") -> ObjectDetectionDataset:
        taxonomy = Taxonomy(entries=(CategoryEntry(source_id=0, name=category),))
        sample = ImageObjectDetectionSample(
            image_id=name,
            file_name=f"{name}.jpg",
            image_bytes=_JPEG + name.encode(),
            width=64,
            height=48,
            split=split,
            detections=(ObjectDetectionAnnotation(bbox=(1, 1, 5, 5), category_id=0, category_name=category),),
        )
        return ObjectDetectionDataset(samples=(sample,), dataset_metadata=DatasetMetadata(taxonomy=taxonomy))

    @pytest.mark.parametrize("metadata_format", ["jsonl", "csv"])
    def test_append_keeps_earlier_rows(self, tmp_path: Path, metadata_format: str) -> None:
        out = tmp_path / "out"
        for name in ("a", "b"):
            write(
                self._one(name), out, output_format="huggingface_vision", mode="append", metadata_format=metadata_format
            )

        ds = load_od(out, dataset_format="huggingface_vision")

        assert [s.image_id for s in ds.samples] == ["data/a.jpg", "data/b.jpg"]
        assert all(len(s.detections) == 1 for s in ds.samples)

    def test_append_adds_new_splits_and_keeps_old_ones(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        write(self._one("a", split="train"), out, output_format="huggingface_vision")
        write(self._one("b", split="test", category="bus"), out, output_format="huggingface_vision", mode="append")
        write(self._one("c", split="train", category="bus"), out, output_format="huggingface_vision", mode="append")

        ds = load_od(out, dataset_format="huggingface_vision")

        assert [(s.image_id, s.detections[0].category_name) for s in ds.samples] == [
            ("train/a.jpg", "car"),
            ("train/c.jpg", "bus"),
            ("test/b.jpg", "bus"),
        ]

    def test_append_never_overwrites_an_image(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        write(self._one("a"), out, output_format="huggingface_vision")
        before = (out / "data" / "a.jpg").read_bytes()
        clash = self._one("a", category="bus")
        clash = replace(clash, samples=tuple(replace(s, image_bytes=b"other") for s in clash.samples))

        write(clash, out, output_format="huggingface_vision", mode="append")

        assert (out / "data" / "a.jpg").read_bytes() == before
        ds = load_od(out, dataset_format="huggingface_vision")
        assert [(s.image_id, s.detections[0].category_name) for s in ds.samples] == [
            ("data/a-1.jpg", "bus"),
            ("data/a.jpg", "car"),
        ]

    def test_append_with_another_metadata_format_is_refused(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        write(self._one("a"), out, output_format="huggingface_vision")
        before = sorted(p.relative_to(out).as_posix() for p in out.rglob("*"))

        with pytest.raises(ValueError, match="pass metadata_format='jsonl'"):
            write(self._one("b"), out, output_format="huggingface_vision", mode="append", metadata_format="csv")

        assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*")) == before

    @pytest.mark.parametrize(
        ("metadata_format", "metadata"),
        [
            ("jsonl", '{"file_name": "a.jpg"\n'),
            ("jsonl", '["not", "an", "object"]\n'),
            ("csv", "file_name,objects\na.jpg,{},extra\n"),
        ],
        ids=["truncated-json", "not-an-object", "csv-extra-column"],
    )
    def test_append_to_malformed_metadata_is_refused_before_writing(
        self, tmp_path: Path, metadata_format: str, metadata: str
    ) -> None:
        out = tmp_path / "out"
        write(self._one("a"), out, output_format="huggingface_vision", metadata_format=metadata_format)
        (out / "data" / f"metadata.{metadata_format}").write_text(metadata, encoding="utf-8")
        before = sorted(p.relative_to(out).as_posix() for p in out.rglob("*"))

        # Into another split, so no image directory but the malformed one's is touched.
        with pytest.raises(ValueError, match=rf"cannot append to .*metadata\.{metadata_format}"):
            write(
                self._one("b", split="train"),
                out,
                output_format="huggingface_vision",
                mode="append",
                metadata_format=metadata_format,
            )

        assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*")) == before

    def test_append_to_parquet_metadata_is_refused(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        (out / "train").mkdir(parents=True)
        (out / "train" / "metadata.parquet").write_bytes(b"PAR1")

        with pytest.raises(ValueError, match="parquet metadata"):
            write(self._one("b"), out, output_format="huggingface_vision", mode="append")

        assert sorted(p.name for p in (out / "train").iterdir()) == ["metadata.parquet"]
        assert not (out / "data").exists()

    def test_append_to_a_root_metadata_file_is_refused(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        (out / "img").mkdir(parents=True)
        (out / "img" / "x.jpg").write_bytes(_JPEG)
        (out / "metadata.jsonl").write_text('{"file_name": "img/x.jpg", "objects": {"bbox": [], "categories": []}}\n')

        with pytest.raises(ValueError, match="root-level metadata file"):
            write(self._one("b"), out, output_format="huggingface_vision", mode="append")

    def test_append_after_an_empty_write_replaces_the_placeholder(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        empty = ObjectDetectionDataset(samples=())
        write(empty, out, output_format="huggingface_vision")
        assert (out / "metadata.jsonl").is_file()

        write(self._one("a"), out, output_format="huggingface_vision", mode="append")
        write(empty, out, output_format="huggingface_vision", mode="append")

        assert not (out / "metadata.jsonl").exists()
        assert [s.image_id for s in load_od(out, dataset_format="huggingface_vision").samples] == ["data/a.jpg"]
