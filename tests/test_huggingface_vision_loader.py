"""Tests for the Hugging Face Vision still-image loaders (IR-3.2-S-2)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from datamaite import (
    DatasetFormat,
    HuggingFaceVisionImageClassificationLoader,
    HuggingFaceVisionObjectDetectionLoader,
    Task,
    load,
    load_ic,
    load_od,
)
from datamaite.image_classification import ImageClassificationDataset
from datamaite.loaders import available_formats, get_loader
from datamaite.object_detection import ObjectDetectionDataset

_JPEG = b"\xff\xd8\xff\xe0 fake jpeg payload"

_LOGGER = "datamaite._formats.huggingface_vision.loader"


def _touch_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_JPEG)
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class TestHuggingFaceVisionRegistry:
    def test_registered_for_both_tasks(self) -> None:
        assert DatasetFormat.HUGGINGFACE_VISION in available_formats(task=Task.IC)
        assert DatasetFormat.HUGGINGFACE_VISION in available_formats(task=Task.OD)
        assert isinstance(get_loader("huggingface_vision", task=Task.IC), HuggingFaceVisionImageClassificationLoader)
        assert isinstance(get_loader("huggingface_vision", task=Task.OD), HuggingFaceVisionObjectDetectionLoader)

    def test_generic_load_without_task_is_ambiguous(self, tmp_path: Path) -> None:
        _touch_image(tmp_path / "cat" / "a.jpg")
        with pytest.raises(ValueError, match="Multiple loaders registered"):
            load(tmp_path, dataset_format="huggingface_vision")

    def test_never_sniffed(self, tmp_path: Path) -> None:
        # Explicit opt-in only: a folder of class folders matches far too much (#40).
        _touch_image(tmp_path / "cat" / "a.jpg")
        assert HuggingFaceVisionImageClassificationLoader.sniff(tmp_path) is False
        assert HuggingFaceVisionObjectDetectionLoader.sniff(tmp_path) is False


class TestHuggingFaceVisionImageClassification:
    def test_split_class_folder_layout(self, tmp_path: Path) -> None:
        _touch_image(tmp_path / "train" / "cat" / "a.jpg")
        _touch_image(tmp_path / "train" / "dog" / "b.jpg")
        _touch_image(tmp_path / "validation" / "cat" / "c.jpg")

        ds = load_ic(tmp_path, dataset_format="huggingface_vision")

        assert isinstance(ds, ImageClassificationDataset)
        assert [(s.split, s.labels[0].category_name) for s in ds.samples] == [
            ("train", "cat"),
            ("train", "dog"),
            ("validation", "cat"),
        ]
        taxonomy = ds.dataset_metadata.taxonomy
        assert taxonomy is not None
        assert [(e.source_id, e.name) for e in taxonomy.entries] == [(0, "cat"), (1, "dog")]
        assert taxonomy.id_density == "dense"
        assert ds.dataset_metadata.splits == ("train", "validation")
        assert ds.dataset_id == "huggingface_vision"

    def test_hf_split_keyword_aliases_recognized(self, tmp_path: Path) -> None:
        # The full ImageFolder split-inference keyword set is recognized:
        # dev -> validation, eval -> test (matching Hugging Face data_files
        # keyword lists), so writer-emitted and HF-conventional dirs agree.
        _touch_image(tmp_path / "dev" / "cat" / "a.jpg")
        _touch_image(tmp_path / "eval" / "cat" / "b.jpg")

        ds = load_ic(tmp_path, dataset_format="huggingface_vision")

        assert [(s.split, s.labels[0].category_name) for s in ds.samples] == [
            ("validation", "cat"),
            ("test", "cat"),
        ]

    def test_unsplit_class_folder_layout(self, tmp_path: Path) -> None:
        _touch_image(tmp_path / "cat" / "a.jpg")
        _touch_image(tmp_path / "dog" / "b.jpg")

        ds = load_ic(tmp_path, dataset_format="huggingface_vision")

        assert [(s.split, s.labels[0].category_name) for s in ds.samples] == [(None, "cat"), (None, "dog")]

    def test_metadata_csv_labels(self, tmp_path: Path) -> None:
        _touch_image(tmp_path / "images" / "a.jpg")
        _touch_image(tmp_path / "images" / "b.jpg")
        (tmp_path / "metadata.csv").write_text(
            "file_name,label\nimages/a.jpg,cat\nimages/b.jpg,dog\n", encoding="utf-8"
        )

        ds = load_ic(tmp_path, dataset_format="huggingface_vision")

        assert [(s.image_id, s.labels[0].category_name) for s in ds.samples] == [
            ("images/a.jpg", "cat"),
            ("images/b.jpg", "dog"),
        ]
        assert ds.samples[0].metadata["metadata_file"] == str(tmp_path / "metadata.csv")

    def test_metadata_missing_file_and_unsafe_names_are_skipped(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        _touch_image(tmp_path / "a.jpg")
        _write_jsonl(
            tmp_path / "metadata.jsonl",
            [
                {"file_name": "a.jpg", "label": "cat"},
                {"file_name": "missing.jpg", "label": "cat"},
                {"file_name": "../escape.jpg", "label": "cat"},
                {"file_name": "notes.txt", "label": "cat"},
                {"label": "cat"},
            ],
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            ds = load_ic(tmp_path, dataset_format="huggingface_vision")

        assert [s.image_id for s in ds.samples] == ["a.jpg"]
        assert "missing image file" in caplog.text
        assert "unsafe file_name" in caplog.text
        assert "unsupported image extension" in caplog.text
        assert "missing file_name" in caplog.text

    def test_unlabeled_metadata_rows_yield_label_free_samples(self, tmp_path: Path) -> None:
        _touch_image(tmp_path / "a.jpg")
        _write_jsonl(tmp_path / "metadata.jsonl", [{"file_name": "a.jpg"}])

        ds = load_ic(tmp_path, dataset_format="huggingface_vision")

        assert ds.samples[0].labels == ()
        assert ds.dataset_metadata.taxonomy is None

    def test_image_extensions_option(self, tmp_path: Path) -> None:
        _touch_image(tmp_path / "cat" / "a.jpg")
        _touch_image(tmp_path / "cat" / "b.png")

        only_png = load_ic(tmp_path, dataset_format="huggingface_vision", image_extensions=".png")

        assert [s.image_id for s in only_png.samples] == ["cat/b.png"]
        with pytest.raises(ValueError, match="safe extensions"):
            load_ic(tmp_path, dataset_format="huggingface_vision", image_extensions="../evil")

    def test_missing_root_returns_empty(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            ds = HuggingFaceVisionImageClassificationLoader().load(tmp_path / "missing")
        assert len(ds) == 0
        assert "not a directory" in caplog.text

    def test_integer_labels_are_ordered_numerically_and_preserve_ids(self, tmp_path: Path) -> None:
        # ClassLabel-style integer folder names must not be re-sorted lexically
        # (0,1,10,11,2,...) nor re-indexed positionally: label "10" keeps id 10.
        labels = [str(index) for index in range(11)]  # "0".."10"
        for index, label in enumerate(labels):
            _touch_image(tmp_path / label / f"img{index}.jpg")

        ds = load_ic(tmp_path, dataset_format="huggingface_vision")

        taxonomy = ds.dataset_metadata.taxonomy
        assert taxonomy is not None
        assert [e.name for e in taxonomy.entries] == labels  # numeric order, not lexical
        assert [e.source_id for e in taxonomy.entries] == list(range(11))
        assert taxonomy.id_density == "dense"
        by_label = {s.labels[0].category_name: s.labels[0] for s in ds.samples}
        assert by_label["10"].source_category_id == 10
        assert by_label["10"].category_id == 10

    def test_metadata_without_label_column_does_not_fabricate_folder_label(self, tmp_path: Path) -> None:
        # A metadata file disables folder-based label inference (HF ImageFolder):
        # a subdir file_name with no label column must not invent a class.
        _touch_image(tmp_path / "cat" / "a.jpg")
        _write_jsonl(tmp_path / "metadata.jsonl", [{"file_name": "cat/a.jpg"}])

        ds = load_ic(tmp_path, dataset_format="huggingface_vision")

        assert ds.samples[0].labels == ()
        assert ds.dataset_metadata.taxonomy is None

    def test_split_dirs_with_class_subdirs_win_over_stray_class_dir(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        # When split-named dirs contain class subdirectories they are unambiguously
        # splits; a stray non-split top-level class dir is ignored (with a warning).
        _touch_image(tmp_path / "train" / "cat" / "a.jpg")
        _touch_image(tmp_path / "validation" / "dog" / "b.jpg")
        _touch_image(tmp_path / "stray" / "c.jpg")

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            ds = load_ic(tmp_path, dataset_format="huggingface_vision")

        assert [(s.split, s.labels[0].category_name) for s in ds.samples] == [
            ("train", "cat"),
            ("validation", "dog"),
        ]
        assert "ignoring non-split top-level dir(s) ['stray']" in caplog.text

    def test_sparse_integer_labels_preserve_ids(self, tmp_path: Path) -> None:
        for label in ["0", "2", "5"]:
            _touch_image(tmp_path / label / f"img{label}.jpg")

        ds = load_ic(tmp_path, dataset_format="huggingface_vision")

        taxonomy = ds.dataset_metadata.taxonomy
        assert taxonomy is not None
        assert [e.source_id for e in taxonomy.entries] == [0, 2, 5]
        assert taxonomy.id_density == "sparse"

    def test_class_dir_named_like_split_is_not_treated_as_split(self, tmp_path: Path) -> None:
        # An unsplit repo with a class literally named "train" must keep it (and
        # its sibling class folders), not silently drop everything but "train/".
        for cls in ["train", "cat", "dog"]:
            _touch_image(tmp_path / cls / f"{cls}.jpg")

        ds = load_ic(tmp_path, dataset_format="huggingface_vision")

        assert sorted(s.labels[0].category_name for s in ds.samples) == ["cat", "dog", "train"]
        assert len(ds.samples) == 3
        assert all(s.split is None for s in ds.samples)


class TestHuggingFaceVisionObjectDetection:
    def _od_root(self, tmp_path: Path) -> Path:
        _touch_image(tmp_path / "train" / "img1.jpg")
        _touch_image(tmp_path / "train" / "img2.jpg")
        _write_jsonl(
            tmp_path / "metadata.jsonl",
            [
                {
                    "file_name": "train/img1.jpg",
                    "width": 640,
                    "height": 480,
                    "objects": {
                        "bbox": [[10, 20, 30, 40], [1, 2, 3, 4]],
                        "categories": [0, 2],
                        "id": [101, 102],
                        "area": [1200.0, 12.0],
                    },
                },
                {"file_name": "train/img2.jpg", "width": 320, "height": 240, "objects": {"bbox": [], "categories": []}},
            ],
        )
        return tmp_path

    def test_metadata_jsonl_objects(self, tmp_path: Path) -> None:
        ds = load_od(self._od_root(tmp_path), dataset_format="huggingface_vision")

        assert isinstance(ds, ObjectDetectionDataset)
        first, second = ds.samples
        assert (first.width, first.height, first.split) == (640, 480, "train")
        assert [(d.bbox, d.category_id, d.source_annotation_id, d.area) for d in first.detections] == [
            ((10.0, 20.0, 30.0, 40.0), 0, 101, 1200.0),
            ((1.0, 2.0, 3.0, 4.0), 2, 102, 12.0),
        ]
        assert second.detections == ()
        taxonomy = ds.dataset_metadata.taxonomy
        assert taxonomy is not None
        assert [(e.source_id, e.name) for e in taxonomy.entries] == [(0, "0"), (2, "2")]
        assert taxonomy.id_density == "sparse"

    def test_metadata_csv_objects_are_json_decoded(self, tmp_path: Path) -> None:
        _touch_image(tmp_path / "img.jpg")
        objects = json.dumps({"bbox": [[5, 6, 7, 8]], "categories": ["person"]})
        (tmp_path / "metadata.csv").write_text(
            'file_name,width,height,objects\nimg.jpg,100,200,"' + objects.replace('"', '""') + '"\n',
            encoding="utf-8",
        )

        ds = load_od(tmp_path, dataset_format="huggingface_vision")

        detection = ds.samples[0].detections[0]
        assert detection.bbox == (5.0, 6.0, 7.0, 8.0)
        assert detection.category_name == "person"
        assert detection.category_id is None
        taxonomy = ds.dataset_metadata.taxonomy
        assert taxonomy is not None
        assert [(e.source_id, e.name) for e in taxonomy.entries] == [("person", "person")]
        assert (ds.samples[0].width, ds.samples[0].height) == (100, 200)

    def test_malformed_objects_are_skipped_per_detection(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        _touch_image(tmp_path / "img.jpg")
        _write_jsonl(
            tmp_path / "metadata.jsonl",
            [
                {
                    "file_name": "img.jpg",
                    "objects": {
                        "bbox": [[1, 2, 3], [1, 2, 0, 4], [1, 2, "x", 4], [5, 6, 7, 8]],
                        "categories": [0, 1, 2, 3],
                    },
                }
            ],
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            ds = load_od(tmp_path, dataset_format="huggingface_vision")

        assert [(d.bbox, d.category_id) for d in ds.samples[0].detections] == [((5.0, 6.0, 7.0, 8.0), 3)]
        assert "expected 4 values" in caplog.text
        assert "non-positive area" in caplog.text
        assert "non-numeric value" in caplog.text

    def test_category_length_mismatch_pairs_to_shorter_list(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        _touch_image(tmp_path / "img.jpg")
        _write_jsonl(
            tmp_path / "metadata.jsonl",
            [{"file_name": "img.jpg", "objects": {"bbox": [[1, 2, 3, 4], [5, 6, 7, 8]], "categories": [9]}}],
        )

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            ds = load_od(tmp_path, dataset_format="huggingface_vision")

        detections = ds.samples[0].detections
        assert [(d.bbox, d.category_id) for d in detections] == [
            ((1.0, 2.0, 3.0, 4.0), 9),
            ((5.0, 6.0, 7.0, 8.0), None),
        ]
        assert "pairing up to the shorter list" in caplog.text

    def test_row_without_objects_is_a_background_image(self, tmp_path: Path) -> None:
        _touch_image(tmp_path / "img.jpg")
        _write_jsonl(tmp_path / "metadata.jsonl", [{"file_name": "img.jpg"}])

        ds = load_od(tmp_path, dataset_format="huggingface_vision")

        assert len(ds) == 1
        assert ds.samples[0].detections == ()

    def test_no_metadata_at_all_loads_empty_with_warning(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        _touch_image(tmp_path / "cat" / "a.jpg")

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            ds = load_od(tmp_path, dataset_format="huggingface_vision")

        assert len(ds) == 0
        assert "requires a metadata file" in caplog.text

    def test_taxonomy_sorts_integer_categories_numerically(self, tmp_path: Path) -> None:
        # >=10 contiguous integer category ids must order 0..N and read as dense,
        # not lexically ordered (0,1,10,11,2,...) which also breaks the density check.
        _touch_image(tmp_path / "img.jpg")
        count = 12
        _write_jsonl(
            tmp_path / "metadata.jsonl",
            [
                {
                    "file_name": "img.jpg",
                    "objects": {"bbox": [[1, 2, 3, 4]] * count, "categories": list(range(count))},
                }
            ],
        )

        ds = load_od(tmp_path, dataset_format="huggingface_vision")

        taxonomy = ds.dataset_metadata.taxonomy
        assert taxonomy is not None
        assert [e.source_id for e in taxonomy.entries] == list(range(count))
        assert taxonomy.id_density == "dense"

    def test_per_split_metadata_files(self, tmp_path: Path) -> None:
        _touch_image(tmp_path / "train" / "img.jpg")
        _write_jsonl(
            tmp_path / "train" / "metadata.jsonl",
            [{"file_name": "img.jpg", "objects": {"bbox": [[1, 2, 3, 4]], "categories": [1]}}],
        )

        ds = load_od(tmp_path, dataset_format="huggingface_vision")

        assert [s.image_id for s in ds.samples] == ["train/img.jpg"]
        assert ds.samples[0].split == "train"
        assert len(ds.samples[0].detections) == 1
