"""Tests for the Hugging Face video classification loader."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from datamaite import (
    DatasetFormat,
    HuggingFaceVideoClassificationLoader,
    VideoClassificationDataset,
    convert,
    dataset_stats,
    load,
    load_vc,
)
from datamaite._cli import main
from datamaite._formats.huggingface_video_classification.loader import (
    load_huggingface_video_classification,
)
from datamaite.loaders import available_formats, get_loader


def _write_video(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real video; decoding is not required by this loader")
    return path


class TestHuggingFaceVideoClassificationRegistry:
    def test_registered_and_public_api(self) -> None:
        assert DatasetFormat.HUGGINGFACE_VIDEO_CLASSIFICATION in available_formats()
        assert isinstance(
            get_loader(DatasetFormat.HUGGINGFACE_VIDEO_CLASSIFICATION), HuggingFaceVideoClassificationLoader
        )
        assert isinstance(get_loader("huggingface_video_classification"), HuggingFaceVideoClassificationLoader)
        assert callable(load_huggingface_video_classification)

    def test_dispatch_loads_huggingface_video_classification(self, tmp_path: Path) -> None:
        _write_video(tmp_path / "cat" / "clip.mp4")

        ds = load(tmp_path, dataset_format="huggingface_video_classification")

        assert isinstance(ds, VideoClassificationDataset)
        assert ds.sample_count == 1
        assert ds.samples[0].label == "cat"

    def test_load_vc_public_entrypoint_loads_huggingface_video_classification(self, tmp_path: Path) -> None:
        _write_video(tmp_path / "cat" / "clip.mp4")

        ds = load_vc(tmp_path)

        assert isinstance(ds, VideoClassificationDataset)
        assert ds.sample_count == 1
        assert ds.samples[0].label == "cat"


class TestHuggingFaceVideoClassificationHappyPath:
    def test_loads_split_class_folder_layout(self, tmp_path: Path) -> None:
        train_cat = _write_video(tmp_path / "train" / "cat" / "a.mp4")
        train_dog = _write_video(tmp_path / "train" / "dog" / "b.MOV")
        test_cat = _write_video(tmp_path / "test" / "cat" / "c.webm")
        _write_video(tmp_path / "train" / "cat" / "nested" / "d.mkv")
        (tmp_path / "train" / "cat" / "notes.txt").write_text("ignored", encoding="utf-8")

        ds = load_huggingface_video_classification(tmp_path)

        assert ds.sample_count == 4
        assert len(ds) == 4  # plain record count, not a MAITE item view.
        assert ds.categories == {
            "huggingface_video_classification/label/cat": 0,
            "huggingface_video_classification/label/dog": 1,
        }
        assert ds.label_names() == {0: "cat", 1: "dog"}

        first = ds.samples[0]
        assert first.video_path == str(train_cat)
        assert first.file_name == "train/cat/a.mp4"
        assert first.split == "train"
        assert first.label == "cat"
        assert first.label_id == 0
        assert first.label_uri == "huggingface_video_classification/label/cat"
        assert first.metadata_path is None
        assert first.size_bytes == train_cat.stat().st_size
        assert first.video_meta == {
            "format": "huggingface_video_classification",
            "source_path": str(train_cat),
            "file_name": "train/cat/a.mp4",
            "split": "train",
            "label": "cat",
            "label_id": 0,
            "label_uri": "huggingface_video_classification/label/cat",
        }

        assert ds.samples[1].video_path == str(tmp_path / "train" / "cat" / "nested" / "d.mkv")
        assert ds.samples[1].label == "cat"
        assert ds.samples[2].video_path == str(train_dog)
        assert ds.samples[2].label_id == 1
        assert ds.samples[3].video_path == str(test_cat)
        assert ds.samples[3].split == "test"

    def test_video_classification_dataset_does_not_masquerade_as_maite_mot(self, tmp_path: Path) -> None:
        _write_video(tmp_path / "cat" / "clip.mp4")

        ds = load_huggingface_video_classification(tmp_path)

        assert ds[0] == ds.samples[0]
        assert "index2label" not in ds.metadata
        assert ds.metadata == {
            "id": "datamaite",
            "task": "vc",
            "maite_protocol": None,
            "labels": {0: "cat"},
        }
        assert not hasattr(ds, "index2label")
        assert not hasattr(ds, "sequence_count")

    def test_loads_metadata_csv_with_explicit_labels_and_extra_fields(self, tmp_path: Path) -> None:
        dog = _write_video(tmp_path / "videos" / "dog.mp4")
        cat = _write_video(tmp_path / "videos" / "cat.mp4")
        (tmp_path / "metadata.csv").write_text(
            "file_name,label,license\nvideos/dog.mp4,dog,cc-by\nvideos/cat.mp4,cat,public-domain\n",
            encoding="utf-8",
        )
        _write_video(tmp_path / "videos" / "unlisted.mp4")

        ds = load_huggingface_video_classification(tmp_path)

        assert [sample.video_path for sample in ds.samples] == [str(cat), str(dog)]
        assert [sample.label for sample in ds.samples] == ["cat", "dog"]
        assert all(sample.metadata_path == str(tmp_path / "metadata.csv") for sample in ds.samples)
        assert ds.samples[0].metadata == {
            "file_name": "videos/cat.mp4",
            "label": "cat",
            "license": "public-domain",
        }
        assert ds.samples[1].metadata["license"] == "cc-by"

    def test_loads_per_split_metadata_jsonl_and_accepts_extension_option(self, tmp_path: Path) -> None:
        clip = _write_video(tmp_path / "train" / "clips" / "first.custom")
        (tmp_path / "train" / "metadata.jsonl").write_text(
            json.dumps({"file_name": "clips/first.custom", "label": 7, "source": "synthetic"}) + "\n",
            encoding="utf-8",
        )

        ds = load_huggingface_video_classification(tmp_path, video_extensions="custom")

        assert ds.sample_count == 1
        sample = ds.samples[0]
        assert sample.video_path == str(clip)
        assert sample.split == "train"
        assert sample.label == "7"
        assert sample.label_id == 0
        assert sample.metadata["source"] == "synthetic"

    def test_distinct_labels_with_same_slug_keep_distinct_ids(self, tmp_path: Path) -> None:
        _write_video(tmp_path / "clips" / "space.mp4")
        _write_video(tmp_path / "clips" / "slash.mp4")
        _write_video(tmp_path / "clips" / "underscore.mp4")
        (tmp_path / "metadata.jsonl").write_text(
            json.dumps({"file_name": "clips/space.mp4", "label": "a b"})
            + "\n"
            + json.dumps({"file_name": "clips/slash.mp4", "label": "a/b"})
            + "\n"
            + json.dumps({"file_name": "clips/underscore.mp4", "label": "a_b"})
            + "\n",
            encoding="utf-8",
        )

        ds = load_huggingface_video_classification(tmp_path)

        assert ds.label_names() == {0: "a b", 1: "a/b", 2: "a_b"}
        assert ds.categories == {
            "huggingface_video_classification/label/a%20b": 0,
            "huggingface_video_classification/label/a%2Fb": 1,
            "huggingface_video_classification/label/a_b": 2,
        }
        assert {sample.label_id for sample in ds.samples} == {0, 1, 2}
        assert {sample.label_uri for sample in ds.samples} == set(ds.categories)
        assert all(sample.label_id in ds.label_names() for sample in ds.samples if sample.label_id is not None)


class TestHuggingFaceVideoClassificationMalformedInputs:
    def test_missing_or_empty_root_returns_empty(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="datamaite._formats.huggingface_video_classification.loader"):
            missing = load_huggingface_video_classification(tmp_path / "missing")
            empty = load_huggingface_video_classification(tmp_path)

        assert missing.sample_count == 0
        assert empty.sample_count == 0
        assert "not a directory" in caplog.text
        assert "No loadable Hugging Face video classification files" in caplog.text

    def test_malformed_metadata_rows_are_skipped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        _write_video(tmp_path / "safe" / "ok.mp4")
        (tmp_path / "metadata.jsonl").write_text(
            "not json\n"
            + json.dumps(["not", "an", "object"])
            + "\n"
            + json.dumps({"label": "missing-file-name"})
            + "\n"
            + json.dumps({"file_name": "../escape.mp4", "label": "bad"})
            + "\n"
            + json.dumps({"file_name": "safe/not-video.txt", "label": "bad"})
            + "\n"
            + json.dumps({"file_name": "safe/missing.mp4", "label": "missing"})
            + "\n"
            + json.dumps({"file_name": "safe/ok.mp4", "label": {"bad": "shape"}})
            + "\n",
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.huggingface_video_classification.loader"):
            ds = load_huggingface_video_classification(tmp_path)

        assert ds.sample_count == 1
        assert ds.samples[0].label == "safe"  # folder-derived fallback after malformed label.
        assert "malformed Hugging Face metadata JSONL row" in caplog.text
        assert "non-object Hugging Face metadata JSONL row" in caplog.text
        assert "missing file_name" in caplog.text
        assert "unsafe file_name" in caplog.text
        assert "unsupported video extension" in caplog.text
        assert "missing file" in caplog.text
        assert "Ignoring non-scalar Hugging Face label value" in caplog.text

    def test_metadata_csv_missing_file_name_returns_empty(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        (tmp_path / "metadata.csv").write_text("path,label\nclip.mp4,cat\n", encoding="utf-8")
        _write_video(tmp_path / "cat" / "clip.mp4")

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.huggingface_video_classification.loader"):
            ds = load_huggingface_video_classification(tmp_path)

        assert ds.sample_count == 0
        assert "missing required file_name column" in caplog.text

    def test_unreadable_parquet_metadata_falls_back_to_folder_layout(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        (tmp_path / "metadata.parquet").write_bytes(b"not a parquet file")
        _write_video(tmp_path / "cat" / "clip.mp4")

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.huggingface_video_classification.loader"):
            ds = load_huggingface_video_classification(tmp_path)

        assert ds.sample_count == 1
        assert ds.samples[0].label == "cat"
        assert "parquet metadata produced no loadable rows" in caplog.text

    def test_invalid_video_extension_option_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="video_extensions"):
            load_huggingface_video_classification(tmp_path, video_extensions="../mp4")


class TestVideoClassificationTaskGuards:
    def test_convert_rejects_video_classification_dataset(self, tmp_path: Path) -> None:
        _write_video(tmp_path / "cat" / "clip.mp4")

        with pytest.raises(TypeError, match="consumes BoxTrackDataset"):
            convert(
                tmp_path,
                tmp_path / "out",
                input_format="huggingface_video_classification",
                output_format="hmie",
            )

    def test_dataset_stats_rejects_video_classification_dataset(self, tmp_path: Path) -> None:
        _write_video(tmp_path / "cat" / "clip.mp4")
        ds = load_huggingface_video_classification(tmp_path)

        with pytest.raises(TypeError, match="box-track datasets only"):
            dataset_stats(ds)

    def test_stats_cli_rejects_video_classification_dataset(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_video(tmp_path / "cat" / "clip.mp4")

        rc = main(["stats", str(tmp_path), "--format", "huggingface_video_classification"])

        captured = capsys.readouterr()
        assert rc == 2
        assert "box-track datasets only" in captured.err
        assert "VideoClassificationDataset" in captured.err
