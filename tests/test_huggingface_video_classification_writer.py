"""Tests for the Hugging Face video classification writer and round trip."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import pytest

from databridge import (
    DatasetFormat,
    HuggingFaceVideoClassificationWriter,
    convert,
    load_huggingface_video_classification,
    write,
)
from databridge.model import BoxTrackDataset, VideoClassificationDataset, VideoClassificationSample
from databridge.writers import available_output_formats, get_writer


def _write_video(path: Path, payload: bytes = b"video") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _sample(
    video_id: int,
    video_path: Path,
    *,
    file_name: str | None = None,
    label: str | None = None,
    split: str | None = None,
    metadata: dict[str, object] | None = None,
) -> VideoClassificationSample:
    return VideoClassificationSample(
        video_id=video_id,
        video_path=str(video_path),
        file_name=file_name or video_path.name,
        label=label,
        split=split,
        metadata=metadata or {},
        size_bytes=video_path.stat().st_size if video_path.exists() else None,
    )


def _dataset(*samples: VideoClassificationSample) -> VideoClassificationDataset:
    labels = sorted({sample.label for sample in samples if sample.label is not None})
    label_ids = {label: index for index, label in enumerate(labels)}
    categories = {f"huggingface_video_classification/label/{label}": label_ids[label] for label in labels}
    return VideoClassificationDataset(
        samples=tuple(samples),
        categories=categories,
        labels={index: label for label, index in label_ids.items()},
    )


def _classification_fingerprint(ds: VideoClassificationDataset) -> list[tuple[object, ...]]:
    return sorted(
        (
            sample.split,
            sample.label,
            Path(sample.video_path).suffix,
            sample.size_bytes,
            tuple(sorted((key, repr(value)) for key, value in sample.metadata.items())),
        )
        for sample in ds.samples
    )


class TestHuggingFaceVideoClassificationWriterRegistry:
    def test_registered_and_public_api(self) -> None:
        assert DatasetFormat.HUGGINGFACE_VIDEO_CLASSIFICATION in available_output_formats()
        assert isinstance(
            get_writer(DatasetFormat.HUGGINGFACE_VIDEO_CLASSIFICATION), HuggingFaceVideoClassificationWriter
        )
        assert isinstance(get_writer("huggingface_video_classification"), HuggingFaceVideoClassificationWriter)


class TestHuggingFaceVideoClassificationWriterHappyPath:
    def test_write_produces_reloadable_metadata_csv_dataset(self, tmp_path: Path) -> None:
        cat = _write_video(tmp_path / "cat.mp4", b"cat-video")
        dog = _write_video(tmp_path / "dog.webm", b"dog-video")
        ds = _dataset(
            _sample(0, cat, file_name="source/cat.mp4", label="cat", split="train", metadata={"source": "unit"}),
            _sample(
                1,
                dog,
                file_name="source/dog.webm",
                label="dog",
                split="validation",
                metadata={"source": "unit"},
            ),
        )

        out = tmp_path / "out"
        files = write(ds, out, output_format="huggingface_video_classification", verbose=True)

        assert out / "metadata.csv" in files
        assert out / "train" / "cat.mp4" in files
        assert out / "validation" / "dog.webm" in files
        assert (out / "train" / "cat.mp4").read_bytes() == b"cat-video"
        with (out / "metadata.csv").open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows == [
            {"file_name": "train/cat.mp4", "label": "cat", "source": "unit"},
            {"file_name": "validation/dog.webm", "label": "dog", "source": "unit"},
        ]
        reloaded = load_huggingface_video_classification(out)
        assert [(sample.split, sample.label) for sample in reloaded.samples] == [
            ("train", "cat"),
            ("validation", "dog"),
        ]
        assert reloaded.categories == {
            "huggingface_video_classification/label/cat": 0,
            "huggingface_video_classification/label/dog": 1,
        }

    def test_round_trip_from_folder_layout(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write_video(src / "train" / "cat" / "a.mp4", b"a")
        _write_video(src / "test" / "dog" / "b.mp4", b"b")
        ds = load_huggingface_video_classification(src)

        write(ds, tmp_path / "out", output_format="huggingface_video_classification")

        assert _classification_fingerprint(load_huggingface_video_classification(tmp_path / "out")) == [
            ("test", "dog", ".mp4", 1, (("file_name", "'test/b.mp4'"), ("label", "'dog'"))),
            ("train", "cat", ".mp4", 1, (("file_name", "'train/a.mp4'"), ("label", "'cat'"))),
        ]

    def test_convert_huggingface_video_classification_end_to_end(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _write_video(src / "train" / "cat" / "a.mp4", b"a")

        files = convert(
            src,
            tmp_path / "out",
            input_format="huggingface_video_classification",
            output_format="huggingface_video_classification",
            verbose=True,
        )

        assert files
        reloaded = load_huggingface_video_classification(tmp_path / "out")
        assert reloaded.sample_count == 1
        assert reloaded.samples[0].label == "cat"

    def test_jsonl_metadata_preserves_json_safe_metadata_and_unsplit_data(self, tmp_path: Path) -> None:
        video = _write_video(tmp_path / "video.mp4", b"video")
        ds = _dataset(_sample(0, video, label="action", metadata={"score": 1.5, "tags": ["a", "b"]}))

        write(ds, tmp_path / "out", output_format="huggingface_video_classification", metadata_format="jsonl")

        row = json.loads((tmp_path / "out" / "metadata.jsonl").read_text(encoding="utf-8"))
        assert row == {"file_name": "data/video.mp4", "label": "action", "score": 1.5, "tags": ["a", "b"]}
        reloaded = load_huggingface_video_classification(tmp_path / "out")
        assert reloaded.samples[0].label == "action"
        assert reloaded.samples[0].split is None

    def test_preserve_splits_can_be_disabled(self, tmp_path: Path) -> None:
        video = _write_video(tmp_path / "video.mp4")
        ds = _dataset(_sample(0, video, label="cat", split="train"))

        write(
            ds,
            tmp_path / "out",
            output_format="huggingface_video_classification",
            split="val",
            preserve_splits=False,
        )

        rows = list(csv.DictReader((tmp_path / "out" / "metadata.csv").open(newline="", encoding="utf-8")))
        assert rows[0]["file_name"] == "validation/video.mp4"

    def test_duplicate_video_names_are_disambiguated(self, tmp_path: Path) -> None:
        first = _write_video(tmp_path / "a" / "clip.mp4", b"first")
        second = _write_video(tmp_path / "b" / "clip.mp4", b"second")
        ds = _dataset(
            _sample(0, first, file_name="clip.mp4", label="class", split="train"),
            _sample(1, second, file_name="clip.mp4", label="class", split="train"),
        )

        write(ds, tmp_path / "out", output_format="huggingface_video_classification")

        assert (tmp_path / "out" / "train" / "clip.mp4").read_bytes() == b"first"
        assert (tmp_path / "out" / "train" / "clip-1.mp4").read_bytes() == b"second"


class TestHuggingFaceVideoClassificationWriterMalformedInputs:
    def test_missing_video_is_skipped_with_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        missing = tmp_path / "missing.mp4"
        ds = _dataset(_sample(0, missing, label="missing"))

        with caplog.at_level(logging.WARNING, logger="databridge._formats.huggingface_video_classification.writer"):
            write(ds, tmp_path / "out", output_format="huggingface_video_classification")

        assert (tmp_path / "out" / "metadata.csv").read_text(encoding="utf-8") == "file_name,label\n"
        assert "missing video" in caplog.text
        assert "No Hugging Face video classification files were written" in caplog.text

    def test_box_track_dataset_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="requires VideoClassificationDataset"):
            write(
                BoxTrackDataset(sequences=(), categories={}),
                tmp_path / "out",
                output_format="huggingface_video_classification",
            )

    def test_invalid_options_raise(self, tmp_path: Path) -> None:
        ds = VideoClassificationDataset(samples=(), categories={})
        with pytest.raises(ValueError, match="metadata_format"):
            write(ds, tmp_path / "out", output_format="huggingface_video_classification", metadata_format="parquet")
        with pytest.raises(ValueError, match="split"):
            write(ds, tmp_path / "out", output_format="huggingface_video_classification", split="../train")
