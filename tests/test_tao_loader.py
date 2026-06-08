"""Tests for the TAO loader."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from databridge import DatasetFormat, load, load_tao
from databridge._formats.tao.loader import TaoLoader
from databridge.loaders import available_formats, get_loader
from databridge.model import BoxTrackDataset


def _write_tao(root: Path, split: str = "train", payload: dict[str, Any] | None = None) -> Path:
    annotations = root / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    path = annotations / f"{split}.json"
    path.write_text(json.dumps(payload if payload is not None else _basic_payload()), encoding="utf-8")
    return path


def _touch_frames(root: Path, *names: str) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a real image")


def _basic_payload() -> dict[str, Any]:
    return {
        "videos": [
            {"id": 10, "name": "video-a", "width": 640, "height": 480, "fps": 30},
            {"id": 20, "name": "empty-video", "width": 320, "height": 240},
        ],
        "images": [
            {
                "id": 100,
                "video_id": 10,
                "file_name": "train/video-a/000001.jpg",
                "frame_index": 0,
                "width": 640,
                "height": 480,
            },
            {
                "id": 101,
                "video_id": 10,
                "file_name": "train/video-a/000002.jpg",
                "frame_index": 1,
                "width": 640,
                "height": 480,
            },
            {
                "id": 200,
                "video_id": 20,
                "file_name": "train/empty-video/000001.jpg",
                "frame_index": 0,
                "width": 320,
                "height": 240,
            },
        ],
        "tracks": [{"id": 5, "video_id": 10, "category_id": 1}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [
            {
                "id": 900,
                "image_id": 100,
                "track_id": 5,
                "category_id": 1,
                "bbox": [10, 20, 30, 40],
                "area": 1200,
                "iscrowd": 0,
                "segmentation": [[10, 20, 40, 20, 40, 60, 10, 60]],
            },
            {"id": 901, "image_id": 101, "track_id": 5, "category_id": 1, "bbox": [12, 22, 30, 40]},
        ],
    }


class TestTaoRegistry:
    def test_registered_and_public_api(self) -> None:
        assert DatasetFormat.TAO in available_formats()
        assert isinstance(get_loader(DatasetFormat.TAO), TaoLoader)
        assert isinstance(get_loader("tao"), TaoLoader)

    def test_dispatch_loads_tao(self, tmp_path: Path) -> None:
        _write_tao(tmp_path)

        ds = load(tmp_path, dataset_format="tao")

        assert isinstance(ds, BoxTrackDataset)
        assert ds.sequence_count == 2
        assert ds.num_boxes == 2


class TestTaoHappyPath:
    def test_loads_official_tao_json_and_empty_video(self, tmp_path: Path) -> None:
        _touch_frames(
            tmp_path,
            "frames/train/video-a/000001.jpg",
            "frames/train/video-a/000002.jpg",
            "frames/train/empty-video/000001.jpg",
        )
        _write_tao(tmp_path)

        ds = load_tao(tmp_path)

        assert ds.sequence_count == 2
        assert len(ds) == 0  # TAO is image-sequence backed, not video-backed.
        assert ds.num_boxes == 2
        assert ds.categories == {"tao/category_1/person": 1}
        assert ds.index2label() == {1: "person"}

        seq = ds.sequences[0]
        assert seq.video_path is None
        assert seq.frame_dir == str(tmp_path / "frames" / "train" / "video-a")
        assert seq.frame_files == (
            str(tmp_path / "frames" / "train" / "video-a" / "000001.jpg"),
            str(tmp_path / "frames" / "train" / "video-a" / "000002.jpg"),
        )
        assert seq.frame_path(0) == tmp_path / "frames" / "train" / "video-a" / "000001.jpg"
        assert seq.frame_filename(1) == "000002.jpg"
        assert seq.fps == 30.0
        assert seq.num_frames == 2
        assert seq.num_frames_exact is True
        assert seq.duration == pytest.approx(2 / 30)
        assert seq.width == 640
        assert seq.height == 480
        assert seq.video_meta["split"] == "train"
        assert seq.video_meta["sequence_name"] == "video-a"

        first = seq.boxes[0]
        assert first.track_uuid == "train:10:5"
        assert first.track_id == 5
        assert first.category_id == 1
        assert first.category_uri == "tao/category_1/person"
        assert first.category_name == "person"
        assert first.bbox == (10.0, 20.0, 30.0, 40.0)
        assert first.frame_index == 0
        assert first.attributes["source_format"] == "tao"
        assert first.attributes["split"] == "train"
        assert first.attributes["tao_image_id"] == 100
        assert first.attributes["tao_annotation_id"] == 900
        assert first.attributes["area"] == 1200
        assert first.attributes["iscrowd"] == 0
        assert first.attributes["segmentation"] == [[10, 20, 40, 20, 40, 60, 10, 60]]

        empty = ds.sequences[1]
        assert empty.video_meta["sequence_name"] == "empty-video"
        assert len(empty.boxes) == 0

    def test_loads_all_standard_splits_found(self, tmp_path: Path) -> None:
        train = _basic_payload()
        validation = _basic_payload()
        validation["videos"][0]["id"] = 30
        validation["videos"][0]["name"] = "video-validation"
        validation["images"] = [
            {
                "id": 300,
                "video_id": 30,
                "file_name": "validation/video/000001.jpg",
                "frame_index": 0,
            }
        ]
        validation["tracks"] = []
        validation["annotations"] = []
        test = {
            "videos": [{"id": 40, "name": "test/video"}],
            "images": [{"id": 400, "video_id": 40, "file_name": "test/video/000001.jpg", "frame_index": 0}],
            "tracks": [],
            "categories": [],
            "annotations": [],
        }
        _write_tao(tmp_path, "train", train)
        _write_tao(tmp_path, "validation", validation)
        _write_tao(tmp_path, "test_without_annotations", test)

        ds = load_tao(tmp_path)

        assert [seq.video_meta["split"] for seq in ds.sequences] == ["train", "train", "validation", "test"]
        assert ds.sequences[-1].frame_path(0) == tmp_path / "frames" / "test" / "video" / "000001.jpg"

    def test_uses_annotation_category_and_warns_on_track_mismatch(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        payload = _basic_payload()
        payload["categories"] = [{"id": 1, "name": "person"}, {"id": 2, "name": "vehicle"}]
        payload["tracks"] = [{"id": 5, "video_id": 10, "category_id": 2}]
        _write_tao(tmp_path, payload=payload)

        with caplog.at_level(logging.WARNING, logger="databridge._formats.tao.loader"):
            ds = load_tao(tmp_path)

        assert ds.sequences[0].boxes[0].category_id == 1
        assert ds.sequences[0].boxes[0].category_name == "person"
        assert "disagrees with track" in caplog.text

    def test_reserved_attribute_names_are_not_clobbered_by_raw_fields(self, tmp_path: Path) -> None:
        payload = _basic_payload()
        payload["annotations"][0].update(
            {
                "source_format": "not-tao",
                "split": "not-train",
                "tao_image_id": "not-100",
                "tao_video_id": "not-10",
                "tao_track_id": "not-5",
                "tao_frame_index": "not-0",
                "tao_annotation_id": "not-900",
            }
        )
        _write_tao(tmp_path, payload=payload)

        attrs = load_tao(tmp_path).sequences[0].boxes[0].attributes

        assert attrs["source_format"] == "tao"
        assert attrs["split"] == "train"
        assert attrs["tao_image_id"] == 100
        assert attrs["tao_video_id"] == 10
        assert attrs["tao_track_id"] == 5
        assert attrs["tao_frame_index"] == 0
        assert attrs["tao_annotation_id"] == 900

    def test_derives_frame_order_when_frame_index_missing(self, tmp_path: Path) -> None:
        payload = {
            "videos": [{"id": 1, "name": "v"}],
            "images": [
                {"id": 2, "video_id": 1, "file_name": "frames/v/b.jpg"},
                {"id": 1, "video_id": 1, "file_name": "frames/v/a.jpg"},
            ],
            "tracks": [{"id": 9, "video_id": 1, "category_id": 7}],
            "categories": [{"id": 7, "name": "thing"}],
            "annotations": [
                {"id": 1, "image_id": 1, "track_id": 9, "category_id": 7, "bbox": [1, 2, 3, 4]},
                {"id": 2, "image_id": 2, "track_id": 9, "category_id": 7, "bbox": [5, 6, 7, 8]},
            ],
        }
        _write_tao(tmp_path, payload=payload)

        ds = load_tao(tmp_path)

        seq = ds.sequences[0]
        assert seq.frame_files == (str(tmp_path / "frames" / "v" / "a.jpg"), str(tmp_path / "frames" / "v" / "b.jpg"))
        assert [box.frame_index for box in seq.boxes] == [0, 1]


class TestTaoMalformedInputs:
    def test_missing_annotations_returns_empty(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="databridge._formats.tao.loader"):
            ds = load_tao(tmp_path)

        assert ds.sequence_count == 0
        assert "no standard annotation files" in caplog.text

    def test_bad_json_returns_empty(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        annotations = tmp_path / "annotations"
        annotations.mkdir()
        (annotations / "train.json").write_text("{not valid json", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger="databridge._formats.tao.loader"):
            ds = load_tao(tmp_path)

        assert ds.sequence_count == 0
        assert "Could not read TAO annotation file" in caplog.text

    def test_unsafe_image_path_is_skipped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        payload = _basic_payload()
        payload["images"][0]["file_name"] = "../escape.jpg"
        _write_tao(tmp_path, payload=payload)

        with caplog.at_level(logging.WARNING, logger="databridge._formats.tao.loader"):
            ds = load_tao(tmp_path)

        assert ds.num_boxes == 1
        assert "unsafe file_name" in caplog.text

    def test_symlink_escape_image_path_is_skipped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        outside = tmp_path.parent / f"outside-{tmp_path.name}"
        outside.mkdir()
        link = tmp_path / "frames" / "link"
        link.parent.mkdir()
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
        payload = _basic_payload()
        payload["images"][0]["file_name"] = "frames/link/escape.jpg"
        _write_tao(tmp_path, payload=payload)

        with caplog.at_level(logging.WARNING, logger="databridge._formats.tao.loader"):
            ds = load_tao(tmp_path)

        assert ds.num_boxes == 1
        assert "resolved path escapes" in caplog.text

    def test_malformed_annotations_are_skipped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        payload = _basic_payload()
        payload["annotations"] = [
            "bad",
            {"id": 1, "image_id": "bad", "track_id": 5, "category_id": 1, "bbox": [1, 2, 3, 4]},
            {"id": 10, "image_id": True, "track_id": 5, "category_id": 1, "bbox": [1, 2, 3, 4]},
            {"id": 2, "image_id": 100, "track_id": "bad", "category_id": 1, "bbox": [1, 2, 3, 4]},
            {"id": 3, "image_id": 100, "track_id": 5, "category_id": 1, "bbox": [1, 2, -3, 4]},
            {"id": 4, "image_id": 100, "track_id": 5, "category_id": 99, "bbox": [1, 2, 3, 4]},
        ]
        _write_tao(tmp_path, payload=payload)

        with caplog.at_level(logging.WARNING, logger="databridge._formats.tao.loader"):
            ds = load_tao(tmp_path)

        assert ds.num_boxes == 1
        assert ds.sequences[0].boxes[0].category_id == 99
        assert ds.sequences[0].boxes[0].category_name == "category_99"
        assert "Skipping malformed TAO annotation" in caplog.text
        assert "invalid image_id" in caplog.text
        assert "invalid track_id" in caplog.text
        assert "invalid bbox" in caplog.text

    def test_frame_path_errors_for_missing_explicit_frame(self, tmp_path: Path) -> None:
        payload = _basic_payload()
        payload["images"][1]["frame_index"] = 3
        _write_tao(tmp_path, payload=payload)

        seq = load_tao(tmp_path).sequences[0]

        assert seq.num_frames == 4
        assert seq.num_frames_exact is False
        assert seq.duration is None
        assert seq.frame_path(0) == tmp_path / "frames" / "train" / "video-a" / "000001.jpg"
        with pytest.raises(ValueError, match="no frame file"):
            seq.frame_path(1)

    def test_huge_declared_frame_index_falls_back_to_dense_order(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        payload = {
            "videos": [{"id": 1, "name": "train/v"}],
            "images": [
                {
                    "id": 1,
                    "video_id": 1,
                    "file_name": "train/v/frame1000001.jpg",
                    "frame_index": 1_000_001,
                }
            ],
            "tracks": [{"id": 1, "video_id": 1, "category_id": 7}],
            "categories": [{"id": 7, "name": "thing"}],
            "annotations": [{"id": 1, "image_id": 1, "track_id": 1, "category_id": 7, "bbox": [1, 2, 3, 4]}],
        }
        _write_tao(tmp_path, payload=payload)

        with caplog.at_level(logging.WARNING, logger="databridge._formats.tao.loader"):
            seq = load_tao(tmp_path).sequences[0]

        assert seq.num_frames == 1
        assert seq.frame_path(0) == tmp_path / "frames" / "train" / "v" / "frame1000001.jpg"
        assert seq.boxes[0].frame_index == 0
        assert seq.boxes[0].attributes["tao_frame_index"] == 1_000_001
        assert "deriving dense frame order" in caplog.text


class TestTaoImageProbe:
    def test_probe_images_uses_opencv_when_available(self, tmp_path: Path) -> None:
        cv2 = pytest.importorskip("cv2")
        np = pytest.importorskip("numpy")
        payload = _basic_payload()
        payload["videos"][0]["width"] = 1
        payload["videos"][0]["height"] = 1
        _write_tao(tmp_path, payload=payload)
        image_path = tmp_path / "frames" / "train" / "video-a" / "000001.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.zeros((12, 34, 3), dtype=np.uint8)
        assert cv2.imwrite(str(image_path), image)

        ds = load_tao(tmp_path, probe_images=True)

        assert ds.sequences[0].width == 34
        assert ds.sequences[0].height == 12
