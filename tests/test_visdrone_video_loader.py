"""Tests for the VisDrone video loader."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from databridge import DatasetFormat, load
from databridge._formats.visdrone.loader import VisDroneVideoLoader, load_visdrone_video
from databridge.loaders import available_formats, get_loader
from databridge.model import BoxTrackDataset


def _write_visdrone_split(
    root: Path,
    *,
    split_name: str = "VisDrone2019-VID-train",
    sequence_name: str = "uav0000013_00000_v",
    rows: list[str] | None = None,
    frame_count: int = 3,
    frame_ext: str = ".jpg",
    make_sequence_dir: bool = True,
) -> Path:
    split = root / split_name
    annotation_dir = split / "annotations"
    annotation_dir.mkdir(parents=True)
    if make_sequence_dir:
        frame_dir = split / "sequences" / sequence_name
        frame_dir.mkdir(parents=True)
        for frame in range(1, frame_count + 1):
            (frame_dir / f"{frame:07d}{frame_ext}").write_bytes(b"not a real image")
    else:
        (split / "sequences").mkdir(parents=True)

    if rows is not None:
        (annotation_dir / f"{sequence_name}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return split


class TestVisDroneVideoRegistry:
    def test_registered_and_public_api(self) -> None:
        assert DatasetFormat.VISDRONE_VIDEO in available_formats()
        assert isinstance(get_loader(DatasetFormat.VISDRONE_VIDEO), VisDroneVideoLoader)
        assert isinstance(get_loader("visdrone_video"), VisDroneVideoLoader)

    def test_dispatch_loads_visdrone_video(self, tmp_path: Path) -> None:
        _write_visdrone_split(tmp_path, rows=["1,1,10,20,30,40,1,4,0,0"])

        ds = load(tmp_path, dataset_format="visdrone_video")

        assert isinstance(ds, BoxTrackDataset)
        assert ds.sequence_count == 1
        assert ds.num_boxes == 1
        assert ds.sequences[0].boxes[0].category_name == "car"


class TestVisDroneVideoHappyPath:
    def test_loads_vid_split_root(self, tmp_path: Path) -> None:
        split = _write_visdrone_split(
            tmp_path,
            rows=[
                "1,1,10,20,30,40,1,4,0,0",
                "2,1,12,22,30,40,1,4,1,2",
                "2,2,50,60,10,15,0,1,0,0",  # ignored by score by default
                "3,-1,5,6,7,8,1,0,0,0",  # ignored-region category by default
                "3,4,1,2,3,4,1,11,0,0",
            ],
        )

        ds = load_visdrone_video(split)

        assert ds.sequence_count == 1
        assert len(ds) == 0  # VisDrone video is image-sequence based, not video-backed.
        assert ds.num_boxes == 3
        assert ds.categories == {"visdrone_video/car": 4, "visdrone_video/others": 11}
        assert ds.index2label() == {4: "car", 11: "others"}

        seq = ds.sequences[0]
        assert seq.video_path is None
        assert seq.frame_dir == str(split / "sequences" / "uav0000013_00000_v")
        assert seq.frame_pattern == "{frame:07d}.jpg"
        assert seq.frame_number_base == 1
        assert seq.frame_filename(0) == "0000001.jpg"
        assert seq.frame_path(0) == split / "sequences" / "uav0000013_00000_v" / "0000001.jpg"
        assert seq.fps == 0.0
        assert seq.num_frames == 3
        assert seq.num_frames_exact is True
        assert seq.duration is None
        assert seq.width is None
        assert seq.height is None
        assert seq.video_meta["format"] == "visdrone_video"
        assert seq.video_meta["variant"] == "vid"
        assert seq.video_meta["split"] == "train"
        assert seq.video_meta["annotation_source"] == "gt"

        first = seq.boxes[0]
        assert first.frame_index == 0  # VisDrone's 1-based frame number becomes model 0-based frame_index.
        assert first.track_uuid == "train:uav0000013_00000_v:gt:1"
        assert first.track_id == 1
        assert first.category_id == 4
        assert first.category_name == "car"
        assert first.bbox == (10.0, 20.0, 30.0, 40.0)
        assert first.attributes["visdrone_frame"] == 1
        assert first.attributes["visdrone_target_id"] == 1
        assert first.attributes["visdrone_category_id"] == 4
        assert first.attributes["confidence"] == 1.0
        assert first.attributes["truncation"] == 0
        assert first.attributes["occlusion"] == 0

    def test_loads_mot_parent_with_multiple_splits(self, tmp_path: Path) -> None:
        _write_visdrone_split(
            tmp_path,
            split_name="VisDrone2019-MOT-val",
            sequence_name="uav-val",
            rows=["1,1,10,20,30,40,1,1,0,0"],
        )
        _write_visdrone_split(
            tmp_path,
            split_name="VisDrone2019-MOT-test-dev",
            sequence_name="uav-test",
            rows=["1,1,10,20,30,40,1,9,0,0"],
        )

        ds = load_visdrone_video(tmp_path)

        assert ds.sequence_count == 2
        assert [seq.video_meta["variant"] for seq in ds.sequences] == ["mot", "mot"]
        assert [seq.video_meta["split"] for seq in ds.sequences] == ["val", "test-dev"]
        assert ds.index2label() == {1: "pedestrian", 9: "bus"}

    def test_explicit_variant_and_fps(self, tmp_path: Path) -> None:
        split = _write_visdrone_split(tmp_path, split_name="custom-split", rows=["2,1,10,20,30,40,1,4,0,0"])

        ds = load_visdrone_video(split, variant="mot", fps=20)

        seq = ds.sequences[0]
        assert seq.video_meta["variant"] == "mot"
        assert seq.video_meta["split"] == "custom-split"
        assert seq.fps == 20.0
        assert seq.num_frames == 3
        assert seq.duration == 0.15

    def test_include_ignored_and_class_filter(self, tmp_path: Path) -> None:
        _write_visdrone_split(
            tmp_path,
            rows=[
                "1,-1,10,20,30,40,0,0,0,0",
                "1,1,10,20,30,40,1,1,0,0",
            ],
        )

        ds = load_visdrone_video(tmp_path, include_ignored=True, classes={0})

        assert ds.num_boxes == 1
        box = ds.sequences[0].boxes[0]
        assert box.track_id == -1
        assert box.category_id == 0
        assert box.category_name == "ignored_region"
        assert ds.categories == {"visdrone_video/ignored_region": 0}

    def test_loads_detection_source(self, tmp_path: Path) -> None:
        _write_visdrone_split(
            tmp_path,
            rows=[
                "1,-1,10,20,30,40,0.42,4,-1,-1",
                "2,7,50,60,10,15,0.75,5,-1,-1",
            ],
        )

        ds = load_visdrone_video(tmp_path, annotation_source="det")

        assert ds.sequence_count == 1
        boxes = ds.sequences[0].boxes
        assert [box.frame_index for box in boxes] == [0, 1]
        assert boxes[0].track_id == -1
        assert boxes[0].category_id == 4
        assert boxes[0].attributes["score"] == 0.42
        assert "truncation" not in boxes[0].attributes
        assert "occlusion" not in boxes[0].attributes
        assert boxes[1].track_id == 7
        assert boxes[1].category_name == "van"


class TestVisDroneVideoMalformedInputs:
    def test_missing_split_root_returns_empty(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="databridge._formats.visdrone.loader"):
            ds = load_visdrone_video(tmp_path)

        assert ds.sequence_count == 0
        assert "must contain sequences/ and annotations/" in caplog.text

    def test_no_annotation_files_returns_empty(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        _write_visdrone_split(tmp_path, rows=None)

        with caplog.at_level(logging.WARNING, logger="databridge._formats.visdrone.loader"):
            ds = load_visdrone_video(tmp_path)

        assert ds.sequence_count == 0
        assert "no .txt annotation files" in caplog.text

    def test_missing_sequence_dir_uses_annotation_frame_count(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write_visdrone_split(
            tmp_path,
            rows=["2,1,10,20,30,40,1,4,0,0"],
            make_sequence_dir=False,
        )

        with caplog.at_level(logging.WARNING, logger="databridge._formats.visdrone.loader"):
            ds = load_visdrone_video(tmp_path)

        seq = ds.sequences[0]
        assert seq.num_frames == 2
        assert seq.num_frames_exact is False
        assert "frame directory is missing" in caplog.text

    def test_malformed_rows_are_skipped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        _write_visdrone_split(
            tmp_path,
            rows=[
                "# comment",
                "1,1,10",  # too few columns
                "0,1,10,20,30,40,1,4,0,0",  # invalid frame
                "1,x,10,20,30,40,1,4,0,0",  # invalid id
                "1,1,10,20,-1,40,1,4,0,0",  # invalid bbox
                "1,1,10,20,30,40,nope,4,0,0",  # invalid score
                "1,1,10,20,30,40,1,car,0,0",  # invalid category
                "1,1,10,20,30,40,1,4,bad,0",  # invalid truncation
                "1,1,10,20,30,40,1,4,0,bad",  # invalid occlusion
                "1,-1,10,20,30,40,1,4,0,0",  # invalid GT id for a non-ignored object
                "4,4,1,2,3,4,1,4,0,0",
            ],
        )

        with caplog.at_level(logging.WARNING, logger="databridge._formats.visdrone.loader"):
            ds = load_visdrone_video(tmp_path)

        assert ds.num_boxes == 1
        assert ds.sequences[0].boxes[0].frame_index == 3
        assert "Skipping malformed VisDrone row" in caplog.text

    def test_does_not_sort_frames_when_probe_disabled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_visdrone_split(tmp_path, rows=["1,1,10,20,30,40,1,4,0,0"])

        def fail_sorted_scan(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("sorted frame scan should not run")

        monkeypatch.setattr("databridge._formats.visdrone.loader._frame_paths", fail_sorted_scan)

        ds = load_visdrone_video(tmp_path)

        assert ds.sequence_count == 1
        assert ds.sequences[0].num_frames == 3

    def test_invalid_options_raise(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="variant"):
            load_visdrone_video(tmp_path, variant="sot")
        with pytest.raises(ValueError, match="annotation_source"):
            load_visdrone_video(tmp_path, annotation_source="tracks")
        with pytest.raises(ValueError, match="classes"):
            load_visdrone_video(tmp_path, classes={"car"})  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="frame_ext"):
            load_visdrone_video(tmp_path, frame_ext="../jpg")
        with pytest.raises(ValueError, match="fps"):
            load_visdrone_video(tmp_path, fps=-1)


class TestVisDroneVideoImageProbe:
    def test_probe_images_uses_opencv_when_available(self, tmp_path: Path) -> None:
        cv2 = pytest.importorskip("cv2")
        np = pytest.importorskip("numpy")
        split = _write_visdrone_split(tmp_path, rows=["1,1,10,20,30,40,1,4,0,0"])
        image = np.zeros((12, 34, 3), dtype=np.uint8)
        assert cv2.imwrite(str(split / "sequences" / "uav0000013_00000_v" / "0000001.jpg"), image)

        ds = load_visdrone_video(tmp_path, probe_images=True)

        assert ds.sequences[0].width == 34
        assert ds.sequences[0].height == 12
