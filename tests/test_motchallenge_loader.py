"""Tests for the MOTChallenge loader."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from datamaite import DatasetFormat, load
from datamaite._formats.motchallenge.loader import MotChallengeLoader, load_motchallenge
from datamaite.loaders import available_formats, get_loader
from datamaite.model import BoxTrackDataset


def _write_mot_sequence(
    root: Path,
    *,
    split: str = "train",
    name: str = "MOT17-02",
    gt_rows: list[str] | None = None,
    det_rows: list[str] | None = None,
    write_seqinfo: bool = True,
    frame_count: int = 3,
    width: int = 640,
    height: int = 480,
    frame_ext: str = ".jpg",
    im_dir: str = "img1",
) -> Path:
    seq = root / split / name
    frame_dir = seq / "img1"
    frame_dir.mkdir(parents=True)
    for frame in range(1, frame_count + 1):
        (frame_dir / f"{frame:06d}{frame_ext}").write_bytes(b"not a real image")

    if write_seqinfo:
        (seq / "seqinfo.ini").write_text(
            "\n".join(
                [
                    "[Sequence]",
                    f"name={name}",
                    f"imDir={im_dir}",
                    "frameRate=30",
                    f"seqLength={frame_count}",
                    f"imWidth={width}",
                    f"imHeight={height}",
                    f"imExt={frame_ext}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    if gt_rows is not None:
        gt_dir = seq / "gt"
        gt_dir.mkdir()
        (gt_dir / "gt.txt").write_text("\n".join(gt_rows) + "\n", encoding="utf-8")
    if det_rows is not None:
        det_dir = seq / "det"
        det_dir.mkdir()
        (det_dir / "det.txt").write_text("\n".join(det_rows) + "\n", encoding="utf-8")
    return seq


class TestMotChallengeRegistry:
    def test_registered_and_public_api(self) -> None:
        assert DatasetFormat.MOTCHALLENGE in available_formats()
        assert isinstance(get_loader(DatasetFormat.MOTCHALLENGE), MotChallengeLoader)
        assert isinstance(get_loader("motchallenge"), MotChallengeLoader)

    def test_dispatch_loads_motchallenge(self, tmp_path: Path) -> None:
        _write_mot_sequence(tmp_path, gt_rows=["1,1,10,20,30,40,1,42,0.9"])
        ds = load(tmp_path, dataset_format="motchallenge", class_names={42: "vehicle"})
        assert isinstance(ds, BoxTrackDataset)
        assert ds.sequence_count == 1
        assert ds.num_boxes == 1
        assert ds.sequences[0].boxes[0].category_name == "vehicle"


class TestMotChallengeHappyPath:
    def test_loads_standard_gt_sequence(self, tmp_path: Path) -> None:
        _write_mot_sequence(
            tmp_path,
            gt_rows=[
                "1,1,10,20,30,40,1,1,0.9",
                "2,1,12,22,30,40,1,1,0.8",
                "2,2,50,60,10,15,0,1,0.5",  # ignored by default
                "3,3,5,6,7,8,1,7,1.0",
            ],
        )

        ds = load_motchallenge(tmp_path)

        assert ds.sequence_count == 1
        assert len(ds) == 0  # MOTChallenge is image-sequence based, not video-backed.
        assert ds.num_boxes == 3
        assert ds.categories == {"motchallenge/pedestrian": 1, "motchallenge/static_person": 7}

        seq = ds.sequences[0]
        assert seq.video_path is None
        assert seq.frame_dir == str(tmp_path / "train" / "MOT17-02" / "img1")
        assert seq.frame_pattern == "{frame:06d}.jpg"
        assert seq.frame_number_base == 1
        assert seq.frame_filename(0) == "000001.jpg"
        assert seq.frame_path(0) == tmp_path / "train" / "MOT17-02" / "img1" / "000001.jpg"
        assert seq.fps == 30.0
        assert seq.num_frames == 3
        assert seq.num_frames_exact is True
        assert seq.duration == 0.1
        assert seq.width == 640
        assert seq.height == 480
        assert seq.video_meta["split"] == "train"
        assert seq.video_meta["annotation_source"] == "gt"

        first = seq.boxes[0]
        assert first.frame_index == 0  # MOT's 1-based frame number becomes model 0-based frame_index.
        assert first.track_uuid == "train:MOT17-02:gt:1"
        assert first.track_id == 1
        assert first.category_id == 1
        assert first.category_name == "pedestrian"
        assert first.bbox == (10.0, 20.0, 30.0, 40.0)
        assert first.attributes["mot_frame"] == 1
        assert first.attributes["confidence"] == 1.0
        assert first.attributes["visibility"] == 0.9

    def test_include_ignored_and_class_filter(self, tmp_path: Path) -> None:
        _write_mot_sequence(
            tmp_path,
            gt_rows=[
                "1,1,10,20,30,40,1,1,0.9",
                "2,2,50,60,10,15,0,1,0.5",
                "3,3,5,6,7,8,1,7,1.0",
            ],
        )

        ds = load_motchallenge(tmp_path, include_ignored=True, classes={1})

        assert ds.num_boxes == 2
        assert [box.attributes["mot_track_id"] for box in ds.sequences[0].boxes] == [1, 2]
        assert all(box.category_name == "pedestrian" for box in ds.sequences[0].boxes)

    def test_custom_class_names_for_mot_style_labels(self, tmp_path: Path) -> None:
        _write_mot_sequence(tmp_path, gt_rows=["1,1,10,20,30,40,1,42,0.9"])

        ds = load_motchallenge(tmp_path, class_names={42: "vehicle"})

        assert ds.categories == {"motchallenge/class_42/vehicle": 42}
        assert ds.index2label() == {42: "vehicle"}
        box = ds.sequences[0].boxes[0]
        assert box.category_id == 42
        assert box.category_uri == "motchallenge/class_42/vehicle"
        assert box.category_name == "vehicle"

    def test_empty_class_names_uses_builtin_and_unknown_fallbacks(self, tmp_path: Path) -> None:
        _write_mot_sequence(
            tmp_path,
            gt_rows=[
                "1,1,10,20,30,40,1,1,0.9",
                "2,2,50,60,10,15,1,42,0.5",
            ],
        )

        ds = load_motchallenge(tmp_path, class_names={})

        assert ds.categories == {"motchallenge/pedestrian": 1, "motchallenge/class_42": 42}
        assert [box.category_name for box in ds.sequences[0].boxes] == ["pedestrian", "class_42"]

    def test_loads_detection_source(self, tmp_path: Path) -> None:
        _write_mot_sequence(
            tmp_path,
            gt_rows=None,
            det_rows=[
                "1,-1,10,20,30,40,0.42,-1,-1,-1",
                "2,7,50,60,10,15,0.75,1,2,3",
            ],
        )

        ds = load_motchallenge(tmp_path, annotation_source="det")

        assert ds.sequence_count == 1
        assert ds.categories == {}
        boxes = ds.sequences[0].boxes
        assert [box.frame_index for box in boxes] == [0, 1]
        assert boxes[0].track_id == -1  # det id -1 receives a negative per-row pseudo id.
        assert boxes[0].category_id == -1
        assert boxes[0].attributes["score"] == 0.42
        assert "world_x" not in boxes[0].attributes
        assert boxes[1].track_id == 7
        assert boxes[1].attributes["world_x"] == 1.0
        assert boxes[1].attributes["world_y"] == 2.0
        assert boxes[1].attributes["world_z"] == 3.0


class TestMotChallengeMalformedInputs:
    def test_missing_selected_annotation_skips_sequence(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        _write_mot_sequence(tmp_path, split="train", name="has-gt", gt_rows=["1,1,10,20,30,40,1,1,1"])
        _write_mot_sequence(tmp_path, split="test", name="no-gt", gt_rows=None)

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.motchallenge.loader"):
            ds = load_motchallenge(tmp_path)

        assert ds.sequence_count == 1
        assert "missing gt annotation file" in caplog.text

    def test_requires_full_benchmark_root(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        seq = _write_mot_sequence(tmp_path, gt_rows=["1,1,10,20,30,40,1,1,1"])

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.motchallenge.loader"):
            ds = load_motchallenge(seq)

        assert ds.sequence_count == 0
        assert "must contain train/ and/or test" in caplog.text

    def test_malformed_rows_are_skipped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        _write_mot_sequence(
            tmp_path,
            gt_rows=[
                "# comment",
                "1,1,10",  # too few columns
                "0,1,10,20,30,40,1,1,1",  # invalid frame
                "1,x,10,20,30,40,1,1,1",  # invalid id
                "1,-1,10,20,30,40,1,1,1",  # invalid GT track id
                "1,1,10,20,-1,40,1,1,1",  # invalid bbox
                "1,1,10,20,30,40,nope,1,1",  # invalid confidence
                "1,1,10,20,30,40,1,not-a-class,1",  # invalid class
                "1,1,10,20,30,40,1,1,not-visible",  # invalid visibility
                "4,4,1,2,3,4,1,1,0.75",
            ],
        )

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.motchallenge.loader"):
            ds = load_motchallenge(tmp_path)

        assert ds.num_boxes == 1
        assert ds.sequences[0].boxes[0].frame_index == 3
        assert "Skipping malformed MOTChallenge row" in caplog.text

    def test_missing_seqinfo_uses_defaults_and_box_frame_count(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write_mot_sequence(
            tmp_path,
            gt_rows=["2,1,10,20,30,40,1,1,1"],
            write_seqinfo=False,
            frame_count=0,
        )

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.motchallenge.loader"):
            ds = load_motchallenge(tmp_path)

        seq = ds.sequences[0]
        assert seq.fps == 0.0
        assert seq.num_frames == 2
        assert seq.num_frames_exact is False
        assert seq.width is None
        assert seq.height is None
        assert "missing seqinfo.ini" in caplog.text

    def test_classes_filter_warns_and_skips_classless_gt_rows(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write_mot_sequence(tmp_path, gt_rows=["1,1,10,20,30,40,1"])

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.motchallenge.loader"):
            ds = load_motchallenge(tmp_path, classes={1})

        assert ds.num_boxes == 0
        assert "classes filter requires a MOT class column" in caplog.text

    def test_unsafe_seqinfo_paths_fall_back_to_safe_values(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        seq = _write_mot_sequence(tmp_path, gt_rows=["1,1,10,20,30,40,1,1,1"])
        (seq / "seqinfo.ini").write_text(
            "\n".join(
                [
                    "[Sequence]",
                    "name=MOT17-02",
                    "imDir=../../outside",
                    "frameRate=30",
                    "seqLength=3",
                    "imWidth=640",
                    "imHeight=480",
                    "imExt=../bad",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.motchallenge.loader"):
            ds = load_motchallenge(tmp_path)

        loaded = ds.sequences[0]
        assert loaded.frame_dir == str(seq / "img1")
        assert loaded.frame_pattern == "{frame:06d}.jpg"
        assert "Unsafe MOTChallenge imDir" in caplog.text
        assert "Unsafe MOTChallenge imExt" in caplog.text

    def test_does_not_scan_frames_when_seqinfo_has_length_and_probe_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_mot_sequence(tmp_path, gt_rows=["1,1,10,20,30,40,1,1,1"])

        def fail_scan(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("frame scan should not run")

        monkeypatch.setattr("datamaite._formats.motchallenge.loader._frame_paths", fail_scan)
        monkeypatch.setattr("datamaite._formats.motchallenge.loader._count_frame_files", fail_scan)

        ds = load_motchallenge(tmp_path)

        assert ds.sequence_count == 1
        assert ds.sequences[0].num_frames == 3

    def test_invalid_options_raise(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="annotation_source"):
            load_motchallenge(tmp_path, annotation_source="tracks")
        with pytest.raises(ValueError, match="classes"):
            load_motchallenge(tmp_path, classes={"pedestrian"})  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="class_names"):
            load_motchallenge(tmp_path, class_names={"pedestrian": "person"})  # type: ignore[dict-item]
        with pytest.raises(ValueError, match="class_names"):
            load_motchallenge(tmp_path, class_names={42: ""})
        with pytest.raises(ValueError, match="class_names"):
            load_motchallenge(tmp_path, class_names=[])  # type: ignore[arg-type]

    def test_det_ignores_classes_filter(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        _write_mot_sequence(tmp_path, gt_rows=None, det_rows=["1,-1,10,20,30,40,0.42,-1,-1,-1"])

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.motchallenge.loader"):
            ds = load_motchallenge(tmp_path, annotation_source="det", classes={1})

        assert ds.num_boxes == 1
        assert "Ignoring classes filter" in caplog.text


class TestMotChallengeImageProbe:
    def test_probe_images_uses_opencv_when_available(self, tmp_path: Path) -> None:
        cv2 = pytest.importorskip("cv2")
        np = pytest.importorskip("numpy")
        seq = _write_mot_sequence(tmp_path, gt_rows=["1,1,10,20,30,40,1,1,1"], width=1, height=1)
        image = np.zeros((12, 34, 3), dtype=np.uint8)
        assert cv2.imwrite(str(seq / "img1" / "000001.jpg"), image)

        ds = load_motchallenge(tmp_path, probe_images=True)

        assert ds.sequences[0].width == 34
        assert ds.sequences[0].height == 12
