"""Tests for the TAO writer and TAO load -> write -> load round trip."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from datamaite import DatasetFormat, TaoWriter, convert, write
from datamaite._formats.tao.loader import load_tao
from datamaite.model import BoxAnnotation, BoxTrackDataset, VideoSequence
from datamaite.writers import available_output_formats, get_writer


def write_tao(root: Path, payload: dict[str, Any], split: str = "train") -> Path:
    annotations = root / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    path = annotations / f"{split}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_tao_frames(root: Path, *names: str) -> None:
    for name in names:
        path = root / "frames" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"frame:{name}".encode())


def tao_payload() -> dict[str, Any]:
    return {
        "videos": [{"id": 10, "name": "video-a", "width": 640, "height": 480, "fps": 30}],
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
        ],
        "tracks": [{"id": 5, "video_id": 10, "category_id": 7}],
        "categories": [{"id": 7, "name": "vehicle"}],
        "annotations": [
            {
                "id": 900,
                "image_id": 100,
                "track_id": 5,
                "category_id": 7,
                "bbox": [10, 20, 30, 40],
                "area": 1200,
                "iscrowd": 0,
                "segmentation": [[10, 20, 40, 20, 40, 60, 10, 60]],
            },
            {"id": 901, "image_id": 101, "track_id": 5, "category_id": 7, "bbox": [12, 22, 30, 40]},
        ],
    }


def _tao_fingerprint(ds: BoxTrackDataset) -> list[tuple[object, ...]]:
    seqs: list[tuple[object, ...]] = []
    for seq in ds.sequences:
        boxes = tuple(
            sorted(
                (
                    box.track_uuid,
                    box.track_id,
                    box.category_id,
                    box.category_uri,
                    box.frame_index,
                    tuple(round(value, 3) for value in box.bbox),
                    tuple(sorted((key, repr(value)) for key, value in box.attributes.items())),
                )
                for box in seq.boxes
            )
        )
        seqs.append(
            (
                seq.video_meta["split"],
                seq.video_meta["source_video_id"],
                seq.video_meta["sequence_name"],
                seq.width,
                seq.height,
                seq.fps,
                seq.num_frames,
                boxes,
            )
        )
    return sorted(seqs)


class TestTaoWriterRegistry:
    def test_registered_and_public_api(self) -> None:
        assert DatasetFormat.TAO in available_output_formats()
        assert isinstance(get_writer(DatasetFormat.TAO), TaoWriter)
        assert isinstance(get_writer("tao"), TaoWriter)


class TestTaoWriterHappyPath:
    def test_write_produces_reloadable_full_tao_dataset(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        write_tao(src, tao_payload())
        write_tao_frames(src, "train/video-a/000001.jpg", "train/video-a/000002.jpg")
        ds = load_tao(src)

        out = tmp_path / "out"
        files = write(ds, out, output_format="tao", verbose=True)

        assert out / "annotations" / "train.json" in files
        assert out / "frames" / "train" / "video-a" / "000001.jpg" in files
        assert (out / "frames" / "train" / "video-a" / "000001.jpg").read_bytes() == b"frame:train/video-a/000001.jpg"
        written = json.loads((out / "annotations" / "train.json").read_text(encoding="utf-8"))
        assert written["categories"] == [{"id": 7, "name": "vehicle"}]
        assert written["videos"][0]["id"] == 10
        assert [image["id"] for image in written["images"]] == [100, 101]
        assert [annotation["id"] for annotation in written["annotations"]] == [900, 901]
        assert written["annotations"][0]["segmentation"] == [[10, 20, 40, 20, 40, 60, 10, 60]]

        assert _tao_fingerprint(load_tao(out)) == _tao_fingerprint(ds)

    def test_convert_tao_to_tao_end_to_end(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        write_tao(src, tao_payload())
        write_tao_frames(src, "train/video-a/000001.jpg", "train/video-a/000002.jpg")

        out = tmp_path / "out"
        files = convert(src, out, input_format="tao", output_format="tao", verbose=True)

        assert files
        assert _tao_fingerprint(load_tao(out)) == _tao_fingerprint(load_tao(src))

    def test_defaults_unknown_split_to_train_and_option_can_override(self, tmp_path: Path) -> None:
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"frame")
        seq = VideoSequence(
            video_id=0,
            video_path=None,
            fps=0.0,
            num_frames=1,
            duration=None,
            annotation_path="unused",
            frame_files=(str(frame),),
            boxes=[],
            num_frames_exact=True,
        )
        ds = BoxTrackDataset(sequences=(seq,), categories={})

        write(ds, tmp_path / "out", output_format="tao", split="validation")

        assert (tmp_path / "out" / "annotations" / "validation.json").is_file()
        assert not (tmp_path / "out" / "annotations" / "train.json").exists()

    def test_generated_frame_paths_include_video_id_to_avoid_sanitized_name_collisions(self, tmp_path: Path) -> None:
        first_frame = tmp_path / "first.jpg"
        second_frame = tmp_path / "second.jpg"
        first_frame.write_bytes(b"PIXELS-OF-SEQ-1")
        second_frame.write_bytes(b"PIXELS-OF-SEQ-2")
        seq1 = VideoSequence(
            video_id=0,
            video_path=None,
            fps=0.0,
            num_frames=1,
            duration=None,
            annotation_path="unused-1",
            frame_files=(str(first_frame),),
            video_meta={"sequence_name": "my clip"},
            boxes=[],
            num_frames_exact=True,
        )
        seq2 = VideoSequence(
            video_id=1,
            video_path=None,
            fps=0.0,
            num_frames=1,
            duration=None,
            annotation_path="unused-2",
            frame_files=(str(second_frame),),
            video_meta={"sequence_name": "my+clip"},
            boxes=[],
            num_frames_exact=True,
        )
        ds = BoxTrackDataset(sequences=(seq1, seq2), categories={})

        write(ds, tmp_path / "out", output_format="tao")
        written = json.loads((tmp_path / "out" / "annotations" / "train.json").read_text(encoding="utf-8"))

        assert (tmp_path / "out" / "frames" / "train" / "0__my_clip" / "000000.jpg").read_bytes() == b"PIXELS-OF-SEQ-1"
        assert (tmp_path / "out" / "frames" / "train" / "1__my_clip" / "000000.jpg").read_bytes() == b"PIXELS-OF-SEQ-2"
        assert [image["file_name"] for image in written["images"]] == [
            "train/0__my_clip/000000.jpg",
            "train/1__my_clip/000000.jpg",
        ]

    def test_unlabeled_boxes_are_written_with_explicit_fallback_category(self, tmp_path: Path) -> None:
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"frame")
        box = BoxAnnotation(
            track_uuid="track-a",
            track_id=1,
            category_id=-1,
            category_uri="",
            category_name=None,
            bbox=(1.0, 2.0, 3.0, 4.0),
            attributes={},
            frame_index=0,
            timestamp=None,
        )
        seq = VideoSequence(
            video_id=0,
            video_path=None,
            fps=0.0,
            num_frames=1,
            duration=None,
            annotation_path="unused",
            frame_files=(str(frame),),
            boxes=[box],
            num_frames_exact=True,
        )
        ds = BoxTrackDataset(sequences=(seq,), categories={})

        write(ds, tmp_path / "out", output_format="tao")
        written = json.loads((tmp_path / "out" / "annotations" / "train.json").read_text(encoding="utf-8"))
        fallback_category = written["categories"][0]
        reloaded = load_tao(tmp_path / "out")

        assert fallback_category["name"] == "unlabeled"
        assert written["annotations"][0]["category_id"] == fallback_category["id"]
        assert written["tracks"][0]["category_id"] == fallback_category["id"]
        assert len(reloaded.sequences[0].boxes) == 1
        assert reloaded.sequences[0].boxes[0].category_name == "unlabeled"

    def test_video_backed_sequence_extracts_frames_with_video_extra(self, tmp_path: Path) -> None:
        cv2 = pytest.importorskip("cv2")
        np = pytest.importorskip("numpy")
        video = tmp_path / "clip.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (8, 6))
        assert writer.isOpened()
        try:
            writer.write(np.full((6, 8, 3), 10, dtype=np.uint8))
            writer.write(np.full((6, 8, 3), 20, dtype=np.uint8))
        finally:
            writer.release()
        box = BoxAnnotation(
            track_uuid="track-a",
            track_id=1,
            category_id=3,
            category_uri="tao/category_3/person",
            category_name="person",
            bbox=(1.0, 2.0, 3.0, 4.0),
            attributes={},
            frame_index=1,
            timestamp=None,
        )
        seq = VideoSequence(
            video_id=0,
            video_path=str(video),
            fps=2.0,
            num_frames=2,
            duration=1.0,
            annotation_path="unused",
            video_meta={"sequence_name": "clip"},
            boxes=[box],
            width=8,
            height=6,
            num_frames_exact=True,
        )
        ds = BoxTrackDataset(sequences=(seq,), categories={"tao/category_3/person": 3})

        write(ds, tmp_path / "out", output_format="tao")
        reloaded = load_tao(tmp_path / "out")

        assert (tmp_path / "out" / "frames" / "train" / "0__clip" / "000000.jpg").is_file()
        assert (tmp_path / "out" / "frames" / "train" / "0__clip" / "000001.jpg").is_file()
        assert reloaded.sequence_count == 1
        assert reloaded.sequences[0].num_frames == 2
        assert reloaded.sequences[0].boxes[0].frame_index == 1
        assert reloaded.sequences[0].boxes[0].bbox == pytest.approx((1.0, 2.0, 3.0, 4.0))


class TestTaoWriterMalformedInputs:
    def test_missing_frame_drops_frame_and_annotation(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        src = tmp_path / "src"
        write_tao(src, tao_payload())
        write_tao_frames(src, "train/video-a/000001.jpg")
        ds = load_tao(src)

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.tao.writer"):
            write(ds, tmp_path / "out", output_format="tao")

        written = json.loads((tmp_path / "out" / "annotations" / "train.json").read_text(encoding="utf-8"))
        assert [image["frame_index"] for image in written["images"]] == [0]
        assert [annotation["id"] for annotation in written["annotations"]] == [900]
        assert "Skipping missing TAO source frame" in caplog.text
        assert "no frame image was written" in caplog.text

    def test_malformed_bbox_is_dropped_with_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"frame")
        bad_box = BoxAnnotation(
            track_uuid="track-a",
            track_id=1,
            category_id=1,
            category_uri="tao/category_1/person",
            category_name="person",
            bbox=(1.0, 2.0, -3.0, 4.0),
            attributes={},
            frame_index=0,
            timestamp=None,
        )
        seq = VideoSequence(
            video_id=0,
            video_path=None,
            fps=0.0,
            num_frames=1,
            duration=None,
            annotation_path="unused",
            frame_files=(str(frame),),
            boxes=[bad_box],
            num_frames_exact=True,
        )
        ds = BoxTrackDataset(sequences=(seq,), categories={"tao/category_1/person": 1})

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.tao.writer"):
            write(ds, tmp_path / "out", output_format="tao")

        written = json.loads((tmp_path / "out" / "annotations" / "train.json").read_text(encoding="utf-8"))
        assert written["annotations"] == []
        assert "bbox is malformed" in caplog.text
