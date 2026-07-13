"""Tests for the MOTChallenge writer and MOTChallenge load -> write -> load round trip."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from datamaite import DatasetFormat, MotChallengeWriter, convert, write
from datamaite._formats.motchallenge.loader import load_motchallenge
from datamaite.model import BoxAnnotation, BoxTrackDataset, VideoSequence
from datamaite.writers import available_output_formats, get_writer

from ._hmie_factory import AnnotationSpec, SnippetSpec, TrackSpec, single_video_dataset


def write_mot_sequence(
    root: Path,
    *,
    split: str = "train",
    name: str = "MOT17-02",
    gt_rows: list[str] | None = None,
    det_rows: list[str] | None = None,
    frame_count: int = 3,
) -> Path:
    seq = root / split / name
    frame_dir = seq / "img1"
    frame_dir.mkdir(parents=True)
    for frame in range(1, frame_count + 1):
        (frame_dir / f"{frame:06d}.jpg").write_bytes(f"frame-{frame}".encode())
    (seq / "seqinfo.ini").write_text(
        "\n".join(
            [
                "[Sequence]",
                f"name={name}",
                "imDir=img1",
                "frameRate=30",
                f"seqLength={frame_count}",
                "imWidth=640",
                "imHeight=480",
                "imExt=.jpg",
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


def _mot_fingerprint(ds: BoxTrackDataset) -> list[tuple[object, ...]]:
    seqs: list[tuple[object, ...]] = []
    for seq in ds.sequences:
        boxes = tuple(
            sorted(
                (
                    box.track_id,
                    box.category_id,
                    box.category_uri,
                    box.category_name,
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
                seq.video_meta["sequence_name"],
                seq.video_meta["annotation_source"],
                seq.frame_pattern,
                seq.frame_number_base,
                seq.fps,
                seq.num_frames,
                seq.width,
                seq.height,
                boxes,
            )
        )
    return sorted(seqs)


class TestMotChallengeWriterRegistry:
    def test_registered_and_public_api(self) -> None:
        assert DatasetFormat.MOTCHALLENGE in available_output_formats()
        assert isinstance(get_writer(DatasetFormat.MOTCHALLENGE), MotChallengeWriter)
        assert isinstance(get_writer("motchallenge"), MotChallengeWriter)


class TestMotChallengeWriterHappyPath:
    def test_write_produces_reloadable_gt_benchmark_root(self, tmp_path: Path) -> None:
        write_mot_sequence(
            tmp_path / "src",
            gt_rows=[
                "1,1,10,20,30,40,1,1,0.9",
                "2,1,12,22,30,40,1,1,0.8",
                "3,2,5,6,7,8,1,7,1",
            ],
        )
        ds = load_motchallenge(tmp_path / "src")

        out = tmp_path / "out"
        files = write(ds, out, output_format="motchallenge", verbose=True)

        assert out / "train" / "MOT17-02" / "seqinfo.ini" in files
        assert out / "train" / "MOT17-02" / "gt" / "gt.txt" in files
        assert out / "train" / "MOT17-02" / "img1" / "000001.jpg" in files
        assert (out / "train" / "MOT17-02" / "img1" / "000001.jpg").read_bytes() == b"frame-1"
        assert (out / "train" / "MOT17-02" / "gt" / "gt.txt").read_text(encoding="utf-8").splitlines() == [
            "1,1,10,20,30,40,1,1,0.9",
            "2,1,12,22,30,40,1,1,0.8",
            "3,2,5,6,7,8,1,7,1",
        ]
        assert _mot_fingerprint(load_motchallenge(out)) == _mot_fingerprint(ds)

    def test_convert_motchallenge_to_motchallenge_end_to_end(self, tmp_path: Path) -> None:
        write_mot_sequence(tmp_path / "src", gt_rows=["1,1,10,20,30,40,1,1,1"])

        files = convert(
            tmp_path / "src", tmp_path / "out", input_format="motchallenge", output_format="motchallenge", verbose=True
        )

        assert files
        assert _mot_fingerprint(load_motchallenge(tmp_path / "out")) == _mot_fingerprint(
            load_motchallenge(tmp_path / "src")
        )

    def test_convert_hmie_to_motchallenge_keeps_every_track(self, tmp_path: Path) -> None:
        # Regression: HMIE track ids are 1-based on purpose. With 0-based ids
        # the GT writer -- which reserves non-positive ids for "no usable
        # track" -- silently dropped the first track of every sequence.
        single_video_dataset(
            tmp_path / "src",
            [
                SnippetSpec(
                    name="video_001_000001",
                    annotation=AnnotationSpec(
                        tracks=[TrackSpec(label="vehicle"), TrackSpec(label="boat")],
                    ),
                )
            ],
        )

        files = convert(
            tmp_path / "src", tmp_path / "out", input_format="hmie", output_format="motchallenge", verbose=True
        )

        gt_path = next(path for path in files if path.name == "gt.txt")
        rows = gt_path.read_text(encoding="utf-8").splitlines()
        assert {int(row.split(",")[1]) for row in rows} == {1, 2}
        assert len(rows) == 10  # 5 labeled frames per track, none dropped

    def test_writes_detection_source(self, tmp_path: Path) -> None:
        write_mot_sequence(
            tmp_path / "src",
            gt_rows=None,
            det_rows=["1,-1,10,20,30,40,0.42,-1,-1,-1", "2,7,50,60,10,15,0.75,1,2,3"],
        )
        ds = load_motchallenge(tmp_path / "src", annotation_source="det")

        write(ds, tmp_path / "out", output_format="motchallenge", annotation_source="det")

        rows = (tmp_path / "out" / "train" / "MOT17-02" / "det" / "det.txt").read_text(encoding="utf-8")
        assert rows.splitlines() == [
            "1,-1,10,20,30,40,0.42,-1,-1,-1",
            "2,7,50,60,10,15,0.75,1,2,3",
        ]
        assert _mot_fingerprint(load_motchallenge(tmp_path / "out", annotation_source="det")) == _mot_fingerprint(ds)

    def test_non_standard_split_defaults_to_train_and_option_can_override(self, tmp_path: Path) -> None:
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
            video_meta={"split": "validation", "sequence_name": "clip"},
            boxes=[],
            num_frames_exact=True,
        )

        write(
            BoxTrackDataset(sequences=(seq,), categories={}),
            tmp_path / "out",
            output_format="motchallenge",
            split="test",
        )

        assert (tmp_path / "out" / "test" / "clip" / "seqinfo.ini").is_file()
        assert not (tmp_path / "out" / "train").exists()

    def test_duplicate_sequence_names_are_disambiguated(self, tmp_path: Path) -> None:
        first = tmp_path / "first.jpg"
        second = tmp_path / "second.jpg"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        seqs = (
            VideoSequence(
                video_id=0,
                video_path=None,
                fps=0.0,
                num_frames=1,
                duration=None,
                annotation_path="unused",
                frame_files=(str(first),),
                video_meta={"sequence_name": "same name"},
                boxes=[],
                num_frames_exact=True,
            ),
            VideoSequence(
                video_id=1,
                video_path=None,
                fps=0.0,
                num_frames=1,
                duration=None,
                annotation_path="unused",
                frame_files=(str(second),),
                video_meta={"sequence_name": "same/name"},
                boxes=[],
                num_frames_exact=True,
            ),
        )

        write(BoxTrackDataset(sequences=seqs, categories={}), tmp_path / "out", output_format="motchallenge")

        assert (tmp_path / "out" / "train" / "same-name" / "img1" / "000001.jpg").read_bytes() == b"first"
        assert (tmp_path / "out" / "train" / "same-name-1" / "img1" / "000001.jpg").read_bytes() == b"second"


class TestMotChallengeWriterMalformedInputs:
    def test_missing_frame_drops_annotation(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"frame")
        missing = tmp_path / "missing.jpg"
        box = BoxAnnotation(
            track_uuid="track-a",
            track_id=1,
            category_id=1,
            category_uri="motchallenge/pedestrian",
            category_name="pedestrian",
            bbox=(1.0, 2.0, 3.0, 4.0),
            attributes={"mot_class_id": 1},
            frame_index=1,
            timestamp=None,
        )
        seq = VideoSequence(
            video_id=0,
            video_path=None,
            fps=0.0,
            num_frames=2,
            duration=None,
            annotation_path="unused",
            frame_files=(str(frame), str(missing)),
            video_meta={"sequence_name": "clip"},
            boxes=[box],
            num_frames_exact=True,
        )

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.motchallenge.writer"):
            write(
                BoxTrackDataset(sequences=(seq,), categories={"motchallenge/pedestrian": 1}),
                tmp_path / "out",
                output_format="motchallenge",
            )

        rows = (tmp_path / "out" / "train" / "clip" / "gt" / "gt.txt").read_text(encoding="utf-8")
        assert rows == ""
        assert "Skipping missing MOTChallenge source frame" in caplog.text
        assert "no frame image was written" in caplog.text

    def test_malformed_gt_boxes_are_dropped_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"frame")
        bad_box = BoxAnnotation(
            track_uuid="track-a",
            track_id=-1,
            category_id=1,
            category_uri="motchallenge/pedestrian",
            category_name="pedestrian",
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
            video_meta={"sequence_name": "clip"},
            boxes=[bad_box],
            num_frames_exact=True,
        )

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.motchallenge.writer"):
            write(BoxTrackDataset(sequences=(seq,), categories={}), tmp_path / "out", output_format="motchallenge")

        assert (tmp_path / "out" / "train" / "clip" / "gt" / "gt.txt").read_text(encoding="utf-8") == ""
        assert "bbox is malformed" in caplog.text

    def test_invalid_options_raise(self, tmp_path: Path) -> None:
        ds = BoxTrackDataset(sequences=(), categories={})
        with pytest.raises(ValueError, match="split"):
            write(ds, tmp_path / "out", output_format="motchallenge", split="validation")
        with pytest.raises(ValueError, match="annotation_source"):
            write(ds, tmp_path / "out", output_format="motchallenge", annotation_source="labels")
        with pytest.raises(ValueError, match="image_extension"):
            write(ds, tmp_path / "out", output_format="motchallenge", image_extension=".gif")


def _classed_box(
    *,
    category_id: int = 3,
    category_name: str | None = "person",
    attributes: dict | None = None,
    track_id: int = 1,
) -> BoxAnnotation:
    return BoxAnnotation(
        track_uuid=f"track-{track_id}",
        track_id=track_id,
        category_id=category_id,
        category_uri=f"src/{category_name or category_id}",
        category_name=category_name,
        bbox=(1.0, 2.0, 3.0, 4.0),
        attributes=attributes or {},
        frame_index=0,
        timestamp=None,
    )


def _one_frame_sequence(tmp_path: Path, boxes: list[BoxAnnotation]) -> BoxTrackDataset:
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
        video_meta={"sequence_name": "clip"},
        boxes=boxes,
        num_frames_exact=True,
    )
    return BoxTrackDataset(sequences=(seq,), categories={})


_MOT_WRITER_LOGGER = "datamaite._formats.motchallenge.writer"


def _gt_rows(out: Path) -> list[str]:
    return (out / "train" / "clip" / "gt" / "gt.txt").read_text(encoding="utf-8").splitlines()


class TestMotChallengeFixedTaxonomy:
    def test_generic_fallback_warns_once_aggregated(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        ds = _one_frame_sequence(
            tmp_path,
            [_classed_box(track_id=1), _classed_box(track_id=2)],  # no mot_class_id attribute
        )
        with caplog.at_level(logging.WARNING, logger=_MOT_WRITER_LOGGER):
            write(ds, tmp_path / "out", output_format="motchallenge")
        fallback = [r for r in caplog.records if "class_map" in r.getMessage()]
        assert len(fallback) == 1
        assert "2 annotation(s)" in fallback[0].getMessage()
        assert "person" in fallback[0].getMessage()
        # The generic ids were still written (warn-then-use).
        assert [row.split(",")[7] for row in _gt_rows(tmp_path / "out")] == ["3", "3"]

    def test_mot_class_id_attribute_stays_quiet(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        ds = _one_frame_sequence(tmp_path, [_classed_box(attributes={"mot_class_id": 1})])
        with caplog.at_level(logging.WARNING, logger=_MOT_WRITER_LOGGER):
            write(ds, tmp_path / "out", output_format="motchallenge")
        assert not [r for r in caplog.records if "class_map" in r.getMessage()]
        assert [row.split(",")[7] for row in _gt_rows(tmp_path / "out")] == ["1"]

    def test_class_map_overrides_attribute_and_maps_by_name(self, tmp_path: Path) -> None:
        ds = _one_frame_sequence(tmp_path, [_classed_box(attributes={"mot_class_id": 1})])
        write(ds, tmp_path / "out", output_format="motchallenge", class_map={"person": 7})
        assert [row.split(",")[7] for row in _gt_rows(tmp_path / "out")] == ["7"]

    def test_class_map_unmapped_boxes_are_dropped_with_one_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        ds = _one_frame_sequence(
            tmp_path,
            [_classed_box(track_id=1), _classed_box(track_id=2, category_name="vehicle", category_id=4)],
        )
        with caplog.at_level(logging.WARNING, logger=_MOT_WRITER_LOGGER):
            write(ds, tmp_path / "out", output_format="motchallenge", class_map={"person": 1})
        assert [row.split(",")[7] for row in _gt_rows(tmp_path / "out")] == ["1"]
        dropped = [r for r in caplog.records if "not present in class_map" in r.getMessage()]
        assert len(dropped) == 1
        assert "vehicle" in dropped[0].getMessage()

    def test_invalid_class_map_raises_before_writing(self, tmp_path: Path) -> None:
        ds = _one_frame_sequence(tmp_path, [_classed_box()])
        out = tmp_path / "out"
        with pytest.raises(ValueError, match="class ids must be >= 1"):
            write(ds, out, output_format="motchallenge", class_map={"person": 0})
        # The dest is never touched: validate_options() raises before write()'s
        # own mkdir/destination-policy handling runs (#55 B10).
        assert not out.exists()

    def test_det_rows_ignore_class_map(self, tmp_path: Path) -> None:
        ds = _one_frame_sequence(tmp_path, [_classed_box(attributes={"score": 0.9})])
        write(
            ds,
            tmp_path / "out",
            output_format="motchallenge",
            annotation_source="det",
            class_map={"no-such-category": 1},
        )
        rows = (tmp_path / "out" / "train" / "clip" / "det" / "det.txt").read_text(encoding="utf-8").splitlines()
        assert len(rows) == 1  # not dropped: det rows carry no class column

    def test_empty_class_map_drops_everything(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        ds = _one_frame_sequence(tmp_path, [_classed_box(attributes={"mot_class_id": 5})])
        with caplog.at_level(logging.WARNING, logger=_MOT_WRITER_LOGGER):
            write(ds, tmp_path / "out", output_format="motchallenge", class_map={})
        assert _gt_rows(tmp_path / "out") == []
        assert any("not present in class_map" in r.getMessage() for r in caplog.records)

    def test_name_present_but_unmatched_falls_through_to_id_key(self, tmp_path: Path) -> None:
        ds = _one_frame_sequence(tmp_path, [_classed_box(category_id=3, category_name="unmatched")])
        write(ds, tmp_path / "out", output_format="motchallenge", class_map={3: 9})
        assert [row.split(",")[7] for row in _gt_rows(tmp_path / "out")] == ["9"]


class TestMotChallengeWriterEmitsWarningsDespiteMidWriteFailure:
    """#55 B2: aggregated warnings must still surface even if a later
    sequence raises mid-write (earlier sequences' output is already on disk)."""

    def test_emit_warnings_runs_even_if_a_later_sequence_raises(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import datamaite._formats.motchallenge.writer as mot_writer_module

        first_frame = tmp_path / "first.jpg"
        second_frame = tmp_path / "second.jpg"
        first_frame.write_bytes(b"first")
        second_frame.write_bytes(b"second")
        seqs = (
            VideoSequence(
                video_id=0,
                video_path=None,
                fps=0.0,
                num_frames=1,
                duration=None,
                annotation_path="unused",
                frame_files=(str(first_frame),),
                video_meta={"sequence_name": "clip-a"},
                boxes=[_classed_box(track_id=1)],  # no mot_class_id -> generic fallback
                num_frames_exact=True,
            ),
            VideoSequence(
                video_id=1,
                video_path=None,
                fps=0.0,
                num_frames=1,
                duration=None,
                annotation_path="unused",
                frame_files=(str(second_frame),),
                video_meta={"sequence_name": "clip-b"},
                boxes=[],
                num_frames_exact=True,
            ),
        )
        ds = BoxTrackDataset(sequences=seqs, categories={})

        calls = {"n": 0}
        original_seqinfo_text = mot_writer_module._seqinfo_text

        def _raise_on_second_call(*args: object, **kwargs: object):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom mid-write")
            return original_seqinfo_text(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(mot_writer_module, "_seqinfo_text", _raise_on_second_call)

        with (
            caplog.at_level(logging.WARNING, logger=_MOT_WRITER_LOGGER),
            pytest.raises(RuntimeError, match="boom mid-write"),
        ):
            write(ds, tmp_path / "out", output_format="motchallenge")

        fallback = [r for r in caplog.records if "class_map" in r.getMessage()]
        assert len(fallback) == 1
        # The first sequence's gt.txt is already on disk despite the raise.
        assert (tmp_path / "out" / "train" / "clip-a" / "gt" / "gt.txt").is_file()
