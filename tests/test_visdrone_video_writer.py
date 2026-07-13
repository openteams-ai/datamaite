"""Tests for the VisDrone video writer and VisDrone load -> write -> load round trip."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from datamaite import DatasetFormat, VisDroneVideoWriter, convert, write
from datamaite._formats.visdrone.loader import load_visdrone_video
from datamaite.model import BoxAnnotation, BoxTrackDataset, VideoSequence
from datamaite.writers import available_output_formats, get_writer


def write_visdrone_split(
    root: Path,
    *,
    split_name: str = "VisDrone2019-VID-train",
    sequence_name: str = "uav0000013_00000_v",
    rows: list[str] | None = None,
    frame_count: int = 3,
) -> Path:
    split = root / split_name
    annotation_dir = split / "annotations"
    frame_dir = split / "sequences" / sequence_name
    annotation_dir.mkdir(parents=True)
    frame_dir.mkdir(parents=True)
    for frame in range(1, frame_count + 1):
        (frame_dir / f"{frame:07d}.jpg").write_bytes(f"frame-{frame}".encode())
    if rows is not None:
        (annotation_dir / f"{sequence_name}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return split


def _visdrone_fingerprint(ds: BoxTrackDataset) -> list[tuple[object, ...]]:
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
                seq.video_meta["variant"],
                seq.video_meta["split"],
                seq.video_meta["sequence_name"],
                seq.video_meta["annotation_source"],
                seq.frame_pattern,
                seq.frame_number_base,
                seq.num_frames,
                boxes,
            )
        )
    return sorted(seqs)


class TestVisDroneVideoWriterRegistry:
    def test_registered_and_public_api(self) -> None:
        assert DatasetFormat.VISDRONE_VIDEO in available_output_formats()
        assert isinstance(get_writer(DatasetFormat.VISDRONE_VIDEO), VisDroneVideoWriter)
        assert isinstance(get_writer("visdrone_video"), VisDroneVideoWriter)


class TestVisDroneVideoWriterHappyPath:
    def test_write_produces_reloadable_vid_split_root(self, tmp_path: Path) -> None:
        write_visdrone_split(
            tmp_path / "src",
            rows=[
                "1,1,10,20,30,40,1,4,0,0",
                "2,1,12,22,30,40,1,4,1,2",
                "3,4,1,2,3,4,1,11,0,0",
            ],
        )
        ds = load_visdrone_video(tmp_path / "src")

        out = tmp_path / "out"
        files = write(ds, out, output_format="visdrone_video", verbose=True)

        root = out / "VisDrone2019-VID-train"
        assert root / "annotations" / "uav0000013_00000_v.txt" in files
        assert root / "sequences" / "uav0000013_00000_v" / "0000001.jpg" in files
        assert (root / "sequences" / "uav0000013_00000_v" / "0000001.jpg").read_bytes() == b"frame-1"
        assert (root / "annotations" / "uav0000013_00000_v.txt").read_text(encoding="utf-8").splitlines() == [
            "1,1,10,20,30,40,1,4,0,0",
            "2,1,12,22,30,40,1,4,1,2",
            "3,4,1,2,3,4,1,11,0,0",
        ]
        assert _visdrone_fingerprint(load_visdrone_video(out)) == _visdrone_fingerprint(ds)

    def test_convert_visdrone_to_visdrone_end_to_end(self, tmp_path: Path) -> None:
        write_visdrone_split(tmp_path / "src", rows=["1,1,10,20,30,40,1,4,0,0"])

        files = convert(
            tmp_path / "src",
            tmp_path / "out",
            input_format="visdrone_video",
            output_format="visdrone_video",
            verbose=True,
        )

        assert files
        assert _visdrone_fingerprint(load_visdrone_video(tmp_path / "out")) == _visdrone_fingerprint(
            load_visdrone_video(tmp_path / "src")
        )

    def test_variant_option_writes_mot_split_root(self, tmp_path: Path) -> None:
        write_visdrone_split(tmp_path / "src", rows=["1,1,10,20,30,40,1,4,0,0"])
        ds = load_visdrone_video(tmp_path / "src")

        write(ds, tmp_path / "out", output_format="visdrone_video", variant="mot", split="validation")

        root = tmp_path / "out" / "VisDrone2019-MOT-train"
        assert (root / "annotations" / "uav0000013_00000_v.txt").is_file()
        reloaded = load_visdrone_video(tmp_path / "out")
        assert reloaded.sequences[0].video_meta["variant"] == "mot"
        assert reloaded.sequences[0].video_meta["split"] == "train"

    def test_preserve_splits_can_be_disabled(self, tmp_path: Path) -> None:
        write_visdrone_split(
            tmp_path / "src",
            split_name="VisDrone2019-VID-val",
            rows=["1,1,10,20,30,40,1,4,0,0"],
        )
        ds = load_visdrone_video(tmp_path / "src")

        write(
            ds,
            tmp_path / "out",
            output_format="visdrone_video",
            split="test-dev",
            preserve_splits=False,
        )

        assert (tmp_path / "out" / "VisDrone2019-VID-test-dev" / "annotations" / "uav0000013_00000_v.txt").is_file()

    def test_writes_detection_source(self, tmp_path: Path) -> None:
        write_visdrone_split(
            tmp_path / "src",
            rows=["1,-1,10,20,30,40,0.42,4,-1,-1", "2,7,50,60,10,15,0.75,5,-1,-1"],
        )
        ds = load_visdrone_video(tmp_path / "src", annotation_source="det")

        write(ds, tmp_path / "out", output_format="visdrone_video", annotation_source="det")

        rows = (tmp_path / "out" / "VisDrone2019-VID-train" / "annotations" / "uav0000013_00000_v.txt").read_text(
            encoding="utf-8"
        )
        assert rows.splitlines() == [
            "1,-1,10,20,30,40,0.42,4,-1,-1",
            "2,7,50,60,10,15,0.75,5,-1,-1",
        ]
        assert _visdrone_fingerprint(
            load_visdrone_video(tmp_path / "out", annotation_source="det")
        ) == _visdrone_fingerprint(ds)

    def test_nonpositive_gt_target_id_is_kept_for_ignored_or_excluded_rows(self, tmp_path: Path) -> None:
        frame_files = []
        for index in range(3):
            frame = tmp_path / f"frame-{index}.jpg"
            frame.write_bytes(f"frame-{index}".encode())
            frame_files.append(str(frame))
        boxes = [
            BoxAnnotation(
                track_uuid="ignored-region",
                track_id=0,
                category_id=0,
                category_uri="visdrone_video/ignored_region",
                category_name="ignored_region",
                bbox=(10, 20, 30, 40),
                attributes={"visdrone_category_id": 0, "visdrone_target_id": 0, "visdrone_score": 1},
                frame_index=0,
                timestamp=None,
            ),
            BoxAnnotation(
                track_uuid="eval-excluded",
                track_id=0,
                category_id=4,
                category_uri="visdrone_video/car",
                category_name="car",
                bbox=(11, 21, 31, 41),
                attributes={"visdrone_category_id": 4, "visdrone_target_id": 0, "visdrone_score": 0},
                frame_index=1,
                timestamp=None,
            ),
            BoxAnnotation(
                track_uuid="invalid-gt",
                track_id=0,
                category_id=4,
                category_uri="visdrone_video/car",
                category_name="car",
                bbox=(12, 22, 32, 42),
                attributes={"visdrone_category_id": 4, "visdrone_target_id": 0, "visdrone_score": 1},
                frame_index=2,
                timestamp=None,
            ),
        ]
        seq = VideoSequence(
            video_id=0,
            video_path=None,
            fps=0.0,
            num_frames=3,
            duration=None,
            annotation_path="unused",
            frame_files=tuple(frame_files),
            video_meta={"sequence_name": "clip"},
            boxes=boxes,
            num_frames_exact=True,
        )

        write(
            BoxTrackDataset(
                sequences=(seq,),
                categories={"visdrone_video/ignored_region": 0, "visdrone_video/car": 4},
            ),
            tmp_path / "out",
            output_format="visdrone_video",
        )

        rows = (tmp_path / "out" / "VisDrone2019-VID-train" / "annotations" / "clip.txt").read_text(encoding="utf-8")
        assert rows.splitlines() == [
            "1,0,10,20,30,40,1,0,-1,-1",
            "2,0,11,21,31,41,0,4,-1,-1",
        ]

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

        write(BoxTrackDataset(sequences=seqs, categories={}), tmp_path / "out", output_format="visdrone_video")

        root = tmp_path / "out" / "VisDrone2019-VID-train" / "sequences"
        assert (root / "same-name" / "0000001.jpg").read_bytes() == b"first"
        assert (root / "same-name-1" / "0000001.jpg").read_bytes() == b"second"


def _classed_box(
    *,
    category_id: int = 4,
    category_name: str | None = "car",
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


def _one_frame_dataset(tmp_path: Path, boxes: list[BoxAnnotation]) -> BoxTrackDataset:
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


_VISDRONE_WRITER_LOGGER = "datamaite._formats.visdrone.writer"


def _annotation_categories(out: Path) -> list[str]:
    ann = out / "VisDrone2019-VID-train" / "annotations" / "clip.txt"
    return [row.split(",")[7] for row in ann.read_text(encoding="utf-8").splitlines()]


class TestVisDroneFixedTaxonomy:
    def test_generic_fallback_warns_once_aggregated(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        ds = _one_frame_dataset(tmp_path, [_classed_box(track_id=1), _classed_box(track_id=2)])
        with caplog.at_level(logging.WARNING, logger=_VISDRONE_WRITER_LOGGER):
            write(ds, tmp_path / "out", output_format="visdrone_video")
        fallback = [r for r in caplog.records if "class_map" in r.getMessage()]
        assert len(fallback) == 1
        assert "2 annotation(s)" in fallback[0].getMessage()
        assert _annotation_categories(tmp_path / "out") == ["4", "4"]

    def test_visdrone_category_id_attribute_stays_quiet(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        ds = _one_frame_dataset(tmp_path, [_classed_box(attributes={"visdrone_category_id": 5})])
        with caplog.at_level(logging.WARNING, logger=_VISDRONE_WRITER_LOGGER):
            write(ds, tmp_path / "out", output_format="visdrone_video")
        assert not [r for r in caplog.records if "class_map" in r.getMessage()]
        assert _annotation_categories(tmp_path / "out") == ["5"]

    def test_class_map_maps_by_name_and_allows_zero(self, tmp_path: Path) -> None:
        ds = _one_frame_dataset(tmp_path, [_classed_box(attributes={"visdrone_category_id": 5})])
        write(ds, tmp_path / "out", output_format="visdrone_video", class_map={"car": 0})
        assert _annotation_categories(tmp_path / "out") == ["0"]

    def test_class_map_unmapped_boxes_are_dropped_with_one_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        ds = _one_frame_dataset(
            tmp_path,
            [_classed_box(track_id=1), _classed_box(track_id=2, category_name="tricycle", category_id=8)],
        )
        with caplog.at_level(logging.WARNING, logger=_VISDRONE_WRITER_LOGGER):
            write(ds, tmp_path / "out", output_format="visdrone_video", class_map={"car": 4})
        assert _annotation_categories(tmp_path / "out") == ["4"]
        dropped = [r for r in caplog.records if "not present in class_map" in r.getMessage()]
        assert len(dropped) == 1
        assert "tricycle" in dropped[0].getMessage()

    def test_invalid_class_map_raises_before_writing(self, tmp_path: Path) -> None:
        ds = _one_frame_dataset(tmp_path, [_classed_box()])
        out = tmp_path / "out"
        with pytest.raises(ValueError, match="class ids must be >= 0"):
            write(ds, out, output_format="visdrone_video", class_map={"car": -1})
        # The dest is never touched: validate_options() raises before write()'s
        # own mkdir/destination-policy handling runs (#55 B10).
        assert not out.exists()

    def test_empty_class_map_drops_everything(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        ds = _one_frame_dataset(tmp_path, [_classed_box(attributes={"visdrone_category_id": 5})])
        with caplog.at_level(logging.WARNING, logger=_VISDRONE_WRITER_LOGGER):
            write(ds, tmp_path / "out", output_format="visdrone_video", class_map={})
        ann = tmp_path / "out" / "VisDrone2019-VID-train" / "annotations" / "clip.txt"
        assert ann.read_text(encoding="utf-8") == ""
        assert any("not present in class_map" in r.getMessage() for r in caplog.records)

    def test_name_present_but_unmatched_falls_through_to_id_key(self, tmp_path: Path) -> None:
        ds = _one_frame_dataset(tmp_path, [_classed_box(category_id=4, category_name="unmatched")])
        write(ds, tmp_path / "out", output_format="visdrone_video", class_map={4: 9})
        assert _annotation_categories(tmp_path / "out") == ["9"]


class TestVisDroneFixedTaxonomyDetSource:
    """#55 B9: VisDrone applies the resolver to det too, unlike MOT."""

    def test_class_map_maps_by_name_and_drops_unmapped_under_det(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        ds = _one_frame_dataset(
            tmp_path,
            [_classed_box(track_id=1), _classed_box(track_id=2, category_name="tricycle", category_id=8)],
        )
        with caplog.at_level(logging.WARNING, logger=_VISDRONE_WRITER_LOGGER):
            write(
                ds,
                tmp_path / "out",
                output_format="visdrone_video",
                annotation_source="det",
                class_map={"car": 4},
            )
        assert _annotation_categories(tmp_path / "out") == ["4"]
        dropped = [r for r in caplog.records if "not present in class_map" in r.getMessage()]
        assert len(dropped) == 1
        assert "tricycle" in dropped[0].getMessage()

    def test_generic_fallback_warns_under_det(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        ds = _one_frame_dataset(tmp_path, [_classed_box(track_id=1), _classed_box(track_id=2)])
        with caplog.at_level(logging.WARNING, logger=_VISDRONE_WRITER_LOGGER):
            write(ds, tmp_path / "out", output_format="visdrone_video", annotation_source="det")
        fallback = [r for r in caplog.records if "class_map" in r.getMessage()]
        assert len(fallback) == 1
        assert _annotation_categories(tmp_path / "out") == ["4", "4"]


class TestVisDroneIgnoredRegionExemption:
    """#55 B3: the category-0 ignored-region exemption must not apply to a
    generic-fallback zero (a category that merely happens to be 0, not a
    deliberately-assigned ignored-region marker)."""

    def test_generic_fallback_zero_category_is_not_exempted(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        box = BoxAnnotation(
            track_uuid="t",
            track_id=0,
            category_id=0,
            category_uri="src/unlabeled",
            category_name=None,
            bbox=(1.0, 2.0, 3.0, 4.0),
            attributes={"visdrone_target_id": 0, "visdrone_score": 1},
            frame_index=0,
            timestamp=None,
        )
        ds = _one_frame_dataset(tmp_path, [box])
        with caplog.at_level(logging.WARNING, logger=_VISDRONE_WRITER_LOGGER):
            write(ds, tmp_path / "out", output_format="visdrone_video")
        ann = tmp_path / "out" / "VisDrone2019-VID-train" / "annotations" / "clip.txt"
        assert ann.read_text(encoding="utf-8") == ""
        assert "target id is not positive" in caplog.text

    def test_attribute_zero_category_is_exempted_as_ignored_region(self, tmp_path: Path) -> None:
        box = BoxAnnotation(
            track_uuid="t",
            track_id=0,
            category_id=0,
            category_uri="src/ignored",
            category_name=None,
            bbox=(1.0, 2.0, 3.0, 4.0),
            attributes={"visdrone_category_id": 0, "visdrone_target_id": 0, "visdrone_score": 1},
            frame_index=0,
            timestamp=None,
        )
        ds = _one_frame_dataset(tmp_path, [box])
        write(ds, tmp_path / "out", output_format="visdrone_video")
        ann = tmp_path / "out" / "VisDrone2019-VID-train" / "annotations" / "clip.txt"
        assert ann.read_text(encoding="utf-8").strip() != ""
        assert ann.read_text(encoding="utf-8").splitlines()[0].split(",")[7] == "0"


class TestVisDroneWriterEmitsWarningsDespiteMidWriteFailure:
    """#55 B2: aggregated warnings must still surface even if a later
    sequence raises mid-write (earlier sequences' output is already on disk)."""

    def test_emit_warnings_runs_even_if_a_later_sequence_raises(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import datamaite._formats.visdrone.writer as visdrone_writer_module

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
                boxes=[_classed_box(track_id=1)],  # no visdrone_category_id -> generic fallback
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
        original_annotation_rows = visdrone_writer_module._annotation_rows

        def _raise_on_second_call(*args: object, **kwargs: object):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom mid-write")
            return original_annotation_rows(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(visdrone_writer_module, "_annotation_rows", _raise_on_second_call)

        with (
            caplog.at_level(logging.WARNING, logger=_VISDRONE_WRITER_LOGGER),
            pytest.raises(RuntimeError, match="boom mid-write"),
        ):
            write(ds, tmp_path / "out", output_format="visdrone_video")

        fallback = [r for r in caplog.records if "class_map" in r.getMessage()]
        assert len(fallback) == 1
        # The first sequence's output is already on disk despite the raise.
        assert (tmp_path / "out" / "VisDrone2019-VID-train" / "sequences" / "clip-a").is_dir()


class TestVisDroneVideoWriterMalformedInputs:
    def test_missing_frame_drops_annotation(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"frame")
        missing = tmp_path / "missing.jpg"
        box = BoxAnnotation(
            track_uuid="track-a",
            track_id=1,
            category_id=4,
            category_uri="visdrone_video/car",
            category_name="car",
            bbox=(1.0, 2.0, 3.0, 4.0),
            attributes={"visdrone_category_id": 4},
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

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.visdrone.writer"):
            write(
                BoxTrackDataset(sequences=(seq,), categories={"visdrone_video/car": 4}),
                tmp_path / "out",
                output_format="visdrone_video",
            )

        rows = (tmp_path / "out" / "VisDrone2019-VID-train" / "annotations" / "clip.txt").read_text(encoding="utf-8")
        assert rows == ""
        assert "Skipping missing VisDrone source frame" in caplog.text
        assert "no frame image was written" in caplog.text

    def test_malformed_gt_boxes_are_dropped_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"frame")
        bad_box = BoxAnnotation(
            track_uuid="track-a",
            track_id=-1,
            category_id=-1,
            category_uri="",
            category_name=None,
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

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.visdrone.writer"):
            write(BoxTrackDataset(sequences=(seq,), categories={}), tmp_path / "out", output_format="visdrone_video")

        rows = (tmp_path / "out" / "VisDrone2019-VID-train" / "annotations" / "clip.txt").read_text(encoding="utf-8")
        assert rows == ""
        assert "bbox is malformed" in caplog.text

    def test_invalid_options_raise(self, tmp_path: Path) -> None:
        ds = BoxTrackDataset(sequences=(), categories={})
        with pytest.raises(ValueError, match="variant"):
            write(ds, tmp_path / "out", output_format="visdrone_video", variant="sot")
        with pytest.raises(ValueError, match="annotation_source"):
            write(ds, tmp_path / "out", output_format="visdrone_video", annotation_source="labels")
        with pytest.raises(ValueError, match="image_extension"):
            write(ds, tmp_path / "out", output_format="visdrone_video", image_extension=".gif")
        with pytest.raises(ValueError, match="split"):
            write(ds, tmp_path / "out", output_format="visdrone_video", split="../train")
