"""Tests for the box-track model helpers, decode backend, and import isolation."""

from __future__ import annotations

import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from datamaite.model import BoxTrackDataset, VideoSequence, category_name_from_uri

from ._maite_factory import WIDGET, box, make_mp4, sequence


class TestModelHelpers:
    def test_boxes_by_frame_groups_and_sorts(self) -> None:
        boxes = [
            box(track_id=0, category_id=1, uri=WIDGET, name="widget", bbox=(0, 0, 1, 1), frame_index=2),
            box(track_id=1, category_id=1, uri=WIDGET, name="widget", bbox=(0, 0, 1, 1), frame_index=0),
            box(track_id=2, category_id=1, uri=WIDGET, name="widget", bbox=(0, 0, 1, 1), frame_index=2),
        ]
        seq = sequence("/tmp/x.mp4", boxes=boxes)  # noqa: S108 - synthetic label
        grouped = seq.boxes_by_frame()
        assert list(grouped) == [0, 2]  # sorted keys; frame 1 absent
        assert len(grouped[2]) == 2

    def test_index2label(self) -> None:
        ds = BoxTrackDataset(sequences=[], categories={WIDGET: 1, "http://example.com/ontology/a/boat": 2})
        assert ds.index2label() == {1: "widget", 2: "boat"}

    def test_category_name_from_uri(self) -> None:
        assert category_name_from_uri("http://example.com/ontology/a/FOO_000") == "FOO_000"
        assert category_name_from_uri("http://example.com/ontology/a/bar/") == "bar"
        assert category_name_from_uri("plain") == "plain"

    def test_video_sequence_new_fields_default_none(self) -> None:
        seq = VideoSequence(video_id=0, video_path=None, fps=30.0, num_frames=None, duration=None, annotation_path="a")
        assert seq.width is None
        assert seq.height is None
        assert seq.size_bytes is None

    def test_len_counts_only_video_bearing_sequences(self) -> None:
        from ._maite_factory import CATEGORIES, sequence

        ds = BoxTrackDataset(
            sequences=[sequence("/tmp/has_video.mp4", video_id=0), sequence(None, video_id=1)],  # noqa: S108
            categories=dict(CATEGORIES),
        )
        assert len(ds) == 1  # the None-video sequence is excluded from the MAITE surface
        assert len(ds.sequences) == 2  # but .sequences keeps every record

    def test_metadata_has_id_and_index2label(self) -> None:
        ds = BoxTrackDataset(sequences=[], categories={WIDGET: 1}, dataset_id="my-set")
        assert ds.metadata["id"] == "my-set"
        assert ds.metadata["index2label"] == {1: "widget"}

    def test_dataset_id_defaults_to_datamaite(self) -> None:
        ds = BoxTrackDataset(sequences=[], categories={})
        assert ds.metadata["id"] == "datamaite"


class TestImportIsolation:
    def test_core_import_does_not_pull_maite_or_av(self) -> None:
        code = (
            "import sys, datamaite; "
            "leaked = [m for m in ('datamaite.maite', 'maite', 'av') if m in sys.modules]; "
            "assert not leaked, leaked"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)  # noqa: S603
        assert result.returncode == 0, result.stderr

    def test_indexing_without_maite_extra_raises_actionable_error(self, monkeypatch) -> None:
        # Simulate the maite extra being absent: force the lazy import inside
        # __getitem__ to fail the way a missing numpy/av would. The dataset needs
        # one video-bearing sequence so __getitem__ reaches the import (it selects
        # from _mot_sequences first).
        monkeypatch.setitem(sys.modules, "datamaite.maite._mot", None)
        ds = BoxTrackDataset(sequences=[sequence("/tmp/x.mp4")], categories={})  # noqa: S108 - synthetic
        with pytest.raises(ImportError, match=r"pip install datamaite\[maite\]"):
            _ = ds[0]


class TestDecodeBackend:
    def test_pyav_decoder_is_a_decoder(self) -> None:
        from datamaite.maite._decode import Decoder, PyAVDecoder, default_decoder

        assert isinstance(PyAVDecoder(), Decoder)
        assert isinstance(default_decoder(), Decoder)

    def test_info_reads_dimensions_and_time_base(self, tmp_path: Path) -> None:
        from datamaite.maite._decode import PyAVDecoder

        video_path = make_mp4(tmp_path / "v.mp4", width=64, height=48)
        info = PyAVDecoder().info(str(video_path))
        assert (info.width, info.height) == (64, 48)
        assert isinstance(info.time_base, Fraction)
        assert info.size_bytes == video_path.stat().st_size

    def test_stream_none_yields_all_frames(self, tmp_path: Path) -> None:
        from datamaite.maite._decode import PyAVDecoder

        video_path = make_mp4(tmp_path / "v.mp4", num_frames=6)
        frames = list(PyAVDecoder().stream(str(video_path), None))
        assert len(frames) == 6
        assert [f.frame_index for f in frames] == list(range(6))

    def test_decode_one_out_of_range_raises(self, tmp_path: Path) -> None:
        from datamaite.maite._decode import PyAVDecoder

        video_path = make_mp4(tmp_path / "v.mp4", num_frames=3)
        with pytest.raises(IndexError):
            PyAVDecoder().decode_one(str(video_path), 999)

    def test_resolve_video_info_falls_back_on_probe_failure(self, tmp_path: Path, caplog) -> None:
        from datamaite.maite._decode import PyAVDecoder, VideoInfo, resolve_video_info

        fallback = VideoInfo(width=1, height=2, time_base=Fraction(1, 30), size_bytes=3)
        missing = str(tmp_path / "nope.mp4")
        with caplog.at_level("WARNING"):
            info = resolve_video_info(missing, PyAVDecoder(), fallback)
        assert info is fallback
        assert "probe failed" in caplog.text

    def test_fallback_video_info_branches(self) -> None:
        from datamaite.maite._common import fallback_video_info

        seq_known = sequence("/tmp/a.mp4", fps=25.0, width=8, height=9, size_bytes=10)  # noqa: S108
        info = fallback_video_info(seq_known)
        assert info.time_base == Fraction(1, 25)
        assert (info.width, info.height, info.size_bytes) == (8, 9, 10)

        seq_no_fps = sequence("/tmp/b.mp4", fps=0.0, width=None, height=None, size_bytes=None)  # noqa: S108
        info2 = fallback_video_info(seq_no_fps)
        assert info2.time_base == Fraction(1, 1000)
        assert (info2.width, info2.height, info2.size_bytes) == (0, 0, 0)


class TestDefaultMaiteSurface:
    """The loaded object IS a MAITE MOT dataset — no adapter call."""

    def test_dataset_is_mot_without_adapter(self, tmp_path: Path) -> None:
        from maite.protocols import multiobject_tracking as mot

        from ._maite_factory import sample_dataset

        ds, _ = sample_dataset(tmp_path)
        assert isinstance(ds, mot.Dataset)

    def test_getitem_returns_mot_triple(self, tmp_path: Path) -> None:
        from ._maite_factory import sample_dataset

        ds, _ = sample_dataset(tmp_path)
        _stream, target, meta = ds[0]
        assert set(meta) >= {"id", "height", "width", "time_base", "size"}
        assert len(target.frame_tracks) == 2  # annotated frames 0 and 2
        assert target.frame_tracks[1].boxes.shape == (2, 4)

    def test_iterating_dataset_yields_mot_triples(self, tmp_path: Path) -> None:
        from ._maite_factory import sample_dataset

        ds, _ = sample_dataset(tmp_path)
        items = list(ds)  # iteration falls back to __getitem__/__len__
        assert len(items) == 1
        _stream, target, _meta = items[0]
        assert len(target.frame_tracks) == 2

    def test_sequences_still_carry_typed_source_fields(self, tmp_path: Path) -> None:
        from ._maite_factory import sample_dataset

        ds, _ = sample_dataset(tmp_path)
        boat = ds.sequences[0].boxes[-1]
        assert boat.category_uri == "http://example.com/ontology/a/boat"
        assert boat.track_uuid == "uuid-1"
        assert boat.keyframe_type == "start"
        assert boat.is_inferred is False


class TestEndToEndFromLoader:
    def test_load_hmie_is_directly_maite_indexable(self, tmp_path: Path) -> None:
        from datamaite._formats.hmie.loader import load_hmie

        from ._hmie_factory import SnippetSpec, single_video_dataset

        single_video_dataset(
            tmp_path,
            snippets=[SnippetSpec(name="video_001_000001", source_designator="SRC1", hash_suffix="abc001")],
        )
        # require_video=True so the loader probes real width/height/size onto
        # the sequence, exercising the loader -> MAITE metadata path.
        ds = load_hmie(tmp_path, require_video=True)
        assert len(ds) >= 1
        _stream, target, meta = ds[0]  # directly indexable -- no adapter call
        assert set(meta) >= {"id", "height", "width", "time_base", "size"}
        assert meta["width"] > 0
        assert meta["height"] > 0
        assert len(target.frame_tracks) >= 1
