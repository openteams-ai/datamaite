"""Tests for the flat-folder MP4 loader (IR-3.3-S-1)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from datamaite import DatasetFormat, FlatMp4Loader, load
from datamaite._formats.flat_mp4.loader import (
    _canonical_codec,
    _fourcc_to_string,
    _probe_mp4_video,
    _VideoProbe,
    load_flat_mp4,
)
from datamaite.loaders import available_formats, get_loader
from datamaite.model import BoxTrackDataset


def _write_mp4(path: Path) -> Path:
    path.write_bytes(b"not a real mp4; probe is monkeypatched in unit tests")
    return path


def _probe(
    *,
    codec: str | None,
    fourcc: str | None,
    opened: bool = True,
    fps: float = 30.0,
    frame_count: int = 90,
    width: int = 640,
    height: int = 480,
    first_frame_decodable: bool = True,
) -> _VideoProbe:
    return _VideoProbe(
        opened=opened,
        codec=codec,
        codec_fourcc=fourcc,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        first_frame_decodable=first_frame_decodable,
    )


class TestFlatMp4Registry:
    def test_registered_and_public_api(self) -> None:
        assert DatasetFormat.FLAT_MP4 in available_formats()
        assert isinstance(get_loader(DatasetFormat.FLAT_MP4), FlatMp4Loader)
        assert isinstance(get_loader("flat_mp4"), FlatMp4Loader)
        assert callable(load_flat_mp4)

    def test_dispatch_loads_flat_mp4(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        _write_mp4(tmp_path / "clip.mp4")
        monkeypatch.setattr(
            "datamaite._formats.flat_mp4.loader._probe_mp4_video",
            lambda _p: _probe(codec="h264", fourcc="avc1"),
        )

        ds = load(tmp_path, dataset_format="flat_mp4")

        assert isinstance(ds, BoxTrackDataset)
        assert ds.sequence_count == 1
        assert ds.sequences[0].video_meta["codec"] == "h264"


class TestFlatMp4HappyPath:
    def test_loads_immediate_h264_and_mpeg2_mp4_children(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        h264 = _write_mp4(tmp_path / "h264.mp4")
        mpeg2 = _write_mp4(tmp_path / "mpeg2.MP4")
        (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
        nested = tmp_path / "nested"
        nested.mkdir()
        _write_mp4(nested / "ignored.mp4")

        probes = {
            h264.name: _probe(codec="h264", fourcc="avc1", fps=30.0, frame_count=90, width=1920, height=1080),
            mpeg2.name: _probe(codec="mpeg2", fourcc="mpg2", fps=25.0, frame_count=50, width=720, height=480),
        }
        monkeypatch.setattr("datamaite._formats.flat_mp4.loader._probe_mp4_video", lambda p: probes[p.name])

        ds = load_flat_mp4(tmp_path)

        assert ds.categories == {}
        assert ds.num_boxes == 0
        assert ds.sequence_count == 2
        assert len(ds) == 2  # video-backed sequences are exposed through the MAITE item view.

        first, second = ds.sequences
        assert Path(first.video_path or "").name == "h264.mp4"
        assert first.annotation_path == str(h264)
        assert first.fps == 30.0
        assert first.num_frames == 90
        assert first.duration == 3.0
        assert first.width == 1920
        assert first.height == 1080
        assert first.size_bytes == h264.stat().st_size
        assert first.num_frames_exact is True
        assert first.boxes == []
        assert first.video_meta == {
            "format": "flat_mp4",
            "container": "mp4",
            "filename": "h264.mp4",
            "source_path": str(h264),
            "codec": "h264",
            "codec_label": "H.264",
            "codec_fourcc": "avc1",
        }

        assert Path(second.video_path or "").name == "mpeg2.MP4"
        assert second.video_meta["codec"] == "mpeg2"
        assert second.video_meta["codec_label"] == "MPEG-2"
        assert second.fps == 25.0
        assert second.num_frames == 50
        assert second.duration == 2.0

    def test_codec_aliases_and_fourcc_decoding(self) -> None:
        def cv_fourcc(token: str) -> int:
            return sum(ord(ch) << (8 * idx) for idx, ch in enumerate(token))

        assert _canonical_codec("avc1") == "h264"
        assert _canonical_codec("H264") == "h264"
        assert _canonical_codec("mpg2") == "mpeg2"
        assert _canonical_codec("MP2V") == "mpeg2"
        assert _canonical_codec("mp4v") is None
        assert _fourcc_to_string(cv_fourcc("avc1")) == "avc1"
        assert _fourcc_to_string(0) is None


class TestFlatMp4MalformedInputs:
    def test_missing_or_empty_root_returns_empty_dataset(
        self,
        tmp_path: Path,
        caplog,
        monkeypatch,  # type: ignore[no-untyped-def]
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="datamaite._formats.flat_mp4.loader"):
            missing = load_flat_mp4(tmp_path / "missing")
            empty = load_flat_mp4(tmp_path)

        assert missing.sequence_count == 0
        assert empty.sequence_count == 0
        assert "not a directory" in caplog.text
        assert "No immediate .mp4 files" in caplog.text

        # Ensure an empty directory did not try to probe anything.
        monkeypatch.setattr(
            "datamaite._formats.flat_mp4.loader._probe_mp4_video",
            lambda _p: (_ for _ in ()).throw(AssertionError("probe should not run")),
        )
        assert load_flat_mp4(tmp_path).sequence_count == 0

    def test_skips_malformed_h264_and_mpeg2_videos(
        self,
        tmp_path: Path,
        caplog,
        monkeypatch,  # type: ignore[no-untyped-def]
    ) -> None:
        h264_empty = _write_mp4(tmp_path / "h264-empty.mp4")
        mpeg2_bad_frame = _write_mp4(tmp_path / "mpeg2-bad-frame.mp4")
        zero_res = _write_mp4(tmp_path / "h264-zero-res.mp4")
        unsupported = _write_mp4(tmp_path / "mpeg4.mp4")

        probes = {
            h264_empty.name: _probe(codec="h264", fourcc="avc1", frame_count=0),
            mpeg2_bad_frame.name: _probe(codec="mpeg2", fourcc="mpg2", first_frame_decodable=False),
            zero_res.name: _probe(codec="h264", fourcc="avc1", width=0, height=0),
            unsupported.name: _probe(codec=None, fourcc="mp4v"),
        }
        monkeypatch.setattr("datamaite._formats.flat_mp4.loader._probe_mp4_video", lambda p: probes[p.name])

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.flat_mp4.loader"):
            ds = load_flat_mp4(tmp_path)

        assert ds.sequence_count == 0
        assert "reports no frames" in caplog.text
        assert "first frame cannot be decoded" in caplog.text
        assert "invalid resolution" in caplog.text
        assert "unsupported codec" in caplog.text

    def test_probe_dependency_failure_skips_video(self, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        _write_mp4(tmp_path / "clip.mp4")
        monkeypatch.setattr(
            "datamaite._formats.flat_mp4.loader._probe_mp4_video",
            lambda _p: _probe(codec=None, fourcc=None, opened=False),
        )

        ds = load_flat_mp4(tmp_path)

        assert ds.sequence_count == 0


class TestFlatMp4RealProbe:
    """Exercise the real OpenCV probe (``_probe_mp4_video``) end to end.

    The unit tests above monkeypatch ``_probe_mp4_video`` so they never run
    the OpenCV integration -- the codec gating, ``CAP_PROP_*`` extraction, and
    fourcc decoding. These tests drive that real path against clips encoded on
    the fly. OpenCV reports a codec's fourcc differently depending on the
    FFmpeg build (for example ``avc1`` is reported as ``h264``), and an
    encoder may be missing entirely, so each accepted-codec case skips when the
    installed build cannot produce that codec.
    """

    @staticmethod
    def _write_real_clip(
        path: Path,
        fourcc: str,
        *,
        frames: int = 8,
        width: int = 64,
        height: int = 48,
        fps: float = 30.0,
    ) -> Path:
        cv2 = pytest.importorskip("cv2")
        np = pytest.importorskip("numpy")
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
        if not writer.isOpened():
            writer.release()
            pytest.skip(f"OpenCV cannot open a VideoWriter for {fourcc!r}")
        try:
            for idx in range(frames):
                writer.write(np.full((height, width, 3), (idx * 10) % 256, dtype=np.uint8))
        finally:
            writer.release()
        if not path.exists() or path.stat().st_size == 0:
            pytest.skip(f"OpenCV produced no output for {fourcc!r}")
        return path

    @pytest.mark.parametrize(
        ("fourcc", "expected_codec", "expected_label"),
        [("avc1", "h264", "H.264"), ("mp2v", "mpeg2", "MPEG-2")],
    )
    def test_loads_real_supported_codec_end_to_end(
        self, tmp_path: Path, fourcc: str, expected_codec: str, expected_label: str
    ) -> None:
        pytest.importorskip("cv2")
        clip = self._write_real_clip(tmp_path / f"{fourcc}.mp4", fourcc, frames=8, width=64, height=48, fps=30.0)

        # Probe the real file first: a build without this encoder silently
        # re-tags to something we (correctly) do not accept, so skip rather
        # than fail on an environment that cannot produce the codec.
        probe = _probe_mp4_video(clip)
        if probe.codec != expected_codec:
            pytest.skip(f"OpenCV encoded {fourcc!r} as {probe.codec_fourcc!r}, not {expected_codec}")
        assert probe.opened is True
        assert probe.first_frame_decodable is True

        ds = load_flat_mp4(tmp_path)

        assert ds.sequence_count == 1
        assert len(ds) == 1  # video-backed -> exposed in the MAITE item view
        seq = ds.sequences[0]
        assert seq.video_meta["codec"] == expected_codec
        assert seq.video_meta["codec_label"] == expected_label
        assert seq.num_frames == 8
        assert seq.num_frames_exact is True
        assert seq.width == 64
        assert seq.height == 48
        assert seq.fps == pytest.approx(30.0, abs=1.0)
        assert seq.duration is not None
        assert seq.duration > 0
        assert seq.boxes == []

    def test_real_unsupported_codec_is_probed_then_skipped(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        pytest.importorskip("cv2")
        # mp4v (MPEG-4 Part 2) is widely encodable and is *not* an IR-3.3-S-1
        # codec, so the loader must open and probe it, then reject it. This
        # deterministically covers the real probe body on any build with cv2.
        clip = self._write_real_clip(tmp_path / "mp4v.mp4", "mp4v", frames=8, width=64, height=48)

        probe = _probe_mp4_video(clip)
        assert probe.opened is True
        assert probe.frame_count == 8
        assert probe.width == 64
        assert probe.height == 48
        assert probe.first_frame_decodable is True
        assert probe.codec_fourcc  # fourcc number decoded to a non-empty token
        assert probe.codec is None  # not an H.264 / MPEG-2 codec

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.flat_mp4.loader"):
            ds = load_flat_mp4(tmp_path)

        assert ds.sequence_count == 0
        assert "unsupported codec" in caplog.text

    def test_real_corrupt_mp4_cannot_be_opened(self, tmp_path: Path, caplog) -> None:  # type: ignore[no-untyped-def]
        pytest.importorskip("cv2")
        # A file with an .mp4 suffix that is not a real video must be probed
        # (the open path runs) and skipped, not crash the load.
        (tmp_path / "broken.mp4").write_bytes(b"\x00\x01\x02 not an mp4 container")

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.flat_mp4.loader"):
            ds = load_flat_mp4(tmp_path)

        assert ds.sequence_count == 0
