"""Tests for video integrity and consistency checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from datamaite._formats.hmie.consistency_checks import check_video_annotation_consistency
from datamaite._formats.hmie.schema import ScaleAnnotation
from datamaite._formats.hmie.video_checks import probe_video
from datamaite._types import Severity
from tests._hmie_factory import VideoSpec, make_video
from tests._scale_factory import default_frame, one_track_annotation


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Create a small synthetic video file with non-flat gradient frames."""
    import cv2

    video_path = tmp_path / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))
    for i in range(30):
        gradient = np.linspace(0, 255, 320, dtype=np.uint8)
        row = np.roll(gradient, i * 5)
        frame = np.tile(row, (240, 1))
        rgb = np.stack([frame, frame, frame], axis=-1)
        writer.write(rgb)
    writer.release()
    return video_path


@pytest.fixture
def flat_video(tmp_path: Path) -> Path:
    """Create a video that is entirely solid black (all frames flat)."""
    import cv2

    video_path = tmp_path / "flat.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (320, 240))
    for _ in range(30):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


@pytest.fixture
def corrupt_video(tmp_path: Path) -> Path:
    """Create a corrupt video file."""
    p = tmp_path / "corrupt.mp4"
    p.write_bytes(b"this is not a video file")
    return p


class TestMidAndLastFrameFailure:
    """Mid/last-frame decode failures are the flagship HMIE failure mode.

    Real HMIE videos commonly open cleanly and decode frame 0 but fail
    halfway through or at the tail. Synthetic mp4v videos with every
    frame as an I-frame cannot reproduce this behavior, so we patch
    ``_decode_frame_at`` (the shared seek-and-decode helper) to fail
    after a configurable number of successful calls.
    """

    def _install_failing_decode(self, monkeypatch, succeed_calls: int) -> None:
        """Patch ``_decode_frame_at`` so only the first ``succeed_calls`` calls succeed.

        Successful calls delegate to the real implementation (so sample,
        mid, and last frames are genuinely decoded from the container);
        calls beyond ``succeed_calls`` return ``None``, simulating a
        seek/decode failure. ``_decode_frame_at`` is called once per
        sampled index plus once each for the mid and last frame, so this
        controls exactly how far into that sequence decoding "fails".
        """
        import datamaite._formats.hmie.video_checks as vc

        real_decode_frame_at = vc._decode_frame_at
        call_count = 0

        def fake_decode_frame_at(container, stream, frame_index, fps):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count > succeed_calls:
                return None
            return real_decode_frame_at(container, stream, frame_index, fps)

        monkeypatch.setattr(vc, "_decode_frame_at", fake_decode_frame_at)

    def test_mid_frame_decode_failure(self, synthetic_video: Path, monkeypatch) -> None:
        """First-frame decode succeeds; every sampled/mid/last decode fails.

        -> video_mid_frame and video_last_frame WARNING.

        Note: video_mid_frame / video_last_frame are WARNING (not ERROR) because
        PyAV's keyframe-based seek semantics on H.264 B-frame pyramids can produce
        false positives even on valid videos. The check is advisory until real
        CDAO data is profiled.
        """
        self._install_failing_decode(monkeypatch, succeed_calls=0)
        _props, findings = probe_video(synthetic_video)
        warning_checks = {f.check for f in findings if f.severity == Severity.WARNING}
        # Both mid and last should be flagged as WARNING
        assert "video_mid_frame" in warning_checks
        assert "video_last_frame" in warning_checks
        # Must NOT be in the ERROR set
        error_checks = {f.check for f in findings if f.severity == Severity.ERROR}
        assert "video_mid_frame" not in error_checks
        assert "video_last_frame" not in error_checks

    def test_last_frame_decode_failure_only(self, synthetic_video: Path, monkeypatch) -> None:
        """Samples + mid-frame succeed; last-frame alone fails.

        Exercises the video_last_frame branch in isolation. ``_decode_frame_at``
        is called once per the 10 sampled indices plus once for the mid frame
        (11 calls); allow those to succeed, then fail the 12th (last-frame) call.
        """
        self._install_failing_decode(monkeypatch, succeed_calls=11)
        _props, findings = probe_video(synthetic_video)
        warning_checks = {f.check for f in findings if f.severity == Severity.WARNING}
        assert "video_last_frame" in warning_checks
        assert "video_mid_frame" not in warning_checks

    def test_no_sample_reads_succeed(self, synthetic_video: Path, monkeypatch) -> None:
        """If EVERY sample decode fails, _check_frame_samples early-returns
        via sampled_count==0 without flagging flat-frames."""
        self._install_failing_decode(monkeypatch, succeed_calls=0)
        _props, findings = probe_video(synthetic_video)
        # video_flat_frames should NOT appear -- sampled_count was 0
        assert not any(f.check == "video_flat_frames" for f in findings)

    def test_exception_during_probe_becomes_finding(self, synthetic_video: Path, monkeypatch) -> None:
        """If PyAV raises during metadata/decode, we must NOT crash the worker.

        Previously the locals frame_count/fps/width/height/first_frame_decodable
        were assigned inside the try block and read after the finally. An
        unexpected exception from container decoding would leave those
        locals unbound, and the function would NameError after closing the
        container -- crashing the worker process and (in parallel mode)
        terminating the whole validation run.

        The fix wraps the metadata/decode block in try/except and emits a
        video_probe_error ERROR finding. This test wraps the real container's
        ``decode()`` so it raises RuntimeError to prove the new error path.
        """
        pytest.importorskip("av")
        import av

        real_open = av.open

        class RaisingContainer:
            def __init__(self, real_container) -> None:  # type: ignore[no-untyped-def]
                self._real = real_container

            @property
            def streams(self):  # type: ignore[no-untyped-def]
                return self._real.streams

            def decode(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
                msg = "simulated codec crash"
                raise RuntimeError(msg)

            def seek(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                return self._real.seek(*args, **kwargs)

            def close(self) -> None:
                self._real.close()

        def fake_open(source, *args, **kwargs):  # type: ignore[no-untyped-def]
            return RaisingContainer(real_open(source, *args, **kwargs))

        monkeypatch.setattr(av, "open", fake_open)

        # Must not raise
        props, findings = probe_video(synthetic_video)

        # Should surface as a Finding, not a NameError / RuntimeError
        assert props.opened is False
        assert any(f.check == "video_probe_error" for f in findings)
        error = next(f for f in findings if f.check == "video_probe_error")
        assert "RuntimeError" in error.message
        assert "simulated codec crash" in error.message


class TestProbeVideo:
    def test_valid_video_properties(self, synthetic_video: Path) -> None:
        props, findings = probe_video(synthetic_video)
        assert props.opened is True
        assert props.fps == 30.0
        assert props.frame_count == 30
        assert props.width == 320
        assert props.height == 240
        assert props.first_frame_decodable is True
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_corrupt_video_returns_unopened_props(self, corrupt_video: Path) -> None:
        props, findings = probe_video(corrupt_video)
        assert props.opened is False
        assert any(f.check == "video_open" for f in findings)

    def test_flat_video_detected(self, flat_video: Path) -> None:
        """A video of all-black frames should be flagged as flat/stuck."""
        _props, findings = probe_video(flat_video)
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert any(f.check == "video_flat_frames" for f in errors)

    @staticmethod
    def _write_mixed_flat_video(path: Path, num_frames: int, flat_fraction: float) -> None:
        """Write a video where the first ``flat_fraction`` of frames are
        solid black and the rest are non-flat gradient frames.

        This matches the sampler's behavior: sampling is uniform across
        the video so the fraction-flat in the file roughly equals the
        fraction-flat in the sampled set.
        """
        import cv2

        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
        flat_count = int(num_frames * flat_fraction)
        for i in range(num_frames):
            if i < flat_count:
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
            else:
                gradient = np.linspace(0, 255, 320, dtype=np.uint8)
                row = np.roll(gradient, i * 5)
                single = np.tile(row, (240, 1))
                frame = np.stack([single, single, single], axis=-1)
            writer.write(frame)
        writer.release()

    def test_mixed_flat_below_threshold_passes(self, tmp_path: Path) -> None:
        """A video with 20% flat frames should NOT trigger video_flat_frames.

        Threshold is 50%. 20% is well below, must pass.
        """
        video_path = tmp_path / "mostly_ok.mp4"
        self._write_mixed_flat_video(video_path, num_frames=30, flat_fraction=0.2)
        _props, findings = probe_video(video_path)
        assert not any(f.check == "video_flat_frames" for f in findings)

    def test_mixed_flat_above_threshold_fails(self, tmp_path: Path) -> None:
        """A video with 80% flat frames should trigger video_flat_frames."""
        video_path = tmp_path / "mostly_flat.mp4"
        self._write_mixed_flat_video(video_path, num_frames=30, flat_fraction=0.8)
        _props, findings = probe_video(video_path)
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert any(f.check == "video_flat_frames" for f in errors)

    def test_single_open_per_probe(self, synthetic_video: Path, monkeypatch) -> None:
        """probe_video should open the video exactly once."""
        pytest.importorskip("av")
        import av

        real_open = av.open
        call_count = 0

        def counting_open(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            return real_open(*args, **kwargs)

        monkeypatch.setattr(av, "open", counting_open)
        probe_video(synthetic_video)
        assert call_count == 1


class TestSilenceAvLogging:
    def test_tolerates_av_without_logging(self) -> None:
        """Variant PyAV builds may omit the av.logging submodule."""
        import datamaite._formats.hmie.video_checks as vc

        class FakeAV:  # no .logging at all
            pass

        # Reset the module-level silenced flag so the call actually runs.
        vc._av_silenced = False
        try:
            vc._silence_av_logging(FakeAV)  # type: ignore[arg-type]
            assert vc._av_silenced is True
        finally:
            vc._av_silenced = False


class TestCheckVideoAnnotationConsistency:
    @staticmethod
    def _consistency(video_path: Path, ann: ScaleAnnotation):  # type: ignore[no-untyped-def]
        """Test helper: probe once, then run the consistency check.

        check_video_annotation_consistency is single-open-only -- callers
        must pass video_props from a prior probe_video() call. This helper
        does that in one shot so each test stays a one-liner.
        """
        props, _ = probe_video(video_path)
        return check_video_annotation_consistency(video_path, ann, props)

    def _make_annotation(self, fps: float = 30.0, afr: float = 5.0, max_key: int = 4) -> ScaleAnnotation:
        data = one_track_annotation(
            afr=afr,
            fps=fps,
            frames=[default_frame(key=i) for i in range(max_key + 1)],
        )
        return ScaleAnnotation.model_validate(data)

    def test_consistent(self, synthetic_video: Path) -> None:
        ann = self._make_annotation(fps=30.0, afr=5.0, max_key=4)
        findings = self._consistency(synthetic_video, ann)
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_fps_mismatch(self, synthetic_video: Path) -> None:
        ann = self._make_annotation(fps=60.0, afr=5.0, max_key=4)
        findings = self._consistency(synthetic_video, ann)
        assert any(f.check == "consistency_fps" for f in findings)

    def test_frame_bounds_exceeded(self, synthetic_video: Path) -> None:
        # 30 frame video, fps=30, afr=5 -> max_key of 100 maps to frame 600, way beyond 30
        ann = self._make_annotation(fps=30.0, afr=5.0, max_key=100)
        findings = self._consistency(synthetic_video, ann)
        assert any(f.check == "consistency_frame_bounds" for f in findings)

    def test_frame_bounds_boundary_case(self, synthetic_video: Path) -> None:
        """Max key that maps exactly to the last valid frame must pass.

        30-frame video, fps=30, afr=5: max_key=4 -> frame# = 24 (valid,
        under 30), must NOT trigger consistency_frame_bounds.
        """
        ann = self._make_annotation(fps=30.0, afr=5.0, max_key=4)
        findings = self._consistency(synthetic_video, ann)
        assert not any(f.check == "consistency_frame_bounds" for f in findings)

    def test_frame_bounds_just_over(self, synthetic_video: Path) -> None:
        """max_key that maps to frame index == video_frame_count triggers the warning."""
        # 30 frame video, fps=30, afr=5 -> max_key=5 -> frame# = 30 (out of bounds, valid is 0..29)
        ann = self._make_annotation(fps=30.0, afr=5.0, max_key=5)
        findings = self._consistency(synthetic_video, ann)
        assert any(f.check == "consistency_frame_bounds" for f in findings)

    def test_consistency_uses_cached_props_without_reopen(self, synthetic_video: Path, monkeypatch) -> None:
        """Passing video_props to consistency check must not open the video again."""
        import cv2

        ann = self._make_annotation(fps=30.0, afr=5.0, max_key=4)
        props, _ = probe_video(synthetic_video)
        assert props.opened

        # Count VideoCapture constructions during the consistency call
        original_cls = cv2.VideoCapture
        call_count = 0

        def counting_capture(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            return original_cls(*args, **kwargs)

        monkeypatch.setattr(cv2, "VideoCapture", counting_capture)
        check_video_annotation_consistency(synthetic_video, ann, video_props=props)
        assert call_count == 0, "consistency check opened the video despite cached props"

    def test_non_finite_video_fps_does_not_crash(self, tmp_path: Path) -> None:
        """NaN/Inf video FPS must emit a finding and not crash frame-bounds math."""
        from datamaite._formats.hmie.video_checks import VideoProperties

        ann = self._make_annotation(fps=30.0, afr=5.0, max_key=4)
        fake_video = tmp_path / "fake.mp4"
        for bad_fps in (float("nan"), float("inf"), float("-inf")):
            props = VideoProperties(
                path=fake_video,
                opened=True,
                fps=bad_fps,
                frame_count=100,
                width=320,
                height=240,
            )
            findings = check_video_annotation_consistency(fake_video, ann, props)
            assert any(f.check == "consistency_fps_invalid" for f in findings), (
                f"expected consistency_fps_invalid for fps={bad_fps}"
            )
            # Must NOT crash, must NOT emit consistency_frame_bounds
            # (ratio computation short-circuited).
            assert not any(f.check == "consistency_frame_bounds" for f in findings)

    def test_non_finite_annotation_fps_is_flagged(self, tmp_path: Path) -> None:
        """NaN annotation FPS must produce a consistency_fps_invalid finding."""
        from datamaite._formats.hmie.video_checks import VideoProperties

        data = one_track_annotation(
            afr=5.0,
            fps=float("nan"),
            frames=[default_frame(key=0)],
        )
        ann = ScaleAnnotation.model_validate(data)

        fake_video = tmp_path / "fake.mp4"
        props = VideoProperties(
            path=fake_video,
            opened=True,
            fps=30.0,
            frame_count=100,
            width=320,
            height=240,
        )
        findings = check_video_annotation_consistency(fake_video, ann, props)
        assert any(f.check == "consistency_fps_invalid" for f in findings)
        # The regular consistency_fps finding must NOT fire for a NaN
        # comparison — that was the silent-false-negative bug.
        assert not any(f.check == "consistency_fps" for f in findings)

    def test_fps_tolerance_within_bounds(self, tmp_path: Path) -> None:
        """29.97 vs 30.00 (NTSC vs clean 30) is within the 0.5 tolerance."""
        import cv2

        video_path = tmp_path / "ntsc.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 29.97, (320, 240))
        for i in range(30):
            gradient = np.linspace(0, 255, 320, dtype=np.uint8)
            row = np.roll(gradient, i * 5)
            frame = np.tile(row, (240, 1))
            rgb = np.stack([frame, frame, frame], axis=-1)
            writer.write(rgb)
        writer.release()

        ann = self._make_annotation(fps=30.0, afr=5.0, max_key=4)
        findings = self._consistency(video_path, ann)
        # 29.97 vs 30.00 diff is 0.03, well under 0.5
        assert not any(f.check == "consistency_fps" for f in findings)

    def test_bbox_at_tolerance_boundary_passes(self, synthetic_video: Path) -> None:
        """A bbox ending exactly at (video_width + 1) must not trigger the warning.

        Scale labelers round coordinates to the nearest pixel, which can produce
        a bbox that extends 1 pixel past the nominal frame edge. The
        _BBOX_PIXEL_TOLERANCE constant permits this; test verifies the boundary.
        """
        # Video is 320x240. Right edge of bbox exactly at 321 (tolerance limit).
        data = one_track_annotation(
            task_id="tolerance-boundary",
            frames=[default_frame(left=0, top=0, height=241, width=321)],
        )
        ann = ScaleAnnotation.model_validate(data)
        findings = self._consistency(synthetic_video, ann)
        # Exactly at the 1-pixel tolerance -- must NOT warn
        assert not any(f.check == "consistency_bbox_bounds" for f in findings)

    def test_bbox_just_past_tolerance_fails(self, synthetic_video: Path) -> None:
        """A bbox ending at (video_width + 2) exceeds the 1-pixel tolerance and must warn."""
        data = one_track_annotation(
            task_id="tolerance-over",
            frames=[default_frame(left=0, top=0, height=242, width=322)],
        )
        ann = ScaleAnnotation.model_validate(data)
        findings = self._consistency(synthetic_video, ann)
        assert any(f.check == "consistency_bbox_bounds" for f in findings)

    def test_bbox_outside_frame(self, synthetic_video: Path) -> None:
        # Video is 320x240, put bbox way outside
        data = one_track_annotation(
            frames=[default_frame(left=300, top=200, height=100, width=100)],
        )
        ann = ScaleAnnotation.model_validate(data)
        findings = self._consistency(synthetic_video, ann)
        assert any(f.check == "consistency_bbox_bounds" for f in findings)


class TestRemoteProbeStreaming:
    """probe_video over a cloud-style (fsspec memory://) root.

    memory:// exercises the real remote branch: _av_source returns a
    seekable fsspec file object and PyAV streams from it directly.
    """

    def test_probe_memory_video_streams(self, memory_root, tmp_path):
        pytest.importorskip("av")
        local = tmp_path / "video_001.mp4"
        make_video(local, VideoSpec())
        remote = memory_root / "seq_mp4" / "video_001.mp4"
        remote.parent.mkdir(parents=True, exist_ok=True)
        remote.write_bytes(local.read_bytes())

        props, findings = probe_video(remote)
        assert props.opened
        assert props.frame_count > 0
        assert props.fps > 0
        assert props.width > 0
        assert props.height > 0
        assert props.path == remote
        assert all(str(f.path) == str(remote) for f in findings)

    def test_probe_memory_matches_local_probe(self, memory_root, tmp_path):
        """Streaming and local probing agree on the same bytes."""
        pytest.importorskip("av")
        local = tmp_path / "video_001.mp4"
        make_video(local, VideoSpec(num_frames=40, fps=20.0))
        remote = memory_root / "video_001.mp4"
        remote.write_bytes(local.read_bytes())

        local_props, _ = probe_video(local)
        remote_props, _ = probe_video(remote)
        assert remote_props.opened
        assert local_props.opened
        assert remote_props.frame_count == local_props.frame_count
        assert remote_props.fps == local_props.fps
        assert (remote_props.width, remote_props.height) == (local_props.width, local_props.height)
        assert remote_props.first_frame_decodable == local_props.first_frame_decodable

    def test_probe_corrupt_remote_video_reports_logical_path(self, memory_root):
        pytest.importorskip("av")
        remote = memory_root / "seq_mp4" / "video_001.mp4"
        remote.parent.mkdir(parents=True, exist_ok=True)
        remote.write_bytes(b"this is not a video file")

        props, findings = probe_video(remote)
        assert not props.opened
        assert any(f.check == "video_open" for f in findings)
        assert all(str(f.path) == str(remote) for f in findings)

    def test_probe_missing_remote_video_is_open_error(self, memory_root):
        pytest.importorskip("av")
        props, findings = probe_video(memory_root / "seq_mp4" / "absent.mp4")
        assert not props.opened
        assert any(f.check == "video_open" for f in findings)


def test_probe_video_reports_missing_av(monkeypatch):
    """Without the fmv extra, probing degrades to a WARNING finding."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "av":
            raise ImportError("No module named 'av'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    props, findings = probe_video(Path("/data/video_001.mp4"))
    assert not props.opened
    assert [f.check for f in findings] == ["video_dependency"]
    assert findings[0].severity == Severity.WARNING
