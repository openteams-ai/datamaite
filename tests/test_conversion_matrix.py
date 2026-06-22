"""Cross-format conversion matrix: read format A, write format B, reload B.

``databridge`` is an N-to-M bridge -- any registered box-track loader feeds the
neutral ``BoxTrackDataset``, which any registered box-track writer serialises.
The per-format ``test_<format>_writer.py`` files cover each writer's *own*
round trip (A->A); this module covers the *cross-format* matrix the bridge
exists to provide: for every ordered pair of writable formats, ``convert`` the
dataset A->B, then reload B *from disk* and confirm the box geometry survived.

"Writable" means a format with both a registered loader and writer; that is the
4x4 matrix of HMIE, MOTChallenge, TAO, and VisDrone video. Cross-format
comparison uses a deliberately *neutral* fingerprint (sequence/frame counts plus
box geometry) -- not the per-format fingerprints -- because category-key
prefixes, reassigned track IDs, split names, and video metadata legitimately
differ across formats.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from databridge import convert
from databridge._types import DatasetFormat
from databridge.loaders import available_formats, load
from databridge.model import BoxTrackDataset
from databridge.writers import available_output_formats

from ._hmie_factory import (
    AnnotationSpec,
    FullVideoSpec,
    SnippetSpec,
    TrackSpec,
    VideoSpec,
    default_happy_dataset,
    make_hmie_dataset,
)
from .test_motchallenge_writer import write_mot_sequence
from .test_tao_writer import tao_payload, write_tao, write_tao_frames
from .test_visdrone_video_writer import write_visdrone_split

# Formats whose writer decodes a video-backed sequence to per-frame images.
# HMIE sequences carry a video file (not pre-extracted frames), so converting
# HMIE -> any of these exercises the OpenCV frame-extraction path.
_FRAME_EXTRACTING_FORMATS = (
    DatasetFormat.MOTCHALLENGE,
    DatasetFormat.TAO,
    DatasetFormat.VISDRONE_VIDEO,
)


def _box_count(ds: BoxTrackDataset) -> int:
    return sum(len(seq.boxes) for seq in ds.sequences)


def _reload_root(dest: Path, output_format: DatasetFormat) -> Path:
    """Path to reload after writing ``output_format`` under ``dest``.

    VisDrone writes one split-root subdirectory; the others reload at ``dest``.
    """
    if output_format is DatasetFormat.VISDRONE_VIDEO:
        return dest / "VisDrone2019-VID-train"
    return dest


# Formats with BOTH a loader and a writer -- the convertible matrix. Derived so
# adding a 5th read+write format auto-expands the matrix (and trips the
# guardrail below if it lacks a source builder).
WRITABLE_FORMATS = sorted(
    set(available_formats()) & set(available_output_formats()),
    key=lambda fmt: fmt.value,
)


# Each writable format -> a zero-knob builder that lays a small valid source
# dataset on disk and returns the path to load. The builders themselves live
# with each format's writer tests (single source of truth); these wrappers just
# pin a small default payload so every matrix cell starts from real boxes.
def _hmie_src(root: Path) -> Path:
    return default_happy_dataset(root)


def _motchallenge_src(root: Path) -> Path:
    write_mot_sequence(
        root,
        gt_rows=[
            "1,1,10,20,30,40,1,1,0.9",
            "2,1,12,22,30,40,1,1,0.8",
            "3,2,5,6,7,8,1,7,1",
        ],
    )
    return root


def _tao_src(root: Path) -> Path:
    payload = tao_payload()
    write_tao(root, payload)
    write_tao_frames(root, *(image["file_name"] for image in payload["images"]))
    return root


def _visdrone_src(root: Path) -> Path:
    write_visdrone_split(
        root,
        rows=[
            "1,1,10,20,30,40,1,4,0,0",
            "2,1,12,22,30,40,1,4,0,0",
            "3,2,5,6,7,8,1,5,0,0",
        ],
    )
    return root


SOURCE_BUILDERS: dict[DatasetFormat, Callable[[Path], Path]] = {
    DatasetFormat.HMIE: _hmie_src,
    DatasetFormat.MOTCHALLENGE: _motchallenge_src,
    DatasetFormat.TAO: _tao_src,
    DatasetFormat.VISDRONE_VIDEO: _visdrone_src,
}


def _neutral_fingerprint(ds: BoxTrackDataset) -> tuple[int, tuple]:
    """Format-agnostic content fingerprint: sequence count + box geometry.

    Captures only what every conversion hop must preserve: how many sequences,
    and per sequence the multiset of ``(frame_index, x, y, w, h)`` boxes
    (coordinates rounded to absorb cross-format float representation).

    Deliberately excluded because they legitimately differ by format:

    * ``num_frames`` -- each format derives it differently (HMIE from the
      labeled-frame span, TAO from the video-metadata field, ...), so a
      video->frame-extraction hop changes it without changing any box.
    * track identity, category keys, split names, and other video metadata.
    """
    seqs = []
    for seq in ds.sequences:
        boxes = tuple(
            sorted(
                (
                    box.frame_index,
                    round(box.bbox[0]),
                    round(box.bbox[1]),
                    round(box.bbox[2]),
                    round(box.bbox[3]),
                )
                for box in seq.boxes
            )
        )
        seqs.append(boxes)
    return (len(ds.sequences), tuple(sorted(seqs)))


class TestMatrixIntegrity:
    """The matrix and its fixtures stay in sync with the registered formats."""

    def test_writable_matrix_matches_registries(self) -> None:
        # The convertible set is exactly (loaders ∩ writers); spelled out so a
        # newly added format is a deliberate change, not a silent one.
        assert set(WRITABLE_FORMATS) == {
            DatasetFormat.HMIE,
            DatasetFormat.MOTCHALLENGE,
            DatasetFormat.TAO,
            DatasetFormat.VISDRONE_VIDEO,
        }

    def test_every_writable_format_has_a_source_builder(self) -> None:
        missing = [fmt.value for fmt in WRITABLE_FORMATS if fmt not in SOURCE_BUILDERS]
        assert not missing, f"no source builder for: {missing}"


@pytest.mark.parametrize("output_format", WRITABLE_FORMATS, ids=lambda f: f.value)
@pytest.mark.parametrize("input_format", WRITABLE_FORMATS, ids=lambda f: f.value)
class TestConversionMatrix:
    """Every (input -> output) pair: convert, reload from disk, compare geometry."""

    def test_convert_then_reload_preserves_geometry(
        self,
        input_format: DatasetFormat,
        output_format: DatasetFormat,
        tmp_path: Path,
    ) -> None:
        src = SOURCE_BUILDERS[input_format](tmp_path / "src")
        baseline = load(src, dataset_format=input_format)
        # Source builders must produce real boxes, else the comparison is vacuous.
        assert any(seq.boxes for seq in baseline.sequences)

        dest = tmp_path / "out"
        files = convert(src, dest, input_format=input_format, output_format=output_format, verbose=True)
        assert files, f"{input_format.value} -> {output_format.value} wrote nothing"

        reloaded = load(dest, dataset_format=output_format)
        assert _neutral_fingerprint(reloaded) == _neutral_fingerprint(baseline), (
            f"geometry changed converting {input_format.value} -> {output_format.value}"
        )


@pytest.mark.parametrize("output_format", _FRAME_EXTRACTING_FORMATS, ids=lambda f: f.value)
class TestVideoBackedConversionIsLossless:
    """Converting a video-backed HMIE dataset must not drop boxes.

    The frame-extracting writers (MOTChallenge, TAO, VisDrone) decode the source
    video to per-frame images and key annotations to those frames. When the
    declared fps matches the real video (the ``default_happy_dataset`` factory
    clamps annotation keys to fit the video length), *every* box has a backing
    frame, so the round trip must preserve the full box count. This guards the
    OpenCV decode path that pre-extracted-frame fixtures never exercise.
    """

    def test_no_boxes_dropped(self, output_format: DatasetFormat, tmp_path: Path) -> None:
        src = default_happy_dataset(tmp_path / "src")
        baseline = load(src, dataset_format=DatasetFormat.HMIE)
        assert _box_count(baseline) > 0

        dest = tmp_path / "out"
        convert(src, dest, input_format=DatasetFormat.HMIE, output_format=output_format)
        reloaded = load(_reload_root(dest, output_format), dataset_format=output_format)

        assert _box_count(reloaded) == _box_count(baseline), (
            f"hmie -> {output_format.value} dropped boxes for well-formed video data"
        )


def _fps_mismatch_hmie_dataset(root: Path) -> Path:
    """Lay a video-backed HMIE dataset whose annotations outrun the real video.

    The annotation declares ``video_fps=60`` while the synthesised mp4 is 30
    frames at 30 fps. The loader maps frame keys with the *declared* fps, so
    boxes land on frame indices (e.g. 36, 48) that do not exist in the decoded
    video -- the deliberate ``warn-fps-mismatch`` scenario, reproduced in-repo
    without external data.
    """
    snippet = SnippetSpec(
        name="video_001_000001",
        video=VideoSpec(num_frames=30, fps=30.0),  # clamp basis: keys fit a 30-frame video
        annotation=AnnotationSpec(
            afr=5.0,
            video_fps=60.0,  # loader's mapping basis: doubles indices past the video length
            tracks=[TrackSpec(num_frames=6, bbox=(10.0, 10.0, 50.0, 40.0))],
        ),
    )
    return make_hmie_dataset(root, [FullVideoSpec(name="video_001_000000", snippets=[snippet])])


class TestFpsMismatchDropsOutOfRangeBoxesWithWarning:
    """Regression: HMIE -> frame format on fps-mismatch data.

    When annotations reference frames the real video does not contain, the
    frame-extracting writers cannot represent those boxes (no frame image
    exists). Per the writer contract they drop the unrepresentable boxes and log
    a WARNING -- they neither crash nor silently relocate the box to a wrong
    frame. The boxes that *do* have a backing frame still round-trip faithfully.
    """

    def test_out_of_range_boxes_are_dropped_and_in_range_survive(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        src = _fps_mismatch_hmie_dataset(tmp_path / "src")
        baseline = load(src, dataset_format=DatasetFormat.HMIE)
        seq = baseline.sequences[0]
        real_frame_count = 30  # frames the synthesised mp4 actually contains
        in_range = [b for b in seq.boxes if b.frame_index < real_frame_count]
        out_of_range = [b for b in seq.boxes if b.frame_index >= real_frame_count]
        # The fixture must actually exercise both sides, else it proves nothing.
        assert in_range, "fixture has no in-range boxes"
        assert out_of_range, "fixture has no out-of-range boxes to drop"

        dest = tmp_path / "out"
        with caplog.at_level("WARNING"):
            convert(src, dest, input_format=DatasetFormat.HMIE, output_format=DatasetFormat.VISDRONE_VIDEO)
        reloaded = load(_reload_root(dest, DatasetFormat.VISDRONE_VIDEO), dataset_format=DatasetFormat.VISDRONE_VIDEO)

        # Loss is exactly the out-of-range boxes -- nothing more, nothing less.
        assert _box_count(reloaded) == len(in_range)
        # Drops are reported, not silent.
        assert any("Dropping" in record.message for record in caplog.records), (
            "out-of-range boxes were dropped without a warning"
        )
        # Surviving boxes keep their original frame positions (no silent clamp).
        survivor_frames = sorted(b.frame_index for b in reloaded.sequences[0].boxes)
        assert survivor_frames == sorted(b.frame_index for b in in_range)
