"""Tests for the HMIE reference writer and load -> write -> load round trip.

The round trip is the proof that the writer architecture works end to end and
that ``BoxTrackDataset`` round-trips everything the IR represents: loading an
HMIE dataset, writing it back out, and reloading recovers the same box/category
content *and* the video-level / task metadata the loader harvested. The only
field the round trip cannot reproduce is the source label-space AFR/key clock,
which the loader normalises into video-frame space at load time by design.
"""

from __future__ import annotations

from pathlib import Path

from databridge import convert, write
from databridge._formats.hmie.loader import load_hmie
from databridge.model import BoxTrackDataset, VideoSequence

from ._hmie_factory import (
    AnnotationSpec,
    SnippetSpec,
    TrackSpec,
    default_happy_dataset,
    single_video_dataset,
)


def _fingerprint(ds: BoxTrackDataset) -> list:
    """Order-independent *box* content per dataset: category/track/frame/bbox + per-frame fields.

    Compared instead of exact equality because reload may order sequences and
    reassign integer ``category_id`` differently; the *content* (ontology URI,
    the stable ``track_uuid``, video-frame index, geometry, per-frame
    attributes, keyframe flags and timestamp the writer emits) is what must
    survive. ``track_uuid`` is asserted rather than the integer ``track_id``
    because the loader reassigns ``track_id`` by encounter order on reload,
    whereas ``track_uuid`` is carried through verbatim.
    """
    seqs = []
    for seq in ds.sequences:
        boxes = sorted(
            (
                b.category_uri,
                b.track_uuid,
                b.frame_index,
                tuple(round(x, 3) for x in b.bbox),
                tuple(sorted(b.attributes.items())),
                b.keyframe_type,
                b.is_inferred,
                round(b.timestamp, 3) if b.timestamp is not None else None,
            )
            for b in seq.boxes
        )
        seqs.append(tuple(boxes))
    return sorted(seqs)


def _meta_fingerprint(ds: BoxTrackDataset) -> list:
    """Order-independent video-level metadata per sequence.

    Asserts the fields the writer now round-trips: ``video_meta`` (codec,
    origin_id, dimensions, ...), task ``metadata``, derived ``width``/
    ``height``/``duration``, and ``status``. Keyed by ``video_id`` so it is
    independent of sequence ordering on reload.
    """
    return sorted(
        (
            seq.video_id,
            tuple(sorted((k, repr(v)) for k, v in seq.video_meta.items())),
            tuple(sorted((k, repr(v)) for k, v in seq.metadata.items())),
            seq.width,
            seq.height,
            round(seq.duration, 3) if seq.duration is not None else None,
            seq.status,
        )
        for seq in ds.sequences
    )


class TestHmieWriter:
    def test_write_produces_a_reloadable_tree(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        default_happy_dataset(src)
        ds = load_hmie(src)

        out = tmp_path / "out"
        files = write(ds, out, output_format="hmie", verbose=True)

        assert files  # wrote something
        assert all(p.exists() for p in files)
        ds2 = load_hmie(out)
        assert ds2.sequence_count == ds.sequence_count

    def test_round_trip_preserves_box_content(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        default_happy_dataset(src)
        ds = load_hmie(src)

        out = tmp_path / "out"
        write(ds, out, output_format="hmie")
        ds2 = load_hmie(out)

        assert set(ds.categories) == set(ds2.categories)
        assert ds.num_boxes == ds2.num_boxes
        assert _fingerprint(ds) == _fingerprint(ds2)

    def test_round_trip_preserves_video_and_task_metadata(self, tmp_path: Path) -> None:
        """The writer round-trips video_meta / task metadata / global attributes.

        Uses a snippet carrying top-level video metadata, a task-level metadata
        object, non-default per-frame attributes, and global event attributes,
        so the round trip exercises every field the writer now serialises back
        (not just box geometry).
        """
        src = tmp_path / "src"
        spec = SnippetSpec(
            name="video_001_000001",
            annotation=AnnotationSpec(
                task_id="t-meta",
                video_meta={
                    "origin_id": "SRC1_100001",
                    "data_source": "unit-test",
                    "codec_name": "h264",
                    "width": 320,
                    "heigth": 240,
                    "duration": 1.0,
                    "nb_frames": 30,
                },
                metadata={"original_filename": "video_001.mp4", "reviewer": "alpha"},
                global_attributes={"weather": "clear", "scene": "daytime"},
                tracks=[
                    TrackSpec(label="boat", num_frames=3, bbox=(5.0, 6.0, 20.0, 15.0)),
                ],
            ),
        )
        single_video_dataset(src, [spec])
        ds = load_hmie(src)

        out = tmp_path / "out"
        write(ds, out, output_format="hmie")
        ds2 = load_hmie(out)

        # Box content (incl. attributes / keyframe flags) and metadata both survive.
        assert _fingerprint(ds) == _fingerprint(ds2)
        assert _meta_fingerprint(ds) == _meta_fingerprint(ds2)
        # Spot-check the harvested values actually made the trip (not just equal-but-empty).
        seq = next(iter(ds2.sequences))
        assert seq.video_meta["origin_id"] == "SRC1_100001"
        assert seq.video_meta["global_attributes"] == {"weather": "clear", "scene": "daytime"}
        assert seq.metadata["original_filename"] == "video_001.mp4"

    def test_convert_hmie_to_hmie_end_to_end(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        default_happy_dataset(src)

        out = tmp_path / "out"
        files = convert(src, out, input_format="hmie", output_format="hmie", verbose=True)

        assert files
        assert _fingerprint(load_hmie(src)) == _fingerprint(load_hmie(out))

    def test_round_trip_annotation_only_sequence(self, tmp_path: Path) -> None:
        """A sequence with no source video round-trips as an orphan annotation.

        Exercises the no-video branch of the writer: the snippet is written
        annotation-only, and the loader reloads it as a video-less sequence
        (``video_path is None``) whose boxes survive.
        """
        src = tmp_path / "src"
        spec = SnippetSpec(
            name="video_001_000001",
            include_video=False,  # orphan annotation, no seq_mp4/*.mp4
            annotation=AnnotationSpec(tracks=[TrackSpec(label="boat", num_frames=2)]),
        )
        single_video_dataset(src, [spec])
        ds = load_hmie(src)
        assert ds.sequence_count == 1
        assert all(s.video_path is None for s in ds.sequences)

        out = tmp_path / "out"
        write(ds, out, output_format="hmie")
        ds2 = load_hmie(out)

        assert ds2.sequence_count == 1
        assert all(s.video_path is None for s in ds2.sequences)
        assert _fingerprint(ds) == _fingerprint(ds2)

    def test_rewrite_to_same_dest_does_not_keep_stale_snippets(self, tmp_path: Path) -> None:
        """Re-writing fewer sequences to an existing dest drops the stale ones.

        Guards the idempotent-output contract: a second ``write()`` with a
        smaller dataset must not leave the first run's extra ``out_*`` snippets
        behind for ``load_hmie`` to reload.
        """
        src = tmp_path / "src"
        default_happy_dataset(src)  # 4 sequences across 2 full-length videos
        ds = load_hmie(src)
        assert ds.sequence_count > 1

        out = tmp_path / "out"
        write(ds, out, output_format="hmie")
        assert load_hmie(out).sequence_count == ds.sequence_count

        # Re-write only the first sequence; the stale snippets must be gone.
        from dataclasses import replace

        smaller = replace(ds, sequences=ds.sequences[:1])
        write(smaller, out, output_format="hmie")
        assert load_hmie(out).sequence_count == 1

    def test_non_mp4_video_preserves_container(self, tmp_path: Path) -> None:
        """A ``.ts`` source video is written under ``seq_ts/*.ts``, not ``.mp4``.

        Guards against copying non-MP4 payloads under an ``.mp4`` name (wrong
        container). Built in-memory so no real video decode is needed.
        """
        video = tmp_path / "clip.ts"
        video.write_bytes(b"\x47 fake transport-stream payload")  # 0x47 = TS sync byte
        seq = VideoSequence(
            video_id=0,
            video_path=str(video),
            fps=30.0,
            num_frames=None,
            duration=None,
            annotation_path="unused",
        )
        ds = BoxTrackDataset(sequences=(seq,), categories={})

        out = tmp_path / "out"
        write(ds, out, output_format="hmie")

        ts_videos = list(out.rglob("seq_ts/*.ts"))
        assert len(ts_videos) == 1
        assert ts_videos[0].read_bytes() == video.read_bytes()  # bytes preserved, not transcoded
        assert not list(out.rglob("seq_mp4/*"))  # not mislabelled as mp4
        # The annotation filename embeds the .ts video name so the loader pairs them.
        assert any(".ts_" in p.name for p in out.rglob("*.json"))
