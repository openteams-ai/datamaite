"""HMIE writer: serialise a :class:`BoxTrackDataset` back to the HMIE layout.

The reference writer for the writer architecture (:mod:`datamaite.writers`)
and the inverse of the HMIE loader. It reconstructs a discoverable HMIE tree
(snippet dir -> ``<labeler>/`` Scale annotation JSON + ``seq_<container>/``
video, preserving the source ``.mp4``/``.ts`` suffix) so that::

    load_hmie(src) -> BoxTrackDataset -> write -> load_hmie(dest)

recovers everything the IR represents -- box/category/track content *and* the
video-level ``video_meta`` / task ``metadata`` / global attributes the loader
harvested. The writer is the inverse of the loader's read path, so each field
is written back where the loader reads it.

The one field the round trip cannot reproduce is the source labeling clock:
the loader normalises Scale frame keys into video-frame space at load time
(``frame_index = floor(key * fps / afr)``) and does not keep the raw key or
``afr``, so the writer emits ``annotation_frame_rate == video fps`` (ratio 1),
giving ``key == frame_index``. Box geometry and frame placement round-trip
exactly; the label-space ``afr``/key encoding is normalised away by design
(see ``_formats/hmie/loader.py`` -- storing the raw key would put boxes and
``num_frames`` on different clocks).

Categories are written as their ontology ``label`` (URI), so they re-resolve
to the same names; the integer ``category_id`` is reassigned on reload
(encounter order), so round-trip equivalence is by ``category_uri``, not by id.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from datamaite._formats.hmie.discovery import _VIDEO_EXTENSIONS
from datamaite._types import DatasetFormat
from datamaite.model import BoxTrackDataset, VideoSequence
from datamaite.writers import Writer, register_writer

logger = logging.getLogger(__name__)


@register_writer
class HmieWriter(Writer[BoxTrackDataset]):
    """Write a :class:`BoxTrackDataset` as an HMIE/Scale on-disk dataset."""

    format = DatasetFormat.HMIE

    def write(
        self, dataset: BoxTrackDataset, dest: str | Path, *, labeler: str = "scale", **_options: Any
    ) -> list[Path]:
        """Serialise ``dataset`` under ``dest`` as HMIE and return files written.

        Each sequence becomes a ``out_<id>_000000/out_<id>_000001/`` snippet
        with the annotation under ``<labeler>/`` and the (copied) source video
        under ``seq_<container>/`` (the source suffix is preserved, e.g.
        ``seq_mp4/*.mp4`` or ``seq_ts/*.ts``). A sequence with no source video
        is written as an annotation-only snippet and logged -- the HMIE loader
        will treat it as an orphan annotation on reload (no video to pair).

        Write is **idempotent over its own output**: any pre-existing
        writer-owned snippet dirs (matching the ``out_*_000000`` naming this
        writer creates) are removed first, so re-writing to a destination with
        fewer sequences does not leave stale snippets that ``load_hmie(dest)``
        would then reload. Other contents of ``dest`` are left untouched.
        """
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        # Clear only directories matching this writer's own output pattern, so a
        # re-run is idempotent without touching anything the writer didn't create.
        for stale in dest.glob("out_*_000000"):
            if stale.is_dir():
                shutil.rmtree(stale)
        written: list[Path] = []
        for seq in dataset.sequences:
            written.extend(_write_sequence(dest, seq, labeler=labeler))
        return written


def _write_sequence(dest: Path, seq: VideoSequence, *, labeler: str) -> list[Path]:
    name = f"out_{seq.video_id:06d}"
    snippet_name = f"{name}_000001"
    snippet_dir = dest / f"{name}_000000" / snippet_name
    written: list[Path] = []

    # Preserve the source container/suffix: HMIE discovery recognises both
    # ``.mp4`` and ``.ts``, so forcing ``.mp4`` would write a ``.ts`` payload
    # under an ``.mp4`` name (wrong container) and break the seq_<ext> dir.
    # Annotation-only snippets default to ``.mp4`` so the loader's filename
    # heuristic still recognises the JSON as a Scale annotation.
    suffix = Path(seq.video_path).suffix.lower() if seq.video_path else ".mp4"
    if suffix not in _VIDEO_EXTENSIONS:
        logger.warning(
            "sequence %s has video suffix %r which HMIE discovery does not recognise "
            "(%s); bytes are preserved but the reloaded snippet will be an orphan",
            seq.video_id,
            suffix,
            ", ".join(sorted(_VIDEO_EXTENSIONS)),
        )
    video_filename = f"{snippet_name}{suffix}"

    # Video container dir mirrors the source: seq_mp4/, seq_ts/, etc. Always
    # created (even with no video) so discovery still identifies the snippet.
    seq_dir = snippet_dir / f"seq_{suffix.lstrip('.')}"
    seq_dir.mkdir(parents=True, exist_ok=True)
    if seq.video_path and Path(seq.video_path).exists():
        video_out = seq_dir / video_filename
        shutil.copyfile(seq.video_path, video_out)
        written.append(video_out)
    else:
        logger.warning(
            "sequence %s has no source video; writing an annotation-only HMIE snippet "
            "(the loader will treat it as an orphan annotation on reload)",
            seq.video_id,
        )

    # The annotation filename embeds the full video filename (``<snippet><ext>``)
    # so the loader's annotation<->video matcher pairs them on reload.
    ann_dir = snippet_dir / labeler
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_path = ann_dir / f"CDAO_OUT_{video_filename}_rt.json"
    ann_path.write_text(json.dumps(_annotation_dict(seq), indent=2), encoding="utf-8")
    written.append(ann_path)
    return written


def _annotation_dict(seq: VideoSequence) -> dict[str, Any]:
    """Rebuild a Scale Video Playback annotation from a sequence's boxes.

    Tracks are grouped by ``track_uuid`` and emitted in ``track_id`` order so
    the loader reassigns the same per-annotation ``track_id``; frames within a
    track are sorted by ``frame_index`` and written with ``key == frame_index``.

    Video-level ``video_meta``, task ``metadata`` and global attributes are
    written back where the loader reads them (top-level keys, the ``metadata``
    object, and ``response.events`` respectively), so they survive the round
    trip rather than being dropped.
    """
    # rate == video fps == afr, so the loader's key->frame_index mapping is the
    # identity (frame# = key * fps / afr = key). Falls back to 1.0 when fps is
    # unknown (still identity, since afr is written equal to it).
    rate = seq.fps if seq.fps and seq.fps > 0 else 1.0

    groups: dict[str, dict[str, Any]] = {}
    for box in seq.boxes:
        group = groups.setdefault(box.track_uuid, {"track_id": box.track_id, "boxes": []})
        group["boxes"].append(box)

    annotations: dict[str, Any] = {}
    for track_uuid, group in sorted(groups.items(), key=lambda kv: (kv[1]["track_id"], kv[0])):
        boxes = sorted(group["boxes"], key=lambda b: b.frame_index)
        # HMIE labels a track once; the model allows per-box category_uri, so a
        # track whose boxes disagree would be silently relabelled to the first.
        # Warn rather than drop -- current loaders never produce this.
        labels = {b.category_uri for b in boxes}
        if len(labels) > 1:
            logger.warning(
                "track %s in sequence %s has boxes with %d different category_uris (%s); "
                "writing the first (%r) for the whole track",
                track_uuid,
                seq.video_id,
                len(labels),
                ", ".join(sorted(labels)),
                boxes[0].category_uri,
            )
        frames = []
        for box in boxes:
            left, top, width, height = box.bbox
            frame: dict[str, Any] = {
                "keyframeType": box.keyframe_type or "middle",
                "isInferredKeyframe": bool(box.is_inferred) if box.is_inferred is not None else False,
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "key": box.frame_index,
                "attributes": dict(box.attributes),
            }
            # Only emit a timestamp when the source had one; fabricating
            # frame_index / rate here would round-trip None as a real value.
            if box.timestamp is not None:
                frame["timestamp_secs"] = box.timestamp
            frames.append(frame)
        annotations[track_uuid] = {
            "label": boxes[0].category_uri,  # ontology URI -> re-resolves to the same category
            "geometry": "box",
            "frames": frames,
        }

    # Round-trip the video-level metadata the loader harvested, as the inverse
    # of _extract_video_meta: its top-level keys (origin_id, codec_name, width,
    # heigth, duration, nb_frames, ...) go back at the top level where the
    # loader reads them, and the merged ``global_attributes`` go back under
    # ``response.events`` (the loader re-merges them identically). The IR keeps
    # only the merged attribute dict, so a single synthetic event carries it.
    video_meta = dict(seq.video_meta)
    global_attributes = video_meta.pop("global_attributes", None)
    events: Any = [{"attributes": global_attributes}] if global_attributes else {}

    # Fall back to the VideoSequence's derived/probed fields for the top-level
    # HMIE metadata keys the loader reads, so dimensions/duration survive even
    # when they came from a video probe (require_video=True) rather than from
    # the source meta dict. Only fill a key the source meta did not already
    # carry, so an explicit source value always wins and we never emit a
    # duplicate dimension key (the loader reads "height" or "heigth").
    if seq.width is not None and "width" not in video_meta:
        video_meta["width"] = seq.width
    if seq.height is not None and not ({"height", "heigth"} & video_meta.keys()):
        video_meta["height"] = seq.height
    if seq.duration is not None and "duration" not in video_meta:
        video_meta["duration"] = seq.duration

    return {
        # Spread video_meta first so the structural keys below always win on any
        # name clash (none today, but keeps the writer robust to new meta keys).
        **video_meta,
        "task_id": f"roundtrip-{seq.video_id}",
        "status": seq.status or "completed",
        "type": "videoannotation",
        # Task-level metadata round-trips at the top level (loader reads it from
        # annotation.metadata). Emitted always; an empty dict reloads as {}.
        "metadata": dict(seq.metadata),
        "params": {
            "annotation_frame_rate": rate,
            "videoMetadata": {"video": {"fps": rate}},
        },
        "response": {"annotations": annotations, "events": events},
    }
