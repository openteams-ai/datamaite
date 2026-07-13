"""MOTChallenge writer: serialise :class:`BoxTrackDataset` to benchmark roots.

The writer emits the standard MOTChallenge sequence layout::

    <dest>/
        train|test/
            <sequence>/
                img1/000001.jpg
                gt/gt.txt | det/det.txt
                seqinfo.ini

MOTChallenge is an image-sequence format. Existing image-sequence inputs copy
source frames into ``img1``; video-backed inputs are decoded to frame images with
OpenCV (``datamaite[fmv]``). Annotation rows are written as either official
``gt.txt`` rows (frame, id, bbox, valid/confidence, class, visibility) or
``det.txt`` rows (frame, id, bbox, score, world coordinates), selected by the
``annotation_source`` writer option.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from datamaite._formats._coerce import coerce_finite_float, coerce_int
from datamaite._formats._fixed_taxonomy import ClassIdResolver, validate_class_map
from datamaite._types import DatasetFormat
from datamaite.model import BoxAnnotation, BoxTrackDataset, VideoSequence
from datamaite.writers import Writer, WriterCapabilities, register_writer

logger = logging.getLogger(__name__)

_VALID_SPLITS = frozenset({"train", "test"})
_VALID_SOURCES = frozenset({"gt", "det"})
_GENERATED_FRAME_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


@dataclass(frozen=True)
class _FrameOutput:
    """One frame image written under a MOTChallenge ``img1`` directory."""

    frame_index: int
    path: Path
    width: int | None = None
    height: int | None = None


@register_writer
class MotChallengeWriter(Writer[BoxTrackDataset]):
    """Write a :class:`BoxTrackDataset` as a MOTChallenge benchmark root."""

    format = DatasetFormat.MOTCHALLENGE
    capabilities: ClassVar[WriterCapabilities] = WriterCapabilities(forbids_dense_remap=True)

    def validate_options(self, **options: Any) -> None:
        """Validate options that can raise, before ``write()``'s destination policy runs (#55 Fix A1).

        Only validates options that are present so absent options never
        duplicate ``write()``'s own defaults; ``write()`` re-validates inline
        (cheap) after normalizing values, which also covers callers invoking
        this writer's ``.write()`` directly.
        """
        if "split" in options:
            _validate_split(options["split"], field="split")
        if "annotation_source" in options:
            _validate_source(options["annotation_source"])
        if "image_extension" in options:
            _validate_image_extension(options["image_extension"])
        validate_class_map(options.get("class_map"), minimum=1, format_label="MOTChallenge")

    def write(
        self,
        dataset: BoxTrackDataset,
        dest: str | Path,
        *,
        split: str = "train",
        preserve_splits: bool = True,
        annotation_source: str = "gt",
        image_extension: str = ".jpg",
        class_map: Mapping[str | int, int] | None = None,
        **_options: Any,
    ) -> list[Path]:
        """Serialise ``dataset`` under ``dest`` as MOTChallenge and return files written.

        Parameters
        ----------
        split
            Fallback split for sequences that do not carry MOT split metadata.
            Must be ``"train"`` or ``"test"``.
        preserve_splits
            When True (default), a sequence with ``video_meta["split"]`` set to
            ``"train"`` or ``"test"`` is written back to that split; otherwise
            the fallback ``split`` is used.
        annotation_source
            ``"gt"`` (default) writes ``gt/gt.txt`` rows with class and
            visibility columns. ``"det"`` writes ``det/det.txt`` rows with
            score and optional world-coordinate columns.
        image_extension
            Extension used for output frame images. Image-sequence inputs copy
            bytes into the selected extension; video-backed inputs are encoded
            through OpenCV using this extension.
        class_map
            Optional explicit mapping from source categories to MOTChallenge
            class ids for GT output. Keys are ``category_name`` strings
            (matched first) or ``category_id`` ints; values must be positive.
            A ``category_name`` key is assumed to identify a single source
            category; if the same name is seen with more than one distinct
            source ``category_id``, they are all silently collapsed onto that
            key's target id, and one aggregated WARNING flags the ambiguity.
            When provided it overrides both the ``mot_class_id`` attribute and
            the generic ``category_id`` fallback; boxes whose category is not
            in the map are dropped and reported in one aggregated warning.
            Without it, boxes lacking ``mot_class_id`` fall back to the
            generic ``category_id`` with one aggregated warning per write.
            ``det`` output has no class column and ignores ``class_map``.

        Notes
        -----
        Existing writer-owned files for the selected sequence names are
        overwritten. Video-backed sequences require OpenCV. Boxes whose frame,
        bbox, track id (for GT), or class id (for GT) cannot be represented are
        dropped with a warning.
        """
        fallback_split = _validate_split(split, field="split")
        source = _validate_source(annotation_source)
        extension = _validate_image_extension(image_extension)
        resolver = ClassIdResolver(
            format_label="MOTChallenge",
            attribute="mot_class_id",
            class_map=validate_class_map(class_map, minimum=1, format_label="MOTChallenge"),
            logger=logger,
            minimum=1,
        )
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        written_seen: set[Path] = set()
        used_names: dict[str, set[str]] = {}

        try:
            for seq in dataset.sequences:
                seq_split = _split_for_sequence(seq, fallback=fallback_split, preserve_splits=preserve_splits)
                sequence_name = _unique_sequence_name(seq, used_names.setdefault(seq_split, set()))
                seq_dir = dest / seq_split / sequence_name
                image_dir = seq_dir / "img1"
                image_dir.mkdir(parents=True, exist_ok=True)

                frame_outputs = _materialize_frames(
                    seq,
                    image_dir=image_dir,
                    sequence_name=sequence_name,
                    image_extension=extension,
                    written=written,
                    written_seen=written_seen,
                )
                if not frame_outputs:
                    logger.warning(
                        "Skipping MOTChallenge sequence %s because no source frames could be written", sequence_name
                    )
                    continue

                rows = _annotation_rows(
                    seq, frame_outputs, source=source, sequence_name=sequence_name, resolver=resolver
                )
                ann_path = seq_dir / source / f"{source}.txt"
                ann_path.parent.mkdir(parents=True, exist_ok=True)
                text = "".join(f"{','.join(_format_field(value) for value in row)}\n" for row in rows)
                ann_path.write_text(text, encoding="utf-8")
                _append_written(written, written_seen, ann_path)

                seqinfo_path = seq_dir / "seqinfo.ini"
                seqinfo_path.write_text(_seqinfo_text(seq, sequence_name, frame_outputs, extension), encoding="utf-8")
                _append_written(written, written_seen, seqinfo_path)
        finally:
            # Emit aggregated warnings even if a later sequence raises mid-write
            # (#55 B2): earlier sequences' output is already on disk, so the
            # resolver's fallback/unmapped/ambiguity tallies for them should
            # still surface rather than being silently lost.
            resolver.emit_warnings()
        return written


def _append_written(written: list[Path], seen: set[Path], path: Path) -> None:
    if path not in seen:
        seen.add(path)
        written.append(path)


def _validate_split(value: str, *, field: str) -> str:
    split = str(value).strip().lower()
    if split not in _VALID_SPLITS:
        raise ValueError(f"{field} must be one of {sorted(_VALID_SPLITS)!r}; got {value!r}")
    return split


def _split_for_sequence(seq: VideoSequence, *, fallback: str, preserve_splits: bool) -> str:
    if not preserve_splits:
        return fallback
    raw = seq.video_meta.get("split")
    if raw is None:
        return fallback
    try:
        return _validate_split(str(raw), field="video_meta['split']")
    except ValueError:
        logger.warning(
            "Sequence %s has non-standard MOTChallenge split %r; writing it to fallback split %r",
            _sequence_label(seq),
            raw,
            fallback,
        )
        return fallback


def _validate_source(value: str) -> str:
    source = str(value).strip().lower()
    if source not in _VALID_SOURCES:
        raise ValueError(f"annotation_source must be one of {sorted(_VALID_SOURCES)!r}; got {value!r}")
    return source


def _validate_image_extension(value: str) -> str:
    extension = str(value).strip().lower()
    if not extension.startswith("."):
        extension = f".{extension}"
    if extension not in _GENERATED_FRAME_EXTENSIONS:
        raise ValueError(f"image_extension must be one of {sorted(_GENERATED_FRAME_EXTENSIONS)!r}; got {value!r}")
    return extension


def _materialize_frames(
    seq: VideoSequence,
    *,
    image_dir: Path,
    sequence_name: str,
    image_extension: str,
    written: list[Path],
    written_seen: set[Path],
) -> dict[int, _FrameOutput]:
    if seq.frame_files or seq.frame_pattern is not None:
        return _copy_image_sequence(
            seq,
            image_dir=image_dir,
            sequence_name=sequence_name,
            image_extension=image_extension,
            written=written,
            written_seen=written_seen,
        )
    if seq.video_path:
        return _extract_video_frames(
            seq,
            image_dir=image_dir,
            sequence_name=sequence_name,
            image_extension=image_extension,
            written=written,
            written_seen=written_seen,
        )
    logger.warning("MOTChallenge writer needs source frames or a video file for sequence %s", sequence_name)
    return {}


def _copy_image_sequence(
    seq: VideoSequence,
    *,
    image_dir: Path,
    sequence_name: str,
    image_extension: str,
    written: list[Path],
    written_seen: set[Path],
) -> dict[int, _FrameOutput]:
    outputs: dict[int, _FrameOutput] = {}
    for frame_index in _image_sequence_frame_indices(seq):
        source = _safe_frame_path(seq, frame_index)
        if source is None:
            continue
        if not source.is_file():
            logger.warning("Skipping missing MOTChallenge source frame for sequence %s: %s", sequence_name, source)
            continue
        dest_path = image_dir / f"{frame_index + 1:06d}{image_extension}"
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        _copy_frame(source, dest_path)
        outputs[frame_index] = _FrameOutput(frame_index=frame_index, path=dest_path, width=seq.width, height=seq.height)
        _append_written(written, written_seen, dest_path)
    return outputs


def _image_sequence_frame_indices(seq: VideoSequence) -> list[int]:
    if seq.frame_files:
        return [idx for idx, frame_file in enumerate(seq.frame_files) if frame_file is not None]
    if seq.num_frames_exact and seq.num_frames is not None:
        return list(range(seq.num_frames))
    return sorted({box.frame_index for box in seq.boxes if box.frame_index >= 0})


def _safe_frame_path(seq: VideoSequence, frame_index: int) -> Path | None:
    try:
        return seq.frame_path(frame_index)
    except (IndexError, ValueError) as exc:
        logger.warning("Skipping frame %s for sequence %s: %s", frame_index, _sequence_label(seq), exc)
        return None


def _copy_frame(source: Path, dest: Path) -> None:
    try:
        same_file = source.resolve(strict=False) == dest.resolve(strict=False)
    except OSError:
        same_file = False
    if same_file:
        return
    shutil.copy2(source, dest)


def _extract_video_frames(  # pragma: no cover - optional OpenCV path covered by integration-style users.
    seq: VideoSequence,
    *,
    image_dir: Path,
    sequence_name: str,
    image_extension: str,
    written: list[Path],
    written_seen: set[Path],
) -> dict[int, _FrameOutput]:
    """Decode every frame of a video-backed sequence into MOT frame images."""
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Writing video-backed sequences to MOTChallenge requires OpenCV. "
            "Install it with: pip install datamaite[fmv]"
        ) from exc

    video_path = Path(seq.video_path or "")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video for MOTChallenge frame extraction: %s", video_path)
        return {}

    if seq.boxes and not seq.num_frames_exact:
        logger.warning(
            "Extracting MOTChallenge frames for sequence %s without an exact video frame count; "
            "annotation frame_index values may be label-space rather than decoded video-frame positions",
            sequence_name,
        )

    outputs: dict[int, _FrameOutput] = {}
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dest_path = image_dir / f"{frame_index + 1:06d}{image_extension}"
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(dest_path), frame):
                raise OSError(f"OpenCV failed to write frame image: {dest_path}")
            height, width = frame.shape[:2]
            outputs[frame_index] = _FrameOutput(
                frame_index=frame_index,
                path=dest_path,
                width=int(width),
                height=int(height),
            )
            _append_written(written, written_seen, dest_path)
            frame_index += 1
    finally:
        cap.release()

    if not outputs:
        logger.warning("Video %s decoded zero frames for MOTChallenge output", video_path)
    if seq.num_frames_exact and seq.num_frames is not None and seq.num_frames != len(outputs):
        logger.warning(
            "Video %s decoded %s frame(s), but sequence metadata expected %s",
            video_path,
            len(outputs),
            seq.num_frames,
        )
    return outputs


def _annotation_rows(
    seq: VideoSequence,
    frame_outputs: dict[int, _FrameOutput],
    *,
    source: str,
    sequence_name: str,
    resolver: ClassIdResolver,
) -> list[list[float | int]]:
    rows: list[list[float | int]] = []
    for box in sorted(seq.boxes, key=lambda item: (item.frame_index, item.track_id, item.track_uuid)):
        row = _row_for_box(box, frame_outputs, source=source, sequence_name=sequence_name, resolver=resolver)
        if row is not None:
            rows.append(row)
    return rows


def _row_for_box(
    box: BoxAnnotation,
    frame_outputs: dict[int, _FrameOutput],
    *,
    source: str,
    sequence_name: str,
    resolver: ClassIdResolver,
) -> list[float | int] | None:
    if box.frame_index not in frame_outputs:
        logger.warning(
            "Dropping MOTChallenge annotation for sequence %s frame %s because no frame image was written",
            sequence_name,
            box.frame_index,
        )
        return None
    bbox = _bbox_tuple(box.bbox)
    if bbox is None:
        logger.warning(
            "Dropping MOTChallenge annotation for sequence %s frame %s because bbox is malformed: %r",
            sequence_name,
            box.frame_index,
            box.bbox,
        )
        return None
    frame = box.frame_index + 1
    if frame <= 0:
        logger.warning(
            "Dropping MOTChallenge annotation for sequence %s because frame_index is negative: %s",
            sequence_name,
            box.frame_index,
        )
        return None

    left, top, width, height = bbox
    if source == "det":
        detection_id = _first_int(coerce_int(box.attributes.get("mot_track_id")), coerce_int(box.track_id))
        if detection_id is None:
            detection_id = -1
        return [
            frame,
            detection_id,
            left,
            top,
            width,
            height,
            _first_float(
                coerce_finite_float(box.attributes.get("score")),
                coerce_finite_float(box.attributes.get("confidence")),
                1.0,
            ),
            _first_float(coerce_finite_float(box.attributes.get("world_x")), -1.0),
            _first_float(coerce_finite_float(box.attributes.get("world_y")), -1.0),
            _first_float(coerce_finite_float(box.attributes.get("world_z")), -1.0),
        ]

    track_id = _first_int(coerce_int(box.attributes.get("mot_track_id")), coerce_int(box.track_id))
    if track_id is None or track_id <= 0:
        logger.warning(
            "Dropping MOTChallenge GT annotation for sequence %s frame %s because track id is not positive",
            sequence_name,
            box.frame_index,
        )
        return None
    resolved = resolver.resolve(box)
    if resolved.class_id is None:
        return None  # unmapped under class_map; the aggregated warning reports it
    class_id = resolved.class_id
    if class_id <= 0:
        logger.warning(
            "Dropping MOTChallenge GT annotation for sequence %s frame %s because class id is not positive",
            sequence_name,
            box.frame_index,
        )
        return None
    return [
        frame,
        track_id,
        left,
        top,
        width,
        height,
        _first_float(coerce_finite_float(box.attributes.get("confidence")), 1.0),
        class_id,
        _first_float(coerce_finite_float(box.attributes.get("visibility")), 1.0),
    ]


def _seqinfo_text(
    seq: VideoSequence,
    sequence_name: str,
    frame_outputs: dict[int, _FrameOutput],
    image_extension: str,
) -> str:
    frame_count = max(frame_outputs, default=-1) + 1
    width = _first_int(
        _coerce_positive_int(seq.width),
        *[_coerce_positive_int(frame.width) for frame in frame_outputs.values()],
        0,
    )
    height = _first_int(
        _coerce_positive_int(seq.height), *[_coerce_positive_int(frame.height) for frame in frame_outputs.values()], 0
    )
    fps = _coerce_positive_float(seq.fps) or 0.0
    return "\n".join(
        [
            "[Sequence]",
            f"name={sequence_name}",
            "imDir=img1",
            f"frameRate={_format_field(fps)}",
            f"seqLength={frame_count}",
            f"imWidth={width}",
            f"imHeight={height}",
            f"imExt={image_extension}",
            "",
        ]
    )


def _sequence_label(seq: VideoSequence) -> str:
    value = seq.video_meta.get("sequence_name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if seq.video_path:
        stem = Path(seq.video_path).stem
        if stem:
            return stem
    return f"sequence_{seq.video_id:06d}"


def _unique_sequence_name(seq: VideoSequence, used: set[str]) -> str:
    base = _safe_name(_sequence_label(seq))
    name = base
    if name in used:
        name = f"{base}-{seq.video_id}"
    suffix = 2
    while name in used:
        name = f"{base}-{seq.video_id}-{suffix}"
        suffix += 1
    used.add(name)
    return name


def _safe_name(value: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-_")
    return candidate or "sequence"


def _bbox_tuple(bbox: object) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, tuple | list) or len(bbox) != 4:
        return None
    values: list[float] = []
    for value in bbox:
        parsed = coerce_finite_float(value)
        if parsed is None:
            return None
        values.append(parsed)
    if values[2] <= 0 or values[3] <= 0:
        return None
    return values[0], values[1], values[2], values[3]


def _first_int(*values: int | None) -> int | None:
    for value in values:
        if value is not None:
            return value
    return None


def _first_float(*values: float | None) -> float:
    for value in values:
        if value is not None:
            return value
    return 0.0


def _coerce_positive_int(value: object) -> int | None:
    parsed = coerce_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _coerce_positive_float(value: object) -> float | None:
    parsed = coerce_finite_float(value)
    return parsed if parsed is not None and parsed > 0 else None


def _format_field(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.12g}"
