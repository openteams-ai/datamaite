"""TAO writer: serialise :class:`BoxTrackDataset` to an official TAO root.

The writer emits the standard TAO shape::

    <dest>/
        annotations/train.json | validation.json | test.json
        frames/<split>/<video_id>__<sequence>/000000.jpg

TAO is an image-sequence format. Image-sequence inputs copy their source frame
files. Video-backed inputs are decoded into JPEG frames with OpenCV, so writing
those sequences requires the optional ``datamaite[fmv]`` extra. Source TAO
``images.file_name`` values are preserved when possible; generated paths include
the unique output video id so sanitized sequence-name collisions cannot overwrite
frame data. The output JSON uses TAO's COCO-style top-level ``videos``,
``images``, ``tracks``, ``categories``, and ``annotations`` arrays.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from datamaite._types import DatasetFormat
from datamaite.model import BoxAnnotation, BoxTrackDataset, VideoSequence, category_name_from_uri
from datamaite.writers import Writer, register_writer

logger = logging.getLogger(__name__)

_STANDARD_SPLITS = frozenset({"train", "validation", "test"})
_ANNOTATION_FILENAMES = {
    "train": "train.json",
    "validation": "validation.json",
    "test": "test.json",
}
_GENERATED_FRAME_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
_TAO_INTERNAL_ATTRS = frozenset(
    {
        "source_format",
        "split",
        "tao_image_id",
        "tao_video_id",
        "tao_track_id",
        "tao_frame_index",
        "tao_annotation_id",
    }
)
_ANNOTATION_CORE_KEYS = frozenset({"id", "image_id", "track_id", "category_id", "bbox"})
_UNLABELED_CATEGORY_KEY = "generated/unlabeled"
_UNLABELED_CATEGORY_NAME = "unlabeled"


@dataclass(frozen=True)
class _FrameOutput:
    """One frame image written under ``dest / "frames"``."""

    frame_index: int
    file_name: str
    path: Path
    width: int | None = None
    height: int | None = None


@dataclass
class _IdAllocator:
    """Allocate deterministic integer IDs while preserving valid preferred IDs."""

    next_id: int = 1
    used: set[int] = field(default_factory=set)

    def reserve(self, preferred: int | None = None) -> int:
        if preferred is not None and preferred >= 0 and preferred not in self.used:
            self.used.add(preferred)
            if preferred >= self.next_id:
                self.next_id = preferred + 1
            return preferred
        while self.next_id in self.used:
            self.next_id += 1
        value = self.next_id
        self.used.add(value)
        self.next_id += 1
        return value


@dataclass(frozen=True)
class _CategoryTable:
    """TAO category records plus a lookup from model boxes to output IDs."""

    records: tuple[dict[str, Any], ...]
    ids_by_key: dict[str, int]

    @classmethod
    def build(cls, dataset: BoxTrackDataset) -> _CategoryTable:
        candidates: dict[str, tuple[int | None, str]] = {}

        for uri, raw_id in sorted(dataset.categories.items(), key=lambda item: (item[1], item[0])):
            key = _category_key(uri, raw_id, None)
            if key is None:
                continue
            candidates.setdefault(key, (_coerce_nonnegative_int(raw_id), _category_name(uri, raw_id, None)))

        for seq in dataset.sequences:
            for box in seq.boxes:
                key = _category_key(box.category_uri, box.category_id, box.category_name)
                if key is None:
                    continue
                candidates.setdefault(
                    key,
                    (
                        _coerce_nonnegative_int(box.category_id),
                        _category_name(box.category_uri, box.category_id, box.category_name),
                    ),
                )

        allocator = _IdAllocator()
        ids_by_key: dict[str, int] = {}
        records: list[dict[str, Any]] = []
        for key, (preferred_id, name) in sorted(
            candidates.items(),
            key=lambda item: (
                item[1][0] is None,
                item[1][0] if item[1][0] is not None else 10**12,
                item[1][1],
                item[0],
            ),
        ):
            category_id = allocator.reserve(preferred_id)
            ids_by_key[key] = category_id
            records.append({"id": category_id, "name": name})
        return cls(records=tuple(sorted(records, key=lambda record: record["id"])), ids_by_key=ids_by_key)

    def id_for_box(self, box: BoxAnnotation) -> int | None:
        key = _category_key(box.category_uri, box.category_id, box.category_name)
        return self.ids_by_key.get(key) if key is not None else None


@dataclass
class _SplitBuilder:
    """Accumulated TAO records for one output split JSON."""

    split: str
    categories: tuple[dict[str, Any], ...]
    videos: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    annotations: list[dict[str, Any]] = field(default_factory=list)
    tracks: dict[int, dict[str, Any]] = field(default_factory=dict)
    video_ids: _IdAllocator = field(default_factory=_IdAllocator)
    image_ids: _IdAllocator = field(default_factory=_IdAllocator)
    annotation_ids: _IdAllocator = field(default_factory=_IdAllocator)
    track_ids: _IdAllocator = field(default_factory=_IdAllocator)
    track_ids_by_key: dict[tuple[int, str], int] = field(default_factory=dict)

    def track_id_for(self, seq_index: int, box: BoxAnnotation) -> int:
        track_key = (seq_index, box.track_uuid or str(box.track_id))
        if track_key not in self.track_ids_by_key:
            self.track_ids_by_key[track_key] = self.track_ids.reserve(
                _first_int(
                    _coerce_nonnegative_int(box.attributes.get("tao_track_id")),
                    _coerce_nonnegative_int(box.track_id),
                )
            )
        return self.track_ids_by_key[track_key]

    def add_track(self, track_id: int, *, video_id: int, category_id: int, seq: VideoSequence) -> None:
        existing = self.tracks.get(track_id)
        if existing is None:
            self.tracks[track_id] = {"id": track_id, "video_id": video_id, "category_id": category_id}
            return
        if existing["video_id"] != video_id:
            logger.warning(
                "TAO track id %s is shared across output videos %s and %s; keeping the first track record",
                track_id,
                existing["video_id"],
                video_id,
            )
        if existing["category_id"] != category_id:
            logger.warning(
                "TAO track id %s in sequence %s has boxes with multiple categories (%s, %s); "
                "keeping the first track category",
                track_id,
                _sequence_name(seq),
                existing["category_id"],
                category_id,
            )

    def payload(self) -> dict[str, Any]:
        return {
            "videos": sorted(self.videos, key=lambda record: record["id"]),
            "images": sorted(self.images, key=lambda record: (record["video_id"], record["frame_index"], record["id"])),
            "tracks": sorted(self.tracks.values(), key=lambda record: record["id"]),
            "categories": list(self.categories),
            "annotations": sorted(self.annotations, key=lambda record: record["id"]),
        }


@register_writer
class TaoWriter(Writer[BoxTrackDataset]):
    """Write a :class:`BoxTrackDataset` as a TAO dataset root."""

    format = DatasetFormat.TAO

    def validate_options(self, **options: Any) -> None:
        """Validate options that can raise, before write()'s destination policy runs (#55 Fix A1).

        Mirrors the inline ``split`` / ``image_extension`` checks in
        ``write()``, but only for options that are present, so a
        ``mode="replace"`` clear never happens ahead of an option error.
        ``write()`` re-validates inline, which also covers direct
        ``Writer.write()`` calls.
        """
        if "split" in options:
            _validate_split(options["split"], field="split")
        if "image_extension" in options:
            _validate_image_extension(options["image_extension"])

    def write(
        self,
        dataset: BoxTrackDataset,
        dest: str | Path,
        *,
        split: str = "train",
        preserve_splits: bool = True,
        image_extension: str = ".jpg",
        **_options: Any,
    ) -> list[Path]:
        """Serialise ``dataset`` under ``dest`` as TAO and return files written.

        Parameters
        ----------
        split
            Fallback split for sequences that do not carry TAO split metadata.
            Must be one of ``"train"``, ``"validation"``, or ``"test"``.
        preserve_splits
            When True (default), a sequence with ``video_meta["split"]`` set to
            a standard TAO split is written back to that split; otherwise the
            fallback ``split`` is used.
        image_extension
            Extension used for frames extracted from video-backed sequences.
            Image-sequence inputs preserve their source extension.

        Notes
        -----
        Image-sequence inputs copy frames. Video-backed inputs are decoded into
        images with OpenCV and therefore require ``datamaite[fmv]``. Video
        extraction currently writes every decoded frame (not only annotated or
        TAO-sampled frames), so long sparse videos can produce large outputs.
        Existing standard annotation JSONs in ``dest/annotations`` are
        overwritten; existing frames not referenced by the new annotations are
        left in place and ignored by the loader.
        """
        fallback_split = _validate_split(split, field="split")
        generated_extension = _validate_image_extension(image_extension)
        dest = Path(dest)
        frames_root = dest / "frames"
        annotations_dir = dest / "annotations"
        frames_root.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)
        _clear_standard_annotations(annotations_dir)

        categories = _CategoryTable.build(dataset)
        splits: dict[str, _SplitBuilder] = {}
        written: list[Path] = []
        written_seen: set[Path] = set()

        for seq_index, seq in enumerate(dataset.sequences):
            seq_split = _split_for_sequence(seq, fallback=fallback_split, preserve_splits=preserve_splits)
            builder = splits.setdefault(seq_split, _SplitBuilder(split=seq_split, categories=categories.records))
            _write_sequence(
                seq_index,
                seq,
                builder=builder,
                categories=categories,
                dest=dest,
                split=seq_split,
                generated_extension=generated_extension,
                written=written,
                written_seen=written_seen,
            )

        if not splits:
            splits[fallback_split] = _SplitBuilder(split=fallback_split, categories=categories.records)

        for split_name, builder in sorted(splits.items()):
            path = annotations_dir / _ANNOTATION_FILENAMES[split_name]
            path.write_text(json.dumps(builder.payload(), indent=2), encoding="utf-8")
            _append_written(written, written_seen, path)
        return written


def _write_sequence(
    seq_index: int,
    seq: VideoSequence,
    *,
    builder: _SplitBuilder,
    categories: _CategoryTable,
    dest: Path,
    split: str,
    generated_extension: str,
    written: list[Path],
    written_seen: set[Path],
) -> None:
    sequence_name = _sequence_name(seq)
    video_id = builder.video_ids.reserve(_preferred_video_id(seq))
    frame_outputs = _materialize_frames(
        seq,
        dest=dest,
        split=split,
        video_id=video_id,
        sequence_name=sequence_name,
        generated_extension=generated_extension,
    )
    if not frame_outputs:
        logger.warning("Skipping TAO sequence %s because no source frames could be written", sequence_name)
        return

    builder.videos.append(_video_record(seq, video_id=video_id, name=sequence_name))
    frame_to_image_id = _add_images(builder, seq, frame_outputs, video_id=video_id, written=written, seen=written_seen)
    _add_annotations(builder, seq_index, seq, categories, frame_to_image_id, video_id=video_id)


def _add_images(
    builder: _SplitBuilder,
    seq: VideoSequence,
    frame_outputs: dict[int, _FrameOutput],
    *,
    video_id: int,
    written: list[Path],
    seen: set[Path],
) -> dict[int, int]:
    frame_to_image_id: dict[int, int] = {}
    for frame_index, frame in sorted(frame_outputs.items()):
        image_id = builder.image_ids.reserve(_preferred_image_id(seq, frame_index))
        frame_to_image_id[frame_index] = image_id
        builder.images.append(_image_record(seq, frame, image_id=image_id, video_id=video_id))
        _append_written(written, seen, frame.path)
    return frame_to_image_id


def _image_record(seq: VideoSequence, frame: _FrameOutput, *, image_id: int, video_id: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": image_id,
        "video_id": video_id,
        "file_name": frame.file_name,
        "frame_index": frame.frame_index,
    }
    width = frame.width or seq.width
    height = frame.height or seq.height
    if width is not None:
        record["width"] = int(width)
    if height is not None:
        record["height"] = int(height)
    return record


def _add_annotations(
    builder: _SplitBuilder,
    seq_index: int,
    seq: VideoSequence,
    categories: _CategoryTable,
    frame_to_image_id: dict[int, int],
    *,
    video_id: int,
) -> None:
    sequence_name = _sequence_name(seq)
    for box in sorted(seq.boxes, key=lambda item: (item.frame_index, item.track_id, item.track_uuid)):
        annotation = _annotation_record(
            builder,
            seq_index,
            seq,
            box,
            categories,
            frame_to_image_id,
            video_id=video_id,
            sequence_name=sequence_name,
        )
        if annotation is not None:
            builder.annotations.append(annotation)


def _annotation_record(
    builder: _SplitBuilder,
    seq_index: int,
    seq: VideoSequence,
    box: BoxAnnotation,
    categories: _CategoryTable,
    frame_to_image_id: dict[int, int],
    *,
    video_id: int,
    sequence_name: str,
) -> dict[str, Any] | None:
    image_id = frame_to_image_id.get(box.frame_index)
    category_id = categories.id_for_box(box)
    bbox = _bbox_list(box.bbox)
    if image_id is None:
        logger.warning(
            "Dropping TAO annotation for sequence %s frame %s because no frame image was written",
            sequence_name,
            box.frame_index,
        )
        return None
    if category_id is None:
        logger.warning(
            "Dropping TAO annotation for sequence %s frame %s because it has no usable category",
            sequence_name,
            box.frame_index,
        )
        return None
    if bbox is None:
        logger.warning(
            "Dropping TAO annotation for sequence %s frame %s because bbox is malformed: %r",
            sequence_name,
            box.frame_index,
            box.bbox,
        )
        return None
    track_id = builder.track_id_for(seq_index, box)
    builder.add_track(track_id, video_id=video_id, category_id=category_id, seq=seq)
    annotation = {
        "id": builder.annotation_ids.reserve(_coerce_nonnegative_int(box.attributes.get("tao_annotation_id"))),
        "image_id": image_id,
        "track_id": track_id,
        "category_id": category_id,
        "bbox": bbox,
    }
    annotation.update(_annotation_attributes(box))
    return annotation


def _append_written(written: list[Path], seen: set[Path], path: Path) -> None:
    if path not in seen:
        seen.add(path)
        written.append(path)


def _clear_standard_annotations(annotations_dir: Path) -> None:
    for filename in set(_ANNOTATION_FILENAMES.values()) | {"test_without_annotations.json"}:
        path = annotations_dir / filename
        if path.exists():
            path.unlink()


def _validate_split(value: str, *, field: str) -> str:
    split = str(value).strip().lower()
    if split == "test_without_annotations":
        return "test"
    if split not in _STANDARD_SPLITS:
        raise ValueError(f"{field} must be one of {sorted(_STANDARD_SPLITS)!r}; got {value!r}")
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
            "Sequence %s has non-standard TAO split %r; writing it to fallback split %r",
            _sequence_name(seq),
            raw,
            fallback,
        )
        return fallback


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
    dest: Path,
    split: str,
    video_id: int,
    sequence_name: str,
    generated_extension: str,
) -> dict[int, _FrameOutput]:
    if seq.frame_files or seq.frame_pattern is not None:
        return _copy_image_sequence(seq, dest=dest, split=split, video_id=video_id, sequence_name=sequence_name)
    if seq.video_path:
        return _extract_video_frames(
            seq,
            dest=dest,
            split=split,
            video_id=video_id,
            sequence_name=sequence_name,
            generated_extension=generated_extension,
        )
    logger.warning("TAO writer needs source frames or a video file for sequence %s", sequence_name)
    return {}


def _copy_image_sequence(
    seq: VideoSequence,
    *,
    dest: Path,
    split: str,
    video_id: int,
    sequence_name: str,
) -> dict[int, _FrameOutput]:
    outputs: dict[int, _FrameOutput] = {}
    for frame_index in _image_sequence_frame_indices(seq):
        source = _safe_frame_path(seq, frame_index)
        if source is None:
            continue
        if not source.is_file():
            logger.warning("Skipping missing TAO source frame for sequence %s: %s", sequence_name, source)
            continue
        file_name = _source_tao_file_name(seq, source, split=split) or _generated_frame_file_name(
            split,
            video_id,
            sequence_name,
            frame_index,
            source.suffix or ".jpg",
        )
        dest_path = dest / "frames" / file_name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        _copy_frame(source, dest_path)
        outputs[frame_index] = _FrameOutput(
            frame_index=frame_index,
            file_name=file_name,
            path=dest_path,
            width=seq.width,
            height=seq.height,
        )
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
        logger.warning("Skipping frame %s for sequence %s: %s", frame_index, _sequence_name(seq), exc)
        return None


def _copy_frame(source: Path, dest: Path) -> None:
    try:
        same_file = source.resolve(strict=False) == dest.resolve(strict=False)
    except OSError:
        same_file = False
    if same_file:
        return
    shutil.copy2(source, dest)


def _extract_video_frames(
    seq: VideoSequence,
    *,
    dest: Path,
    split: str,
    video_id: int,
    sequence_name: str,
    generated_extension: str,
) -> dict[int, _FrameOutput]:
    """Decode every frame of a video-backed sequence into TAO frame images.

    Known limitation: TAO datasets are often sparsely sampled, but this emits
    every decoded video frame so output ``frame_index`` values stay direct
    decoded-frame positions.
    """
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Writing video-backed sequences to TAO requires OpenCV. Install it with: pip install datamaite[fmv]"
        ) from exc

    video_path = Path(seq.video_path or "")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video for TAO frame extraction: %s", video_path)
        return {}

    if seq.boxes and not seq.num_frames_exact:
        logger.warning(
            "Extracting TAO frames for sequence %s without an exact video frame count; "
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
            file_name = _generated_frame_file_name(split, video_id, sequence_name, frame_index, generated_extension)
            dest_path = dest / "frames" / file_name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(dest_path), frame):
                raise OSError(f"OpenCV failed to write frame image: {dest_path}")
            height, width = frame.shape[:2]
            outputs[frame_index] = _FrameOutput(
                frame_index=frame_index,
                file_name=file_name,
                path=dest_path,
                width=int(width),
                height=int(height),
            )
            frame_index += 1
    finally:
        cap.release()

    if not outputs:
        logger.warning("Video %s decoded zero frames for TAO output", video_path)
    if seq.num_frames_exact and seq.num_frames is not None and seq.num_frames != len(outputs):
        logger.warning(
            "Video %s decoded %s frame(s), but sequence metadata expected %s",
            video_path,
            len(outputs),
            seq.num_frames,
        )
    return outputs


def _source_tao_file_name(seq: VideoSequence, source: Path, *, split: str) -> str | None:
    if seq.video_meta.get("format") != "tao":
        return None
    annotation_file = seq.video_meta.get("annotation_file") or seq.annotation_path
    if not annotation_file:
        return None
    source_root = Path(str(annotation_file)).parent.parent
    try:
        rel = source.resolve(strict=False).relative_to((source_root / "frames").resolve(strict=False))
    except (OSError, ValueError):
        return None
    if rel.parts and rel.parts[0] in _STANDARD_SPLITS and rel.parts[0] != split:
        return None
    file_name = rel.as_posix()
    return file_name if _safe_tao_file_name(file_name) else None


def _safe_tao_file_name(file_name: str) -> bool:
    if "\\" in file_name:
        return False
    posix = PurePosixPath(file_name)
    return (
        not posix.is_absolute()
        and bool(posix.parts)
        and all(part not in {"", ".", ".."} and ":" not in part for part in posix.parts)
    )


def _generated_frame_file_name(split: str, video_id: int, sequence_name: str, frame_index: int, suffix: str) -> str:
    suffix = suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
    if not re.fullmatch(r"\.[a-z0-9]+", suffix):
        suffix = ".jpg"
    first_part, *remaining_parts = _safe_sequence_parts(sequence_name)
    sequence_dir = f"{video_id}__{first_part}"
    return PurePosixPath(split, sequence_dir, *remaining_parts, f"{frame_index:06d}{suffix}").as_posix()


def _safe_sequence_parts(sequence_name: str) -> tuple[str, ...]:
    raw_parts = [part for part in str(sequence_name).replace("\\", "/").split("/") if part]
    safe_parts: list[str] = []
    for raw in raw_parts:
        part = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
        if part and part not in {".", ".."}:
            safe_parts.append(part)
    return tuple(safe_parts or ("sequence",))


def _preferred_video_id(seq: VideoSequence) -> int | None:
    return _first_int(
        _coerce_nonnegative_int(seq.video_meta.get("source_video_id")),
        _coerce_nonnegative_int(seq.video_id),
    )


def _preferred_image_id(seq: VideoSequence, frame_index: int) -> int | None:
    ids = {
        image_id
        for box in seq.boxes
        if box.frame_index == frame_index
        for image_id in [_coerce_nonnegative_int(box.attributes.get("tao_image_id"))]
        if image_id is not None
    }
    return next(iter(ids)) if len(ids) == 1 else None


def _video_record(seq: VideoSequence, *, video_id: int, name: str) -> dict[str, Any]:
    raw = seq.video_meta.get("video")
    record = _json_object(raw) if isinstance(raw, dict) else {}
    record.update({"id": video_id, "name": name})
    if seq.width is not None:
        record["width"] = int(seq.width)
    if seq.height is not None:
        record["height"] = int(seq.height)
    if seq.fps and seq.fps > 0:
        record["fps"] = float(seq.fps)
    return record


def _sequence_name(seq: VideoSequence) -> str:
    value = seq.video_meta.get("sequence_name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    raw_video = seq.video_meta.get("video")
    if isinstance(raw_video, dict):
        raw_name = raw_video.get("name") or raw_video.get("file_name")
        if isinstance(raw_name, str) and raw_name.strip():
            return raw_name.strip()
    return f"sequence_{seq.video_id:06d}"


def _bbox_list(bbox: object) -> list[float] | None:
    if not isinstance(bbox, tuple | list) or len(bbox) != 4:
        return None
    values: list[float] = []
    for value in bbox:
        parsed = _coerce_finite_float(value)
        if parsed is None:
            return None
        values.append(parsed)
    if values[2] <= 0 or values[3] <= 0:
        return None
    return values


def _annotation_attributes(box: BoxAnnotation) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for key, value in sorted(box.attributes.items()):
        if key in _TAO_INTERNAL_ATTRS or key in _ANNOTATION_CORE_KEYS:
            continue
        safe = _json_safe(value)
        if safe is not _UNSAFE:
            attrs[key] = safe
        else:
            logger.warning("Dropping non-JSON-serializable TAO annotation attribute %r", key)
    return attrs


_UNSAFE = object()


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _UNSAFE
    if isinstance(value, list | tuple):
        items = [_json_safe(item) for item in value]
        return _UNSAFE if any(item is _UNSAFE for item in items) else items
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                return _UNSAFE
            safe = _json_safe(item)
            if safe is _UNSAFE:
                return _UNSAFE
            result[key] = safe
        return result
    return _UNSAFE


def _json_object(value: dict[str, Any]) -> dict[str, Any]:
    safe = _json_safe(value)
    return dict(safe) if isinstance(safe, dict) else {}


def _category_key(category_uri: str | None, category_id: object, category_name: str | None) -> str | None:
    if category_uri:
        return category_uri
    parsed_id = _coerce_nonnegative_int(category_id)
    if parsed_id is not None:
        return f"generated/category_{parsed_id}/{category_name or f'category_{parsed_id}'}"
    if category_name:
        return f"generated/name/{category_name}"
    return _UNLABELED_CATEGORY_KEY


def _category_name(category_uri: str | None, category_id: object, category_name: str | None) -> str:
    if category_name:
        return category_name
    if category_uri:
        name = category_name_from_uri(category_uri)
        if name:
            return name
    parsed_id = _coerce_nonnegative_int(category_id)
    return f"category_{parsed_id}" if parsed_id is not None else _UNLABELED_CATEGORY_NAME


def _first_int(*values: int | None) -> int | None:
    for value in values:
        if value is not None:
            return value
    return None


def _coerce_nonnegative_int(value: object) -> int | None:
    number = _coerce_finite_float(value)
    if number is None or not number.is_integer() or number < 0:
        return None
    return int(number)


def _coerce_finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
