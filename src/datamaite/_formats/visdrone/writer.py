"""VisDrone video writer: serialise :class:`BoxTrackDataset` to official splits.

The writer emits one or more official VisDrone video split roots::

    <dest>/
        VisDrone2019-VID-train/ | VisDrone2019-MOT-train/
            sequences/<sequence>/0000001.jpg
            annotations/<sequence>.txt

Both VisDrone Object Detection in Videos (``variant="vid"``) and
Multi-Object Tracking (``variant="mot"``) use the same ten-column row shape;
the writer option chooses the output variant. Existing image-sequence inputs
copy source frames directly. Video-backed inputs are decoded to frame images
with OpenCV (``datamaite[fmv]``).
"""

from __future__ import annotations

import logging
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datamaite._types import DatasetFormat
from datamaite.model import BoxAnnotation, BoxTrackDataset, VideoSequence
from datamaite.writers import Writer, register_writer

logger = logging.getLogger(__name__)

_VALID_VARIANTS = frozenset({"auto", "vid", "mot"})
_VALID_SOURCES = frozenset({"gt", "det"})
_GENERATED_FRAME_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "validation": "val",
    "valid": "val",
    "val": "val",
    "test-dev": "test-dev",
    "test_dev": "test-dev",
    "testdev": "test-dev",
    "test-challenge": "test-challenge",
    "test_challenge": "test-challenge",
    "testchallenge": "test-challenge",
    "test": "test",
}
_SPLIT_ORDER = {"train": 0, "val": 1, "test-dev": 2, "test-challenge": 3, "test": 4}


@dataclass(frozen=True)
class _FrameOutput:
    """One frame image written under a VisDrone ``sequences/<name>`` directory."""

    frame_index: int
    path: Path
    width: int | None = None
    height: int | None = None


@register_writer
class VisDroneVideoWriter(Writer[BoxTrackDataset]):
    """Write a :class:`BoxTrackDataset` as VisDrone VID or MOT video split roots."""

    format = DatasetFormat.VISDRONE_VIDEO

    def write(
        self,
        dataset: BoxTrackDataset,
        dest: str | Path,
        *,
        variant: str = "auto",
        split: str = "train",
        preserve_splits: bool = True,
        annotation_source: str = "gt",
        image_extension: str = ".jpg",
        **_options: Any,
    ) -> list[Path]:
        """Serialise ``dataset`` under ``dest`` as VisDrone video and return files written.

        Parameters
        ----------
        variant
            ``"vid"`` writes Object Detection in Videos split roots,
            ``"mot"`` writes Multi-Object Tracking split roots, and ``"auto"``
            (default) preserves ``video_meta["variant"]`` when it is ``"vid"``
            or ``"mot"`` and otherwise falls back to ``"vid"``.
        split
            Fallback split name. Common aliases such as ``"validation"`` are
            normalised to VisDrone's ``"val"``.
        preserve_splits
            When True (default), a sequence with ``video_meta["split"]`` is
            written back to that split; otherwise the fallback ``split`` is used.
        annotation_source
            ``"gt"`` (default) requires positive target IDs for non-ignored
            objects. ``"det"`` allows non-positive detection IDs.
        image_extension
            Extension used for output frame images.

        Notes
        -----
        The output variant is user-configurable because VisDrone's VID and MOT
        video tasks share row columns but differ in split-root naming. Boxes
        whose frame, bbox, target id, or category id cannot be represented are
        dropped with a warning.
        """
        requested_variant = _validate_variant(variant)
        fallback_split = _normalize_split(split, field="split")
        source = _validate_source(annotation_source)
        extension = _validate_image_extension(image_extension)
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)

        written: list[Path] = []
        written_seen: set[Path] = set()
        used_names: dict[tuple[str, str], set[str]] = {}

        for seq in dataset.sequences:
            seq_variant = _variant_for_sequence(seq, requested=requested_variant)
            seq_split = _split_for_sequence(seq, fallback=fallback_split, preserve_splits=preserve_splits)
            split_root = dest / f"VisDrone2019-{seq_variant.upper()}-{seq_split}"
            sequence_name = _unique_sequence_name(seq, used_names.setdefault((seq_variant, seq_split), set()))
            frame_dir = split_root / "sequences" / sequence_name
            frame_dir.mkdir(parents=True, exist_ok=True)

            frame_outputs = _materialize_frames(
                seq,
                frame_dir=frame_dir,
                sequence_name=sequence_name,
                image_extension=extension,
                written=written,
                written_seen=written_seen,
            )
            if not frame_outputs:
                logger.warning("Skipping VisDrone sequence %s because no source frames could be written", sequence_name)
                continue

            rows = _annotation_rows(seq, frame_outputs, source=source, sequence_name=sequence_name)
            ann_path = split_root / "annotations" / f"{sequence_name}.txt"
            ann_path.parent.mkdir(parents=True, exist_ok=True)
            text = "".join(f"{','.join(_format_field(value) for value in row)}\n" for row in rows)
            ann_path.write_text(text, encoding="utf-8")
            _append_written(written, written_seen, ann_path)

        return written


def _append_written(written: list[Path], seen: set[Path], path: Path) -> None:
    if path not in seen:
        seen.add(path)
        written.append(path)


def _validate_variant(value: str) -> str:
    variant = str(value).strip().lower()
    if variant not in _VALID_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(_VALID_VARIANTS)!r}; got {value!r}")
    return variant


def _variant_for_sequence(seq: VideoSequence, *, requested: str) -> str:
    if requested != "auto":
        return requested
    raw = seq.video_meta.get("variant")
    variant = str(raw).strip().lower() if raw is not None else ""
    if variant in {"vid", "mot"}:
        return variant
    if raw is not None:
        logger.warning(
            "Sequence %s has non-standard VisDrone variant %r; writing it as 'vid'",
            _sequence_label(seq),
            raw,
        )
    return "vid"


def _normalize_split(value: str, *, field: str) -> str:
    raw = str(value).strip()
    normalized = _SPLIT_ALIASES.get(raw.lower().replace("_", "-"), raw)
    if _safe_name(normalized) != normalized:
        raise ValueError(f"{field} must be a safe VisDrone split name; got {value!r}")
    return normalized


def _split_for_sequence(seq: VideoSequence, *, fallback: str, preserve_splits: bool) -> str:
    if not preserve_splits:
        return fallback
    raw = seq.video_meta.get("split")
    if raw is None:
        return fallback
    try:
        return _normalize_split(str(raw), field="video_meta['split']")
    except ValueError:
        logger.warning(
            "Sequence %s has unsafe VisDrone split %r; writing it to fallback split %r",
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
    frame_dir: Path,
    sequence_name: str,
    image_extension: str,
    written: list[Path],
    written_seen: set[Path],
) -> dict[int, _FrameOutput]:
    if seq.frame_files or seq.frame_pattern is not None:
        return _copy_image_sequence(
            seq,
            frame_dir=frame_dir,
            sequence_name=sequence_name,
            image_extension=image_extension,
            written=written,
            written_seen=written_seen,
        )
    if seq.video_path:
        return _extract_video_frames(
            seq,
            frame_dir=frame_dir,
            sequence_name=sequence_name,
            image_extension=image_extension,
            written=written,
            written_seen=written_seen,
        )
    logger.warning("VisDrone writer needs source frames or a video file for sequence %s", sequence_name)
    return {}


def _copy_image_sequence(
    seq: VideoSequence,
    *,
    frame_dir: Path,
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
            logger.warning("Skipping missing VisDrone source frame for sequence %s: %s", sequence_name, source)
            continue
        dest_path = frame_dir / f"{frame_index + 1:07d}{image_extension}"
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
    frame_dir: Path,
    sequence_name: str,
    image_extension: str,
    written: list[Path],
    written_seen: set[Path],
) -> dict[int, _FrameOutput]:
    """Decode every frame of a video-backed sequence into VisDrone frame images."""
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Writing video-backed sequences to VisDrone requires OpenCV. Install it with: pip install datamaite[fmv]"
        ) from exc

    video_path = Path(seq.video_path or "")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video for VisDrone frame extraction: %s", video_path)
        return {}

    if seq.boxes and not seq.num_frames_exact:
        logger.warning(
            "Extracting VisDrone frames for sequence %s without an exact video frame count; "
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
            dest_path = frame_dir / f"{frame_index + 1:07d}{image_extension}"
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
        logger.warning("Video %s decoded zero frames for VisDrone output", video_path)
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
) -> list[list[float | int]]:
    rows: list[list[float | int]] = []
    for box in sorted(seq.boxes, key=lambda item: (item.frame_index, item.track_id, item.track_uuid)):
        row = _row_for_box(box, frame_outputs, source=source, sequence_name=sequence_name)
        if row is not None:
            rows.append(row)
    return rows


def _row_for_box(
    box: BoxAnnotation,
    frame_outputs: dict[int, _FrameOutput],
    *,
    source: str,
    sequence_name: str,
) -> list[float | int] | None:
    if box.frame_index not in frame_outputs:
        logger.warning(
            "Dropping VisDrone annotation for sequence %s frame %s because no frame image was written",
            sequence_name,
            box.frame_index,
        )
        return None
    bbox = _bbox_tuple(box.bbox)
    if bbox is None:
        logger.warning(
            "Dropping VisDrone annotation for sequence %s frame %s because bbox is malformed: %r",
            sequence_name,
            box.frame_index,
            box.bbox,
        )
        return None
    frame = box.frame_index + 1
    if frame <= 0:
        logger.warning(
            "Dropping VisDrone annotation for sequence %s because frame_index is negative: %s",
            sequence_name,
            box.frame_index,
        )
        return None

    category_id = _first_int(_coerce_int(box.attributes.get("visdrone_category_id")), _coerce_int(box.category_id))
    if category_id is None or category_id < 0:
        logger.warning(
            "Dropping VisDrone annotation for sequence %s frame %s because category id is negative or missing",
            sequence_name,
            box.frame_index,
        )
        return None
    score = _first_float(
        _coerce_finite_float(box.attributes.get("visdrone_score")),
        _coerce_finite_float(box.attributes.get("score")),
        _coerce_finite_float(box.attributes.get("confidence")),
        1.0,
    )
    target_id = _first_int(_coerce_int(box.attributes.get("visdrone_target_id")), _coerce_int(box.track_id))
    if target_id is None:
        target_id = -1
    if source == "gt" and target_id <= 0 and category_id != 0 and score > 0:
        logger.warning(
            "Dropping VisDrone GT annotation for sequence %s frame %s because target id is not positive",
            sequence_name,
            box.frame_index,
        )
        return None

    truncation = _coerce_int(box.attributes.get("truncation"))
    if truncation is None:
        truncation = -1
    occlusion = _coerce_int(box.attributes.get("occlusion"))
    if occlusion is None:
        occlusion = -1

    left, top, width, height = bbox
    return [
        frame,
        target_id,
        left,
        top,
        width,
        height,
        score,
        category_id,
        truncation,
        occlusion,
    ]


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
        parsed = _coerce_finite_float(value)
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


def _coerce_int(value: object) -> int | None:
    number = _coerce_finite_float(value)
    if number is None or not number.is_integer():
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


def _format_field(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.12g}"
