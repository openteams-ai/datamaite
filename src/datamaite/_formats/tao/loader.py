"""TAO (Tracking Any Object) dataset loader.

Official TAO annotations are COCO-style JSON files with top-level ``videos``,
``images``, ``annotations``, ``tracks``, and ``categories`` arrays. This loader
reads standard TAO dataset roots::

    <root>/
        annotations/train.json
        annotations/validation.json
        annotations/test.json or annotations/test_without_annotations.json
        frames/...

The loader is intentionally best-effort, like the existing HMIE and
MOTChallenge loaders: malformed records are skipped with warnings while the
rest of the dataset continues to load.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from datamaite._types import DatasetFormat
from datamaite.loaders import Loader, register_loader
from datamaite.model import BoxAnnotation, BoxTrackDataset, VideoSequence

logger = logging.getLogger(__name__)

_STANDARD_ANNOTATION_FILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("train", ("train.json",)),
    ("validation", ("validation.json",)),
    ("test", ("test.json", "test_without_annotations.json")),
)
_CORE_ANNOTATION_KEYS = frozenset({"id", "image_id", "track_id", "category_id", "bbox"})
_MAX_DECLARED_FRAME_TABLE_LENGTH = 1_000_000


@dataclass(frozen=True)
class _TaoVideo:
    """Parsed TAO video metadata."""

    id: int
    name: str
    width: int | None = None
    height: int | None = None
    fps: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _TaoImage:
    """Parsed TAO image/frame metadata."""

    id: int
    video_id: int
    file_name: str
    path: Path
    frame_index: int | None = None
    width: int | None = None
    height: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _FrameTables:
    """Model-frame mappings derived from TAO images."""

    frame_by_image_id: dict[int, int]
    files_by_video_id: dict[int, tuple[str | None, ...]]


@dataclass(frozen=True)
class _TaoContext:
    """Shared indexes for one TAO annotation file."""

    root: Path
    split: str
    annotation_path: Path
    categories: dict[str, int]
    category_names: dict[int, str]
    videos: dict[int, _TaoVideo]
    images: dict[int, _TaoImage]
    images_by_video: dict[int, list[_TaoImage]]
    tracks: dict[int, dict[str, Any]]
    frame_tables: _FrameTables


@register_loader
class TaoLoader(Loader):
    """Loader for official TAO dataset roots."""

    format = DatasetFormat.TAO

    def load(
        self,
        root: str | Path,
        *,
        probe_images: bool = False,
        **_: Any,
    ) -> BoxTrackDataset:
        """Read a TAO dataset root into :class:`BoxTrackDataset`.

        Parameters
        ----------
        root
            Standard TAO root. The loader auto-discovers
            ``annotations/train.json``, ``annotations/validation.json``, and
            either ``annotations/test.json`` or the official
            ``annotations/test_without_annotations.json`` when present.
        probe_images
            When True, use OpenCV (``pip install datamaite[fmv]``) to read
            one frame image per sequence and fill/override dimensions.

        Returns
        -------
        BoxTrackDataset
            Loaded TAO videos as image-sequence-backed ``VideoSequence``
            records. Empty when no standard TAO annotation files are found or
            all discovered files are malformed.
        """
        root = Path(root)
        annotation_files = _annotation_files(root)
        if not annotation_files:
            logger.warning("TAO root has no standard annotation files under %s", root / "annotations")
            return BoxTrackDataset(sequences=(), categories={})

        categories: dict[str, int] = {}
        category_names: dict[int, str] = {}
        sequences: list[VideoSequence] = []
        for split, annotation_path in annotation_files:
            data = _read_json(annotation_path)
            if data is None:
                continue
            loaded = _load_annotation_file(
                root,
                split=split,
                annotation_path=annotation_path,
                data=data,
                categories=categories,
                category_names=category_names,
                probe_images=probe_images,
                next_video_id=len(sequences),
            )
            sequences.extend(loaded)

        logger.info("Loaded %d TAO sequence(s), %d categories from %s", len(sequences), len(categories), root)
        return BoxTrackDataset(sequences=tuple(sequences), categories=categories)


def load_tao(root: str | Path, *, probe_images: bool = False) -> BoxTrackDataset:
    """Load an official TAO dataset root.

    Equivalent to ``datamaite.load(root, dataset_format="tao", ...)``. See
    :meth:`TaoLoader.load` for parameter semantics.
    """
    return TaoLoader().load(root, probe_images=probe_images)


def _annotation_files(root: Path) -> list[tuple[str, Path]]:
    """Return standard TAO annotation files in deterministic split order."""
    annotation_dir = root / "annotations"
    files: list[tuple[str, Path]] = []
    for split, filenames in _STANDARD_ANNOTATION_FILES:
        for filename in filenames:
            path = annotation_dir / filename
            if path.is_file():
                files.append((split, path))
                break
    return files


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a TAO JSON file with stdlib JSON, isolated for future streaming swap."""
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        logger.warning("Could not read TAO annotation file %s: %s", path, exc)
        return None
    if not isinstance(data, dict):
        logger.warning("Skipping TAO annotation file whose top-level JSON is not an object: %s", path)
        return None
    return data


def _load_annotation_file(
    root: Path,
    *,
    split: str,
    annotation_path: Path,
    data: dict[str, Any],
    categories: dict[str, int],
    category_names: dict[int, str],
    probe_images: bool,
    next_video_id: int,
) -> list[VideoSequence]:
    """Load one split annotation JSON into sequence records."""
    _index_categories(data.get("categories"), category_names, categories, annotation_path=annotation_path)
    videos = _index_videos(data.get("videos"), annotation_path=annotation_path)
    images, images_by_video = _index_images(root, data.get("images"), annotation_path=annotation_path)
    tracks = _index_tracks(data.get("tracks"), annotation_path=annotation_path)
    frame_tables = _build_frame_tables(images_by_video, annotation_path=annotation_path)
    context = _TaoContext(
        root=root,
        split=split,
        annotation_path=annotation_path,
        categories=categories,
        category_names=category_names,
        videos=videos,
        images=images,
        images_by_video=images_by_video,
        tracks=tracks,
        frame_tables=frame_tables,
    )
    boxes_by_video = _boxes_by_video(data.get("annotations"), context)

    sequences: list[VideoSequence] = []
    for video_id in _video_ids_to_load(videos, images_by_video):
        seq = _build_sequence(
            context,
            video_id=video_id,
            boxes=boxes_by_video.get(video_id, []),
            probe_images=probe_images,
            next_video_id=next_video_id + len(sequences),
        )
        if seq is not None:
            sequences.append(seq)
    return sequences


def _index_categories(
    raw_categories: object,
    category_names: dict[int, str],
    categories: dict[str, int],
    *,
    annotation_path: Path,
) -> None:
    """Index TAO categories by raw sparse category id."""
    for raw in _as_list(raw_categories, field="categories", path=annotation_path):
        if not isinstance(raw, dict):
            logger.warning("Skipping malformed TAO category in %s", annotation_path)
            continue
        category_id = _parse_int(raw.get("id"))
        name = raw.get("name")
        if category_id is None or not isinstance(name, str) or not name.strip():
            logger.warning("Skipping TAO category with missing/invalid id or name in %s", annotation_path)
            continue
        name = name.strip()
        existing = category_names.get(category_id)
        if existing is not None and existing != name:
            logger.warning(
                "TAO category id %s has conflicting names %r and %r; keeping first",
                category_id,
                existing,
                name,
            )
            continue
        category_names[category_id] = name
        categories.setdefault(_category_uri(category_id, name), category_id)


def _index_videos(raw_videos: object, *, annotation_path: Path) -> dict[int, _TaoVideo]:
    """Index valid TAO video records by id."""
    videos: dict[int, _TaoVideo] = {}
    for raw in _as_list(raw_videos, field="videos", path=annotation_path):
        if not isinstance(raw, dict):
            logger.warning("Skipping malformed TAO video in %s", annotation_path)
            continue
        video_id = _parse_int(raw.get("id"))
        if video_id is None:
            logger.warning("Skipping TAO video with missing/invalid id in %s", annotation_path)
            continue
        name = str(raw.get("name") or raw.get("file_name") or video_id)
        videos[video_id] = _TaoVideo(
            id=video_id,
            name=name,
            width=_coerce_positive_int(raw.get("width")),
            height=_coerce_positive_int(raw.get("height")),
            fps=_coerce_positive_float(raw.get("fps")) or 0.0,
            raw=dict(raw),
        )
    return videos


def _index_tracks(raw_tracks: object, *, annotation_path: Path) -> dict[int, dict[str, Any]]:
    """Index valid TAO track records by id."""
    tracks: dict[int, dict[str, Any]] = {}
    for raw in _as_list(raw_tracks, field="tracks", path=annotation_path):
        if not isinstance(raw, dict):
            logger.warning("Skipping malformed TAO track in %s", annotation_path)
            continue
        track_id = _parse_int(raw.get("id"))
        if track_id is None:
            logger.warning("Skipping TAO track with missing/invalid id in %s", annotation_path)
            continue
        tracks[track_id] = dict(raw)
    return tracks


def _index_images(
    root: Path,
    raw_images: object,
    *,
    annotation_path: Path,
) -> tuple[dict[int, _TaoImage], dict[int, list[_TaoImage]]]:
    """Index valid TAO image records by id and video id."""
    images: dict[int, _TaoImage] = {}
    by_video: dict[int, list[_TaoImage]] = defaultdict(list)
    for raw in _as_list(raw_images, field="images", path=annotation_path):
        image = _parse_image(root, raw, annotation_path=annotation_path)
        if image is None:
            continue
        images[image.id] = image
        by_video[image.video_id].append(image)
    return images, dict(by_video)


def _parse_image(root: Path, raw: object, *, annotation_path: Path) -> _TaoImage | None:
    """Parse one TAO image record."""
    if not isinstance(raw, dict):
        logger.warning("Skipping malformed TAO image in %s", annotation_path)
        return None
    image_id = _parse_int(raw.get("id"))
    video_id = _parse_int(raw.get("video_id"))
    file_name = raw.get("file_name")
    if image_id is None or video_id is None or not isinstance(file_name, str) or not file_name.strip():
        logger.warning("Skipping TAO image with missing/invalid id, video_id, or file_name in %s", annotation_path)
        return None
    image_path = _resolve_image_path(root, file_name, annotation_path=annotation_path)
    if image_path is None:
        return None
    frame_index = _parse_int(raw.get("frame_index"))
    if frame_index is not None and frame_index < 0:
        logger.warning("Ignoring negative TAO frame_index for image %s in %s", image_id, annotation_path)
        frame_index = None
    return _TaoImage(
        id=image_id,
        video_id=video_id,
        file_name=file_name,
        path=image_path,
        frame_index=frame_index,
        width=_coerce_positive_int(raw.get("width")),
        height=_coerce_positive_int(raw.get("height")),
        raw=dict(raw),
    )


def _build_frame_tables(
    images_by_video: Mapping[int, list[_TaoImage]],
    *,
    annotation_path: Path,
) -> _FrameTables:
    """Build model 0-based frame indexes and explicit frame-file tables."""
    frame_by_image_id: dict[int, int] = {}
    files_by_video_id: dict[int, tuple[str | None, ...]] = {}
    for video_id, images in images_by_video.items():
        use_declared = images and all(image.frame_index is not None for image in images)
        if use_declared:
            files = _declared_frame_files(video_id, images, frame_by_image_id, annotation_path=annotation_path)
        else:
            if any(image.frame_index is not None for image in images):
                logger.warning("TAO video %s mixes present/missing frame_index values; deriving order", video_id)
            files = _derived_frame_files(images, frame_by_image_id)
        files_by_video_id[video_id] = files
    return _FrameTables(frame_by_image_id=frame_by_image_id, files_by_video_id=files_by_video_id)


def _declared_frame_files(
    video_id: int,
    images: list[_TaoImage],
    frame_by_image_id: dict[int, int],
    *,
    annotation_path: Path,
) -> tuple[str | None, ...]:
    """Build frame files from explicit TAO ``image.frame_index`` values."""
    max_frame = max(image.frame_index or 0 for image in images)
    if max_frame + 1 > _MAX_DECLARED_FRAME_TABLE_LENGTH:
        logger.warning(
            "TAO video %s has maximum frame_index %s in %s; deriving dense frame order instead of "
            "allocating a sparse frame table",
            video_id,
            max_frame,
            annotation_path,
        )
        return _derived_frame_files_by_declared_order(images, frame_by_image_id)

    files: list[str | None] = [None] * (max_frame + 1)
    for image in sorted(images, key=lambda item: (item.frame_index or 0, item.file_name, item.id)):
        frame_index = image.frame_index
        if frame_index is None:
            continue
        frame_by_image_id[image.id] = frame_index
        if files[frame_index] is None:
            files[frame_index] = str(image.path)
        else:
            logger.warning(
                "TAO video %s has duplicate frame_index %s in %s; keeping first frame file",
                video_id,
                frame_index,
                annotation_path,
            )
    return tuple(files)


def _derived_frame_files_by_declared_order(
    images: list[_TaoImage], frame_by_image_id: dict[int, int]
) -> tuple[str | None, ...]:
    """Build dense frame files sorted by declared frame_index when the source is too sparse."""
    files: list[str | None] = []
    for frame_index, image in enumerate(
        sorted(images, key=lambda item: (item.frame_index or 0, item.file_name, item.id))
    ):
        frame_by_image_id[image.id] = frame_index
        files.append(str(image.path))
    return tuple(files)


def _derived_frame_files(images: list[_TaoImage], frame_by_image_id: dict[int, int]) -> tuple[str | None, ...]:
    """Build frame files by stable filename/id sorting when TAO frame_index is absent."""
    files: list[str | None] = []
    for frame_index, image in enumerate(sorted(images, key=lambda item: (item.file_name, item.id))):
        frame_by_image_id[image.id] = frame_index
        files.append(str(image.path))
    return tuple(files)


def _boxes_by_video(raw_annotations: object, context: _TaoContext) -> dict[int, list[BoxAnnotation]]:
    """Parse annotations and group resulting boxes by video id."""
    boxes_by_video: dict[int, list[BoxAnnotation]] = defaultdict(list)
    for raw in _as_list(raw_annotations, field="annotations", path=context.annotation_path):
        parsed = _parse_annotation(raw, context)
        if parsed is None:
            continue
        video_id, box = parsed
        boxes_by_video[video_id].append(box)
    return dict(boxes_by_video)


def _parse_annotation(raw: object, context: _TaoContext) -> tuple[int, BoxAnnotation] | None:
    """Parse one TAO annotation into a model box."""
    if not isinstance(raw, dict):
        logger.warning("Skipping malformed TAO annotation in %s", context.annotation_path)
        return None
    image = _image_for_annotation(raw, context)
    if image is None:
        return None
    track_id = _parse_int(raw.get("track_id"))
    if track_id is None:
        logger.warning("Skipping TAO annotation with missing/invalid track_id in %s", context.annotation_path)
        return None
    bbox = _parse_bbox(raw.get("bbox"))
    if bbox is None:
        logger.warning("Skipping TAO annotation with missing/invalid bbox in %s", context.annotation_path)
        return None

    category_id = _category_id_for_annotation(raw, track_id=track_id, context=context)
    category_id_value, category_uri, category_name = _category_fields(category_id, context)
    frame_index = context.frame_tables.frame_by_image_id[image.id]
    attributes = _annotation_attributes(raw, context=context, image=image, track_id=track_id)
    box = BoxAnnotation(
        track_uuid=f"{context.split}:{image.video_id}:{track_id}",
        track_id=track_id,
        category_id=category_id_value,
        category_uri=category_uri,
        category_name=category_name,
        bbox=bbox,
        attributes=attributes,
        frame_index=frame_index,
        timestamp=None,
    )
    return image.video_id, box


def _image_for_annotation(raw: dict[str, Any], context: _TaoContext) -> _TaoImage | None:
    """Return the indexed image for an annotation, or None with a warning."""
    image_id = _parse_int(raw.get("image_id"))
    if image_id is None:
        logger.warning("Skipping TAO annotation with missing/invalid image_id in %s", context.annotation_path)
        return None
    image = context.images.get(image_id)
    if image is None or image.id not in context.frame_tables.frame_by_image_id:
        logger.warning(
            "Skipping TAO annotation that references unknown image_id %s in %s",
            image_id,
            context.annotation_path,
        )
        return None
    return image


def _category_id_for_annotation(raw: dict[str, Any], *, track_id: int, context: _TaoContext) -> int | None:
    """Resolve category id, preferring the annotation and warning on track mismatches."""
    annotation_category = _parse_int(raw.get("category_id"))
    track = context.tracks.get(track_id)
    track_category = _parse_int(track.get("category_id")) if track else None
    if annotation_category is not None and track_category is not None and annotation_category != track_category:
        logger.warning(
            "TAO annotation category_id %s disagrees with track %s category_id %s in %s; using annotation",
            annotation_category,
            track_id,
            track_category,
            context.annotation_path,
        )
    return annotation_category if annotation_category is not None else track_category


def _category_fields(category_id: int | None, context: _TaoContext) -> tuple[int, str, str | None]:
    """Return BoxAnnotation category fields for a TAO category id."""
    if category_id is None:
        return -1, "", None
    name = context.category_names.get(category_id, f"category_{category_id}")
    uri = _category_uri(category_id, name)
    context.categories.setdefault(uri, category_id)
    return category_id, uri, name


def _annotation_attributes(
    raw: dict[str, Any],
    *,
    context: _TaoContext,
    image: _TaoImage,
    track_id: int,
) -> dict[str, Any]:
    """Build source-preserving attributes for one TAO annotation."""
    attrs: dict[str, Any] = {key: value for key, value in raw.items() if key not in _CORE_ANNOTATION_KEYS}
    attrs.update(
        {
            "source_format": "tao",
            "split": context.split,
            "tao_image_id": image.id,
            "tao_video_id": image.video_id,
            "tao_track_id": track_id,
        }
    )
    if image.frame_index is not None:
        attrs["tao_frame_index"] = image.frame_index
    annotation_id = raw.get("id")
    if annotation_id is not None:
        attrs["tao_annotation_id"] = annotation_id
    return attrs


def _build_sequence(
    context: _TaoContext,
    *,
    video_id: int,
    boxes: list[BoxAnnotation],
    probe_images: bool,
    next_video_id: int,
) -> VideoSequence | None:
    """Build one VideoSequence from indexed TAO frames and boxes."""
    frame_files = context.frame_tables.files_by_video_id.get(video_id, ())
    if not frame_files:
        logger.warning("Skipping TAO video %s with no loadable images in %s", video_id, context.annotation_path)
        return None
    video = context.videos.get(video_id) or _TaoVideo(id=video_id, name=str(video_id))
    width, height = _sequence_dimensions(video, context.images_by_video.get(video_id, []))
    if probe_images:
        probed = _probe_image(_first_frame_file(frame_files))
        if probed is not None:
            width, height = probed
    num_frames = len(frame_files)
    num_frames_exact = all(frame_file is not None for frame_file in frame_files)
    duration = num_frames / video.fps if num_frames_exact and video.fps > 0 else None
    # For explicit frame_files, frame_dir is informational only; frame_path()
    # and frame_filename() always prefer the explicit table over frame_pattern.
    frame_dir = _common_frame_dir(frame_files)
    return VideoSequence(
        video_id=next_video_id,
        video_path=None,
        fps=video.fps,
        num_frames=num_frames,
        duration=duration,
        annotation_path=str(context.annotation_path),
        frame_files=frame_files,
        frame_dir=frame_dir,
        video_meta={
            "format": "tao",
            "split": context.split,
            "source_video_id": video.id,
            "sequence_name": video.name,
            "annotation_file": str(context.annotation_path),
            "video": dict(video.raw),
        },
        boxes=boxes,
        width=width,
        height=height,
        num_frames_exact=num_frames_exact,
    )


def _sequence_dimensions(video: _TaoVideo, images: list[_TaoImage]) -> tuple[int | None, int | None]:
    """Return sequence dimensions from video metadata, then first image metadata."""
    width = video.width
    height = video.height
    if width is not None and height is not None:
        return width, height
    ordered = sorted(
        images,
        key=lambda item: (item.frame_index if item.frame_index is not None else 10**12, item.file_name, item.id),
    )
    for image in ordered:
        width = width or image.width
        height = height or image.height
        if width is not None and height is not None:
            return width, height
    return width, height


def _probe_image(path: Path | None) -> tuple[int, int] | None:
    """Probe one image with OpenCV, returning (width, height)."""
    if path is None:
        return None
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("OpenCV not installed; skipping TAO image probing (install datamaite[fmv])")
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        logger.warning("Could not read TAO frame image for probing: %s", path)
        return None
    height, width = image.shape[:2]
    return int(width), int(height)


def _first_frame_file(frame_files: Iterable[str | None]) -> Path | None:
    """Return the first present frame file from a frame file table."""
    for frame_file in frame_files:
        if frame_file is not None:
            return Path(frame_file)
    return None


def _common_frame_dir(frame_files: Iterable[str | None]) -> str | None:
    """Return a common frame directory when every present frame shares one."""
    parents = {str(Path(frame_file).parent) for frame_file in frame_files if frame_file is not None}
    return next(iter(parents)) if len(parents) == 1 else None


def _video_ids_to_load(videos: Mapping[int, _TaoVideo], images_by_video: Mapping[int, list[_TaoImage]]) -> list[int]:
    """Return video ids with images, preserving the videos array order when available."""
    ordered = [video_id for video_id in videos if video_id in images_by_video]
    extra = sorted(video_id for video_id in images_by_video if video_id not in videos)
    return ordered + extra


def _resolve_image_path(root: Path, file_name: str, *, annotation_path: Path) -> Path | None:
    """Resolve a TAO image file name safely under the dataset root.

    Official TAO ``images.file_name`` values are relative to ``root / "frames"``
    (for example ``train/YFCC100M/.../frame0391.jpg``). Some derived datasets
    already include the leading ``frames/`` component, so keep those relative to
    ``root`` rather than adding ``frames`` twice.
    """
    if "\\" in file_name:
        logger.warning("Skipping TAO image with unsafe file_name %r in %s", file_name, annotation_path)
        return None
    posix = PurePosixPath(file_name.strip())
    if not posix.parts or posix.is_absolute() or any(part in {"..", ""} or ":" in part for part in posix.parts):
        logger.warning("Skipping TAO image with unsafe file_name %r in %s", file_name, annotation_path)
        return None
    base = root if posix.parts[0] == "frames" else root / "frames"
    candidate = base.joinpath(*posix.parts)
    if not _is_within_root(candidate, root):
        logger.warning(
            "Skipping TAO image whose resolved path escapes the dataset root: %r in %s",
            file_name,
            annotation_path,
        )
        return None
    return candidate


def _is_within_root(path: Path, root: Path) -> bool:
    """Return True if ``path`` resolves under ``root``, catching symlink escapes."""
    try:
        resolved_path = path.resolve(strict=False)
        resolved_root = root.resolve(strict=False)
        resolved_path.relative_to(resolved_root)
    except (OSError, ValueError):
        return False
    return True


def _parse_bbox(value: object) -> tuple[float, float, float, float] | None:
    """Parse a TAO bbox as finite xywh with positive width and height."""
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    left = _parse_float(value[0])
    top = _parse_float(value[1])
    width = _parse_float(value[2])
    height = _parse_float(value[3])
    if left is None or top is None or width is None or height is None:
        return None
    if width <= 0 or height <= 0:
        return None
    return left, top, width, height


def _as_list(value: object, *, field: str, path: Path) -> list[object]:
    """Return a list field, logging and returning [] when malformed."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    logger.warning("TAO field %r is not a list in %s; ignoring", field, path)
    return []


def _category_uri(category_id: int, name: str) -> str:
    """Build a stable URI for a TAO category id/name pair."""
    safe_name = name.strip().replace("/", "_") or f"category_{category_id}"
    return f"tao/category_{category_id}/{safe_name}"


def _parse_int(value: object) -> int | None:
    """Parse an integer field, accepting integer-looking floats like ``1.0``."""
    number = _parse_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _parse_float(value: object) -> float | None:
    """Parse a finite float field."""
    if isinstance(value, bool):
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _coerce_positive_float(value: object) -> float | None:
    """Coerce a config value to a finite, positive float."""
    parsed = _parse_float(value)
    return parsed if parsed is not None and parsed > 0 else None


def _coerce_positive_int(value: object) -> int | None:
    """Coerce a config value to a finite, positive integer."""
    parsed = _parse_int(value)
    return parsed if parsed is not None and parsed > 0 else None
