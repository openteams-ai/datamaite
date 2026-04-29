"""HMIE dataset folder discovery.

Walks the HMIE directory hierarchy on SUNet and pairs annotation JSONs
with their corresponding video files.

The SUNet HMIE layout varies across dataset families but follows a
common pattern at the snippet level:

    <batch_dir>/
        <snippet_name>_<id>_<seq>/       # snippet directory
            <snippet_name>.json          # video metadata (NOT an annotation)
            scale/ | <labeler>/          # annotation subdirectory
                <annotation>.json        # Scale annotation (has task_id)
            seq_mp4/                     # video container (always present)
                *.mp4
            seq_ts/                      # alternative video container (some datasets)
                *.ts
            mapp_metadata/ | 0601_metadata/  # pipeline metadata (ignored)
                *.json

Discovery is **snippet-centric**: it identifies snippet directories by
looking for ``seq_*`` video containers, then searches for annotation
JSONs in subdirectories of the snippet (never at snippet level, which
is always video metadata).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# Video container directories follow the ``seq_<format>`` pattern.
_SEQ_DIR_RE = re.compile(r"^seq_\w+$")

# Video file extensions recognised inside seq_* directories.
_VIDEO_EXTENSIONS = frozenset({".mp4", ".ts"})

# Directories whose contents should never be treated as annotations.
# These hold pipeline/ingest metadata, not Scale exports.
_METADATA_DIR_SUFFIXES = ("_metadata",)


@dataclass(frozen=True)
class SnippetPair:
    """A matched annotation JSON and video file for one snippet."""

    annotation_path: Path
    video_path: Path | None


@dataclass
class DiscoveryResult:
    """Result of walking an HMIE dataset root."""

    pairs: list[SnippetPair] = field(default_factory=list)
    orphan_annotations: list[Path] = field(default_factory=list)
    orphan_videos: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    # (video_dir, count) for every seq_* dir containing multiple videos.
    # _find_video deterministically picks the lexicographic first, but the
    # extras are silently ignored -- surface them as warnings.
    multi_video_dirs: list[tuple[Path, int]] = field(default_factory=list)


def _is_metadata_dir(name: str) -> bool:
    """Return True if ``name`` looks like a metadata directory."""
    return any(name.endswith(suffix) for suffix in _METADATA_DIR_SUFFIXES)


def _is_annotation_dir(dirname: str) -> bool:
    """Return True if a subdirectory can contain annotations.

    Annotation directories are any non-metadata, non-seq_* subdirectory
    of a snippet. In practice: ``scale/``, labeler dirs like
    ``labeler_alpha/``, or any other directory that isn't infrastructure.
    """
    if _is_metadata_dir(dirname):
        return False
    return not _SEQ_DIR_RE.match(dirname)


def discover_hmie_pairs(root: Path) -> DiscoveryResult:
    """Walk an HMIE dataset root and discover annotation/video pairs.

    Uses a **snippet-centric**, **location-based** approach:

    1. Single ``os.walk`` pass to collect JSON and video files.
    2. Derive snippet directories from ``seq_*`` video containers.
    3. For each snippet, annotations are JSONs in subdirectories
       (``scale/``, labeler dirs). Snippet-level JSONs are metadata.
    4. Pair annotations with the best video (prefer ``seq_mp4``).
    5. Snippets with no annotation subdirectory have no annotations.
    """
    if not root.is_dir():
        return DiscoveryResult(errors=[f"Root path is not a directory: {root}"])

    annotation_files, video_dirs, snippet_dirs = _collect_files(root)

    if not snippet_dirs:
        logger.info("No snippet directories (containing seq_*/) found under %s", root)
        return DiscoveryResult(errors=[f"No snippet directories (containing seq_*/) found under {root}"])

    logger.info(
        "Walk complete: %d annotation files, %d video dirs, %d snippet dirs under %s",
        len(annotation_files),
        len(video_dirs),
        len(snippet_dirs),
        root,
    )

    result = _build_pairs(annotation_files, video_dirs, snippet_dirs)

    # When we came up empty, check whether the dataset used the
    # batch-level ``scale/`` layout that this snippet-centric walker
    # intentionally does not support -- silent ignoring produces a
    # confusing "no annotations found" error on datasets that clearly
    # have annotations on disk.
    if not result.pairs:
        batch_scale_jsons = _batch_level_scale_jsons(root)
        if batch_scale_jsons:
            result.errors.append(
                f"Found {len(batch_scale_jsons)} annotation file(s) under "
                f"{root / 'scale'}/ (batch-level layout). The validator "
                "currently pairs annotations per-snippet only; batch-level "
                "annotations are not yet supported and were ignored."
            )

    return result


def _batch_level_scale_jsons(root: Path) -> list[Path]:
    """Return annotation JSONs at ``root/scale/*.json`` if that dir exists.

    Some batches use a ``scale/`` directory at batch root (sibling to
    snippet dirs) instead of the per-snippet labeler-subdir layout. The
    walker only handles the per-snippet variant, so callers use this
    detector to produce a helpful error instead of silently dropping
    the data. See ``docs/architecture.md`` for the full layout taxonomy.
    """
    scale_dir = root / "scale"
    if not scale_dir.is_dir():
        return []
    return sorted(p for p in scale_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json")


def _collect_files(root: Path) -> tuple[list[Path], dict[Path, list[Path]], set[Path]]:
    """Single os.walk pass collecting annotation files, videos, and snippet dirs.

    Returns (annotation_files, video_dirs, snippet_dirs) where:
    - annotation_files: ``.json`` files inside annotation subdirectories
      of snippet dirs (not at snippet level, not in metadata/seq dirs)
    - video_dirs: maps each ``seq_*`` directory to its video file paths
    - snippet_dirs: directories that contain at least one ``seq_*`` child
    """
    annotation_files: list[Path] = []
    video_dirs: dict[Path, list[Path]] = {}
    snippet_dirs: set[Path] = set()
    annotation_parent_dirs: set[Path] = set()

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)

        # Prune metadata directories from traversal.
        dirnames[:] = _prune_metadata(dirnames, current)

        # Register snippet dirs and their annotation subdirs.
        _register_snippet(current, dirnames, snippet_dirs, annotation_parent_dirs)

        # Collect files based on directory role.
        _collect_dir_files(current, filenames, video_dirs, annotation_files, annotation_parent_dirs)

    return annotation_files, video_dirs, snippet_dirs


def _prune_metadata(dirnames: list[str], current: Path) -> list[str]:
    """Remove metadata directories from os.walk traversal."""
    pruned = [d for d in dirnames if _is_metadata_dir(d)]
    if pruned:
        logger.debug("Pruning metadata dirs in %s: %s", current, pruned)
    return [d for d in dirnames if not _is_metadata_dir(d)]


def _register_snippet(
    current: Path,
    dirnames: list[str],
    snippet_dirs: set[Path],
    annotation_parent_dirs: set[Path],
) -> None:
    """If current dir has seq_* children, register it as a snippet."""
    if not any(_SEQ_DIR_RE.match(d) for d in dirnames):
        return
    snippet_dirs.add(current)
    logger.debug("Snippet dir identified: %s", current)
    for d in dirnames:
        if _is_annotation_dir(d):
            annotation_parent_dirs.add(current / d)
            logger.debug("  Annotation subdir: %s/", d)


def _collect_dir_files(
    current: Path,
    filenames: list[str],
    video_dirs: dict[Path, list[Path]],
    annotation_files: list[Path],
    annotation_parent_dirs: set[Path],
) -> None:
    """Collect video files from seq_* dirs and annotation JSONs from annotation subdirs."""
    is_seq_dir = _SEQ_DIR_RE.match(current.name) is not None
    for filename in filenames:
        if is_seq_dir and _is_video_file(filename):
            video_dirs.setdefault(current, []).append(current / filename)
        elif current in annotation_parent_dirs and filename.endswith(".json"):
            annotation_files.append(current / filename)
            logger.debug("  Annotation file: %s/%s", current.name, filename)


def _index_by_snippet(annotation_files: list[Path]) -> dict[Path, list[Path]]:
    """Index annotation files by their snippet directory.

    Annotations always live at ``snippet_dir/<subdir>/file.json``, so
    ``parent.parent`` is the snippet dir.
    """
    anns_by_snippet: dict[Path, list[Path]] = {}
    for ann in annotation_files:
        snippet_dir = ann.parent.parent
        anns_by_snippet.setdefault(snippet_dir, []).append(ann)
    return anns_by_snippet


def _emit_pairs_for_snippet(
    snippet_dir: Path,
    ann_paths: list[Path],
    seq_groups: dict[str, list[Path]],
) -> tuple[list[SnippetPair], set[Path], list[tuple[Path, int]]]:
    """Pair this snippet's annotations with its best video.

    Returns ``(pairs, matched_videos, multi_video_extras)``. If the
    snippet has no annotations, returns empty collections and logs a
    debug message so operators can follow why a seq_* dir produced no
    output.
    """
    video_path, extras = _pick_best_video(seq_groups)

    if not ann_paths:
        logger.debug(
            "Snippet %s: no annotation subdir with JSONs (seq_dirs=%s)",
            snippet_dir.name,
            list(seq_groups.keys()),
        )
        return [], set(), extras

    pairs: list[SnippetPair] = []
    matched: set[Path] = set()
    for ann_path in ann_paths:
        if video_path is not None:
            matched.add(video_path)
        pairs.append(SnippetPair(annotation_path=ann_path, video_path=video_path))
        logger.debug(
            "Paired: %s -> %s",
            ann_path.name,
            video_path.name if video_path else "<no video>",
        )
    return pairs, matched, extras


def _build_pairs(
    annotation_files: list[Path],
    video_dirs: dict[Path, list[Path]],
    snippet_dirs: set[Path],
) -> DiscoveryResult:
    """Match annotations to videos via snippet directories."""
    # Map snippet_dir -> {seq_dir_name: [video_paths]}
    snippet_videos: dict[Path, dict[str, list[Path]]] = {sd: {} for sd in snippet_dirs}
    for seq_dir, vids in video_dirs.items():
        snippet_dir = seq_dir.parent
        snippet_videos.setdefault(snippet_dir, {})[seq_dir.name] = vids

    anns_by_snippet = _index_by_snippet(annotation_files)

    pairs: list[SnippetPair] = []
    matched_videos: set[Path] = set()
    multi_video_dirs: list[tuple[Path, int]] = []

    for snippet_dir, seq_groups in sorted(snippet_videos.items()):
        ann_paths = sorted(anns_by_snippet.get(snippet_dir, []))
        snippet_pairs, snippet_matched, extras = _emit_pairs_for_snippet(snippet_dir, ann_paths, seq_groups)
        pairs.extend(snippet_pairs)
        matched_videos.update(snippet_matched)
        multi_video_dirs.extend(extras)

    if not pairs:
        logger.info(
            "No annotations found in subdirs of %d snippet dirs (dataset may not be annotated)",
            len(snippet_videos),
        )
        return DiscoveryResult(
            errors=[f"No annotation files found in subdirectories of {len(snippet_videos)} snippet directories"]
        )

    orphan_annotations = [p.annotation_path for p in pairs if p.video_path is None]
    all_videos = {v for vids in video_dirs.values() for v in vids}
    orphan_videos = sorted(all_videos - matched_videos)

    logger.info(
        "Discovery complete: %d pairs, %d orphan annotations, %d orphan videos, %d multi-video dirs",
        len(pairs),
        len(orphan_annotations),
        len(orphan_videos),
        len(multi_video_dirs),
    )

    return DiscoveryResult(
        pairs=pairs,
        orphan_annotations=orphan_annotations,
        orphan_videos=orphan_videos,
        multi_video_dirs=multi_video_dirs,
    )


def _is_video_file(filename: str) -> bool:
    """Check if a filename has a recognised video extension."""
    return Path(filename).suffix.lower() in _VIDEO_EXTENSIONS


def _has_snippet_children(path: Path) -> bool:
    """Return True if ``path`` itself directly contains snippet directories.

    A snippet directory is one whose children include a ``seq_*`` video
    container (see ``_SEQ_DIR_RE``). This is used by ``find_batch_roots``
    to decide whether a given directory is already a batch (contains
    snippet dirs directly) or is a parent of multiple batches.
    """
    if not path.is_dir():
        return False
    try:
        children = list(path.iterdir())
    except OSError:
        return False
    for child in children:
        if not child.is_dir():
            continue
        try:
            grand = list(child.iterdir())
        except OSError:
            continue
        if any(g.is_dir() and _SEQ_DIR_RE.match(g.name) for g in grand):
            return True
    return False


def find_batch_roots(root: Path) -> list[Path]:
    """Find batch directories under ``root``.

    A *batch directory* is one that directly contains snippet
    directories (i.e. directories whose children include a ``seq_*``
    video container).

    Behaviour:

    - If ``root`` itself is a batch directory, returns ``[root]``.
    - Otherwise, returns the immediate subdirectories of ``root`` that
      are batch directories, sorted by name.

    Used by the CLI to decide whether to run validation once (single
    batch) or iterate over each discovered batch (multi-batch mode).
    Shares the ``seq_*`` detection rule with the full discovery walk,
    so there is a single source of truth for "what counts as a
    snippet".
    """
    if not root.is_dir():
        return []
    if _has_snippet_children(root):
        return [root]
    return sorted(d for d in root.iterdir() if d.is_dir() and _has_snippet_children(d))


def _pick_best_video(
    seq_groups: dict[str, list[Path]],
) -> tuple[Path | None, list[tuple[Path, int]]]:
    """Pick the best video from a snippet's seq_* directories.

    Preference order: ``seq_mp4`` first, then other ``seq_*`` dirs
    alphabetically. Within a directory, picks the lexicographic first
    video file.

    Returns (chosen_video, multi_video_entries) where multi_video_entries
    is a list of (dir_path, count) for dirs with multiple videos.
    """
    multi: list[tuple[Path, int]] = []

    # Prefer seq_mp4 if available
    preferred_order = sorted(seq_groups.keys(), key=lambda k: (k != "seq_mp4", k))

    for seq_name in preferred_order:
        vids = sorted(seq_groups[seq_name])
        if not vids:
            continue

        if len(vids) > 1:
            seq_dir = vids[0].parent
            multi.append((seq_dir, len(vids)))
            logger.warning(
                "Multiple video files in %s (%d), using %s",
                seq_dir,
                len(vids),
                vids[0].name,
            )

        return vids[0], multi

    return None, multi
