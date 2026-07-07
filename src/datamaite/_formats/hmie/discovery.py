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
import re
from collections.abc import Iterator
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
    """A matched annotation JSON and video file for one snippet.

    ``snippet_dir`` is the snippet directory this pair belongs to, used to
    count distinct snippets even when annotations are centralised in a
    batch-level ``scale/`` dir (where ``annotation_path.parent.parent`` would
    collapse every annotation onto the batch root). It is the matched video's
    snippet dir for batch-level pairs, the snippet dir for per-snippet pairs,
    and ``None`` only when neither is determinable.
    """

    annotation_path: Path
    video_path: Path | None
    snippet_dir: Path | None = None


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

    Uses a **snippet-centric**, **location-based** approach, plus a
    **batch-level ``scale/``** path that *merges* with it (rather than only
    falling back when per-snippet discovery is empty):

    1. Single traversal pass (:func:`_walk`) to collect JSON and video files.
    2. Derive snippet directories from ``seq_*`` video containers.
    3. For each snippet, annotations are JSONs in subdirectories
       (``scale/``, labeler dirs). Snippet-level JSONs are metadata.
    4. Any ``scale/`` directory that is *not* inside a snippet (i.e. a
       batch-level ``scale/`` holding annotations for sibling snippets) is
       paired by matching each annotation's embedded video filename to the
       videos within that batch. These pairs merge with the per-snippet ones,
       so a tree mixing both layouts -- or a parent of several batches each
       with its own ``scale/`` -- is fully discovered.
    5. Annotations with the best video (prefer ``seq_mp4``); unmatched ones
       become orphan annotations.
    """
    if not root.is_dir():
        return DiscoveryResult(errors=[f"Root path is not a directory: {root}"])

    annotation_files, video_dirs, snippet_dirs, scale_dirs = _collect_files(root)

    # Batch-level scale dirs are those whose parent is not itself a snippet
    # dir (a snippet's own scale/ is already handled by per-snippet discovery).
    batch_scale_dirs = sorted(d for d in scale_dirs if d.parent not in snippet_dirs)

    if not snippet_dirs and not batch_scale_dirs:
        logger.info("No snippet directories (containing seq_*/) found under %s", root)
        return DiscoveryResult(errors=[f"No snippet directories (containing seq_*/) found under {root}"])

    logger.info(
        "Walk complete: %d annotation files, %d video dirs, %d snippet dirs, %d batch-level scale dirs under %s",
        len(annotation_files),
        len(video_dirs),
        len(snippet_dirs),
        len(batch_scale_dirs),
        root,
    )

    result = _build_pairs(annotation_files, video_dirs, snippet_dirs)
    batch_pairs = _build_batch_scale_pairs(batch_scale_dirs, video_dirs)

    pairs = result.pairs + batch_pairs
    if not pairs:
        return DiscoveryResult(
            errors=[f"No annotation files found under {len(snippet_dirs)} snippet dirs or batch-level scale/ dirs"]
        )

    all_videos = {v for vids in video_dirs.values() for v in vids}
    matched = {p.video_path for p in pairs if p.video_path is not None}
    orphan_annotations = [p.annotation_path for p in pairs if p.video_path is None]
    orphan_videos = sorted(all_videos - matched)
    logger.info(
        "Discovery complete: %d pairs (%d batch-level), %d orphan annotations, %d orphan videos",
        len(pairs),
        len(batch_pairs),
        len(orphan_annotations),
        len(orphan_videos),
    )
    return DiscoveryResult(
        pairs=pairs,
        orphan_annotations=orphan_annotations,
        orphan_videos=orphan_videos,
        multi_video_dirs=result.multi_video_dirs,
    )


def _build_batch_scale_pairs(
    scale_dirs: list[Path],
    video_dirs: dict[Path, list[Path]],
) -> list[SnippetPair]:
    """Pair batch-level ``scale/`` annotations with videos, scoped per batch.

    Each batch-level ``scale/`` (sibling to snippet dirs, not inside one)
    holds annotations for that batch's snippets. There is no directory
    relationship to pair on, so each Scale annotation name embeds its source
    video filename, matched (:func:`match_annotation_to_video`) against the
    videos *within that batch* (the scale dir's parent subtree) -- scoping per
    batch avoids cross-batch mismatches. Non-annotation JSONs (``metadata.json``
    and the like, whose names embed no video filename) are skipped.

    Each pair's ``snippet_dir`` is the matched video's snippet dir
    (``video.parent.parent``), or ``None`` when unmatched. The caller merges
    these with the per-snippet pairs.
    """
    pairs: list[SnippetPair] = []
    for scale_dir in scale_dirs:
        batch = scale_dir.parent
        batch_videos = sorted(v for vids in video_dirs.values() for v in vids if batch in v.parents)
        for ann_path in sorted(scale_dir.glob("*.json")):
            if not _looks_like_scale_annotation_name(ann_path.name):
                logger.debug("Skipping non-annotation JSON in %s: %s", scale_dir, ann_path.name)
                continue
            video_path = match_annotation_to_video(ann_path.name, batch_videos)
            pairs.append(
                SnippetPair(
                    annotation_path=ann_path,
                    video_path=video_path,
                    snippet_dir=video_path.parent.parent if video_path is not None else None,
                )
            )
    return pairs


def _looks_like_scale_annotation_name(name: str) -> bool:
    """Heuristic: does a filename look like a Scale annotation (vs metadata)?

    Scale exports embed the source video filename: ``<prefix>_<video>.<ext>_<hash>.json``
    (e.g. ``CDAO_SRC1_clip.mp4_abc.json``), so an annotation name contains a
    video-extension token. Centralised ``scale/`` dirs can also hold
    non-annotation JSON (``metadata.json``, ``seqinfo.json``) that embed no
    video name. Filename-only so discovery stays filesystem-bound (no parse).
    """
    lower = name.lower()
    return any(ext in lower for ext in _VIDEO_EXTENSIONS)


def match_annotation_to_video(annotation_name: str, videos: list[Path]) -> Path | None:
    """Return the video whose filename is embedded in a Scale annotation name.

    Scale annotation names embed the source video filename followed by a
    hash: ``<prefix>_<video-name>_<hash>.json`` (e.g.
    ``CDAO_SRC1_clip_a.mp4_abc.json``). The match is anchored on both sides
    -- the embedded name must be preceded by ``_`` (or the start) and
    followed by ``_`` or ``.`` -- so a shorter stem (``clip``) does not match
    an annotation for a longer one (``clip_a``). When a shorter embedded name
    also matches (e.g. ``a.mp4`` inside ``clip_a.mp4``), the longest filename
    wins. If two *distinct* videos share the winning filename (same basename
    in different directories), the match is **ambiguous** and ``None`` is
    returned (logged) -- an arbitrary pick is worse than an orphan. Single
    source of truth, used by batch-level discovery and the loader's override
    mode.
    """
    matches: list[Path] = []
    for video in videos:
        token = video.name  # full filename incl. extension, e.g. "clip_a.mp4"
        idx = annotation_name.find(token)
        while idx != -1:
            before_ok = idx == 0 or annotation_name[idx - 1] == "_"
            after = idx + len(token)
            after_ok = after >= len(annotation_name) or annotation_name[after] in "_."
            if before_ok and after_ok:
                matches.append(video)
                break
            idx = annotation_name.find(token, idx + 1)
    if not matches:
        return None
    longest = max(len(v.name) for v in matches)
    best = [v for v in matches if len(v.name) == longest]
    if len(best) > 1:
        logger.warning(
            "Ambiguous video match for annotation %r: %d candidates tie (%s); treating as orphan",
            annotation_name,
            len(best),
            ", ".join(sorted(v.name for v in best)),
        )
        return None
    return best[0]


def _walk(root: Path) -> Iterator[tuple[Path, list[str], list[str]]]:
    """``os.walk``-style top-down traversal that also works on UPath roots.

    Yields ``(dirpath, dirnames, filenames)`` with ``dirpath`` as a path
    object, preserving the ``os.walk`` semantics discovery relies on:
    top-down order, in-place pruning of ``dirnames`` honoured, unreadable
    directories skipped (``onerror=None`` behaviour), and symlinked
    directories listed but not descended into (``followlinks=False``).
    Entries are yielded in sorted order so traversal is deterministic on
    every backend.
    """
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda entry: entry.name)
        except Exception as exc:
            # Object-store backends surface throttling/auth as non-OSError
            # exception families; the walk contract is "one bad entry never
            # kills discovery", so skip this dir rather than abort.
            logger.debug("Skipping unreadable directory %s: %s", current, type(exc).__name__)
            continue
        dirnames: list[str] = []
        filenames: list[str] = []
        for entry in entries:
            try:
                is_dir = entry.is_dir()
            except Exception as exc:
                # Mirror os.walk: a stat failure on one entry means "not a
                # directory", never an aborted walk. Object stores raise
                # non-OSError families here too.
                logger.debug("Treating %s as non-directory after %s", entry, type(exc).__name__)
                is_dir = False
            (dirnames if is_dir else filenames).append(entry.name)
        yield current, dirnames, filenames
        # Reversed so the LIFO stack visits children in sorted order. The
        # caller may have pruned dirnames in place while we were suspended
        # at the yield, so dirnames is re-read here, after the yield.
        for name in reversed(dirnames):
            child = current / name
            try:
                is_symlink = child.is_symlink()
            except Exception as exc:
                # Mirror os.walk: if the symlink check fails, treat the
                # child as a plain directory and keep walking. Object stores
                # raise non-OSError families here too.
                logger.debug("Treating %s as non-symlink after %s", child, type(exc).__name__)
                is_symlink = False
            if not is_symlink:
                stack.append(child)


def _collect_files(root: Path) -> tuple[list[Path], dict[Path, list[Path]], set[Path], set[Path]]:
    """Single traversal pass (:func:`_walk`) collecting annotation files, videos, snippet/scale dirs.

    Returns (annotation_files, video_dirs, snippet_dirs, scale_dirs) where:
    - annotation_files: ``.json`` files inside annotation subdirectories
      of snippet dirs (not at snippet level, not in metadata/seq dirs)
    - video_dirs: maps each ``seq_*`` directory to its video file paths
    - snippet_dirs: directories that contain at least one ``seq_*`` child
    - scale_dirs: every directory named ``scale`` (both per-snippet and
      batch-level; the caller partitions them by whether the parent is a
      snippet dir)
    """
    annotation_files: list[Path] = []
    video_dirs: dict[Path, list[Path]] = {}
    snippet_dirs: set[Path] = set()
    scale_dirs: set[Path] = set()
    annotation_parent_dirs: set[Path] = set()

    for current, dirnames, filenames in _walk(root):
        # Prune metadata directories from traversal.
        dirnames[:] = _prune_metadata(dirnames, current)

        if current.name == "scale":
            scale_dirs.add(current)

        # Register snippet dirs and their annotation subdirs.
        _register_snippet(current, dirnames, snippet_dirs, annotation_parent_dirs)

        # Collect files based on directory role.
        _collect_dir_files(current, filenames, video_dirs, annotation_files, annotation_parent_dirs)

    return annotation_files, video_dirs, snippet_dirs, scale_dirs


def _prune_metadata(dirnames: list[str], current: Path) -> list[str]:
    """Remove metadata directories from the traversal."""
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
        pairs.append(SnippetPair(annotation_path=ann_path, video_path=video_path, snippet_dir=snippet_dir))
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

    # No early error on empty pairs: a tree may have its annotations only in
    # a batch-level scale/ dir, which the caller merges in. The caller emits
    # the final "nothing found" error if both paths come up empty.
    if not pairs:
        logger.info(
            "No per-snippet annotations under %d snippet dirs (may be batch-level scale/ or unannotated)",
            len(snippet_videos),
        )

    orphan_annotations = [p.annotation_path for p in pairs if p.video_path is None]
    all_videos = {v for vids in video_dirs.values() for v in vids}
    orphan_videos = sorted(all_videos - matched_videos)

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
