"""Dataset validation utilities."""

from __future__ import annotations

import logging
import os
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from databridge._cache import ValidationCache
from databridge._formats.hmie import (
    DiscoveryResult,
    SnippetPair,
    check_annotation_schema,
    check_video_annotation_consistency,
    discover_hmie_pairs,
    probe_video,
)
from databridge._formats.hmie.categories import SKIP_VIDEO_CONSISTENCY, SKIP_VIDEO_INTEGRITY
from databridge._types import DatasetFormat, Finding, Severity, ValidationResult

logger = logging.getLogger(__name__)


def validate(
    path: str | Path,
    dataset_format: DatasetFormat | str = DatasetFormat.HMIE,
    *,
    check_video_integrity: bool = True,
    workers: int | None = None,
    max_findings_per_check: int | None = None,
    progress_callback: Callable[[], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
    cache: ValidationCache | None = None,
) -> ValidationResult:
    """Validate a dataset at the given path.

    Parameters
    ----------
    path
        Root directory of the dataset.
    dataset_format
        Dataset format to validate against. Renamed from ``format`` to
        avoid shadowing the builtin.
    check_video_integrity
        If True, attempt to open FMV files to verify they are not corrupted.
        Requires the ``video`` extra (``pip install databridge[video]``).
    workers
        Number of worker processes for parallel per-pair validation.
        ``None`` (default) uses ``os.cpu_count()``. Pass ``1`` to force
        synchronous execution (useful for debugging or when the process
        pool overhead exceeds the gain on small datasets).
    max_findings_per_check
        Upper bound on how many individual findings to keep in the returned
        ``findings`` list per check name. ``None`` (default) keeps every
        finding. Use to bound memory when validating very large datasets.
        The ``finding_counts`` Counter on the result always reflects the
        true per-check total, even when findings are capped.

    Returns
    -------
    ValidationResult
        Contains pass/fail status, findings (possibly capped),
        finding_counts (always complete), and label histogram.
    """
    path = Path(path)
    if isinstance(dataset_format, str):
        dataset_format = DatasetFormat(dataset_format.lower())

    if dataset_format == DatasetFormat.HMIE:
        return _validate_hmie(
            path,
            check_video=check_video_integrity,
            workers=workers,
            max_findings_per_check=max_findings_per_check,
            progress_callback=progress_callback,
            status_callback=status_callback,
            cache=cache,
        )

    raise NotImplementedError(f"Validation for format {dataset_format.value!r} is not implemented")


def validate_annotation(
    annotation_path: str | Path,
    video_path: str | Path | None = None,
    *,
    dataset_format: DatasetFormat | str = DatasetFormat.HMIE,
    check_video_integrity: bool = True,
) -> ValidationResult:
    """Validate a single annotation file and optionally its paired video.

    This is the lower-level entry point for validating individual files
    rather than discovering datasets in a directory tree.

    Parameters mirror :func:`validate` where applicable (same
    ``check_video_integrity`` kwarg name for consistency) and accept an
    explicit ``dataset_format`` so non-HMIE formats can be added without
    hardcoding the result tag.
    """
    annotation_path = Path(annotation_path)
    video_path = Path(video_path) if video_path is not None else None
    if isinstance(dataset_format, str):
        dataset_format = DatasetFormat(dataset_format.lower())

    if dataset_format != DatasetFormat.HMIE:
        raise NotImplementedError(f"Annotation validation for format {dataset_format.value!r} is not implemented")

    findings, label_counter = _validate_pair(annotation_path, video_path, check_video=check_video_integrity)
    has_errors = any(f.severity == Severity.ERROR for f in findings)
    finding_counts: Counter[str] = Counter(f.check for f in findings)
    severity_counts: dict[str, Counter[str]] = {"error": Counter(), "warning": Counter()}
    for f in findings:
        severity_counts[f.severity.value][f.check] += 1
    return ValidationResult(
        dataset_path=annotation_path,
        dataset_format=dataset_format,
        passed=not has_errors,
        findings=findings,
        label_histogram=label_counter,
        finding_counts=finding_counts,
        finding_severity_counts=severity_counts,
        # Video checks are skipped both when integrity is disabled AND when
        # no video was paired: _validate_pair() returns early on
        # ``video_path is None`` regardless of check_video_integrity, so
        # without this the report would render FMV as a clean PASS when in
        # fact nothing ran. The banner copy is reason-neutral so it reads
        # correctly for either cause.
        skipped_checks=_video_skips(check_video_integrity and video_path is not None),
    )


def validate_batches(
    root: str | Path | Iterable[Path],
    *,
    dataset_format: DatasetFormat | str = DatasetFormat.HMIE,
    check_video_integrity: bool = True,
    workers: int | None = None,
    max_findings_per_check: int | None = None,
    progress_callback: Callable[[], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
    cache: ValidationCache | None = None,
) -> Iterator[tuple[Path, ValidationResult]]:
    """Validate multiple batches, yielding ``(batch_path, result)`` pairs.

    Parameters
    ----------
    root
        Either a single directory (``str`` or ``Path``) to discover batches
        under via :func:`find_batch_roots`, or an iterable of batch paths
        the caller has already discovered.
    dataset_format, check_video_integrity, workers, max_findings_per_check,
    progress_callback, status_callback, cache
        Forwarded to :func:`validate` for every batch.

    Yields
    ------
    tuple[Path, ValidationResult]
        One tuple per batch, in ``find_batch_roots`` order when ``root`` is
        a path, or in iteration order when ``root`` is an iterable.

    Raises
    ------
    ValueError
        If ``root`` is a path and discovery returns no batches -- a typo'd
        root is the common error, so we raise rather than silently yield
        nothing. An explicit empty iterable is respected (yields nothing).

    Notes
    -----
    Generator, not list: at scale (tens of thousands of batches) the caller
    should aggregate incrementally rather than buffer every result.

    Per-batch crash isolation: if :func:`validate` raises for a batch, the
    exception is converted into a ``ValidationResult`` carrying a single
    ``validate_crash`` ERROR finding. The loop continues with the next
    batch. This mirrors the ``worker_crash`` pattern used inside
    :func:`validate` for per-pair failures.
    """
    if isinstance(root, (str, Path)):
        from databridge._formats.hmie.discovery import find_batch_roots

        batch_paths: list[Path] = find_batch_roots(Path(root))
        if not batch_paths:
            msg = f"No batches found under {root}"
            raise ValueError(msg)
    else:
        batch_paths = list(root)

    if isinstance(dataset_format, str):
        dataset_format = DatasetFormat(dataset_format.lower())

    for batch in batch_paths:
        try:
            result = validate(
                batch,
                dataset_format=dataset_format,
                check_video_integrity=check_video_integrity,
                workers=workers,
                max_findings_per_check=max_findings_per_check,
                progress_callback=progress_callback,
                status_callback=status_callback,
                cache=cache,
            )
        except Exception as exc:
            result = _build_validate_crash_result(batch, dataset_format, exc)
        yield batch, result


def _build_validate_crash_result(
    batch: Path,
    dataset_format: DatasetFormat,
    exc: BaseException,
) -> ValidationResult:
    """Convert an uncaught exception from ``validate()`` into a
    ``ValidationResult`` carrying a single ``validate_crash`` ERROR finding.

    Shared by :func:`validate_batches` and ``_cli._run_batches`` so both
    multi-batch paths apply the same isolation: one bad batch becomes one
    failed result, not a run-wide abort.
    """
    logger.exception("validate() raised for batch %s", batch, exc_info=exc)
    crash = Finding(
        severity=Severity.ERROR,
        path=batch,
        check="validate_crash",
        message=f"{type(exc).__name__}: {exc}",
    )
    return ValidationResult(
        dataset_path=batch,
        dataset_format=dataset_format,
        passed=False,
        findings=[crash],
        label_histogram=Counter(),
        finding_counts=Counter({"validate_crash": 1}),
        finding_severity_counts={
            "error": Counter({"validate_crash": 1}),
            "warning": Counter(),
        },
    )


def _validate_pair(
    annotation_path: Path,
    video_path: Path | None,
    *,
    check_video: bool,
) -> tuple[list[Finding], Counter[str]]:
    """Validate a single annotation + optional video pair.

    Shared pipeline used by both validate_annotation() (single-pair API)
    and _validate_hmie() (directory walker). Does:
      1. Annotation schema + value checks (including label validation).
      2. Video probe (single open) for integrity findings.
      3. Video-annotation consistency cross-check, skipped when either the
         annotation failed to parse or the video couldn't be opened.

    Returns (findings, label_histogram). Callers aggregate the label
    histogram across pairs for the final ValidationResult.
    """
    findings: list[Finding] = []

    # Annotation schema + label histogram for this pair
    schema_findings, annotation, label_counter = check_annotation_schema(annotation_path)
    findings.extend(schema_findings)

    if video_path is None or not check_video:
        return findings, label_counter

    if not video_path.exists():
        findings.append(
            Finding(
                severity=Severity.ERROR,
                path=video_path,
                check="video_missing",
                message="Video file does not exist",
            )
        )
        return findings, label_counter

    # Video integrity via single open
    video_props, video_findings = probe_video(video_path)
    findings.extend(video_findings)

    # Consistency cross-check whenever we successfully opened the capture and
    # extracted metadata. We do NOT gate on video ERROR findings: things like
    # video_mid_frame, video_last_frame, video_flat_frames can all ERROR while
    # fps/frame_count/width/height are still authoritative. Skipping in that
    # case silently hides real annotation-vs-video mismatches we should catch.
    #
    # The individual consistency helpers have their own guards for missing
    # FPS/AFR/dimensions, so this is safe.
    if annotation is not None and video_props.opened:
        findings.extend(check_video_annotation_consistency(video_path, annotation, video_props=video_props))

    return findings, label_counter


def _add_discovery_findings(
    discovery: DiscoveryResult,
    path: Path,
    accumulator: _FindingAccumulator,
) -> None:
    """Convert discovery-level issues into findings on the accumulator."""
    for err in discovery.errors:
        # "No annotation files" is a coverage issue (structure is fine),
        # everything else (no snippet dirs, not a directory) is structural.
        check = "no_annotations" if "No annotation files" in err else "discovery"
        accumulator.add(Finding(severity=Severity.ERROR, path=path, check=check, message=err))

    # Orphan annotations are ERROR to match validate_annotation()'s
    # video_missing behavior: an annotation without a video is not
    # ML-usable, so the dataset should fail. Orphan videos stay WARNING
    # below because an extra video without annotations is merely extra
    # content, not broken data.
    for orphan in discovery.orphan_annotations:
        accumulator.add(
            Finding(
                severity=Severity.ERROR,
                path=orphan,
                check="orphan_annotation",
                message="Annotation has no matching video in any seq_* directory",
            )
        )

    for orphan in discovery.orphan_videos:
        accumulator.add(
            Finding(
                severity=Severity.WARNING,
                path=orphan,
                check="orphan_video",
                message="Video has no matching CDAO annotation",
            )
        )

    # seq_mp4 directories with more than one mp4 are ambiguous; the
    # discovery layer deterministically picks the lexicographic first,
    # but surface the extras so the user knows half the content may be
    # silently ignored. Only a WARNING -- the chosen video is still valid.
    for mp4_dir, count in discovery.multi_video_dirs:
        accumulator.add(
            Finding(
                severity=Severity.WARNING,
                path=mp4_dir,
                check="multiple_videos_in_seq_mp4",
                message=f"{count} video files in {mp4_dir.name}/; only the lexicographic first is validated",
            )
        )


def _validate_hmie(
    path: Path,
    *,
    check_video: bool = True,
    workers: int | None = None,
    max_findings_per_check: int | None = None,
    progress_callback: Callable[[], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
    cache: ValidationCache | None = None,
) -> ValidationResult:
    """Validate an HMIE/Scale dataset at a directory path.

    Discovers annotation/video pairs using the HMIE folder structure,
    then validates each pair (in parallel when ``workers != 1``).

    Findings are appended to an accumulator that is capped per check
    name via ``max_findings_per_check`` (None = unlimited). The
    ``finding_counts`` Counter tracks the real per-check totals
    regardless of the cap, so callers always get accurate counts.
    """
    accumulator = _FindingAccumulator(max_findings_per_check)

    def _status(msg: str) -> None:
        if status_callback is not None:
            status_callback(msg)

    if not path.exists():
        accumulator.add(Finding(severity=Severity.ERROR, path=path, check="path_exists", message="Path does not exist"))
        return _build_result(path, accumulator, Counter())

    if not path.is_dir():
        accumulator.add(
            Finding(severity=Severity.ERROR, path=path, check="path_is_dir", message="Path is not a directory")
        )
        return _build_result(path, accumulator, Counter())

    # Discover annotation/video pairs
    _status("Scanning directory...")
    discovery = discover_hmie_pairs(path)
    _add_discovery_findings(discovery, path, accumulator)

    n_pairs = len(discovery.pairs)
    if n_pairs > 0:
        _status(f"Found {n_pairs:,} pairs, validating...")
    else:
        _status("No pairs found")

    # Validate each discovered pair via the shared pipeline
    aggregate_labels: Counter[str] = Counter()

    if cache is not None:
        _validate_pairs_cached(
            discovery.pairs,
            cache,
            accumulator,
            aggregate_labels,
            check_video=check_video,
            workers=workers,
            progress_callback=progress_callback,
        )
    else:
        _validate_pairs_parallel(
            discovery.pairs,
            accumulator,
            aggregate_labels,
            check_video=check_video,
            workers=workers,
            progress_callback=progress_callback,
        )

    # Count unique snippet directories. Multiple labelers per snippet produce
    # multiple pairs for the same snippet dir, so deduplicate by snippet dir.
    # Prefer pair.snippet_dir (correct for batch-level scale/ layouts, where
    # annotation_path.parent.parent collapses every annotation onto the batch
    # root); fall back to parent.parent for the per-snippet labeler layout.
    # Note: orphan_videos is a list of video paths, not snippet dirs, so it is
    # NOT added to this count -- adding it inflates snippet_count incorrectly.
    snippet_dirs_seen = set()
    for pair in discovery.pairs:
        snippet_dirs_seen.add(pair.snippet_dir or pair.annotation_path.parent.parent)
    snippet_count = len(snippet_dirs_seen)
    annotation_count = len(discovery.pairs)

    return _build_result(
        path,
        accumulator,
        aggregate_labels,
        snippet_count,
        annotation_count,
        cache_hits=cache.stats.hits if cache else 0,
        cache_misses=cache.stats.misses if cache else 0,
        skipped_checks=_video_skips(check_video),
    )


class _FindingAccumulator:
    """Collects findings with an optional per-check cap.

    Keeps the real per-check total in ``counts`` even when individual
    findings are dropped because of the cap. Tracks whether any ERROR
    was ever seen so the pass/fail verdict is cap-independent.
    """

    def __init__(self, max_per_check: int | None) -> None:
        self._max_per_check = max_per_check
        self.findings: list[Finding] = []
        self.counts: Counter[str] = Counter()
        self.severity_counts: dict[str, Counter[str]] = {
            "error": Counter(),
            "warning": Counter(),
        }
        self.has_error: bool = False

    def add(self, finding: Finding) -> None:
        self.counts[finding.check] += 1
        self.severity_counts[finding.severity.value][finding.check] += 1
        if finding.severity == Severity.ERROR:
            self.has_error = True
        if self._max_per_check is None or self.counts[finding.check] <= self._max_per_check:
            self.findings.append(finding)


def _video_skips(check_video: bool) -> set[str]:
    """Logical checks skipped when video integrity is disabled."""
    if check_video:
        return set()
    return {SKIP_VIDEO_INTEGRITY, SKIP_VIDEO_CONSISTENCY}


def _build_result(
    path: Path,
    accumulator: _FindingAccumulator,
    aggregate_labels: Counter[str],
    snippet_count: int = 0,
    annotation_count: int = 0,
    cache_hits: int = 0,
    cache_misses: int = 0,
    skipped_checks: set[str] | None = None,
) -> ValidationResult:
    return ValidationResult(
        dataset_path=path,
        dataset_format=DatasetFormat.HMIE,
        passed=not accumulator.has_error,
        findings=accumulator.findings,
        label_histogram=aggregate_labels,
        finding_counts=accumulator.counts,
        finding_severity_counts=accumulator.severity_counts,
        snippet_count=snippet_count,
        annotation_count=annotation_count,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        skipped_checks=skipped_checks or set(),
    )


def _validate_pairs_parallel(
    pairs: list[SnippetPair],
    accumulator: _FindingAccumulator,
    aggregate_labels: Counter[str],
    *,
    check_video: bool,
    workers: int | None,
    progress_callback: Callable[[], None] | None = None,
) -> None:
    """Validate pairs using the parallel/serial worker pool (no cache)."""
    for _pair, pair_findings, pair_labels in _validate_and_yield(pairs, check_video=check_video, workers=workers):
        for finding in pair_findings:
            accumulator.add(finding)
        aggregate_labels.update(pair_labels)
        if progress_callback is not None:
            progress_callback()


def _validate_pairs_cached(
    pairs: list[SnippetPair],
    cache: ValidationCache,
    accumulator: _FindingAccumulator,
    aggregate_labels: Counter[str],
    *,
    check_video: bool,
    workers: int | None = None,
    progress_callback: Callable[[], None] | None = None,
) -> None:
    """Validate pairs with cache, using parallel workers for uncached pairs.

    Phase 1: Check cache for all pairs (fast, main process).
    Phase 2: Send uncached pairs through the parallel worker pool.
    Phase 3: Store results in cache as they return.
    """
    # Phase 1: check cache for all pairs, collect uncached ones
    uncached: list[SnippetPair] = []
    for pair in pairs:
        hit = cache.lookup(pair.annotation_path, pair.video_path, check_video=check_video)
        if hit is not None:
            for finding in hit.findings:
                accumulator.add(finding)
            aggregate_labels.update(hit.labels)
            if progress_callback is not None:
                progress_callback()
        else:
            uncached.append(pair)

    if not uncached:
        return

    # Phase 2: validate uncached pairs in parallel, store as they complete.
    # Skip caching any result that contains a worker_crash finding -- those
    # indicate transient failures (OOM, pickle, killed worker) and must not
    # be persisted as cache hits on subsequent runs.
    for pair, pair_findings, pair_labels in _validate_and_yield(uncached, check_video=check_video, workers=workers):
        for finding in pair_findings:
            accumulator.add(finding)
        aggregate_labels.update(pair_labels)
        if not any(f.check == "worker_crash" for f in pair_findings):
            cache.store(pair.annotation_path, pair.video_path, pair_findings, pair_labels, check_video=check_video)
        if progress_callback is not None:
            progress_callback()


def _validate_and_yield(
    pairs: list[SnippetPair],
    *,
    check_video: bool,
    workers: int | None,
) -> Iterator[tuple[SnippetPair, list[Finding], Counter[str]]]:
    """Yield (pair, findings, labels) per snippet. Completion order under parallel mode, input order serial."""
    resolved_workers = workers if workers is not None else (os.cpu_count() or 1)
    resolved_workers = min(resolved_workers, len(pairs))

    if len(pairs) <= 1 or resolved_workers == 1:
        for pair in pairs:
            findings, labels = _safe_validate_pair(pair, check_video=check_video)
            yield pair, findings, labels
        return

    with ProcessPoolExecutor(max_workers=resolved_workers) as pool:
        future_to_pair = {pool.submit(_safe_validate_pair, pair, check_video=check_video): pair for pair in pairs}
        for future in as_completed(future_to_pair):
            pair = future_to_pair[future]
            # Belt-and-braces: _safe_validate_pair catches in-worker exceptions, but pool-level
            # failures (pickle errors, OOM kills, SIGSEGV in cv2) still surface here.
            try:
                findings, labels = future.result()
            except Exception as e:
                logger.exception("Worker future raised for %s", pair.annotation_path)
                findings = [
                    Finding(
                        severity=Severity.ERROR,
                        path=pair.annotation_path,
                        check="worker_crash",
                        message=f"Worker future raised {type(e).__name__}: {e}",
                    )
                ]
                labels = Counter()
            yield pair, findings, labels


def _safe_validate_pair(pair: SnippetPair, *, check_video: bool) -> tuple[list[Finding], Counter[str]]:
    """Wrap _validate_pair so worker exceptions become worker_crash findings, not run-killers."""
    try:
        return _validate_pair(pair.annotation_path, pair.video_path, check_video=check_video)
    except Exception as e:
        logger.exception("Worker raised while validating %s", pair.annotation_path)
        crash_finding = Finding(
            severity=Severity.ERROR,
            path=pair.annotation_path,
            check="worker_crash",
            message=f"Worker raised {type(e).__name__}: {e}",
        )
        return [crash_finding], Counter()
