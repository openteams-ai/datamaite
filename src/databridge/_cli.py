"""CLI entrypoint for databridge."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from databridge._types import DatasetFormat

if TYPE_CHECKING:
    from databridge._cache import ValidationCache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="databridge",
        description="Dataset validation, loading, and conversion.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show individual findings.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress output (for scripts).")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging (very detailed).")
    sub = parser.add_subparsers(dest="command")

    val_parser = sub.add_parser("validate", help="Validate a dataset.")
    val_parser.add_argument("path", type=Path, help="Path to the dataset root.")
    val_parser.add_argument(
        "--format",
        default=DatasetFormat.HMIE.value,
        choices=[f.value for f in DatasetFormat],
        help="Dataset format (default: hmie).",
    )
    val_parser.add_argument("--skip-video-check", action="store_true", help="Skip FMV integrity checks.")
    val_parser.add_argument(
        "--output",
        "-o",
        nargs="?",
        const="auto",
        default=None,
        help=(
            "Write report to a file. Format based on extension:"
            " .html (interactive report), .json (machine-readable),"
            " .txt (plain text)."
        ),
    )
    val_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Number of worker processes for parallel per-pair validation. "
            "Default: os.cpu_count(). Use 1 to force serial execution."
        ),
    )
    val_parser.add_argument(
        "--max-findings-per-check",
        type=int,
        default=None,
        help=(
            "Upper bound on how many individual findings to keep per check name. "
            "Use to bound memory on very large datasets. finding_counts remains "
            "accurate even when findings are capped."
        ),
    )
    val_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip validation cache for this run (neither read nor write).",
    )
    val_parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete validation cache before running (forces full rescan).",
    )
    output_group = val_parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Emit the full validation result as a single JSON object on stdout.",
    )
    output_group.add_argument(
        "--jsonl",
        action="store_true",
        help=(
            "Emit newline-separated JSONL: one summary record followed by one "
            "finding record per line. Convenient for piping to jq or grep."
        ),
    )

    stats_parser = sub.add_parser("stats", help="Summarize a dataset's duration/frame/box distributions.")
    stats_parser.add_argument("path", type=Path, help="Path to the dataset root.")
    stats_parser.add_argument(
        "--format",
        default=DatasetFormat.HMIE.value,
        choices=[f.value for f in DatasetFormat],
        help="Dataset format (default: hmie).",
    )
    stats_parser.add_argument(
        "--require-video",
        action="store_true",
        help="Probe each video for true frame counts (slower; needs the `video` extra).",
    )
    stats_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the statistics as a single JSON object on stdout.",
    )

    args = parser.parse_args(argv)

    if hasattr(args, "workers") and args.workers is not None:
        if args.workers < 1:
            parser.error("--workers must be >= 1")
        args.workers = min(args.workers, 64)

    # Configure logging based on flags.
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    elif args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "validate":
        return _cmd_validate(args)

    if args.command == "stats":
        return _cmd_stats(args)

    return 0


def _is_interactive() -> bool:
    """Return True if stderr is a TTY (safe to write progress indicators)."""
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


def _find_batch_dirs(root: Path) -> list[Path]:
    """Find batch directories under ``root``.

    Thin adapter over ``discovery.find_batch_roots`` so CLI call sites
    keep a stable name while the snippet-detection rule lives in the
    discovery module (the single source of truth).
    """
    from databridge._formats.hmie.discovery import find_batch_roots

    return find_batch_roots(root)


def _rpad(text: str, width: int) -> str:
    """Right-align text to visible width, accounting for ANSI escape codes."""
    import re

    visible_len = len(re.sub(r"\033\[[0-9;]*m", "", text))
    pad = max(0, width - visible_len)
    return " " * pad + text


def _print_batch_table(results: list[tuple[Path, Any]]) -> None:
    """Print a multi-batch summary table with aligned columns."""
    from databridge._formats.hmie.categories import _categorize_findings
    from databridge._types import _dim, _status_indicator

    name_w = max(len(d.name) for d, _ in results)
    name_w = max(name_w, 5)
    col = 6  # column width for status indicators

    # Header
    print(
        f"  {'Batch':<{name_w}s}  {'Snippets':>8s}  {'Annots':>6s}"
        f"  {'Struct':>{col}s}  {'Video':>{col}s}  {'Cover':>{col}s}  {'Scale':>{col}s}"
        f"  {'Errors':>6s}  {'Warns':>6s}"
    )
    print("  " + "-" * (name_w + 60))

    for batch_dir, result in results:
        cats = _categorize_findings(result.finding_severity_counts)
        s_errs, _ = cats["structure"]
        struct_failed = s_errs > 0

        # Uncapped totals from severity_counts — result.findings may be
        # truncated when max_findings_per_check is set.
        error_count = sum(result.finding_severity_counts.get("error", Counter()).values())
        warn_count = sum(result.finding_severity_counts.get("warning", Counter()).values())

        s_ind = _status_indicator(*cats["structure"])

        c_errs, _ = cats["coverage"]
        no_annotations = result.annotation_count == 0 and c_errs > 0

        if struct_failed:
            snip_str = "       -"
            ann_str = "     -"
            v_ind = _dim("N/A")
            c_ind = _dim("N/A")
            sc_ind = _dim("N/A")
        elif no_annotations:
            snip_str = f"{result.snippet_count:>8,}"
            ann_str = "     -"
            v_ind = _dim("N/A")
            c_ind = _status_indicator(*cats["coverage"])
            sc_ind = _dim("N/A")
        else:
            snip_str = f"{result.snippet_count:>8,}"
            ann_str = f"{result.annotation_count:>6,}"
            v_ind = _status_indicator(*cats["video"])
            c_ind = _status_indicator(*cats["coverage"])
            sc_ind = _status_indicator(*cats["scale_spec"])

        print(
            f"  {batch_dir.name:<{name_w}s}  {snip_str}  {ann_str}"
            f"  {_rpad(s_ind, col)}  {_rpad(v_ind, col)}  {_rpad(c_ind, col)}  {_rpad(sc_ind, col)}"
            f"  {error_count:>6,}  {warn_count:>6,}"
        )

    print()


def _resolve_output_path(args: argparse.Namespace, root: Path) -> None:
    """Resolve --output to a Path, generating an auto filename when needed."""
    if args.output == "auto":
        from datetime import datetime, timezone

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
        dataset_name = root.name.replace(" ", "_")
        args.output = Path(f"databridge-report-{dataset_name}-{timestamp}.html")
    if args.output and isinstance(args.output, str):
        args.output = Path(args.output)


def _init_cache(args: argparse.Namespace) -> ValidationCache | None:
    """Create the validation cache unless --no-cache is set; honour --clean."""
    from databridge._cache import ValidationCache

    if getattr(args, "no_cache", False):
        return None
    cache = ValidationCache()
    if getattr(args, "clean", False):
        cache.clear()
    return cache


def _install_sigint_flush(cache: ValidationCache | None) -> Callable[[], None]:
    """Install a SIGINT handler that flushes the cache before re-raising.

    At 60K SUNet scale, Ctrl-C during a long run would otherwise lose
    up to 49 pending cache entries (we batch commits every 50 writes).
    Returns a callable that restores the previous handler.
    """
    import signal

    prev_sigint = signal.getsignal(signal.SIGINT)

    def _on_sigint(_signum: int, _frame: object) -> None:
        if cache is not None:
            cache.flush()
            cache.close()
        signal.signal(signal.SIGINT, prev_sigint)
        raise KeyboardInterrupt

    with contextlib.suppress(ValueError):
        signal.signal(signal.SIGINT, _on_sigint)

    def _restore() -> None:
        with contextlib.suppress(ValueError, TypeError):
            signal.signal(signal.SIGINT, prev_sigint)

    return _restore


def _cmd_stats(args: argparse.Namespace) -> int:
    """Run the stats subcommand: load the dataset and print distributions."""
    from databridge._stats import dataset_stats, format_stats
    from databridge.loaders import load

    show = _show_progress(args)
    if show:
        sys.stderr.write(f"\r\033[K  Loading {args.path}...\n")
        sys.stderr.flush()

    ds = load(args.path, dataset_format=args.format, require_video=args.require_video)
    stats = dataset_stats(ds)

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print(format_stats(stats, root=str(args.path)))
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Run the validate subcommand."""
    root = args.path
    _resolve_output_path(args, root)
    batch_dirs = _find_batch_dirs(root)
    cache = _init_cache(args)
    restore_sigint = _install_sigint_flush(cache)

    try:
        if len(batch_dirs) <= 1:
            return _validate_single(args, root, cache)
        return _validate_multi(args, root, batch_dirs, cache)
    finally:
        if cache:
            cache.close()
        restore_sigint()


def _show_progress(args: argparse.Namespace) -> bool:
    """Return True if we should show progress output."""
    if getattr(args, "quiet", False):
        return False
    return _is_interactive()


def _make_status_callback(show: bool) -> Callable[[str], None]:
    """Create a status callback that prints phase messages to stderr."""

    def _on_status(msg: str) -> None:
        if show:
            sys.stderr.write(f"\r\033[K  {msg}\n")
            sys.stderr.flush()

    return _on_status


def _print_result(args: argparse.Namespace, result: Any) -> None:
    """Print validation result in the requested format."""
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    elif args.jsonl:
        print(result.to_jsonl())
    else:
        verbose = getattr(args, "verbose", False)
        print(result.summary(show_findings=verbose))
        if not verbose and result.findings:
            print("  Use -v to show individual findings.")
            print()

    if args.output:
        ext = args.output.suffix.lower()
        if ext == ".html":
            from databridge._report import render_html_report

            args.output.write_text(render_html_report(result), encoding="utf-8")
        elif ext == ".json":
            args.output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        else:
            # File output must not contain ANSI escape codes.
            report = result.summary(show_findings=True, max_findings=None, use_color=False)
            args.output.write_text(report, encoding="utf-8")
        # Status line goes to stderr so --json/--jsonl stdout stays pipe-clean.
        print(f"  Report written to {args.output}", file=sys.stderr)
        print(file=sys.stderr)


def _validate_single(
    args: argparse.Namespace,
    path: Path,
    cache: ValidationCache | None = None,
) -> int:
    """Validate a single dataset and print detailed output."""
    from databridge.validation import validate

    show = _show_progress(args)
    count = 0

    def _on_pair_done() -> None:
        nonlocal count
        count += 1
        if show:
            sys.stderr.write(f"\r\033[K  Validating... {count:,} pairs completed")
            sys.stderr.flush()

    result = validate(
        path,
        dataset_format=args.format,
        check_video_integrity=not args.skip_video_check,
        workers=args.workers,
        max_findings_per_check=args.max_findings_per_check,
        progress_callback=_on_pair_done,
        status_callback=_make_status_callback(show),
        cache=cache,
    )
    if show:
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()

    _print_result(args, result)

    if not args.json and not args.jsonl and cache and (cache.stats.hits + cache.stats.misses) > 0:
        total = cache.stats.hits + cache.stats.misses
        print(f"  Cache: {cache.stats.hits:,} cached, {cache.stats.misses:,} validated (of {total:,} pairs)")
        print()

    if not result.passed:
        return 2
    if result.findings:
        return 1
    return 0


def _make_batch_progress(batch_idx: int, total_batches: int, batch_name: str, show: bool) -> Callable[[], None]:
    """Create a progress callback for a specific batch, binding loop variables."""
    count = 0

    def _on_pair_done() -> None:
        nonlocal count
        count += 1
        if show:
            sys.stderr.write(f"\r\033[K  [{batch_idx}/{total_batches}] {batch_name}  {count:,} pairs...")
            sys.stderr.flush()

    return _on_pair_done


def _run_batches(
    args: argparse.Namespace,
    batch_dirs: list[Path],
    cache: ValidationCache | None = None,
) -> list[tuple[Path, Any]]:
    """Run validation for each batch directory and return results.

    Mirrors ``validate_batches`` on crash isolation: if ``validate()`` raises
    for one batch, build a ``validate_crash`` result via the shared helper
    so the rest of the run still proceeds. The library helper and the CLI
    both converge on the same finding shape.
    """
    from databridge._types import DatasetFormat
    from databridge.validation import _build_validate_crash_result, validate

    results: list[tuple[Path, Any]] = []
    n_batches = len(batch_dirs)
    show = _show_progress(args)
    dataset_format = args.format if isinstance(args.format, DatasetFormat) else DatasetFormat(args.format.lower())

    for i, batch_dir in enumerate(batch_dirs, 1):
        callback = _make_batch_progress(i, n_batches, batch_dir.name, show)
        prefix = f"[{i}/{n_batches}] {batch_dir.name}"

        def _on_status(msg: str, _prefix: str = prefix) -> None:
            if show:
                sys.stderr.write(f"\r\033[K  {_prefix}  {msg}")
                sys.stderr.flush()

        if show:
            sys.stderr.write(f"  {prefix}...\n")
            sys.stderr.flush()

        try:
            result = validate(
                batch_dir,
                dataset_format=dataset_format,
                check_video_integrity=not args.skip_video_check,
                workers=args.workers,
                max_findings_per_check=args.max_findings_per_check,
                progress_callback=callback,
                status_callback=_on_status,
                cache=cache,
            )
        except Exception as exc:
            result = _build_validate_crash_result(batch_dir, dataset_format, exc)
        results.append((batch_dir, result))

        if show:
            sys.stderr.write(f"\r\033[K  [{i}/{n_batches}] {batch_dir.name}  done\n")
            sys.stderr.flush()

    return results


def _multi_exit_code(results: list[tuple[Path, Any]]) -> int:
    """Return exit code for a multi-batch run.

    Matches single-batch semantics: 2 if any batch has ERROR findings,
    1 if any batch has WARNING findings (but no errors across the run),
    else 0. Previously this collapsed warnings into "pass" and only
    distinguished clean vs errored, which disagreed with the README.
    """
    if any(not r.passed for _, r in results):
        return 2
    if any(r.findings for _, r in results):
        return 1
    return 0


def _validate_multi(
    args: argparse.Namespace,
    root: Path,
    batch_dirs: list[Path],
    cache: ValidationCache | None = None,
) -> int:
    """Validate multiple batch directories and print a summary table."""
    from databridge._types import (
        _bold,
        _green,
        _red,
        _yellow,
    )

    results = _run_batches(args, batch_dirs, cache)

    if args.json:
        all_results = [r.to_dict() for _, r in results]
        print(json.dumps(all_results, indent=2))
        return _multi_exit_code(results)

    if args.jsonl:
        for _, r in results:
            print(r.to_jsonl())
        return _multi_exit_code(results)

    # Print table
    print()
    print(f"  {_bold(str(root))}")
    print("  " + "=" * 68)
    print()

    _print_batch_table(results)

    total = len(results)
    failed = sum(1 for _, r in results if not r.passed)
    warned = sum(1 for _, r in results if r.passed and r.findings)
    clean = total - failed - warned
    print(f"  {_green(f'{clean} clean')} · {_yellow(f'{warned} warnings')} · {_red(f'{failed} failed')} (of {total})")
    print()

    print("  Validate a specific batch for details:")
    print(f"    databridge validate {root}/<batch-name>")
    print()

    # Write full report if requested
    if args.output:
        ext = args.output.suffix.lower()
        if ext == ".html":
            from databridge._report import render_html_report_multi

            args.output.write_text(
                render_html_report_multi(results, root),
                encoding="utf-8",
            )
        elif ext == ".json":
            all_results = [r.to_dict() for _, r in results]
            args.output.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        else:
            report_lines = [f"Multi-batch validation: {root}\n"]
            for _batch_dir, result in results:
                # File output must not contain ANSI escape codes.
                report_lines.append(result.summary(show_findings=True, max_findings=None, use_color=False))
                report_lines.append("")
            args.output.write_text("\n".join(report_lines), encoding="utf-8")
        # Status line goes to stderr so --json/--jsonl stdout stays pipe-clean.
        print(f"  Report written to {args.output}", file=sys.stderr)
        print(file=sys.stderr)

    return _multi_exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
