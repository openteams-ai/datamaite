"""Contract tests for the HMIE validator tutorial notebook.

The notebook at ``docs/tool-usage/validators/hmie.ipynb`` imports from the
public ``datamaite`` surface and calls methods with specific kwargs.
These tests codify that contract so a rename or signature change in the
library fails CI before it ships, rather than surfacing as an
``AttributeError`` at the first line of the customer's run.

The tests do not execute the notebook end-to-end (that requires SUNet
data); they just verify every symbol / attribute / kwarg the notebook
references.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path

import pytest

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "docs" / "tool-usage" / "validators" / "hmie.ipynb"


@pytest.fixture(scope="module")
def notebook_source() -> str:
    """Concatenate all code-cell source into one string for substring searches."""
    nb = json.loads(NOTEBOOK_PATH.read_text())
    sources = ["".join(cell.get("source", [])) for cell in nb["cells"] if cell["cell_type"] == "code"]
    return "\n".join(sources)


def test_notebook_parses_as_valid_ipynb() -> None:
    nb = json.loads(NOTEBOOK_PATH.read_text())
    assert nb["nbformat"] == 4
    assert any(c["cell_type"] == "code" for c in nb["cells"])


def test_notebook_imports_resolve() -> None:
    """Every symbol the notebook imports must exist in datamaite's public API."""
    import datamaite
    from datamaite import (  # noqa: F401
        ValidationCache,
        ValidationResult,
        find_batch_roots,
        render_html_report,
        render_html_report_multi,
        validate,
        validate_batches,
    )

    assert datamaite.__version__


def test_validation_cache_default_path_is_static() -> None:
    """Notebook calls ValidationCache.default_path() as a classmethod/static."""
    from datamaite import ValidationCache

    path = ValidationCache.default_path()
    assert path is not None
    assert "datamaite" in str(path)


def test_validation_cache_has_stats_with_hits_and_misses() -> None:
    """Notebook reads cache.stats.hits / cache.stats.misses to compute per-batch deltas."""
    from datamaite import ValidationCache

    cache = ValidationCache(db_path=None)
    try:
        assert hasattr(cache, "stats"), "ValidationCache must expose .stats"
        assert hasattr(cache.stats, "hits")
        assert hasattr(cache.stats, "misses")
    finally:
        cache.close()


def test_validate_accepts_notebook_kwargs() -> None:
    """Notebook passes these kwargs; validate() must accept every one."""
    from datamaite import validate

    sig = inspect.signature(validate)
    for kw in (
        "workers",
        "check_video_integrity",
        "max_findings_per_check",
        "cache",
        "progress_callback",
        "status_callback",
    ):
        assert kw in sig.parameters, f"validate() missing kwarg: {kw}"


def test_validation_result_has_fields_notebook_reads() -> None:
    """Notebook reads these attrs on ValidationResult; they must exist."""
    from datamaite import ValidationResult

    fields = {f.name for f in dataclasses.fields(ValidationResult)}
    for attr in (
        "snippet_count",
        "annotation_count",
        "finding_severity_counts",
        "passed",
        "cache_hits",
        "cache_misses",
    ):
        assert attr in fields, f"ValidationResult missing field: {attr}"


def test_result_summary_accepts_notebook_kwargs() -> None:
    """Notebook calls result.summary(show_findings=, max_findings=, use_color=)."""
    from datamaite import ValidationResult

    sig = inspect.signature(ValidationResult.summary)
    for kw in ("show_findings", "max_findings", "use_color"):
        assert kw in sig.parameters, f"ValidationResult.summary missing kwarg: {kw}"


def test_find_batch_roots_accepts_path_returns_list(tmp_path: Path) -> None:
    """Notebook calls find_batch_roots(BATCH_ROOT) and iterates the result."""
    from datamaite import find_batch_roots

    out = find_batch_roots(tmp_path)
    assert isinstance(out, list)


def test_validate_batches_accepts_notebook_kwargs() -> None:
    """Notebook calls validate_batches(BATCH_ROOT, check_video_integrity=CHECK_VIDEO);
    the signature must accept that kwarg."""
    from datamaite import validate_batches

    sig = inspect.signature(validate_batches)
    assert "check_video_integrity" in sig.parameters, "validate_batches missing kwarg: check_video_integrity"


def test_render_html_report_accepts_result(tmp_path: Path) -> None:
    """Notebook calls render_html_report(r) where r is a ValidationResult."""
    from datamaite import DatasetFormat, ValidationResult, render_html_report

    r = ValidationResult(dataset_path=tmp_path, dataset_format=DatasetFormat.HMIE)
    html = render_html_report(r)
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html


def test_render_html_report_multi_accepts_results(tmp_path: Path) -> None:
    """Notebook calls render_html_report_multi(results, BATCH_ROOT) where results
    is a list of (batch_path, ValidationResult) pairs from validate_batches."""
    from datamaite import DatasetFormat, ValidationResult, render_html_report_multi

    results = [
        (tmp_path / "batch_a", ValidationResult(dataset_path=tmp_path / "batch_a", dataset_format=DatasetFormat.HMIE)),
        (tmp_path / "batch_b", ValidationResult(dataset_path=tmp_path / "batch_b", dataset_format=DatasetFormat.HMIE)),
    ]
    html = render_html_report_multi(results, tmp_path)
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html


def test_notebook_uses_default_path_not_db_path(notebook_source: str) -> None:
    """Regression guard: notebook must NOT access cache.db_path (doesn't exist)."""
    assert "cache.db_path" not in notebook_source, (
        "Notebook references cache.db_path but ValidationCache has no such attribute; "
        "use ValidationCache.default_path() instead."
    )
