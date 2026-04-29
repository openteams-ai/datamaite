"""Tests for HMIE finding categorization."""

from __future__ import annotations

from collections import Counter

import pytest

from databridge._formats.hmie.categories import _CHECK_CATEGORIES, _categorize_findings


class TestCategorizeFindings:
    def test_empty_counts(self) -> None:
        cats = _categorize_findings({})
        assert cats == {"structure": (0, 0), "video": (0, 0), "coverage": (0, 0), "scale_spec": (0, 0)}

    def test_all_four_buckets(self) -> None:
        counts = {
            "error": Counter({"discovery": 1, "video_open": 2, "no_annotations": 4, "annotation_schema": 8}),
            "warning": Counter(),
        }
        cats = _categorize_findings(counts)
        assert cats["structure"] == (1, 0)
        assert cats["video"] == (2, 0)
        assert cats["coverage"] == (4, 0)
        assert cats["scale_spec"] == (8, 0)

    def test_only_warnings(self) -> None:
        counts = {"error": Counter(), "warning": Counter({"orphan_annotation": 3, "video_fps": 2})}
        cats = _categorize_findings(counts)
        assert cats["coverage"] == (0, 3)
        assert cats["video"] == (0, 2)

    def test_same_check_both_severities(self) -> None:
        # Error and warning streams must not cross-pollute.
        counts = {"error": Counter({"consistency_fps": 1}), "warning": Counter({"consistency_fps": 2})}
        cats = _categorize_findings(counts)
        assert cats["scale_spec"] == (1, 2)

    def test_unknown_check_raises(self) -> None:
        # Loud failure forces new checks to register in _CHECK_CATEGORIES.
        # A silent fallback would let a future structure/video/coverage
        # check land in scale_spec and quietly distort the dashboard.
        with pytest.raises(KeyError):
            _categorize_findings({"error": Counter({"not_a_real_check": 1}), "warning": Counter()})

    def test_unknown_warning_check_raises(self) -> None:
        # Symmetric to test_unknown_check_raises for the warning path.
        with pytest.raises(KeyError):
            _categorize_findings({"error": Counter(), "warning": Counter({"not_a_real_check": 1})})

    def test_worker_crash_registered(self) -> None:
        # Pool-level crashes emit a worker_crash finding; the mapping must
        # exist or the dashboard categorization itself crashes on the
        # exact runs that already had a problem.
        assert _CHECK_CATEGORIES["worker_crash"] == "scale_spec"

    def test_box_missing_fields_registered(self) -> None:
        # Regression: was fired in annotation_checks but missing from
        # _CHECK_CATEGORIES, only working via the now-removed fallback.
        assert _CHECK_CATEGORIES["annotation_box_missing_fields"] == "scale_spec"

    def test_consistency_fps_invalid_registered(self) -> None:
        # Regression: was fired in consistency_checks but missing from
        # _CHECK_CATEGORIES, only working via the now-removed fallback.
        assert _CHECK_CATEGORIES["consistency_fps_invalid"] == "scale_spec"
