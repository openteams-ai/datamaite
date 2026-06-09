"""Tests that run against real HMIE data on SUNet.

These tests are skipped by default. To run them locally when you have
access to real HMIE data, set the DATABRIDGE_HMIE_ROOT environment
variable to the dataset root and invoke pytest with -m real_data:

    export DATABRIDGE_HMIE_ROOT=/path/to/hmie/dataset
    uv run pytest -m real_data

These tests verify that the validator's assumptions about the HMIE folder
structure (derived from ri-demo and jatic/program-tasks#143) hold against
the actual data on the CDAO SUNet machine.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from databridge._formats.hmie.discovery import discover_hmie_pairs
from databridge._types import Severity
from databridge.validation import validate

pytestmark = pytest.mark.real_data


@pytest.fixture
def hmie_root() -> Path:
    root = os.environ.get("DATABRIDGE_HMIE_ROOT")
    if not root:
        pytest.skip("DATABRIDGE_HMIE_ROOT not set")
    path = Path(root)
    if not path.is_dir():
        pytest.skip(f"DATABRIDGE_HMIE_ROOT does not exist: {path}")
    return path


class TestRealHmieDiscovery:
    def test_discovery_finds_pairs(self, hmie_root: Path) -> None:
        """Discovery should find at least one annotation/video pair."""
        result = discover_hmie_pairs(hmie_root)
        assert result.errors == [], f"Discovery errors: {result.errors}"
        assert len(result.pairs) > 0, "No CDAO/seq_mp4 pairs found -- folder structure assumptions may be wrong"

    def test_most_pairs_have_videos(self, hmie_root: Path) -> None:
        """Most discovered annotations should have matching videos."""
        result = discover_hmie_pairs(hmie_root)
        with_video = [p for p in result.pairs if p.video_path is not None]
        ratio = len(with_video) / max(len(result.pairs), 1)
        assert ratio > 0.5, f"Only {ratio:.0%} of annotations have matching videos; structure may differ"


class TestRealHmieValidation:
    def test_validate_small_subset(self, hmie_root: Path) -> None:
        """Run the full validator against real data and report stats.

        This test does not assert PASS -- real data is expected to have
        failures (that's the whole point of databridge). It asserts that
        validation completes without crashing and produces a report.
        """
        result = validate(hmie_root)
        errors = [f for f in result.findings if f.severity == Severity.ERROR]
        warnings = [f for f in result.findings if f.severity == Severity.WARNING]

        # Print for human inspection when running manually
        print("\n=== Real HMIE Validation Report ===")
        print(f"Root: {hmie_root}")
        print(f"Errors: {len(errors)}")
        print(f"Warnings: {len(warnings)}")

        # Group by check name for a quick histogram
        from collections import Counter

        check_counts = Counter(f.check for f in result.findings)
        print("\nFindings by check:")
        for check, count in check_counts.most_common():
            print(f"  {check}: {count}")

        # The only hard assertion: validation completed
        assert result.dataset_path == hmie_root


class TestRealHmieLoader:
    def test_load_produces_sequences(self, hmie_root: Path) -> None:
        """Load real data and report what the dataloader produced.

        Verifies the loader's discovery + parsing assumptions hold against
        the actual SUNet layout: at least one sequence with boxes, and a
        non-empty category map.
        """
        from databridge._formats.hmie.loader import load_hmie

        ds = load_hmie(hmie_root)

        print("\n=== Real HMIE Load Report ===")
        print(f"Root: {hmie_root}")
        print(f"Sequences: {len(ds.sequences)}")
        print(f"Total boxes: {ds.num_boxes}")
        print(f"Categories: {len(ds.categories)}")

        assert len(ds.sequences) > 0, "Loader found no sequences -- folder/schema assumptions may be wrong"
        assert ds.num_boxes > 0, "Sequences loaded but no boxes parsed"
