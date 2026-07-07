"""Opt-in integration tests against a real cloud object store.

Requires ``DATAMAITE_CLOUD_TEST_ROOT`` (an ``s3://`` or ``gs://`` URL of a
small HMIE-layout tree with generic test content) plus provider credentials
in the environment, and the matching backend extra installed. Run with:

    DATAMAITE_CLOUD_TEST_ROOT=s3://bucket/prefix poetry run pytest -m integration tests/e2e/test_cloud_integration.py
"""

from __future__ import annotations

import os

import pytest

from datamaite import load_mot, validate

_CLOUD_ROOT = os.environ.get("DATAMAITE_CLOUD_TEST_ROOT")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _CLOUD_ROOT, reason="DATAMAITE_CLOUD_TEST_ROOT not set"),
]


def test_load_with_video_probe_streams():
    # require_video forces a real probe per snippet; on S3/GCS this
    # exercises PyAV streaming over a ranged-read fsspec file object end to end.
    ds = load_mot(_CLOUD_ROOT, dataset_format="hmie", require_video=True)
    assert ds.sequence_count > 0
    assert all(seq.num_frames_exact for seq in ds.sequences)


def test_validate_full_integrity():
    result = validate(_CLOUD_ROOT, workers=1)
    assert result.annotation_count > 0
    scheme = _CLOUD_ROOT.split("://", 1)[0] + "://"
    assert all(str(f.path).startswith(scheme) for f in result.findings), "findings must carry logical dataset paths"
