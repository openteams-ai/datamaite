"""Smoke tests for package."""

from __future__ import annotations


def test_import() -> None:
    import databridge

    assert databridge.__version__


def test_public_api() -> None:
    from databridge import (
        DatasetFormat,
        Finding,
        MotChallengeLoader,
        Severity,
        TaoLoader,
        ValidationResult,
        VisDroneVideoLoader,
        load_motchallenge,
        load_tao,
        load_visdrone_video,
        validate,
        validate_annotation,
    )

    assert DatasetFormat.HMIE.value == "hmie"
    assert DatasetFormat.MOTCHALLENGE.value == "motchallenge"
    assert DatasetFormat.TAO.value == "tao"
    assert DatasetFormat.VISDRONE_VIDEO.value == "visdrone_video"
    assert Severity.ERROR.value == "error"
    assert Severity.WARNING.value == "warning"
    assert Finding is not None
    assert ValidationResult is not None
    assert MotChallengeLoader is not None
    assert TaoLoader is not None
    assert VisDroneVideoLoader is not None
    assert callable(load_motchallenge)
    assert callable(load_tao)
    assert callable(load_visdrone_video)
    assert callable(validate)
    assert callable(validate_annotation)


def test_version_module_shape() -> None:
    """The committed _version.py fallback exposes the attrs __init__.py imports.

    Regression guard: if someone re-adds src/databridge/_version.py to
    .gitignore, `poetry install` + `pytest` on a fresh clone will fail
    because hatch-vcs doesn't run for editable installs.
    """
    from databridge import _version

    assert isinstance(_version.__version__, str)
    assert _version.__version__  # non-empty
    assert isinstance(_version.__version_tuple__, tuple)
    assert len(_version.__version_tuple__) >= 1


def test_version_tuple_parsing() -> None:
    """_as_tuple preserves numeric segments and falls back to strings."""
    from databridge._version import _as_tuple

    assert _as_tuple("0.1.0") == (0, 1, 0)
    assert _as_tuple("0.1.dev59") == (0, 1, "dev59")
    assert _as_tuple("0.0.0+unknown") == (0, 0, 0, "unknown")
