"""Smoke tests for package."""

from __future__ import annotations


def test_import() -> None:
    import databridge

    assert databridge.__version__ == "0.1.0"


def test_public_api() -> None:
    from databridge import DatasetFormat, Finding, Severity, ValidationResult

    assert DatasetFormat.HMIE.value == "hmie"
    assert Severity.ERROR.value == "error"
    assert Severity.WARNING.value == "warning"
    assert Finding is not None
    assert ValidationResult is not None
