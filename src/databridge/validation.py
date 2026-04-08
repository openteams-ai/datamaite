"""Dataset validation utilities.

This module provides validation functions for verifying dataset structure,
file integrity, and annotation schema conformance.
"""

from __future__ import annotations

from pathlib import Path

from databridge._types import DatasetFormat, ValidationResult


def validate(
    path: str | Path,
    format: DatasetFormat | str = DatasetFormat.HMIE,  # noqa: A002
    *,
    check_video_integrity: bool = True,
) -> ValidationResult:
    """Validate a dataset at the given path.

    Parameters
    ----------
    path
        Root directory of the dataset.
    format
        Dataset format to validate against.
    check_video_integrity
        If True, attempt to open FMV files to verify they are not corrupted.
        Requires the ``video`` extra (``pip install databridge[video]``).

    Returns
    -------
    ValidationResult
        Contains pass/fail status and a list of findings.
    """
    path = Path(path)
    if isinstance(format, str):
        format = DatasetFormat(format.lower())  # noqa: A001

    if format == DatasetFormat.HMIE:
        return _validate_hmie(path, check_video_integrity=check_video_integrity)

    msg = f"Unsupported format: {format}"
    raise ValueError(msg)


def _validate_hmie(
    path: Path,
    *,
    check_video_integrity: bool = True,
) -> ValidationResult:
    """Validate an HMIE/Scale dataset. Implementation in #634."""
    _ = check_video_integrity
    return ValidationResult(dataset_path=path, dataset_format=DatasetFormat.HMIE)
