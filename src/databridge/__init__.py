"""Databridge -- A unified framework for dataset loading, conversion, and quality validation."""

import logging

from databridge._types import DatasetFormat, Finding, Severity, ValidationResult
from databridge._version import __version__, __version_tuple__
from databridge.validation import validate, validate_annotation

# Library convention: attach a NullHandler so downstream applications
# that don't configure logging don't see "No handler found" warnings.
# Consumers who want databridge logs can add their own handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "DatasetFormat",
    "Finding",
    "Severity",
    "ValidationResult",
    "__version__",
    "__version_tuple__",
    "validate",
    "validate_annotation",
]
