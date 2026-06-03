"""Databridge -- A unified framework for dataset loading, conversion, and quality validation."""

import logging

from databridge._cache import ValidationCache
from databridge._formats.hmie.discovery import find_batch_roots
from databridge._report import render_html_report
from databridge._stats import dataset_stats
from databridge._types import DatasetFormat, Finding, Severity, ValidationResult
from databridge._version import __version__, __version_tuple__
from databridge.dataloader import HmieLoader, load_hmie
from databridge.loaders import Loader, available_formats, get_loader, load, register_loader
from databridge.model import BoxAnnotation, BoxTrackDataset, VideoSequence
from databridge.validation import validate, validate_annotation, validate_batches

# Library convention: attach a NullHandler so downstream applications
# that don't configure logging don't see "No handler found" warnings.
# Consumers who want databridge logs can add their own handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "BoxAnnotation",
    "BoxTrackDataset",
    "DatasetFormat",
    "Finding",
    "HmieLoader",
    "Loader",
    "Severity",
    "ValidationCache",
    "ValidationResult",
    "VideoSequence",
    "__version__",
    "__version_tuple__",
    "available_formats",
    "dataset_stats",
    "find_batch_roots",
    "get_loader",
    "load",
    "load_hmie",
    "register_loader",
    "render_html_report",
    "validate",
    "validate_annotation",
    "validate_batches",
]
