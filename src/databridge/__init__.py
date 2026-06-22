"""Databridge -- A unified framework for dataset loading, conversion, and quality validation."""

import logging
from importlib import import_module
from typing import TYPE_CHECKING

from databridge._cache import ValidationCache
from databridge._formats.hmie.discovery import find_batch_roots
from databridge._report import render_html_report, render_html_report_multi
from databridge._stats import dataset_stats
from databridge._types import DatasetFormat, Finding, Severity, Task, ValidationResult
from databridge._version import __version__, __version_tuple__
from databridge.conversion import convert
from databridge.loaders import Loader, available_formats, get_loader, load, load_mot, register_loader
from databridge.model import (
    BoxAnnotation,
    BoxTrackDataset,
    VideoClassificationDataset,
    VideoClassificationSample,
    VideoSequence,
    VisionDataset,
)
from databridge.taxonomy import CategoryEntry, Taxonomy
from databridge.validation import validate, validate_annotation, validate_batches
from databridge.writers import Writer, available_output_formats, get_writer, register_writer, write

if TYPE_CHECKING:
    from databridge._formats.flat_mp4.loader import FlatMp4Loader
    from databridge._formats.hmie.loader import HmieLoader
    from databridge._formats.hmie.writer import HmieWriter
    from databridge._formats.huggingface_video_classification.loader import (
        HuggingFaceVideoClassificationLoader,
        load_huggingface_video_classification,
    )
    from databridge._formats.huggingface_video_classification.writer import HuggingFaceVideoClassificationWriter
    from databridge._formats.motchallenge.loader import MotChallengeLoader
    from databridge._formats.motchallenge.writer import MotChallengeWriter
    from databridge._formats.tao.loader import TaoLoader
    from databridge._formats.tao.writer import TaoWriter
    from databridge._formats.visdrone.loader import VisDroneVideoLoader
    from databridge._formats.visdrone.writer import VisDroneVideoWriter

# Library convention: attach a NullHandler so downstream applications
# that don't configure logging don't see "No handler found" warnings.
# Consumers who want databridge logs can add their own handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())

_LAZY_EXPORTS = {
    "FlatMp4Loader": ("databridge._formats.flat_mp4.loader", "FlatMp4Loader"),
    "HmieLoader": ("databridge._formats.hmie.loader", "HmieLoader"),
    "HmieWriter": ("databridge._formats.hmie.writer", "HmieWriter"),
    "HuggingFaceVideoClassificationLoader": (
        "databridge._formats.huggingface_video_classification.loader",
        "HuggingFaceVideoClassificationLoader",
    ),
    "HuggingFaceVideoClassificationWriter": (
        "databridge._formats.huggingface_video_classification.writer",
        "HuggingFaceVideoClassificationWriter",
    ),
    "MotChallengeLoader": ("databridge._formats.motchallenge.loader", "MotChallengeLoader"),
    "MotChallengeWriter": ("databridge._formats.motchallenge.writer", "MotChallengeWriter"),
    "TaoLoader": ("databridge._formats.tao.loader", "TaoLoader"),
    "TaoWriter": ("databridge._formats.tao.writer", "TaoWriter"),
    "VisDroneVideoLoader": ("databridge._formats.visdrone.loader", "VisDroneVideoLoader"),
    "VisDroneVideoWriter": ("databridge._formats.visdrone.writer", "VisDroneVideoWriter"),
    "load_huggingface_video_classification": (
        "databridge._formats.huggingface_video_classification.loader",
        "load_huggingface_video_classification",
    ),
}


def __getattr__(name: str) -> object:
    """Lazily resolve format-specific public exports."""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "BoxAnnotation",
    "BoxTrackDataset",
    "CategoryEntry",
    "DatasetFormat",
    "Finding",
    "FlatMp4Loader",
    "HmieLoader",
    "HmieWriter",
    "HuggingFaceVideoClassificationLoader",
    "HuggingFaceVideoClassificationWriter",
    "Loader",
    "MotChallengeLoader",
    "MotChallengeWriter",
    "Severity",
    "TaoLoader",
    "TaoWriter",
    "Task",
    "Taxonomy",
    "ValidationCache",
    "ValidationResult",
    "VideoClassificationDataset",
    "VideoClassificationSample",
    "VideoSequence",
    "VisDroneVideoLoader",
    "VisDroneVideoWriter",
    "VisionDataset",
    "Writer",
    "__version__",
    "__version_tuple__",
    "available_formats",
    "available_output_formats",
    "convert",
    "dataset_stats",
    "find_batch_roots",
    "get_loader",
    "get_writer",
    "load",
    "load_huggingface_video_classification",
    "load_mot",
    "register_loader",
    "register_writer",
    "render_html_report",
    "render_html_report_multi",
    "validate",
    "validate_annotation",
    "validate_batches",
    "write",
]
