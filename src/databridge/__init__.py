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
from databridge.loaders import Loader, available_formats, get_loader, load, register_loader
from databridge.model import BoxAnnotation, BoxTrackDataset, VideoSequence
from databridge.taxonomy import CategoryEntry, Taxonomy
from databridge.validation import validate, validate_annotation, validate_batches
from databridge.writers import Writer, available_output_formats, get_writer, register_writer, write

if TYPE_CHECKING:
    from databridge._formats.flat_mp4.loader import FlatMp4Loader, load_flat_mp4
    from databridge._formats.hmie.loader import HmieLoader, load_hmie
    from databridge._formats.hmie.writer import HmieWriter
    from databridge._formats.motchallenge.loader import MotChallengeLoader, load_motchallenge
    from databridge._formats.tao.loader import TaoLoader, load_tao
    from databridge._formats.visdrone.loader import VisDroneVideoLoader, load_visdrone_video

# Library convention: attach a NullHandler so downstream applications
# that don't configure logging don't see "No handler found" warnings.
# Consumers who want databridge logs can add their own handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())

_LAZY_EXPORTS = {
    "FlatMp4Loader": ("databridge._formats.flat_mp4.loader", "FlatMp4Loader"),
    "HmieLoader": ("databridge._formats.hmie.loader", "HmieLoader"),
    "HmieWriter": ("databridge._formats.hmie.writer", "HmieWriter"),
    "MotChallengeLoader": ("databridge._formats.motchallenge.loader", "MotChallengeLoader"),
    "TaoLoader": ("databridge._formats.tao.loader", "TaoLoader"),
    "VisDroneVideoLoader": ("databridge._formats.visdrone.loader", "VisDroneVideoLoader"),
    "load_flat_mp4": ("databridge._formats.flat_mp4.loader", "load_flat_mp4"),
    "load_hmie": ("databridge._formats.hmie.loader", "load_hmie"),
    "load_motchallenge": ("databridge._formats.motchallenge.loader", "load_motchallenge"),
    "load_tao": ("databridge._formats.tao.loader", "load_tao"),
    "load_visdrone_video": ("databridge._formats.visdrone.loader", "load_visdrone_video"),
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
    "Loader",
    "MotChallengeLoader",
    "Severity",
    "TaoLoader",
    "Task",
    "Taxonomy",
    "ValidationCache",
    "ValidationResult",
    "VideoSequence",
    "VisDroneVideoLoader",
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
    "load_flat_mp4",
    "load_hmie",
    "load_motchallenge",
    "load_tao",
    "load_visdrone_video",
    "register_loader",
    "register_writer",
    "render_html_report",
    "render_html_report_multi",
    "validate",
    "validate_annotation",
    "validate_batches",
    "write",
]
