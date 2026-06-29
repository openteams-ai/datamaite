"""Datamaite -- A unified framework for dataset loading, conversion, and quality validation."""

import logging
from importlib import import_module
from typing import TYPE_CHECKING

from datamaite._cache import ValidationCache
from datamaite._formats.hmie.discovery import find_batch_roots
from datamaite._report import render_html_report, render_html_report_multi
from datamaite._stats import dataset_stats
from datamaite._types import DatasetFormat, Finding, Severity, Task, ValidationResult
from datamaite._version import __version__, __version_tuple__
from datamaite.conversion import convert
from datamaite.image_classification import ImageClassificationDataset, load_ic
from datamaite.loaders import (
    Loader,
    available_formats,
    available_loader_keys,
    get_loader,
    load,
    load_mot,
    register_loader,
)
from datamaite.model import (
    BoxAnnotation,
    BoxTrackDataset,
    VideoClassificationDataset,
    VideoClassificationSample,
    VideoSequence,
    VisionDataset,
)
from datamaite.object_detection import ObjectDetectionDataset, load_od
from datamaite.records import (
    ClassificationLabel,
    DatasetMetadata,
    ImageClassificationSample,
    ImageObjectDetectionSample,
    ImageRecord,
    ObjectDetectionAnnotation,
)
from datamaite.taxonomy import CategoryEntry, Taxonomy
from datamaite.validation import validate, validate_annotation, validate_batches
from datamaite.writers import (
    Writer,
    WriterCapabilities,
    available_output_formats,
    available_writer_keys,
    get_writer,
    register_writer,
    write,
)

if TYPE_CHECKING:
    from datamaite._formats.coco.loader import CocoLoader
    from datamaite._formats.coco.writer import CocoWriter
    from datamaite._formats.flat_mp4.loader import FlatMp4Loader
    from datamaite._formats.hmie.loader import HmieLoader
    from datamaite._formats.hmie.writer import HmieWriter
    from datamaite._formats.huggingface_video_classification.loader import (
        HuggingFaceVideoClassificationLoader,
        load_huggingface_video_classification,
    )
    from datamaite._formats.huggingface_video_classification.writer import HuggingFaceVideoClassificationWriter
    from datamaite._formats.motchallenge.loader import MotChallengeLoader
    from datamaite._formats.motchallenge.writer import MotChallengeWriter
    from datamaite._formats.tao.loader import TaoLoader
    from datamaite._formats.tao.writer import TaoWriter
    from datamaite._formats.visdrone.loader import VisDroneVideoLoader
    from datamaite._formats.visdrone.writer import VisDroneVideoWriter
    from datamaite._formats.yolo.classification import (
        YoloImageClassificationLoader,
        YoloImageClassificationWriter,
        load_yolo_image_classification,
    )

# Library convention: attach a NullHandler so downstream applications
# that don't configure logging don't see "No handler found" warnings.
# Consumers who want datamaite logs can add their own handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())

_LAZY_EXPORTS = {
    "CocoLoader": ("datamaite._formats.coco.loader", "CocoLoader"),
    "CocoWriter": ("datamaite._formats.coco.writer", "CocoWriter"),
    "FlatMp4Loader": ("datamaite._formats.flat_mp4.loader", "FlatMp4Loader"),
    "HmieLoader": ("datamaite._formats.hmie.loader", "HmieLoader"),
    "HmieWriter": ("datamaite._formats.hmie.writer", "HmieWriter"),
    "HuggingFaceVideoClassificationLoader": (
        "datamaite._formats.huggingface_video_classification.loader",
        "HuggingFaceVideoClassificationLoader",
    ),
    "HuggingFaceVideoClassificationWriter": (
        "datamaite._formats.huggingface_video_classification.writer",
        "HuggingFaceVideoClassificationWriter",
    ),
    "MotChallengeLoader": ("datamaite._formats.motchallenge.loader", "MotChallengeLoader"),
    "MotChallengeWriter": ("datamaite._formats.motchallenge.writer", "MotChallengeWriter"),
    "TaoLoader": ("datamaite._formats.tao.loader", "TaoLoader"),
    "TaoWriter": ("datamaite._formats.tao.writer", "TaoWriter"),
    "VisDroneVideoLoader": ("datamaite._formats.visdrone.loader", "VisDroneVideoLoader"),
    "VisDroneVideoWriter": ("datamaite._formats.visdrone.writer", "VisDroneVideoWriter"),
    "YoloImageClassificationLoader": ("datamaite._formats.yolo.classification", "YoloImageClassificationLoader"),
    "YoloImageClassificationWriter": ("datamaite._formats.yolo.classification", "YoloImageClassificationWriter"),
    "load_huggingface_video_classification": (
        "datamaite._formats.huggingface_video_classification.loader",
        "load_huggingface_video_classification",
    ),
    "load_yolo_image_classification": (
        "datamaite._formats.yolo.classification",
        "load_yolo_image_classification",
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
    "ClassificationLabel",
    "CocoLoader",
    "CocoWriter",
    "DatasetFormat",
    "DatasetMetadata",
    "Finding",
    "FlatMp4Loader",
    "HmieLoader",
    "HmieWriter",
    "HuggingFaceVideoClassificationLoader",
    "HuggingFaceVideoClassificationWriter",
    "ImageClassificationDataset",
    "ImageClassificationSample",
    "ImageObjectDetectionSample",
    "ImageRecord",
    "Loader",
    "MotChallengeLoader",
    "MotChallengeWriter",
    "ObjectDetectionAnnotation",
    "ObjectDetectionDataset",
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
    "WriterCapabilities",
    "YoloImageClassificationLoader",
    "YoloImageClassificationWriter",
    "__version__",
    "__version_tuple__",
    "available_formats",
    "available_loader_keys",
    "available_output_formats",
    "available_writer_keys",
    "convert",
    "dataset_stats",
    "find_batch_roots",
    "get_loader",
    "get_writer",
    "load",
    "load_huggingface_video_classification",
    "load_ic",
    "load_mot",
    "load_od",
    "load_yolo_image_classification",
    "register_loader",
    "register_writer",
    "render_html_report",
    "render_html_report_multi",
    "validate",
    "validate_annotation",
    "validate_batches",
    "write",
]
