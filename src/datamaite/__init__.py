"""Datamaite -- A unified framework for dataset loading, conversion, and quality validation."""

import logging
from importlib import import_module
from typing import TYPE_CHECKING

from datamaite._cache import ValidationCache
from datamaite._formats.hmie.discovery import find_batch_roots
from datamaite._report import render_html_report, render_html_report_multi
from datamaite._stats import dataset_stats
from datamaite._types import DatasetFormat, Finding, Severity, Task, ValidationResult, WriteMode
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
    load_vc,
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
    from datamaite._formats.flat_images.loader import FlatImagesLoader
    from datamaite._formats.flat_mp4.loader import FlatMp4Loader
    from datamaite._formats.hmie.loader import HmieLoader
    from datamaite._formats.hmie.writer import HmieWriter
    from datamaite._formats.huggingface_video_classification.loader import (
        HuggingFaceVideoClassificationLoader,
    )
    from datamaite._formats.huggingface_video_classification.writer import HuggingFaceVideoClassificationWriter
    from datamaite._formats.huggingface_vision.loader import (
        HuggingFaceVisionImageClassificationLoader,
        HuggingFaceVisionObjectDetectionLoader,
    )
    from datamaite._formats.huggingface_vision.writer import (
        HuggingFaceVisionImageClassificationWriter,
        HuggingFaceVisionObjectDetectionWriter,
    )
    from datamaite._formats.motchallenge.loader import MotChallengeLoader
    from datamaite._formats.motchallenge.writer import MotChallengeWriter
    from datamaite._formats.tao.loader import TaoLoader
    from datamaite._formats.tao.writer import TaoWriter
    from datamaite._formats.visdrone.loader import VisDroneVideoLoader
    from datamaite._formats.visdrone.static_loader import (
        VisDroneImageClassificationLoader,
        VisDroneObjectDetectionLoader,
    )
    from datamaite._formats.visdrone.static_writer import (
        VisDroneImageClassificationWriter,
        VisDroneObjectDetectionWriter,
    )
    from datamaite._formats.visdrone.writer import VisDroneVideoWriter
    from datamaite._formats.yolo.loader import YoloImageClassificationLoader, YoloObjectDetectionLoader
    from datamaite._formats.yolo.writer import YoloImageClassificationWriter, YoloObjectDetectionWriter

# Library convention: attach a NullHandler so downstream applications
# that don't configure logging don't see "No handler found" warnings.
# Consumers who want datamaite logs can add their own handlers.
logging.getLogger(__name__).addHandler(logging.NullHandler())

_LAZY_EXPORTS = {
    "CocoLoader": ("datamaite._formats.coco.loader", "CocoLoader"),
    "CocoWriter": ("datamaite._formats.coco.writer", "CocoWriter"),
    "FlatImagesLoader": ("datamaite._formats.flat_images.loader", "FlatImagesLoader"),
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
    "HuggingFaceVisionImageClassificationLoader": (
        "datamaite._formats.huggingface_vision.loader",
        "HuggingFaceVisionImageClassificationLoader",
    ),
    "HuggingFaceVisionImageClassificationWriter": (
        "datamaite._formats.huggingface_vision.writer",
        "HuggingFaceVisionImageClassificationWriter",
    ),
    "HuggingFaceVisionObjectDetectionLoader": (
        "datamaite._formats.huggingface_vision.loader",
        "HuggingFaceVisionObjectDetectionLoader",
    ),
    "HuggingFaceVisionObjectDetectionWriter": (
        "datamaite._formats.huggingface_vision.writer",
        "HuggingFaceVisionObjectDetectionWriter",
    ),
    "MotChallengeLoader": ("datamaite._formats.motchallenge.loader", "MotChallengeLoader"),
    "MotChallengeWriter": ("datamaite._formats.motchallenge.writer", "MotChallengeWriter"),
    "TaoLoader": ("datamaite._formats.tao.loader", "TaoLoader"),
    "TaoWriter": ("datamaite._formats.tao.writer", "TaoWriter"),
    "VisDroneObjectDetectionLoader": (
        "datamaite._formats.visdrone.static_loader",
        "VisDroneObjectDetectionLoader",
    ),
    "VisDroneImageClassificationLoader": (
        "datamaite._formats.visdrone.static_loader",
        "VisDroneImageClassificationLoader",
    ),
    "VisDroneImageClassificationWriter": (
        "datamaite._formats.visdrone.static_writer",
        "VisDroneImageClassificationWriter",
    ),
    "VisDroneObjectDetectionWriter": (
        "datamaite._formats.visdrone.static_writer",
        "VisDroneObjectDetectionWriter",
    ),
    "VisDroneVideoLoader": ("datamaite._formats.visdrone.loader", "VisDroneVideoLoader"),
    "VisDroneVideoWriter": ("datamaite._formats.visdrone.writer", "VisDroneVideoWriter"),
    "YoloImageClassificationLoader": ("datamaite._formats.yolo.loader", "YoloImageClassificationLoader"),
    "YoloImageClassificationWriter": ("datamaite._formats.yolo.writer", "YoloImageClassificationWriter"),
    "YoloObjectDetectionLoader": ("datamaite._formats.yolo.loader", "YoloObjectDetectionLoader"),
    "YoloObjectDetectionWriter": ("datamaite._formats.yolo.writer", "YoloObjectDetectionWriter"),
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
    "FlatImagesLoader",
    "FlatMp4Loader",
    "HmieLoader",
    "HmieWriter",
    "HuggingFaceVideoClassificationLoader",
    "HuggingFaceVideoClassificationWriter",
    "HuggingFaceVisionImageClassificationLoader",
    "HuggingFaceVisionImageClassificationWriter",
    "HuggingFaceVisionObjectDetectionLoader",
    "HuggingFaceVisionObjectDetectionWriter",
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
    "VisDroneImageClassificationLoader",
    "VisDroneImageClassificationWriter",
    "VisDroneObjectDetectionLoader",
    "VisDroneObjectDetectionWriter",
    "VisDroneVideoLoader",
    "VisDroneVideoWriter",
    "VisionDataset",
    "WriteMode",
    "Writer",
    "WriterCapabilities",
    "YoloImageClassificationLoader",
    "YoloImageClassificationWriter",
    "YoloObjectDetectionLoader",
    "YoloObjectDetectionWriter",
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
    "load_ic",
    "load_mot",
    "load_od",
    "load_vc",
    "register_loader",
    "register_writer",
    "render_html_report",
    "render_html_report_multi",
    "validate",
    "validate_annotation",
    "validate_batches",
    "write",
]
