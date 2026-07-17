"""Hugging Face Vision (still-image) format package."""

from datamaite._formats.huggingface_vision.loader import (
    HuggingFaceVisionImageClassificationLoader,
    HuggingFaceVisionObjectDetectionLoader,
)
from datamaite._formats.huggingface_vision.writer import (
    HuggingFaceVisionImageClassificationWriter,
    HuggingFaceVisionObjectDetectionWriter,
)

__all__ = [
    "HuggingFaceVisionImageClassificationLoader",
    "HuggingFaceVisionImageClassificationWriter",
    "HuggingFaceVisionObjectDetectionLoader",
    "HuggingFaceVisionObjectDetectionWriter",
]
