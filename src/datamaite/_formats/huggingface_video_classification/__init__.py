"""Hugging Face Video Classification format package."""

from datamaite._formats.huggingface_video_classification.loader import (
    HuggingFaceVideoClassificationLoader,
    load_huggingface_video_classification,
)
from datamaite._formats.huggingface_video_classification.writer import HuggingFaceVideoClassificationWriter

__all__ = [
    "HuggingFaceVideoClassificationLoader",
    "HuggingFaceVideoClassificationWriter",
    "load_huggingface_video_classification",
]
