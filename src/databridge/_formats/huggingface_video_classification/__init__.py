"""Hugging Face Video Classification format package."""

from databridge._formats.huggingface_video_classification.loader import (
    HuggingFaceVideoClassificationLoader,
    load_huggingface_video_classification,
)

__all__ = ["HuggingFaceVideoClassificationLoader", "load_huggingface_video_classification"]
