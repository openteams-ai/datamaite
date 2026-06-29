"""COCO object-detection format package."""

from datamaite._formats.coco.loader import CocoLoader, load_coco
from datamaite._formats.coco.writer import CocoWriter

__all__ = ["CocoLoader", "CocoWriter", "load_coco"]
