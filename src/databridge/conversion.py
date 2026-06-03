"""Dataset format conversion utilities.

Converters consume the neutral :class:`databridge.model.BoxTrackDataset`
model and emit an output format (MOTChallenge, YOLO, COCO, ...). Binding
converters to ``BoxTrackDataset`` rather than to a specific loader is what
lets any loader feed any converter; nothing here imports from
``dataloader.py``.
"""

from __future__ import annotations
