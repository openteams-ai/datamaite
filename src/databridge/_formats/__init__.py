"""Per-format internal modules.

Each subpackage (e.g. ``hmie``) contains the schema, discovery, and
check functions specific to one dataset format. The top-level
``validation`` module dispatches to the appropriate format package
based on ``DatasetFormat``.

This layout keeps format-specific logic (currently HMIE validation helpers;
future YOLO/COCO validators/loaders and VisDrone validators) behind clear
ownership boundaries.
"""
