"""Per-format internal modules.

Each subpackage (e.g. ``hmie``) contains the schema, discovery, and
check functions specific to one dataset format. The top-level
``validation`` module dispatches to the appropriate format package
based on ``DatasetFormat``.

This layout keeps HMIE logic isolated from future YOLO/COCO/VisDrone
implementations and gives each format a clear ownership boundary.
"""
