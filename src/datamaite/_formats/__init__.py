"""Format-specific implementation packages.

Each subpackage owns the parser/loader, writer, schema, discovery, and
validation helpers for one dataset format. Top-level modules such as
:mod:`datamaite.loaders`, :mod:`datamaite.writers`, and
:mod:`datamaite.validation` remain format-agnostic dispatch/orchestration
layers; concrete format code lives under ``_formats/<format>/``.
"""
