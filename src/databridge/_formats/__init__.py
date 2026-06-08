"""Format-specific implementation packages.

Each subpackage owns the parser/loader, writer, schema, discovery, and
validation helpers for one dataset format. Top-level modules such as
:mod:`databridge.loaders`, :mod:`databridge.writers`, and
:mod:`databridge.validation` remain format-agnostic dispatch/orchestration
layers; concrete format code lives under ``_formats/<format>/``.
"""
