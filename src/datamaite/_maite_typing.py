"""Type-only bridge to MAITE's dataset protocols.

MAITE's ``Dataset`` protocol declares ``metadata: DatasetMetadata`` as a plain
(writable) attribute. Pyright and mypy only accept a matching *writable*
attribute of exactly that type: a ``@property`` (with or without a setter under
pyright) or a frozen-dataclass field is rejected as read-only. So type checkers
see the attribute declaration below, while at runtime ``metadata`` is a
property that builds a fresh dict from ``dataset_id`` and ``index2label()`` on
every access, as it always has. Edits to the returned dict therefore never
reach the dataset, matching MAITE's ``ReadOnly`` keys.

The one mismatch: type checkers accept ``ds.metadata = ...``, but the frozen
dataclass rejects it at runtime with ``FrozenInstanceError``.

Upstream fix requested: https://github.com/mit-ll-ai-technology/maite/issues/46
(declare ``metadata`` as a read-only property on MAITE's protocols). Once
datamaite's minimum ``maite`` version includes that, drop the
``TYPE_CHECKING`` split below and make ``metadata`` a plain ``@property``
returning ``DatasetMetadata``.

The declaration lives on this plain (non-dataclass) base because, in a
dataclass body, it would become a frozen field. ``maite`` is only imported
under ``TYPE_CHECKING``; runtime ``import datamaite`` never touches it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maite.protocols import DatasetMetadata


def build_dataset_metadata(dataset_id: str, index2label: dict[int, str]) -> DatasetMetadata:
    """Build MAITE ``DatasetMetadata``; a typed function, so the checker verifies the dict."""
    return {"id": dataset_id, "index2label": index2label}


class MaiteDatasetMetadataField:
    """Declares MAITE's ``metadata: DatasetMetadata``; a derived property at runtime."""

    if TYPE_CHECKING:
        metadata: DatasetMetadata
    else:

        @property
        def metadata(self):
            """MAITE ``DatasetMetadata``: dataset id + ``index2label`` map, rebuilt on each access."""
            return build_dataset_metadata(self.dataset_id, self.index2label())
