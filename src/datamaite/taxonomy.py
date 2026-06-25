"""Shared category taxonomy for every task (MOT / OD / IC).

The current FMV model stores categories as ``dict[str, int]`` (URI -> id), which
cannot hold a string/synset source id, a supercategory, or an eval-excluded
marker -- the TAO loader already drops synset/supercategory because of this. The
:class:`Taxonomy` here is the source-preserving replacement shared across tasks:
it keeps the **source** category id (int *or* string *or* none), the display
name, optional ``supercategory``/``synset`` provenance, and per-format flags, and
it derives the **dense contiguous** ids a writer like YOLO needs as a projection
rather than mutating the stored ids.

Key rule (avoids silent class merges): identity is ``(source_dataset, source_id)``
-- never the bare id -- so merging two datasets that both use class index ``0``
keeps them distinct.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

SourceId = int | str | None


@dataclass(frozen=True)
class CategoryEntry:
    """One category, preserving the source identity plus display/provenance fields."""

    source_id: SourceId
    name: str
    supercategory: str | None = None
    synset: str | None = None
    # VisDrone categories 0 (ignored regions) and 11 (others) are real and must
    # be emitted, but are excluded from evaluation. Marks them so a writer does
    # not drop or renumber them.
    eval_excluded: bool = False
    # Provenance of the owning dataset; ``None`` inherits the Taxonomy's
    # ``source_dataset``. Set per-entry only after a merge, to keep identity.
    source_dataset: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Taxonomy:
    """An ordered set of categories with source-preserving ids and a dense projection.

    ``ordered_names`` is the positional truth for formats whose id *is* a
    position (YOLO ``names``, HuggingFace ``ClassLabel``); when not supplied it is
    derived from ``entries`` order. ``id_density`` records whether the source ids
    are sparse (COCO 1..90 with gaps) or dense-contiguous (YOLO/HF), so a writer
    knows whether its dense remap is the identity or a real renumber.
    """

    entries: tuple[CategoryEntry, ...]
    source_dataset: str = "datamaite"
    id_density: str = "sparse"  # 'sparse' | 'dense'
    ordered_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.id_density not in ("sparse", "dense"):
            raise ValueError(f"id_density must be 'sparse' or 'dense', got {self.id_density!r}")
        if not isinstance(self.entries, tuple):
            object.__setattr__(self, "entries", tuple(self.entries))
        if not self.ordered_names:
            object.__setattr__(self, "ordered_names", tuple(e.name for e in self.entries))

    # -- lookups ----------------------------------------------------------
    def by_source_id(self, source_id: SourceId) -> CategoryEntry | None:
        """First entry with this source id, or ``None``.

        A convenience lookup for the common **single-source** taxonomy where
        source ids are unique. In a merged (multi-source) taxonomy a bare id can
        repeat across datasets, so this returns the *first* match; a
        source-dataset-aware lookup is added with the consumer that needs it.
        """
        for e in self.entries:
            if e.source_id == source_id:
                return e
        return None

    def by_name(self, name: str) -> CategoryEntry | None:
        for e in self.entries:
            if e.name == name:
                return e
        return None

    # -- MAITE / display maps --------------------------------------------
    def index2label(self) -> dict[int, str]:
        """``{int source_id: name}`` for entries whose source id is an int.

        Same *shape* (``dict[int, str]``) as ``BoxTrackDataset.index2label``, but
        the value is the stored ``CategoryEntry.name`` verbatim -- not a
        URI-derived segment (``BoxTrackDataset`` runs names through
        ``category_name_from_uri``). When the categories→taxonomy migration lands
        the two are reconciled at that boundary. String-id formats (Pascal VOC,
        KITTI) contribute nothing here; use :meth:`dense_index2label` for a MAITE
        target label space.
        """
        return {
            e.source_id: e.name
            for e in self.entries
            if isinstance(e.source_id, int) and not isinstance(e.source_id, bool)
        }

    def dense_index2label(self) -> dict[int, str]:
        """``{dense_index: name}`` over all entries -- the MAITE target label space."""
        return dict(enumerate(e.name for e in self.entries))

    # -- writer-side projection ------------------------------------------
    def dense_ids(self) -> dict[SourceId, int]:
        """Map each entry's source id to a dense, contiguous 0-based index.

        The projection a writer such as YOLO applies; the stored source ids are
        never mutated. For an already-dense source the map is the identity.

        Requires **unique** source ids. A merged (multi-source) taxonomy can
        repeat a bare ``source_id`` (``ds_a:0`` and ``ds_b:0`` are kept distinct
        by :meth:`merge`), for which a ``source_id -> index`` map is ill-defined.
        Rather than silently collapse two classes onto one index, this **raises**
        ``ValueError`` -- use :meth:`dense_index2label` (positional, collision-free)
        for a merged taxonomy's label space.
        """
        ids = [e.source_id for e in self.entries]
        if len(ids) != len(set(ids)):
            dupes = sorted({str(i) for i in ids if ids.count(i) > 1})
            raise ValueError(
                f"dense_ids() needs unique source ids; duplicates {dupes} "
                "(a merged/multi-source taxonomy?) -- use dense_index2label() for a positional label space"
            )
        return {e.source_id: i for i, e in enumerate(self.entries)}

    # -- merge ------------------------------------------------------------
    def merge(self, *others: Taxonomy) -> Taxonomy:
        """Union this taxonomy with others, keyed on ``(source_dataset, source_id)``.

        Entries with the same ``(source_dataset, source_id)`` collapse (first
        wins); entries that merely share a bare id across *different* datasets
        stay distinct -- so merging two YOLO datasets does not silently fuse
        their class ``0``. Merged entries carry an explicit ``source_dataset`` so
        provenance survives. The result is ``sparse`` (a union is not guaranteed
        contiguous) and re-derives ``ordered_names`` from the merged order.
        """
        merged: list[CategoryEntry] = []
        seen: set[tuple[str, SourceId]] = set()
        for tax in (self, *others):
            for e in tax.entries:
                owner = e.source_dataset or tax.source_dataset
                key = (owner, e.source_id)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(replace(e, source_dataset=owner))
        # Positional, 1:1 with entries (NOT name-deduped) -- ordered_names is the
        # positional label space, so collapsing duplicate names here would make
        # it shorter than entries and silently re-fuse the cross-dataset entries
        # this merge exists to keep distinct.
        names = tuple(e.name for e in merged)
        sources = tuple(dict.fromkeys([self.source_dataset, *(o.source_dataset for o in others)]))
        return Taxonomy(
            entries=tuple(merged),
            source_dataset="+".join(sources),
            id_density="sparse",
            ordered_names=names,
        )
