"""Tests for the shared category Taxonomy."""

from __future__ import annotations

import pytest

from databridge import CategoryEntry, Taxonomy


def _coco_like() -> Taxonomy:
    # sparse source ids with a gap, supercategory provenance
    return Taxonomy(
        entries=(
            CategoryEntry(source_id=1, name="person", supercategory="person"),
            CategoryEntry(source_id=3, name="car", supercategory="vehicle"),
            CategoryEntry(source_id=90, name="toothbrush", supercategory="indoor"),
        ),
        source_dataset="coco",
        id_density="sparse",
    )


class TestConstruction:
    def test_ordered_names_derived_from_entries(self) -> None:
        tax = _coco_like()
        assert tax.ordered_names == ("person", "car", "toothbrush")

    def test_explicit_ordered_names_preserved(self) -> None:
        tax = Taxonomy(
            entries=(CategoryEntry(0, "a"), CategoryEntry(1, "b")),
            ordered_names=("a", "b"),
            id_density="dense",
        )
        assert tax.ordered_names == ("a", "b")

    def test_invalid_id_density_raises(self) -> None:
        with pytest.raises(ValueError, match="id_density"):
            Taxonomy(entries=(), id_density="bogus")


class TestLookups:
    def test_by_source_id_and_name(self) -> None:
        tax = _coco_like()
        assert tax.by_source_id(3).name == "car"
        assert tax.by_name("person").source_id == 1
        assert tax.by_source_id(999) is None
        assert tax.by_name("missing") is None


class TestLabelMaps:
    def test_index2label_int_ids_only(self) -> None:
        tax = Taxonomy(
            entries=(
                CategoryEntry(source_id=1, name="person"),
                CategoryEntry(source_id="car", name="car"),  # VOC-style string id
            )
        )
        # only the int-id entry appears in the MAITE-style map
        assert tax.index2label() == {1: "person"}

    def test_index2label_excludes_bool(self) -> None:
        # guard: bool is an int subclass but must not leak in as a category id
        tax = Taxonomy(entries=(CategoryEntry(source_id=True, name="weird"),))
        assert tax.index2label() == {}

    def test_dense_index2label_covers_all(self) -> None:
        tax = _coco_like()
        assert tax.dense_index2label() == {0: "person", 1: "car", 2: "toothbrush"}


class TestDenseProjection:
    def test_dense_ids_are_contiguous_positions(self) -> None:
        tax = _coco_like()
        # sparse source ids 1/3/90 project to dense 0/1/2
        assert tax.dense_ids() == {1: 0, 3: 1, 90: 2}

    def test_dense_ids_identity_for_dense_source(self) -> None:
        tax = Taxonomy(
            entries=(CategoryEntry(0, "a"), CategoryEntry(1, "b"), CategoryEntry(2, "c")),
            id_density="dense",
        )
        assert tax.dense_ids() == {0: 0, 1: 1, 2: 2}

    def test_dense_ids_raises_on_duplicate_source_ids(self) -> None:
        # merged/multi-source taxonomy with a repeated bare id: dense_ids() must
        # fail loud rather than silently collapse two classes onto one index.
        a = Taxonomy(entries=(CategoryEntry(0, "cat"),), source_dataset="ds_a", id_density="dense")
        b = Taxonomy(entries=(CategoryEntry(0, "dog"),), source_dataset="ds_b", id_density="dense")
        merged = a.merge(b)
        with pytest.raises(ValueError, match="unique source ids"):
            merged.dense_ids()
        # the collision-free positional space still works
        assert merged.dense_index2label() == {0: "cat", 1: "dog"}


class TestMerge:
    def test_merge_keeps_cross_dataset_index_zero_distinct(self) -> None:
        a = Taxonomy(entries=(CategoryEntry(0, "cat"),), source_dataset="ds_a", id_density="dense")
        b = Taxonomy(entries=(CategoryEntry(0, "dog"),), source_dataset="ds_b", id_density="dense")
        merged = a.merge(b)
        # both class-0 entries survive (keyed on (source_dataset, source_id)), so
        # the merge does NOT silently fuse two datasets' class 0.
        assert [e.name for e in merged.entries] == ["cat", "dog"]
        # the robust (collision-free) dense space is positional:
        assert merged.dense_index2label() == {0: "cat", 1: "dog"}
        # provenance preserved per entry
        assert {e.source_dataset for e in merged.entries} == {"ds_a", "ds_b"}
        assert merged.source_dataset == "ds_a+ds_b"

    def test_merge_dedups_within_same_dataset(self) -> None:
        a = Taxonomy(entries=(CategoryEntry(1, "person"),), source_dataset="coco")
        b = Taxonomy(entries=(CategoryEntry(1, "person"), CategoryEntry(2, "car")), source_dataset="coco")
        merged = a.merge(b)
        assert [e.name for e in merged.entries] == ["person", "car"]
        assert merged.source_dataset == "coco"

    def test_merge_ordered_names_stays_aligned_with_entries(self) -> None:
        # regression: same name across datasets must NOT collapse ordered_names
        # (it is the positional label space; collapsing re-fuses the distinct
        # entries the merge keeps apart).
        a = Taxonomy(entries=(CategoryEntry(0, "person"),), source_dataset="a", id_density="dense")
        b = Taxonomy(entries=(CategoryEntry(0, "person"),), source_dataset="b", id_density="dense")
        merged = a.merge(b)
        assert len(merged.entries) == 2
        assert len(merged.ordered_names) == len(merged.entries) == 2
        assert merged.ordered_names == ("person", "person")
        assert merged.dense_index2label() == {0: "person", 1: "person"}

    def test_merge_result_is_sparse(self) -> None:
        a = Taxonomy(entries=(CategoryEntry(0, "a"),), source_dataset="x", id_density="dense")
        b = Taxonomy(entries=(CategoryEntry(0, "b"),), source_dataset="y", id_density="dense")
        assert a.merge(b).id_density == "sparse"


class TestProvenanceFields:
    def test_eval_excluded_and_synset_retained(self) -> None:
        tax = Taxonomy(
            entries=(
                CategoryEntry(source_id=0, name="ignored", eval_excluded=True),
                CategoryEntry(source_id=5, name="dog", synset="dog.n.01"),
            ),
            source_dataset="visdrone",
        )
        assert tax.by_source_id(0).eval_excluded is True
        assert tax.by_source_id(5).synset == "dog.n.01"
