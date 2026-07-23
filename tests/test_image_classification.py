"""Image-classification dataset abstraction tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from datamaite import (
    BoxTrackDataset,
    ClassificationLabel,
    DatasetFormat,
    DatasetMetadata,
    ImageClassificationDataset,
    ImageClassificationSample,
    ImageObjectDetectionSample,
    ObjectDetectionAnnotation,
    Task,
)
from datamaite.loaders import LoaderKey, available_formats, available_loader_keys, get_loader
from datamaite.taxonomy import CategoryEntry, Taxonomy


def _png_bytes(width: int = 3, height: int = 2) -> bytes:
    ok, encoded = cv2.imencode(".png", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


class TestImageClassificationDataset:
    def test_dataset_is_maite_image_classification_surface(self) -> None:
        taxonomy = Taxonomy(
            entries=(CategoryEntry(source_id=0, name="cat"), CategoryEntry(source_id=1, name="dog")),
            id_density="dense",
        )
        ds = ImageClassificationDataset(
            samples=(
                ImageClassificationSample(
                    image_id="train/dog/a.png",
                    image_bytes=_png_bytes(),
                    file_name="train/dog/a.png",
                    width=3,
                    height=2,
                    split="train",
                    labels=(ClassificationLabel(category_id=1, source_category_id=1, category_name="dog"),),
                ),
            ),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy, splits=("train",)),
            dataset_id="ic-set",
        )

        assert len(ds) == 1
        assert ds.task is Task.IC
        assert ds.metadata == {"id": "ic-set", "index2label": {0: "cat", 1: "dog"}}

        image, target, metadata = ds[0]
        assert image.shape == (3, 2, 3)  # CHW
        assert target.tolist() == [0.0, 1.0]
        assert metadata == {"id": "train/dog/a.png", "split": "train", "height": 2, "width": 3}

    def test_score_weighted_target_uses_label_score(self) -> None:
        taxonomy = Taxonomy(
            entries=(CategoryEntry(source_id=0, name="cat"), CategoryEntry(source_id=1, name="dog")),
            id_density="dense",
        )
        ds = ImageClassificationDataset(
            samples=(
                ImageClassificationSample(
                    image_id="a",
                    image_bytes=_png_bytes(),
                    labels=(ClassificationLabel(category_id=1, source_category_id=1, score=0.6),),
                ),
            ),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy),
        )

        _image, target, _meta = ds[0]
        # A probabilistic label lands its score in the target, not a hard 1.0.
        assert target.tolist() == [0.0, pytest.approx(0.6)]

    def test_sparse_taxonomy_resolves_index_by_category_name(self) -> None:
        # source id 7 is absent from the dense projection; the name resolves it
        # to the entry's *position* (1), not the raw id.
        taxonomy = Taxonomy(
            entries=(CategoryEntry(source_id=2, name="car"), CategoryEntry(source_id=7, name="person")),
            id_density="sparse",
        )
        ds = ImageClassificationDataset(
            samples=(
                ImageClassificationSample(
                    image_id="a",
                    image_bytes=_png_bytes(),
                    labels=(ClassificationLabel(category_id=99, category_name="person"),),
                ),
            ),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy),
        )

        _image, target, _meta = ds[0]
        assert target.tolist() == [0.0, 1.0]

    def test_duplicate_source_ids_pick_first_entry_like_the_writer(self) -> None:
        # A merged taxonomy can repeat a bare source id; dense_ids() is then
        # ill-defined. The MAITE target must agree with the writer's
        # by_source_id() (first match) -- here index 0 ("a"), never index 1.
        taxonomy = Taxonomy(
            entries=(CategoryEntry(source_id=0, name="a"), CategoryEntry(source_id=0, name="b")),
            id_density="sparse",
        )
        ds = ImageClassificationDataset(
            samples=(
                ImageClassificationSample(
                    image_id="x",
                    image_bytes=_png_bytes(),
                    labels=(ClassificationLabel(category_id=0, source_category_id=0),),
                ),
            ),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy),
        )

        assert taxonomy.by_source_id(0) is not None
        assert taxonomy.by_source_id(0).name == "a"
        _image, target, _meta = ds[0]
        assert target.tolist() == [1.0, 0.0]

    def test_taxonomyless_target_uses_integer_source_id(self) -> None:
        # No taxonomy: the integer source id is the dense index and the width is
        # inferred from it (documented best-effort, ragged-across-samples) path.
        ds = ImageClassificationDataset(
            samples=(
                ImageClassificationSample(
                    image_id="a",
                    image_bytes=_png_bytes(),
                    labels=(ClassificationLabel(category_id=2, source_category_id=2),),
                ),
            ),
            dataset_metadata=DatasetMetadata(),
        )

        _image, target, _meta = ds[0]
        assert target.tolist() == [0.0, 0.0, 1.0]


class TestTaskAwareRegistry:
    def test_yolo_ic_loader_key_is_registered(self) -> None:
        key = LoaderKey(task=Task.IC, format=DatasetFormat.YOLO, variant="default")
        assert key in available_loader_keys()
        assert DatasetFormat.YOLO in available_formats(task=Task.IC)
        assert get_loader("yolo", task=Task.IC).task is Task.IC


class TestMotTaxonomyView:
    def test_box_track_dataset_materializes_taxonomy_from_legacy_categories(self) -> None:
        ds = BoxTrackDataset(sequences=(), categories={"tao/category_3/person": 3})

        taxonomy = ds.dataset_metadata.taxonomy
        assert taxonomy is not None
        assert taxonomy.by_source_id(3) == CategoryEntry(
            source_id=3,
            name="person",
            source_dataset="datamaite",
            attributes={"category_uri": "tao/category_3/person"},
        )
        assert ds.index2label() == {3: "person"}

    def test_preserves_explicit_source_dataset(self) -> None:
        ds = BoxTrackDataset(
            sequences=(),
            categories={"tao/category_3/person": 3},
            dataset_id="display-id",
            dataset_metadata=DatasetMetadata(source_dataset="tao"),
        )

        assert ds.dataset_metadata.source_dataset == "tao"
        assert ds.dataset_metadata.taxonomy is not None
        assert ds.dataset_metadata.taxonomy.source_dataset == "tao"


class TestObjectDetectionRecordCompatibility:
    def test_positional_detections_are_not_bound_to_split(self) -> None:
        detection = ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=1)

        sample = ImageObjectDetectionSample(1, None, None, "a.jpg", 10, 20, (detection,))

        assert sample.split is None
        assert sample.detections == (detection,)


def test_no_path_import_side_effects(tmp_path: Path) -> None:
    # Smoke assertion that the new abstractions do not require path existence at construction.
    sample = ImageClassificationSample(
        image_id="missing",
        path_or_uri=str(tmp_path / "does-not-exist.jpg"),
        labels=(ClassificationLabel(category_id=0, category_name="missing"),),
    )
    assert sample.path_or_uri is not None


class TestImageClassificationFieldwise:
    """MAITE ``FieldwiseDataset`` surface: get_input / get_target / get_metadata."""

    def _ds(self) -> ImageClassificationDataset:
        taxonomy = Taxonomy(
            entries=(CategoryEntry(source_id=0, name="cat"), CategoryEntry(source_id=1, name="dog")),
            id_density="dense",
        )
        return ImageClassificationDataset(
            samples=(
                ImageClassificationSample(
                    image_id="train/dog/a.png",
                    image_bytes=_png_bytes(),
                    file_name="train/dog/a.png",
                    width=3,
                    height=2,
                    split="train",
                    labels=(ClassificationLabel(category_id=1, source_category_id=1, category_name="dog"),),
                ),
            ),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy, splits=("train",)),
            dataset_id="ic-set",
        )

    def test_has_fieldwise_methods(self) -> None:
        ds = self._ds()
        for name in ("get_input", "get_target", "get_metadata"):
            assert callable(getattr(ds, name))

    def test_fieldwise_matches_getitem(self) -> None:
        ds = self._ds()
        image, target, metadata = ds[0]
        assert np.array_equal(ds.get_input(0), image)
        assert np.array_equal(ds.get_target(0), target)
        assert ds.get_metadata(0) == metadata

    def test_target_and_metadata_need_no_image_decode(self) -> None:
        # A sample with known dims but NO decodable image data: fieldwise target
        # and metadata must still resolve, proving neither decodes the image.
        ds = ImageClassificationDataset(
            samples=(
                ImageClassificationSample(
                    image_id="x",
                    width=3,
                    height=2,
                    split="train",
                    labels=(ClassificationLabel(category_id=1, source_category_id=1, category_name="dog"),),
                ),
            ),
            dataset_metadata=DatasetMetadata(
                taxonomy=Taxonomy(
                    entries=(CategoryEntry(source_id=0, name="cat"), CategoryEntry(source_id=1, name="dog")),
                    id_density="dense",
                )
            ),
        )
        assert ds.get_target(0).tolist() == [0.0, 1.0]
        assert ds.get_metadata(0) == {"id": "x", "split": "train", "height": 2, "width": 3}
        # get_input has no bytes/path to decode -> it must fail, confirming the
        # others genuinely avoided the decode path.
        with pytest.raises(ValueError, match="neither path_or_uri nor image_bytes"):
            ds.get_input(0)
