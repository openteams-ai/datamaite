"""Tests for the ObjectDetectionDataset model + its MAITE OD surface."""

from __future__ import annotations

import pytest

from datamaite import (
    CategoryEntry,
    DatasetMetadata,
    ImageObjectDetectionSample,
    ObjectDetectionAnnotation,
    ObjectDetectionDataset,
    Taxonomy,
)


def _ds() -> ObjectDetectionDataset:
    tax = Taxonomy(entries=(CategoryEntry(1, "person"), CategoryEntry(3, "car")), source_dataset="coco")
    samples = (
        ImageObjectDetectionSample(
            image_id=100,
            file_name="a.jpg",
            width=640,
            height=480,
            detections=(
                ObjectDetectionAnnotation(bbox=(10.0, 20.0, 30.0, 40.0), category_id=1, category_name="person"),
                ObjectDetectionAnnotation(bbox=(0.0, 0.0, 5.0, 5.0), category_id=3, category_name="car"),
            ),
        ),
        ImageObjectDetectionSample(image_id=101, file_name="b.jpg", width=640, height=480, detections=()),
    )
    return ObjectDetectionDataset(
        samples=samples, dataset_metadata=DatasetMetadata(taxonomy=tax, source_dataset="coco")
    )


class TestRegionKeywordOnly:
    """`ImageRecord.region` is keyword-only so positional construction is stable.

    A base-class field would otherwise slot ahead of the subclass `detections`
    field, silently binding a detection tuple into `region`.
    """

    def test_positional_construction_binds_detections_not_region(self) -> None:
        det = ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=1)
        # Positional order: image_id, path_or_uri, image_bytes, file_name, width,
        # height, split, metadata, detections — region is NOT in this sequence.
        sample = ImageObjectDetectionSample(7, None, None, "a.jpg", 640, 480, "train", {}, (det,))
        assert sample.detections == (det,)
        assert sample.split == "train"
        assert sample.region is None

    def test_region_is_keyword_only(self) -> None:
        det = ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=1)
        with pytest.raises(TypeError):
            # A tenth positional arg has no field to bind to now that region is
            # keyword-only; this must raise rather than absorb it into region.
            ImageObjectDetectionSample(7, None, None, "a.jpg", 640, 480, "train", {}, (det,), (0.0, 0.0, 1.0, 1.0))

    def test_region_accepted_as_keyword(self) -> None:
        sample = ImageObjectDetectionSample(image_id=7, region=(0.0, 0.0, 1.0, 1.0))
        assert sample.region == (0.0, 0.0, 1.0, 1.0)


class TestModel:
    def test_len_and_counts(self) -> None:
        ds = _ds()
        assert len(ds) == 2
        assert ds.sample_count == 2
        assert ds.num_detections == 2

    def test_metadata_and_index2label(self) -> None:
        ds = _ds()
        assert ds.index2label() == {1: "person", 3: "car"}
        assert ds.metadata == {"id": "datamaite", "index2label": {1: "person", 3: "car"}}

    def test_index2label_empty_without_taxonomy(self) -> None:
        ds = ObjectDetectionDataset(samples=(ImageObjectDetectionSample(image_id=0),))
        assert ds.index2label() == {}

    def test_iter_samples_yields_records(self) -> None:
        ds = _ds()
        assert [s.image_id for s in ds.iter_samples()] == [100, 101]

    def test_task_marker_is_od(self) -> None:
        from datamaite import Task

        # Parity with BoxTrackDataset (Task.MOT) / VideoClassificationDataset
        # (Task.VC); task-aware writer dispatch keys on it.
        assert _ds().task is Task.OD

    def test_samples_coerced_to_tuple(self) -> None:
        ds = ObjectDetectionDataset(samples=[ImageObjectDetectionSample(image_id=0)])  # type: ignore[arg-type]
        assert isinstance(ds.samples, tuple)


class TestStructuralConformance:
    def test_is_maite_od_dataset(self) -> None:
        from maite.protocols import object_detection as od_protocols

        assert isinstance(_ds(), od_protocols.Dataset)

    def test_target_is_maite_od_target(self, tmp_path) -> None:
        cv2 = pytest.importorskip("cv2")
        np = pytest.importorskip("numpy")
        from maite.protocols import object_detection as od_protocols

        img = tmp_path / "img.png"
        cv2.imwrite(str(img), np.zeros((2, 2, 3), dtype=np.uint8))
        sample = ImageObjectDetectionSample(
            image_id=1,
            path_or_uri=str(img),
            detections=(ObjectDetectionAnnotation(bbox=(0.0, 0.0, 1.0, 1.0), category_id=1),),
        )
        _image, target, _meta = ObjectDetectionDataset(samples=(sample,))[0]
        assert isinstance(target, od_protocols.ObjectDetectionTarget)


class TestMaiteOdSurface:
    def test_getitem_decodes_and_builds_target(self, tmp_path) -> None:
        cv2 = pytest.importorskip("cv2")
        np = pytest.importorskip("numpy")
        img = tmp_path / "img.png"
        cv2.imwrite(str(img), np.full((4, 6, 3), 128, dtype=np.uint8))  # H=4, W=6

        sample = ImageObjectDetectionSample(
            image_id=7,
            path_or_uri=str(img),
            detections=(ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=5, score=0.9),),
        )
        ds = ObjectDetectionDataset(samples=(sample,))
        image, target, meta = ds[0]

        assert image.shape == (3, 4, 6)  # (C, H, W)
        assert image.dtype == np.uint8
        # canonical xywh (1,2,3,4) -> xyxy (1,2,4,6)
        assert target.boxes.tolist() == [[1.0, 2.0, 4.0, 6.0]]
        assert target.labels.tolist() == [5]
        assert target.scores.tolist() == pytest.approx([0.9])
        assert meta == {"id": 7, "height": 4, "width": 6}

    def test_getitem_empty_detections(self, tmp_path) -> None:
        cv2 = pytest.importorskip("cv2")
        np = pytest.importorskip("numpy")
        img = tmp_path / "img.png"
        cv2.imwrite(str(img), np.zeros((2, 2, 3), dtype=np.uint8))
        ds = ObjectDetectionDataset(samples=(ImageObjectDetectionSample(image_id=1, path_or_uri=str(img)),))
        _image, target, _meta = ds[0]
        assert target.boxes.shape == (0, 4)
        assert target.labels.shape == (0,)
        assert target.scores.shape == (0,)

    def test_non_integer_category_ids_map_to_sentinel(self, tmp_path) -> None:
        cv2 = pytest.importorskip("cv2")
        np = pytest.importorskip("numpy")
        img = tmp_path / "img.png"
        cv2.imwrite(str(img), np.zeros((2, 2, 3), dtype=np.uint8))
        sample = ImageObjectDetectionSample(
            image_id=1,
            path_or_uri=str(img),
            detections=(
                ObjectDetectionAnnotation(bbox=(0.0, 0.0, 1.0, 1.0), category_id=None),
                ObjectDetectionAnnotation(bbox=(0.0, 0.0, 1.0, 1.0), category_id="widget"),
                ObjectDetectionAnnotation(bbox=(0.0, 0.0, 1.0, 1.0), category_id=True),  # bool is not a label
                ObjectDetectionAnnotation(bbox=(0.0, 0.0, 1.0, 1.0), category_id=7),
            ),
        )
        _image, target, _meta = ObjectDetectionDataset(samples=(sample,))[0]
        assert target.labels.tolist() == [-1, -1, -1, 7]

    def test_getitem_missing_image_source_raises(self) -> None:
        pytest.importorskip("cv2")
        ds = ObjectDetectionDataset(samples=(ImageObjectDetectionSample(image_id=1),))  # no path/bytes
        with pytest.raises(ValueError, match="neither path_or_uri nor image_bytes"):
            _ = ds[0]

    def test_getitem_unreadable_image_raises(self, tmp_path) -> None:
        pytest.importorskip("cv2")
        bad = tmp_path / "notimage.png"
        bad.write_text("not an image")
        ds = ObjectDetectionDataset(samples=(ImageObjectDetectionSample(image_id=1, path_or_uri=str(bad)),))
        with pytest.raises(OSError, match="could not decode"):
            _ = ds[0]


class TestObjectDetectionFieldwise:
    """MAITE ``FieldwiseDataset`` surface: get_input / get_target / get_metadata."""

    def test_has_fieldwise_methods(self) -> None:
        ds = _ds()
        for name in ("get_input", "get_target", "get_metadata"):
            assert callable(getattr(ds, name))

    def test_target_and_metadata_need_no_image_decode(self) -> None:
        # _ds() samples carry width/height but no decodable image (file_name only):
        # fieldwise target + metadata resolve without a decode.
        ds = _ds()
        target = ds.get_target(0)
        assert target.boxes.shape == (2, 4)
        assert target.labels.tolist() == [1, 3]
        assert ds.get_metadata(0) == {"file_name": "a.jpg", "id": 100, "height": 480, "width": 640}
        # Empty-detection sample still yields an empty, well-formed target.
        empty = ds.get_target(1)
        assert empty.boxes.shape == (0, 4)

    def test_fieldwise_matches_getitem(self, tmp_path) -> None:
        cv2 = pytest.importorskip("cv2")
        import numpy as np

        img = tmp_path / "a.jpg"
        cv2.imwrite(str(img), np.full((4, 6, 3), 128, dtype=np.uint8))  # H=4, W=6
        ds = ObjectDetectionDataset(
            samples=(
                ImageObjectDetectionSample(
                    image_id=7,
                    path_or_uri=str(img),
                    file_name="a.jpg",
                    detections=(ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=1),),
                ),
            ),
        )
        image, target, metadata = ds[0]
        assert np.array_equal(ds.get_input(0), image)
        got = ds.get_target(0)
        assert np.array_equal(got.boxes, target.boxes)
        assert np.array_equal(got.labels, target.labels)
        assert ds.get_metadata(0) == metadata


class TestOdDatumMetadataExtras:
    """#79: source-preserving per-image metadata is surfaced as flat datum-metadata keys."""

    def test_surfaces_file_name_and_extras_with_reserved_precedence(self, tmp_path) -> None:
        cv2 = pytest.importorskip("cv2")
        import numpy as np

        img = tmp_path / "img.png"
        cv2.imwrite(str(img), np.full((4, 6, 3), 128, dtype=np.uint8))  # H=4, W=6
        sample = ImageObjectDetectionSample(
            image_id=100,
            path_or_uri=str(img),
            file_name="a.jpg",
            width=6,
            height=4,
            metadata={
                "license": 1,
                "date_captured": "2021-01-01",
                "flickr_url": "http://f/x",
                "coco_url": "http://c/x",
                "id": "SHOULD_NOT_WIN",  # reserved-key collision
                "width": -5,  # invalid raw value a COCO loader may leave visible
            },
            detections=(),
        )
        ds = ObjectDetectionDataset(samples=(sample,))
        _img, _target, meta = ds[0]
        assert meta["id"] == 100  # typed id wins over passthrough
        assert meta["file_name"] == "a.jpg"
        assert meta["width"] == 6  # typed dims win over -5
        assert meta["height"] == 4
        assert meta["license"] == 1
        assert meta["date_captured"] == "2021-01-01"
        assert meta["flickr_url"] == "http://f/x"
        assert meta["coco_url"] == "http://c/x"
        assert ds.get_metadata(0) == meta

    def test_bare_sample_metadata_is_exactly_id_height_width(self, tmp_path) -> None:
        cv2 = pytest.importorskip("cv2")
        import numpy as np

        img = tmp_path / "img.png"
        cv2.imwrite(str(img), np.zeros((3, 5, 3), dtype=np.uint8))
        sample = ImageObjectDetectionSample(image_id=7, path_or_uri=str(img), width=5, height=3)
        _img, _t, meta = ObjectDetectionDataset(samples=(sample,))[0]
        assert meta == {"id": 7, "height": 3, "width": 5}
