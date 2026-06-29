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
