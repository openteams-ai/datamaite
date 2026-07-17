"""Optional compatibility tests: writer output loads with the real Hugging Face ``datasets``.

datamaite deliberately has no ``datasets`` dependency, so these tests skip
unless the package is installed (e.g. ``uv run --with datasets --with-editable
. -- pytest tests/test_huggingface_vision_datasets_compat.py``). They pin the
claim that the writers emit the *local ImageFolder-compatible layout* —
``datasets.load_dataset("imagefolder", data_dir=...)`` reads it back —
especially the OD ``metadata.jsonl`` ``objects`` convention.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from datamaite import Task
from datamaite.image_classification import ImageClassificationDataset
from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import (
    ClassificationLabel,
    ImageClassificationSample,
    ImageObjectDetectionSample,
    ObjectDetectionAnnotation,
)
from datamaite.writers import get_writer

datasets = pytest.importorskip("datasets")

# A real, decodable 1x1 PNG so ``datasets`` image-feature inference succeeds.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _load_imagefolder(data_dir: Path, cache_dir: Path):  # type: ignore[no-untyped-def]
    return datasets.load_dataset("imagefolder", data_dir=str(data_dir), cache_dir=str(cache_dir))


class TestHuggingFaceDatasetsCompatibility:
    def test_od_jsonl_output_loads_with_datasets(self, tmp_path: Path) -> None:
        sample = ImageObjectDetectionSample(
            image_id="img1.png",
            image_bytes=_PNG,
            file_name="img1.png",
            width=1,
            height=1,
            split="train",
            detections=(
                ObjectDetectionAnnotation(
                    bbox=(10.0, 20.0, 30.0, 40.0),
                    category_id=0,
                    category_name="person",
                    source_annotation_id=101,
                    area=1200.0,
                ),
                ObjectDetectionAnnotation(bbox=(1.0, 2.0, 3.0, 4.0), category_id=1, category_name="dog"),
            ),
        )
        dest = tmp_path / "dest"
        get_writer("huggingface_vision", task=Task.OD).write(ObjectDetectionDataset(samples=(sample,)), dest)

        loaded = _load_imagefolder(dest, tmp_path / "hf-cache")

        assert list(loaded.keys()) == ["train"]
        split = loaded["train"]
        assert split.num_rows == 1
        # Read the objects column only (no image decode needed for the claim).
        objects = split["objects"][0]
        assert objects["bbox"] == [[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]]
        assert objects["categories"] == ["person", "dog"]
        assert objects["id"] == [101, None]
        assert (split["width"][0], split["height"][0]) == (1, 1)

    def test_ic_class_folder_output_loads_with_datasets(self, tmp_path: Path) -> None:
        samples = (
            ImageClassificationSample(
                image_id="a.png",
                image_bytes=_PNG,
                file_name="a.png",
                split="train",
                labels=(ClassificationLabel(category_id=0, category_name="cat"),),
            ),
            ImageClassificationSample(
                image_id="b.png",
                image_bytes=_PNG,
                file_name="b.png",
                split="train",
                labels=(ClassificationLabel(category_id=1, category_name="dog"),),
            ),
        )
        dest = tmp_path / "dest"
        get_writer("huggingface_vision", task=Task.IC).write(ImageClassificationDataset(samples=samples), dest)

        loaded = _load_imagefolder(dest, tmp_path / "hf-cache")

        assert list(loaded.keys()) == ["train"]
        split = loaded["train"]
        assert split.num_rows == 2
        label_feature = split.features["label"]
        assert sorted(label_feature.names) == ["cat", "dog"]
        assert sorted(label_feature.int2str(value) for value in split["label"]) == ["cat", "dog"]
