"""What each format in the harness can carry, and how to write it by hand.

Every format in the harness declares two things:

* ``losses`` -- the :class:`Loss` values a comparison through the format
  ignores, and nothing else. They are declared in three groups, so that a
  choice is never mistaken for a limit of the format:

  - ``cannot_store``: the format has no standard place for the value;
  - ``derives``: the format or its loader makes a value up, such as an area
    from the box or a new id;
  - ``chooses``: the format could keep the value, but datamaite's writer or
    loader does not. Each of these could be fixed instead.
* ``author`` -- a writer for the format's on-disk layout that does not use
  datamaite, so that loading a hand-written original tests the loader on its
  own instead of only against datamaite's own writer.

See ``tests/README.md`` ("Conversion harness") for how a format joins.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import count
from pathlib import Path

from datamaite._types import DatasetFormat, Task
from datamaite.loaders import available_formats
from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import ImageObjectDetectionSample
from datamaite.writers import available_output_formats

from .compare import SPLIT_ALIASES, Loss


@dataclass(frozen=True)
class FormatSpec:
    """One format's allowed losses, grouped by why it allows them, and its hand-written layout."""

    format: DatasetFormat
    author: Callable[[ObjectDetectionDataset, Path], Path]
    cannot_store: frozenset[Loss] = frozenset()
    derives: frozenset[Loss] = frozenset()
    chooses: frozenset[Loss] = frozenset()

    def __post_init__(self) -> None:
        if len(self.cannot_store) + len(self.derives) + len(self.chooses) != len(self.losses):
            raise ValueError(f"{self.format.value}: a loss is declared in more than one group")

    @property
    def losses(self) -> frozenset[Loss]:
        return self.cannot_store | self.derives | self.chooses


def _image_bytes(sample: ImageObjectDetectionSample) -> bytes:
    if sample.image_bytes is not None:
        return sample.image_bytes
    assert sample.path_or_uri is not None
    return Path(sample.path_or_uri).read_bytes()


def _class_ids(dataset: ObjectDetectionDataset) -> list[int]:
    """The taxonomy's class ids in entry order, which is YOLO's class order."""
    taxonomy = dataset.dataset_metadata.taxonomy
    assert taxonomy is not None, "the hand-written layouts need a taxonomy"
    return [entry.source_id for entry in taxonomy.entries]  # type: ignore[misc]


def _names(dataset: ObjectDetectionDataset) -> dict[int, str]:
    taxonomy = dataset.dataset_metadata.taxonomy
    assert taxonomy is not None
    return {entry.source_id: entry.name for entry in taxonomy.entries}  # type: ignore[misc]


def author_yolo(dataset: ObjectDetectionDataset, root: Path) -> Path:
    """Ultralytics layout: ``images/<split>/``, ``labels/<split>/``, ``data.yaml`` with dense class indices."""
    index = {class_id: position for position, class_id in enumerate(_class_ids(dataset))}
    names = _names(dataset)
    splits: dict[str, None] = {}
    for sample in dataset.samples:
        split = sample.split or "train"
        splits[split] = None
        image = root / "images" / split / sample.file_name
        label = (root / "labels" / split / sample.file_name).with_suffix(".txt")
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(_image_bytes(sample))
        rows = []
        for detection in sample.detections:
            x, y, w, h = detection.bbox
            rows.append(
                f"{index[detection.category_id]} {(x + w / 2) / sample.width:.6f} {(y + h / 2) / sample.height:.6f} "
                f"{w / sample.width:.6f} {h / sample.height:.6f}"
            )
        label.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")
    # data.yaml keys are train/val/test; any other split keeps its own name as the key.
    yaml_lines = ["path: ."]
    yaml_lines += [f"{SPLIT_ALIASES.get(split, split)}: images/{split}" for split in splits]
    yaml_lines.append("names:")
    # Quoted (JSON strings are valid YAML), so names such as "yes" or "1" stay strings.
    yaml_lines += [
        f"  {position}: {json.dumps(names[class_id])}" for position, class_id in enumerate(_class_ids(dataset))
    ]
    (root / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return root


def _coco_categories(dataset: ObjectDetectionDataset) -> list[dict[str, object]]:
    """Categories as COCO tools write them, with LVIS's ``synset`` and any other class attributes as extra keys."""
    taxonomy = dataset.dataset_metadata.taxonomy
    assert taxonomy is not None
    categories = []
    for entry in taxonomy.entries:
        category: dict[str, object] = {"id": entry.source_id, "name": entry.name}
        if entry.supercategory is not None:
            category["supercategory"] = entry.supercategory
        if entry.synset is not None:
            category["synset"] = entry.synset
        categories.append({**entry.attributes, **category})
    return categories


def _annotation_ids(dataset: ObjectDetectionDataset) -> Callable[[ImageObjectDetectionSample, int], int]:
    """Each box's COCO id: its source annotation id, else the next id no box has."""
    taken = {d.source_annotation_id for s in dataset.samples for d in s.detections}
    fresh = (i for i in count(1) if i not in taken)
    ids = {
        (id(sample), index): detection.source_annotation_id
        if isinstance(detection.source_annotation_id, int)
        else next(fresh)
        for sample in dataset.samples
        for index, detection in enumerate(sample.detections)
    }
    return lambda sample, index: ids[(id(sample), index)]


def author_coco(dataset: ObjectDetectionDataset, root: Path) -> Path:
    """COCO layout: ``annotations/instances.json``, or ``instances_<split>.json`` per split, images at ``file_name``.

    Like a labelling tool's export, every annotation has an ``area`` (the box's
    when the source has none) and a ``segmentation`` (empty when there is
    none). Image metadata, box attributes and class attributes are extra keys
    on their objects, and the dataset's ``info`` and ``licenses`` are kept.
    """
    by_split: dict[str | None, list[ImageObjectDetectionSample]] = {}
    for sample in dataset.samples:
        by_split.setdefault(sample.split, []).append(sample)
    single = len(by_split) == 1
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    annotation_id = _annotation_ids(dataset)
    metadata = dataset.dataset_metadata
    for split, samples in by_split.items():
        images, annotations = [], []
        for sample in samples:
            image = root / sample.file_name
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(_image_bytes(sample))
            core = {
                "id": sample.image_id,
                "file_name": sample.file_name,
                "width": sample.width,
                "height": sample.height,
            }
            images.append({**sample.metadata, **core})
            for index, detection in enumerate(sample.detections):
                x, y, w, h = detection.bbox
                annotations.append(
                    {
                        **detection.attributes,
                        "id": annotation_id(sample, index),
                        "image_id": sample.image_id,
                        "category_id": detection.category_id,
                        "bbox": [x, y, w, h],
                        "area": detection.area if detection.area is not None else w * h,
                        "segmentation": detection.segmentation if detection.segmentation is not None else [],
                        "iscrowd": detection.iscrowd,
                    }
                )
        document = {
            "info": metadata.info,
            "licenses": list(metadata.licenses),
            "images": images,
            "annotations": annotations,
            "categories": _coco_categories(dataset),
        }
        file_name = "instances.json" if single else f"instances_{split}.json"
        (root / "annotations" / file_name).write_text(json.dumps(document), encoding="utf-8")
    return root


FORMATS: dict[DatasetFormat, FormatSpec] = {
    DatasetFormat.COCO: FormatSpec(
        DatasetFormat.COCO,
        author=author_coco,
        cannot_store=frozenset(
            {
                # There is no list of split names.
                Loss.SPLIT_LIST,
                # An annotation has one category_id; a separate source class id has nowhere to go.
                Loss.SOURCE_CATEGORY_IDS,
                # Ground-truth annotations have no score.
                Loss.SCORE,
                # Nothing records whether class ids are dense; the loader marks every taxonomy sparse.
                Loss.ID_DENSITY,
                # Categories have no evaluation flag or source dataset, and a file has no extras.
                Loss.CLASS_EVAL_EXCLUDED,
                Loss.CLASS_SOURCE_DATASET,
                Loss.DATASET_EXTRAS,
            }
        ),
        derives=frozenset(
            {
                # Every annotation has an area and an id; a box with none gets w*h and a new id.
                Loss.AREA_FROM_BOX,
                Loss.NEW_ANNOTATION_IDS,
            }
        ),
        chooses=frozenset(
            {
                # The writer puts a single split in instances.json, which has no split name,
                # though instances_<split>.json would keep it. Several splits are #131.
                Loss.SINGLE_SPLIT_NAME,
            }
        ),
    ),
    DatasetFormat.YOLO: FormatSpec(
        DatasetFormat.YOLO,
        author=author_yolo,
        cannot_store=frozenset(
            {
                # Image ids are file paths and class ids are list positions.
                Loss.IMAGE_IDS,
                # Label files are named after image stems: a.png and a.jpg in one split can't both keep their names.
                Loss.LABEL_FILE_STEMS,
                Loss.LABEL_IDS,
                Loss.SOURCE_CATEGORY_IDS,
                # data.yaml names hold only class names.
                Loss.LABEL_DETAILS,
                # A ground-truth label row is a class and a box: no crowd flag, score,
                # polygon, area, attributes or annotation id.
                Loss.CROWD,
                Loss.SCORE,
                Loss.SEGMENTATION,
                Loss.AREA,
                Loss.BOX_ATTRIBUTES,
                Loss.ANNOTATION_IDS,
                # Nothing holds per-image metadata or the dataset's info, licenses and extras.
                Loss.IMAGE_METADATA,
                Loss.INFO,
                Loss.LICENSES,
                Loss.DATASET_EXTRAS,
            }
        ),
        derives=frozenset(
            {
                # The split list is the image folders found.
                Loss.SPLIT_LIST_FROM_IMAGES,
                # The loader records each image's and box's source files and rows.
                Loss.YOLO_SOURCE_DETAILS,
                # The loader marks the taxonomy dense when its class ids are 0..n-1.
                Loss.ID_DENSITY_FROM_IDS,
            }
        ),
        chooses=frozenset(
            {
                # The writer renames validation and valid to val, and training to train,
                # and puts images with no split in train rather than in a flat folder.
                Loss.SPLIT_ALIASES,
                Loss.DEFAULT_SPLIT,
            }
        ),
    ),
}

# Object-detection formats with a loader and a writer that the harness does
# not cover yet (issue #149 adds them). A new read+write format must be added
# to FORMATS or here; test_harness_covers_every_od_format checks that.
NOT_YET_COVERED = frozenset({DatasetFormat.FLAT_IMAGES, DatasetFormat.HUGGINGFACE_VISION, DatasetFormat.VISDRONE})


def convertible_od_formats() -> list[DatasetFormat]:
    """Object-detection formats with both a loader and a writer, from the registries."""
    return sorted(
        set(available_formats(task=Task.OD)) & set(available_output_formats(task=Task.OD)), key=lambda f: f.value
    )


def harness_formats() -> list[DatasetFormat]:
    """The registered formats the harness covers, in a stable order."""
    return [fmt for fmt in convertible_od_formats() if fmt in FORMATS]


def loss_stages(formats: Iterable[DatasetFormat | str]) -> tuple[frozenset[Loss], ...]:
    """Each format's losses, in the order the data passes through ``formats``, for ``compare(allowed=...)``."""
    return tuple(FORMATS[DatasetFormat(fmt)].losses for fmt in formats)
