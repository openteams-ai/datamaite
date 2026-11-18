"""Write ``real_world_expected.json``: what each real-world export holds, read without datamaite.

``test_real_world_exports.py`` checks each loader call against this file. It
is written from the exports' own files (COCO JSON, YOLO label files and
``data.yaml``/``classes.txt``, Hugging Face ``metadata.jsonl``/``metadata.parquet``)
with nothing but ``json``, ``yaml``, ``pyarrow`` and Pillow, so a loader bug
cannot leak into it. Regenerate it when the example data changes::

    python tests/e2e/real_world_expected.py "$DATAMAITE_DATASETS_ROOT"

Each case maps to ``labels`` (``{id: name}``), ``label_details`` (for each
COCO category with them: a non-empty ``supercategory``, LVIS's ``synset``, and
any other keys as ``attributes``), the COCO ``info`` and ``licenses``, and
``images``: ``key``, ``split``, ``sha256`` of the file (``null`` when the
export's path cannot reach it), ``width``/``height`` (the stored size for
COCO; the displayed, EXIF-rotated size otherwise), ``metadata`` (the other
keys of a COCO image; YOLO and Hugging Face images have none) and ``boxes`` as
``[class id, class name, x, y, width, height, iscrowd]`` in pixels. A COCO box
has an eighth item with the rest of its annotation: ``id``, ``area``,
``segmentation`` (``null`` when empty) and ``attributes`` (every other key).

Conventions a correct loader follows, which this file encodes:

* YOLO split names are ``train``, ``val`` and ``test``; YOLO boxes are scaled
  by the displayed image size, as Ultralytics does.
* A COCO annotation file named ``instances_<Subset>.json`` holds split ``<Subset>``.
* Hugging Face boxes are ``[x, y, width, height]``. With no ``ClassLabel``
  table, a class is named as its id, and only used classes exist.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml
from PIL import Image

OUT = Path(__file__).with_name("real_world_expected.json")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
EXIF_ORIENTATION = 274

# case id -> (reader, export folder, reader arguments)
COCO_CASES: dict[str, tuple[str, list[tuple[str, str, str | None]]]] = {
    # (annotation file, image folder, split)
    "coco-cvat": (
        "coco/cvat",
        [
            ("annotations/instances_Train.json", "images/Train", "Train"),
            ("annotations/instances_Validation.json", "images/Validation", "Validation"),
        ],
    ),
    "coco-cvat-train": ("coco/cvat", [("annotations/instances_Train.json", "images/Train", "Train")]),
    "coco-cvat-validation": (
        "coco/cvat",
        [("annotations/instances_Validation.json", "images/Validation", "Validation")],
    ),
    "coco-label-studio": ("coco/label-studio", [("result.json", "", None)]),
    "coco-fiftyone": ("coco/fiftyone", [("labels.json", "data", None)]),
    "coco-roboflow-train": ("coco/roboflow/train", [("_annotations.coco.json", "", None)]),
    "coco-roboflow-valid": ("coco/roboflow/valid", [("_annotations.coco.json", "", None)]),
    "coco-roboflow-test": ("coco/roboflow/test", [("_annotations.coco.json", "", None)]),
    "coco-exif-orientation": ("coco/exif-orientation", [("annotations/instances.json", "", None)]),
}
YOLO_CASES: dict[str, tuple[str, str, list[tuple[str | None, str, str]]]] = {
    # (class names file, [(split, image folder, label folder)])
    "yolo-cvat-ultralytics": (
        "yolo/cvat-ultralytics",
        "data.yaml",
        [("train", "images/Train", "labels/Train"), ("val", "images/Validation", "labels/Validation")],
    ),
    "yolo-label-studio": ("yolo/label-studio", "classes.txt", [(None, "images", "labels")]),
    "yolo-ultralytics-od": (
        "yolo/ultralytics-od",
        "data.yaml",
        [("train", "images/train", "labels/train"), ("val", "images/val", "labels/val")],
    ),
    "yolo-roboflow-od": (
        "yolo/roboflow-od",
        "data.yaml",
        [
            ("train", "train/images", "train/labels"),
            ("val", "valid/images", "valid/labels"),
            ("test", "test/images", "test/labels"),
        ],
    ),
    "yolo-exif-orientation": (
        "yolo/exif-orientation",
        "data.yaml",
        [("train", "images/train", "labels/train"), ("val", "images/val", "labels/val")],
    ),
}
HF_CASES: dict[str, tuple[str, str]] = {
    "hf-imagefolder-od": ("huggingface_vision/imagefolder-od", "metadata.jsonl"),
    "hf-parquet-od": ("huggingface_vision/parquet-od", "metadata.parquet"),
}
HF_SPLITS = ("train", "validation", "test")


def _image(path: Path) -> tuple[str | None, tuple[int, int] | None]:
    """The file's sha256 and displayed size, or ``(None, None)`` when it is not there."""
    if not path.is_file():
        return None, None
    with Image.open(path) as image:
        width, height = image.size
        if image.getexif().get(EXIF_ORIENTATION) in (5, 6, 7, 8):
            width, height = height, width
    return hashlib.sha256(path.read_bytes()).hexdigest(), (width, height)


def _num(value: float) -> float:
    return round(float(value), 4)


# Keys a COCO image, annotation and category use for the fields above; any other key is recorded as is.
_COCO_IMAGE_KEYS = {"id", "file_name", "width", "height"}
_COCO_ANNOTATION_KEYS = {"id", "image_id", "category_id", "bbox", "area", "iscrowd", "segmentation"}
_COCO_CATEGORY_KEYS = {"id", "name", "supercategory", "synset"}


def _coco_details(category: dict[str, Any]) -> dict[str, Any]:
    details: dict[str, Any] = {}
    if category.get("supercategory"):
        details["supercategory"] = category["supercategory"]
    if category.get("synset"):
        details["synset"] = category["synset"]
    attributes = {k: v for k, v in category.items() if k not in _COCO_CATEGORY_KEYS}
    if attributes:
        details["attributes"] = attributes
    return details


def _coco_box(annotation: dict[str, Any], labels: dict[int, str]) -> list[Any]:
    rest = {
        "id": annotation["id"],
        "area": annotation.get("area"),
        "segmentation": annotation.get("segmentation") or None,
        "attributes": {k: v for k, v in annotation.items() if k not in _COCO_ANNOTATION_KEYS},
    }
    category_id = annotation["category_id"]
    bbox = [_num(value) for value in annotation["bbox"]]
    return [category_id, labels[category_id], *bbox, int(annotation.get("iscrowd", 0)), rest]


def coco_case(root: Path, files: list[tuple[str, str, str | None]]) -> dict[str, Any]:
    labels: dict[int, str] = {}
    details: dict[int, dict[str, Any]] = {}
    images = []
    info: dict[str, Any] = {}
    licenses: list[dict[str, Any]] = []
    for annotation_file, image_dir, split in files:
        data = json.loads((root / annotation_file).read_text(encoding="utf-8"))
        # Several files of one export repeat the same info and licenses; the first file's are kept.
        info = info or data.get("info") or {}
        licenses = licenses or data.get("licenses") or []
        for category in data["categories"]:
            labels[category["id"]] = category["name"]
            if _coco_details(category):
                details[category["id"]] = _coco_details(category)
        by_image: dict[int, list[dict[str, Any]]] = {image["id"]: [] for image in data["images"]}
        for annotation in data["annotations"]:
            by_image[annotation["image_id"]].append(annotation)
        for image in data["images"]:
            sha, size = _image(root / image_dir / image["file_name"])
            if size is not None and size != (image["width"], image["height"]):
                stored = f"{image['width']}x{image['height']}"
                print(f"note: {annotation_file} {image['file_name']}: stored {stored}, pixels {size[0]}x{size[1]}")
            annotations = by_image[image["id"]]
            images.append(
                {
                    "key": image["file_name"],
                    "split": split,
                    "sha256": sha,
                    "width": image["width"],
                    "height": image["height"],
                    "metadata": {k: v for k, v in image.items() if k not in _COCO_IMAGE_KEYS},
                    "boxes": [_coco_box(a, labels) for a in annotations],
                }
            )
    return {"labels": labels, "label_details": details, "info": info, "licenses": licenses, "images": images}


def _yolo_names(path: Path) -> dict[int, str]:
    if path.name == "classes.txt":
        return dict(enumerate(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()))
    names = yaml.safe_load(path.read_text(encoding="utf-8"))["names"]
    return {int(k): str(v) for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(map(str, names)))


def yolo_case(root: Path, names_file: str, splits: list[tuple[str | None, str, str]]) -> dict[str, Any]:
    labels = _yolo_names(root / names_file)
    images = []
    for split, image_dir, label_dir in splits:
        for path in sorted(p for p in (root / image_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            sha, size = _image(path)
            assert size is not None
            width, height = size
            label_file = root / label_dir / f"{path.stem}.txt"
            rows = label_file.read_text(encoding="utf-8").split("\n") if label_file.is_file() else []
            boxes = []
            for row in rows:
                if not row.strip():
                    continue
                class_id, cx, cy, w, h = row.split()[:5]
                cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
                cid = int(class_id)
                boxes.append([cid, labels[cid], _num(cx - w / 2), _num(cy - h / 2), _num(w), _num(h), 0])
            images.append(
                {
                    "key": path.name,
                    "split": split,
                    "sha256": sha,
                    "width": width,
                    "height": height,
                    "metadata": {},
                    "boxes": boxes,
                }
            )
    return {"labels": labels, "label_details": {}, "images": images}


def hf_case(root: Path, metadata: str) -> dict[str, Any]:
    names: list[str] | None = None
    rows: list[tuple[str, dict[str, Any]]] = []
    for split in HF_SPLITS:
        path = root / split / metadata
        if metadata.endswith(".parquet"):
            table = pq.read_table(path)
            features = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]
            names = features["objects"]["categories"]["feature"]["names"]
            rows += [(split, row) for row in table.to_pylist()]
        else:
            rows += [
                (split, json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
    used = sorted({c for _, row in rows for c in row["objects"]["categories"]})
    labels = dict(enumerate(names)) if names is not None else {c: str(c) for c in used}
    images = []
    for split, row in rows:
        sha, size = _image(root / split / row["file_name"])
        assert size is not None
        objects = row["objects"]
        images.append(
            {
                "key": row["file_name"],
                "split": split,
                "sha256": sha,
                "width": size[0],
                "height": size[1],
                "metadata": {},
                "boxes": [
                    [c, labels[c], *map(_num, bbox), 0]
                    for bbox, c in zip(objects["bbox"], objects["categories"], strict=True)
                ],
            }
        )
    return {"labels": labels, "label_details": {}, "images": images}


def _dumps(cases: dict[str, Any]) -> str:
    """JSON with one line per image, so a change to the data shows as a readable diff."""
    blocks = []
    for case_id, case in cases.items():
        head = {k: v for k, v in case.items() if k != "images"}
        lines = ",\n".join(f"   {json.dumps(image)}" for image in case["images"])
        blocks.append(f' {json.dumps(case_id)}: {json.dumps(head)[:-1]}, "images": [\n{lines}\n ]}}')
    return "{\n" + ",\n".join(blocks) + "\n}\n"


def main(datasets_root: Path) -> None:
    cases: dict[str, Any] = {}
    for case_id, (folder, files) in COCO_CASES.items():
        cases[case_id] = coco_case(datasets_root / folder, files)
    for case_id, (folder, names_file, splits) in YOLO_CASES.items():
        cases[case_id] = yolo_case(datasets_root / folder, names_file, splits)
    for case_id, (folder, metadata) in HF_CASES.items():
        cases[case_id] = hf_case(datasets_root / folder, metadata)
    for case in cases.values():
        case["images"].sort(key=lambda image: (image["key"], image["split"] or ""))
    OUT.write_text(_dumps(dict(sorted(cases.items()))), encoding="utf-8")
    print(f"wrote {OUT} ({sum(len(c['images']) for c in cases.values())} images)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
