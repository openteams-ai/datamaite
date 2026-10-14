"""End-to-end: on-disk datasets -> public loaders -> color mode -> MAITE tooling (#84).

Hermetic (``tmp_path`` / ``memory://``), so it runs in every pipeline, but it
drives only the public surface a consumer uses: real files in each supported
still-image format are loaded with ``load_od`` / ``load_ic``, the view is set
with ``with_image_options``, and the dataset is consumed by MAITE's own
``maite.tasks.predict`` / ``evaluate``.

The oracle is independent of datamaite's decode path: every expected image is
decoded straight from the file on disk with OpenCV (or taken from the raw
SafeTensors tensor) and compared pixel-for-pixel with what MAITE received.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from datamaite import load_ic, load_od, write

cv2 = pytest.importorskip("cv2")
pytest.importorskip("maite")

MODES = ("rgb", "gray", "native")

# Distinct sizes so a sample mix-up cannot pass on shape alone.
_SOURCES = {
    "gray": np.random.default_rng(0).integers(0, 256, (24, 32), dtype=np.uint8),
    "rgb": np.random.default_rng(1).integers(0, 256, (30, 40, 3), dtype=np.uint8),
    "rgba": np.random.default_rng(2).integers(0, 256, (20, 36, 4), dtype=np.uint8),
}
# One box per image: inside, edge-straddling (clamps to 10x10), and full frame.
_BOXES = {"gray": [2, 3, 10, 8], "rgb": [30, 20, 15, 15], "rgba": [0, 0, 36, 20]}
_CATEGORY = {"gray": 1, "rgb": 2, "rgba": 1}


# --- independent oracle -----------------------------------------------------


def _expected_from_file(path: str, mode: str, region: tuple[float, float, float, float] | None) -> np.ndarray:
    """What ``mode`` must yield for ``path``, decoded with OpenCV directly."""
    if mode == "rgb":
        hwc = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1]
    elif mode == "gray":
        hwc = cv2.imread(path, cv2.IMREAD_GRAYSCALE)[:, :, None]
    else:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        hwc = raw[:, :, None] if raw.ndim == 2 else raw
        if hwc.shape[2] == 3:
            hwc = hwc[:, :, ::-1]
        elif hwc.shape[2] == 4:
            hwc = hwc[:, :, [2, 1, 0, 3]]
    if region is not None:
        left, top, width, height = region
        hwc = hwc[max(0, round(top)) : round(top + height), max(0, round(left)) : round(left + width)]
    return np.transpose(hwc, (2, 0, 1))


def _native_channels(stem: str) -> int:
    return {"gray": 1, "rgb": 3, "rgba": 4}[stem]


def _channels(stem: str, mode: str) -> int:
    return {"rgb": 3, "gray": 1, "native": _native_channels(stem)}[mode]


# --- fixtures: real on-disk datasets -----------------------------------------


def _write_images(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for stem, pixels in _SOURCES.items():
        cv2.imwrite(str(directory / f"{stem}.png"), pixels)


def _coco_root(root: Path) -> Path:
    _write_images(root / "images")
    images, annotations = [], []
    for index, (stem, pixels) in enumerate(_SOURCES.items(), start=1):
        height, width = pixels.shape[:2]
        images.append({"id": index, "file_name": f"images/{stem}.png", "width": width, "height": height})
        annotations.append({"id": 100 + index, "image_id": index, "category_id": _CATEGORY[stem], "bbox": _BOXES[stem]})
    payload = {
        "categories": [{"id": 1, "name": "car"}, {"id": 2, "name": "pedestrian"}],
        "images": images,
        "annotations": annotations,
    }
    (root / "annotations").mkdir(parents=True)
    (root / "annotations" / "instances.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def roots(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Every OD/IC on-disk layout, each produced from files, not in-memory records."""
    base = tmp_path_factory.mktemp("color_mode_e2e")
    coco = _coco_root(base / "coco")
    source = load_od(coco, dataset_format="coco")
    for fmt in ("yolo", "visdrone", "flat_images"):
        write(source, base / fmt, output_format=fmt)
    yolo_ic = base / "yolo_ic"
    for stem, pixels in _SOURCES.items():
        class_dir = yolo_ic / "train" / ("car" if _CATEGORY[stem] == 1 else "pedestrian")
        class_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(class_dir / f"{stem}.png"), pixels)
    return {
        "coco": coco,
        "yolo": base / "yolo",
        "visdrone": base / "visdrone" / "VisDrone2019-DET-train",
        "flat_images": base / "flat_images",
        "yolo_ic": yolo_ic,
    }


def _stem(sample: Any) -> str:
    return Path(str(sample.path_or_uri)).stem


OD_FORMATS = ("coco", "yolo", "visdrone", "flat_images")
IC_FORMATS = (("yolo_ic", "yolo"), ("visdrone", "visdrone"))


def _od(roots: dict[str, Path], fmt: str):  # type: ignore[no-untyped-def]
    return load_od(roots[fmt], dataset_format=fmt)


def _ic(roots: dict[str, Path], key: str, fmt: str):  # type: ignore[no-untyped-def]
    return load_ic(roots[key], dataset_format=fmt)


# --- MAITE consumers ----------------------------------------------------------


class _RecordingModel:
    """A MAITE model that records every input batch it is handed."""

    metadata: ClassVar[dict[str, str]] = {"id": "recording-model"}

    def __init__(self, make_target) -> None:  # type: ignore[no-untyped-def]
        self.inputs: list[np.ndarray] = []
        self._make_target = make_target

    def __call__(self, batch):  # type: ignore[no-untyped-def]
        self.inputs.extend(np.asarray(x) for x in batch)
        return [self._make_target() for _ in batch]


class _RecordingMetric:
    """A MAITE metric that records the ground-truth targets it is updated with."""

    metadata: ClassVar[dict[str, str]] = {"id": "recording-metric"}

    def __init__(self) -> None:
        self.targets: list[Any] = []

    def reset(self) -> None:
        self.targets = []

    def update(self, preds, targets, metadata=None) -> None:  # type: ignore[no-untyped-def]  # noqa: ARG002 - MAITE Metric signature
        self.targets.extend(targets)

    def compute(self) -> dict[str, Any]:
        return {"n": len(self.targets)}


def _empty_od_target():  # type: ignore[no-untyped-def]
    from datamaite.maite._common import EMPTY_BOXES, EMPTY_LABELS, EMPTY_SCORES
    from datamaite.maite._od import ObjectDetectionTarget

    return ObjectDetectionTarget(EMPTY_BOXES, EMPTY_LABELS, EMPTY_SCORES)


def _predict_inputs(dataset, make_target, batch_size: int = 2) -> list[np.ndarray]:  # type: ignore[no-untyped-def]
    from maite.tasks import predict

    model = _RecordingModel(make_target)
    preds, _ = predict(model=model, dataset=dataset, batch_size=batch_size)
    assert sum(len(batch) for batch in preds) == len(dataset)
    return model.inputs


def _od_target_key(target: Any) -> tuple[Any, ...]:
    return (target.boxes.tolist(), target.labels.tolist(), target.scores.tolist())


# --- object detection -------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fmt", OD_FORMATS)
class TestObjectDetectionEndToEnd:
    def test_maite_predict_receives_exact_pixels(self, roots, fmt: str, mode: str) -> None:  # type: ignore[no-untyped-def]
        from maite.protocols import object_detection as od_protocols

        ds = _od(roots, fmt).with_image_options(color_mode=mode)
        assert isinstance(ds, od_protocols.Dataset)
        assert len(ds) == len(_SOURCES)
        inputs = _predict_inputs(ds, _empty_od_target)
        for sample, image in zip(ds.samples, inputs, strict=True):
            expected = _expected_from_file(str(sample.path_or_uri), mode, None)
            assert image.shape == (_channels(_stem(sample), mode), *_SOURCES[_stem(sample)].shape[:2])
            assert image.dtype == np.uint8
            np.testing.assert_array_equal(image, expected)

    def test_targets_and_metadata_do_not_depend_on_mode(self, roots, fmt: str, mode: str) -> None:  # type: ignore[no-untyped-def]
        from maite.tasks import evaluate

        base = _od(roots, fmt)
        ds = base.with_image_options(color_mode=mode)
        for index in range(len(ds)):
            image, target, meta = ds[index]
            ref_image, ref_target, ref_meta = base[index]
            assert _od_target_key(target) == _od_target_key(ref_target)
            assert meta == ref_meta
            assert (meta["height"], meta["width"]) == image.shape[1:] == ref_image.shape[1:]
            assert ds.get_metadata(index) == meta

        metric = _RecordingMetric()
        result, _, _ = evaluate(model=_RecordingModel(_empty_od_target), metric=metric, dataset=ds, batch_size=2)
        assert result == {"n": len(ds)}
        assert [_od_target_key(t) for t in metric.targets] == [_od_target_key(base[i][1]) for i in range(len(base))]


# --- image classification -----------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("key", "fmt"), IC_FORMATS, ids=[k for k, _ in IC_FORMATS])
class TestImageClassificationEndToEnd:
    def test_maite_predict_receives_exact_pixels(self, roots, key: str, fmt: str, mode: str) -> None:  # type: ignore[no-untyped-def]
        from maite.protocols import image_classification as ic_protocols

        ds = _ic(roots, key, fmt).with_image_options(color_mode=mode)
        assert isinstance(ds, ic_protocols.Dataset)
        assert len(ds) == len(_SOURCES)
        num_classes = len(ds.index2label())
        inputs = _predict_inputs(ds, lambda: np.zeros(num_classes, dtype=np.float32))
        for sample, image in zip(ds.samples, inputs, strict=True):
            expected = _expected_from_file(str(sample.path_or_uri), mode, sample.region)
            assert image.shape[0] == _channels(_stem(sample), mode)
            np.testing.assert_array_equal(image, expected)

    def test_targets_and_metadata_do_not_depend_on_mode(self, roots, key: str, fmt: str, mode: str) -> None:  # type: ignore[no-untyped-def]
        base = _ic(roots, key, fmt)
        ds = base.with_image_options(color_mode=mode)
        for index in range(len(ds)):
            image, target, meta = ds[index]
            ref_image, ref_target, ref_meta = base[index]
            np.testing.assert_array_equal(target, ref_target)
            assert meta == ref_meta
            # VisDrone crops clamp at the image edge; metadata must follow the decoded crop.
            assert (meta["height"], meta["width"]) == image.shape[1:] == ref_image.shape[1:]
            assert ds.get_metadata(index) == meta


# --- remote storage ------------------------------------------------------------


def _upload(local_root: Path, remote_root: Any) -> None:
    for path in local_root.rglob("*"):
        if path.is_file():
            target = remote_root / path.relative_to(local_root).as_posix()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())


@pytest.mark.parametrize("mode", MODES)
def test_memory_backed_coco_matches_local(roots, memory_root, mode: str) -> None:  # type: ignore[no-untyped-def]
    """The same COCO root served from ``memory://`` decodes identically in every mode."""
    _upload(roots["coco"], memory_root)
    remote = load_od(str(memory_root), dataset_format="coco").with_image_options(color_mode=mode)
    local = _od(roots, "coco").with_image_options(color_mode=mode)
    remote_inputs = _predict_inputs(remote, _empty_od_target)
    local_inputs = _predict_inputs(local, _empty_od_target)
    assert [str(s.path_or_uri).startswith("memory://") for s in remote.samples] == [True] * len(_SOURCES)
    for remote_image, local_image in zip(remote_inputs, local_inputs, strict=True):
        np.testing.assert_array_equal(remote_image, local_image)


# --- SafeTensors ---------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
def test_flat_images_safetensors_end_to_end(tmp_path: Path, mode: str) -> None:
    st_numpy = pytest.importorskip("safetensors.numpy")
    tensors = {
        "gray": _SOURCES["gray"],  # (H, W)
        "rgb": np.ascontiguousarray(np.transpose(_SOURCES["rgb"], (2, 0, 1))),  # CHW
        "rgba": _SOURCES["rgba"],  # HWC, 4 channels
    }
    for stem, tensor in tensors.items():
        st_numpy.save_file({"image": tensor}, str(tmp_path / f"{stem}.safetensors"))

    ds = load_od(tmp_path, dataset_format="flat_images").with_image_options(color_mode=mode)
    inputs = _predict_inputs(ds, _empty_od_target)
    by_stem = {_stem(sample): image for sample, image in zip(ds.samples, inputs, strict=True)}
    assert set(by_stem) == set(tensors)

    rgb_views = {
        "gray": np.repeat(_SOURCES["gray"][:, :, None], 3, axis=2),
        "rgb": _SOURCES["rgb"],
        "rgba": _SOURCES["rgba"][:, :, :3],
    }
    for stem, image in by_stem.items():
        if mode == "rgb":
            expected = np.transpose(rgb_views[stem], (2, 0, 1))
        elif mode == "native":
            native = _SOURCES[stem] if _SOURCES[stem].ndim == 3 else _SOURCES[stem][:, :, None]
            expected = np.transpose(native, (2, 0, 1))
        else:
            source = _SOURCES[stem]
            gray = (
                source if source.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(source[:, :, :3]), cv2.COLOR_RGB2GRAY)
            )
            expected = gray[None, :, :]
        assert image.shape == expected.shape
        # BT.601 luma is float-rounded here and fixed-point in OpenCV: allow 1.
        tolerance = 1 if mode == "gray" and stem != "gray" else 0
        assert int(np.abs(image.astype(int) - expected.astype(int)).max()) <= tolerance


# --- writing is unaffected ------------------------------------------------------


@pytest.mark.parametrize("fmt", ["coco", "yolo", "visdrone"])
def test_write_ignores_color_mode(roots, tmp_path: Path, fmt: str) -> None:
    """The mode is a MAITE view: written datasets are byte-identical and reload as rgb."""
    base = _od(roots, "coco")

    def tree(root: Path) -> dict[str, bytes]:
        return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}

    write(base, tmp_path / "rgb", output_format=fmt)
    write(base.with_image_options(color_mode="gray"), tmp_path / "gray", output_format=fmt)
    assert tree(tmp_path / "gray") == tree(tmp_path / "rgb")

    reload_root = tmp_path / "gray" / "VisDrone2019-DET-train" if fmt == "visdrone" else tmp_path / "gray"
    assert load_od(reload_root, dataset_format=fmt).image_color_mode == "rgb"


# --- multispectral / SAR TIFFs (synthetic Sentinel-like fixtures) ----------------

_RASTER_BANDS = {"synthetic_s2_13band_uint16": (13, np.uint16), "synthetic_s1_2band_float32": (2, np.float32)}


@pytest.fixture(scope="module")
def raster_root(tmp_path_factory: pytest.TempPathFactory, synthetic_rasters: Path) -> Path:
    """COCO root over the synthetic rasters: one box each, one edge-straddling."""
    root = tmp_path_factory.mktemp("rasters")
    (root / "images").mkdir()
    images, annotations = [], []
    for index, stem in enumerate(_RASTER_BANDS, start=1):
        source = synthetic_rasters / f"{stem}.tif"
        (root / "images" / source.name).write_bytes(source.read_bytes())
        images.append({"id": index, "file_name": f"images/{source.name}", "width": 256, "height": 256})
        annotations.append({"id": 10 + index, "image_id": index, "category_id": 1, "bbox": [240, 16 * index, 32, 20]})
    payload = {"categories": [{"id": 1, "name": "field"}], "images": images, "annotations": annotations}
    (root / "annotations").mkdir()
    (root / "annotations" / "instances.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def test_multispectral_and_sar_reach_maite_unchanged(raster_root: Path) -> None:
    """COCO over Sentinel-2/-1-like rasters -> native -> MAITE predict/evaluate."""
    import tifffile
    from maite.tasks import evaluate

    base = load_od(raster_root, dataset_format="coco")
    ds = base.with_image_options(color_mode="native")
    inputs = _predict_inputs(ds, _empty_od_target)
    for sample, image in zip(ds.samples, inputs, strict=True):
        bands, dtype = _RASTER_BANDS[_stem(sample)]
        assert image.shape == (bands, 256, 256)
        assert image.dtype == dtype
        np.testing.assert_array_equal(image, np.transpose(tifffile.imread(str(sample.path_or_uri)), (2, 0, 1)))
    for index in range(len(ds)):
        # get_target never decodes, so the rgb view (which cannot read these rasters) is a fair reference.
        assert _od_target_key(ds[index][1]) == _od_target_key(base.get_target(index))
        assert (ds.get_metadata(index)["height"], ds.get_metadata(index)["width"]) == (256, 256)
    metric = _RecordingMetric()
    result, _, _ = evaluate(model=_RecordingModel(_empty_od_target), metric=metric, dataset=ds, batch_size=2)
    assert result == {"n": 2}


def test_multispectral_crops_through_visdrone_ic(raster_root: Path, tmp_path: Path) -> None:
    """VisDrone IC object crops of the rasters keep every band, clamped at the edge."""
    import tifffile

    write(
        load_od(raster_root, dataset_format="coco"), tmp_path / "vd", output_format="visdrone", class_map={"field": 11}
    )
    ds = load_ic(tmp_path / "vd" / "VisDrone2019-DET-train", dataset_format="visdrone").with_image_options(
        color_mode="native"
    )
    assert ds.samples
    num_classes = len(ds.index2label())
    inputs = _predict_inputs(ds, lambda: np.zeros(num_classes, dtype=np.float32))
    for sample, image in zip(ds.samples, inputs, strict=True):
        left, top, width, height = sample.region
        full = tifffile.imread(str(sample.path_or_uri))
        crop = full[round(top) : round(top + height), round(left) : min(256, round(left + width))]
        assert image.shape == (_RASTER_BANDS[_stem(sample)][0], 20, 16)  # 240 + 32 clamps to 256
        np.testing.assert_array_equal(image, np.transpose(crop, (2, 0, 1)))


@pytest.mark.parametrize("mode", ["rgb", "gray"])
def test_rasters_in_rgb_or_gray_fail_with_a_hint(raster_root: Path, mode: str) -> None:
    ds = load_od(raster_root, dataset_format="coco").with_image_options(color_mode=mode)
    with pytest.raises(OSError, match="color_mode='native'"):
        ds[0]
