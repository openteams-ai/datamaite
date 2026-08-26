"""Registry-wide memory:// read/write/convert contract tests."""

from __future__ import annotations

import pickle
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from datamaite import (
    ClassificationLabel,
    DatasetMetadata,
    ImageClassificationDataset,
    ImageClassificationSample,
    ImageObjectDetectionSample,
    ObjectDetectionAnnotation,
    ObjectDetectionDataset,
    Taxonomy,
    load,
    write,
)
from datamaite._io import source_path, source_storage_context
from datamaite._types import DatasetFormat, Task
from datamaite.loaders import available_loader_keys
from datamaite.model import VideoClassificationDataset, VideoClassificationSample
from datamaite.taxonomy import CategoryEntry
from datamaite.writers import available_writer_keys

from ._maite_factory import CATEGORIES, WIDGET, box, sequence


def _png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    assert cv2.imwrite(str(path), image)
    return path


def _ic_dataset(tmp_path: Path) -> ImageClassificationDataset:
    image = _png(tmp_path / "ic.png")
    taxonomy = Taxonomy(entries=(CategoryEntry(source_id=1, name="pedestrian"),), id_density="sparse")
    sample = ImageClassificationSample(
        image_id="ic",
        path_or_uri=str(image),
        file_name="ic.png",
        width=8,
        height=6,
        split="train",
        labels=(ClassificationLabel(category_id=1, source_category_id=1, category_name="pedestrian"),),
    )
    return ImageClassificationDataset(samples=(sample,), dataset_metadata=DatasetMetadata(taxonomy=taxonomy))


def _od_dataset(tmp_path: Path) -> ObjectDetectionDataset:
    image = _png(tmp_path / "image.png")
    taxonomy = Taxonomy(entries=(CategoryEntry(source_id=1, name="pedestrian"),), id_density="sparse")
    sample = ImageObjectDetectionSample(
        image_id=1,
        path_or_uri=str(image),
        file_name="image.png",
        width=8,
        height=6,
        split="train",
        detections=(
            ObjectDetectionAnnotation(
                bbox=(1.0, 1.0, 4.0, 3.0),
                category_id=1,
                source_category_id=1,
                category_name="pedestrian",
                attributes={"visdrone_category_id": 1, "mot_class_id": 1},
            ),
        ),
    )
    return ObjectDetectionDataset(samples=(sample,), dataset_metadata=DatasetMetadata(taxonomy=taxonomy))


def _vc_dataset(tmp_path: Path) -> VideoClassificationDataset:
    tmp_path.mkdir(parents=True, exist_ok=True)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video payload")
    sample = VideoClassificationSample(
        video_id=1,
        video_path=str(video),
        file_name="clip.mp4",
        label="pedestrian",
        label_id=0,
        split="train",
    )
    return VideoClassificationDataset(samples=(sample,), categories={"pedestrian": 0}, labels={0: "pedestrian"})


def _mot_dataset(tmp_path: Path):  # type: ignore[no-untyped-def]
    from datamaite.model import BoxTrackDataset

    tmp_path.mkdir(parents=True, exist_ok=True)
    video = tmp_path / "video.mp4"
    container = av.open(str(video), "w")
    stream = container.add_stream("libx264", rate=10)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"
    for _ in range(3):
        frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    boxes = [
        box(
            track_id=1,
            category_id=1,
            uri=WIDGET,
            name="widget",
            bbox=(1, 1, 8, 8),
            frame_index=0,
        )
    ]
    seq = sequence(str(video), fps=10, num_frames=3, num_frames_exact=True, boxes=boxes)
    return BoxTrackDataset(sequences=(seq,), categories=dict(CATEGORIES))


def _dataset_for(task: Task, tmp_path: Path):  # type: ignore[no-untyped-def]
    if task is Task.IC:
        return _ic_dataset(tmp_path)
    if task is Task.OD:
        return _od_dataset(tmp_path)
    if task is Task.VC:
        return _vc_dataset(tmp_path)
    return _mot_dataset(tmp_path)


def _content_count(dataset: object) -> int:
    if hasattr(dataset, "sample_count"):
        return int(dataset.sample_count)  # type: ignore[attr-defined]
    return int(dataset.sequence_count)  # type: ignore[attr-defined]


def _reload_root(root, key):  # type: ignore[no-untyped-def]
    if key.format is DatasetFormat.VISDRONE and key.task in {Task.IC, Task.OD}:
        return root / "VisDrone2019-DET-train"
    return root


def _image_sample_signature(sample):  # type: ignore[no-untyped-def]
    return (
        sample.image_id,
        sample.file_name,
        sample.width,
        sample.height,
        sample.split,
        sample.region,
        sample.labels if hasattr(sample, "labels") else sample.detections,
    )


def _semantic_metadata(value):  # type: ignore[no-untyped-def]
    """Drop backend locations while retaining source-format metadata."""
    if isinstance(value, dict):
        return {
            key: _semantic_metadata(item)
            for key, item in value.items()
            if key not in {"source_path", "annotation_file", "metadata_file", "sequence_dir", "seqinfo_path"}
        }
    if isinstance(value, (list, tuple)):
        return tuple(_semantic_metadata(item) for item in value)
    return value


def _sequence_signature(sequence):  # type: ignore[no-untyped-def]
    return (
        sequence.video_id,
        sequence.fps,
        sequence.num_frames,
        sequence.duration,
        sequence.status,
        _semantic_metadata(sequence.video_meta),
        _semantic_metadata(sequence.metadata),
        sequence.boxes,
        sequence.width,
        sequence.height,
        sequence.size_bytes,
        sequence.frame_pattern,
        sequence.frame_number_base,
        sequence.num_frames_exact,
    )


def _read_media(dataset, path: str) -> bytes:  # type: ignore[no-untyped-def]
    with source_storage_context(dataset):
        return source_path(path).read_bytes()


def _assert_dataset_parity(expected, actual, task: Task) -> None:  # type: ignore[no-untyped-def]
    assert _content_count(actual) == _content_count(expected)
    if task in {Task.IC, Task.OD}:
        assert expected.index2label() == actual.index2label()
        assert [_image_sample_signature(sample) for sample in expected.samples] == [
            _image_sample_signature(sample) for sample in actual.samples
        ]
        for index in range(len(expected)):
            expected_metadata = expected.get_metadata(index)
            actual_metadata = actual.get_metadata(index)
            for key in ("id", "file_name", "split", "width", "height", "source_format"):
                assert actual_metadata.get(key) == expected_metadata.get(key)
            np.testing.assert_array_equal(actual.get_input(index), expected.get_input(index))
        return
    if task is Task.MOT:
        assert actual.categories == expected.categories
        assert [_sequence_signature(sequence) for sequence in actual.sequences] == [
            _sequence_signature(sequence) for sequence in expected.sequences
        ]
        expected_mot = expected.with_mot_options(empty_frame_policy="all")
        actual_mot = actual.with_mot_options(empty_frame_policy="all")
        expected_frame = next(iter(expected_mot[0][0]))
        actual_frame = next(iter(actual_mot[0][0]))
        np.testing.assert_array_equal(actual_frame.pixels, expected_frame.pixels)
        return
    assert actual.categories == expected.categories
    assert actual.labels == expected.labels

    def vc_signature(sample):  # type: ignore[no-untyped-def]
        return (
            sample.video_id,
            sample.file_name,
            sample.label,
            sample.label_id,
            sample.label_uri,
            sample.split,
            _semantic_metadata(sample.video_meta),
            _semantic_metadata(sample.metadata),
            sample.size_bytes,
        )

    assert [vc_signature(sample) for sample in actual.samples] == [vc_signature(sample) for sample in expected.samples]
    for expected_sample, actual_sample in zip(expected.samples, actual.samples, strict=True):
        assert _read_media(actual, actual_sample.video_path) == _read_media(expected, expected_sample.video_path)


WRITER_KEYS = available_writer_keys()
_AUTODETECT_EXCLUSIONS = {
    (Task.OD, DatasetFormat.FLAT_IMAGES),  # intentionally explicit: any folder of images would match
    (Task.IC, DatasetFormat.HUGGINGFACE_VISION),  # folder layout is structurally identical to YOLO IC
    (Task.IC, DatasetFormat.VISDRONE),  # also matches YOLO's broad class-folder sniffer
}
AUTODETECT_KEYS = [key for key in WRITER_KEYS if (key.task, key.format) not in _AUTODETECT_EXCLUSIONS]


def test_empty_source_storage_override_clears_dataset_credentials() -> None:
    from datamaite._io import source_path, source_storage_context

    dataset = ImageClassificationDataset(samples=(), _storage_options={"secret_marker": "inherited"})
    with source_storage_context(dataset, {}):
        path = source_path("memory://bucket/image.png")
    assert "secret_marker" not in path.storage_options


def test_video_runtime_objects_do_not_pickle_credentials() -> None:
    from datamaite.maite._decode import PyAVDecoder

    decoder = PyAVDecoder({"secret_marker": "must-not-serialize"})
    stream = decoder.stream("memory://bucket/video.mp4", [0])

    assert b"must-not-serialize" not in pickle.dumps(decoder)
    assert b"must-not-serialize" not in pickle.dumps(stream)

    from datamaite.model import BoxTrackDataset

    payload = pickle.dumps(BoxTrackDataset(sequences=(), categories={}, _decoder=decoder))
    rebound = pickle.loads(payload).with_storage_options(  # noqa: S301 - trusted in-process round-trip
        {"secret_marker": "worker-local"}
    )
    assert rebound._decoder._storage_options == {"secret_marker": "worker-local"}

    from datamaite.model import VideoSequence

    frame_sequence = VideoSequence(
        video_id=1,
        video_path=None,
        fps=1.0,
        num_frames=1,
        duration=1.0,
        annotation_path="memory://bucket/ann.txt",
        frame_files=("memory://bucket/frame.png",),
        num_frames_exact=True,
    )
    frame_dataset = BoxTrackDataset(
        sequences=(frame_sequence,),
        categories={},
        _storage_options={"secret_marker": "must-not-serialize"},
    )
    image_stream = frame_dataset.with_mot_options(empty_frame_policy="all")[0][0]
    assert b"must-not-serialize" not in pickle.dumps(image_stream)


class TestRegistryCoverage:
    def test_every_reader_declares_remote_support(self) -> None:
        from datamaite.loaders import get_loader

        assert all(
            get_loader(key.format, task=key.task, variant=key.variant).supports_remote
            for key in available_loader_keys()
        )

    def test_every_registered_format_has_reader_and_writer(self) -> None:
        reader_keys = {(key.task, key.format, key.variant) for key in available_loader_keys()}
        writer_keys = {(key.task, key.format, key.variant) for key in available_writer_keys()}
        assert reader_keys == writer_keys


@pytest.mark.parametrize("key", WRITER_KEYS, ids=lambda key: f"{key.task.value}-{key.format.value}")
def test_writer_storage_directions_and_remote_reload(key, tmp_path: Path, memory_root) -> None:  # type: ignore[no-untyped-def]
    dataset = _dataset_for(key.task, tmp_path / "source")
    local_reference = tmp_path / "local-reference"
    assert write(
        dataset,
        local_reference,
        output_format=key.format,
        output_variant=key.variant,
        verbose=True,
    )
    loaded_local = load(
        _reload_root(local_reference, key),
        dataset_format=key.format,
        task=key.task,
        registry_variant=key.variant,
    )
    remote_a = memory_root / "matrix" / key.task.value / key.format.value / "a"

    local_to_cloud = write(
        dataset,
        remote_a,
        output_format=key.format,
        output_variant=key.variant,
        verbose=True,
    )
    assert local_to_cloud
    loaded_remote = load(
        _reload_root(remote_a, key),
        dataset_format=key.format,
        task=key.task,
        registry_variant=key.variant,
        storage_options={"secret_marker": "must-not-serialize"},
    )
    assert _content_count(loaded_remote) > 0
    _assert_dataset_parity(loaded_local, loaded_remote, key.task)
    payload = pickle.dumps(loaded_remote)
    assert b"must-not-serialize" not in payload
    loaded_remote = pickle.loads(payload)  # noqa: S301 - trusted in-process round-trip
    if key.task is Task.MOT:
        mot_dataset = loaded_remote.with_mot_options(empty_frame_policy="all")
        frame = next(iter(mot_dataset[0][0]))
        assert frame.pixels.shape[0] == 3
    elif key.task in {Task.IC, Task.OD}:
        assert loaded_remote.get_input(0).shape[0] == 3

    local_b = tmp_path / "cloud-to-local"
    cloud_to_local = write(
        loaded_remote,
        local_b,
        output_format=key.format,
        output_variant=key.variant,
        verbose=True,
    )
    assert cloud_to_local
    loaded_cloud_to_local = load(
        _reload_root(local_b, key),
        dataset_format=key.format,
        task=key.task,
        registry_variant=key.variant,
    )
    assert _content_count(loaded_cloud_to_local) > 0
    _assert_dataset_parity(loaded_local, loaded_cloud_to_local, key.task)

    remote_c = memory_root / "matrix" / key.task.value / key.format.value / "c"
    cloud_to_cloud = write(
        loaded_remote,
        remote_c,
        output_format=key.format,
        output_variant=key.variant,
        verbose=True,
    )
    assert cloud_to_cloud
    loaded_cloud_to_cloud = load(
        _reload_root(remote_c, key),
        dataset_format=key.format,
        task=key.task,
        registry_variant=key.variant,
    )
    assert _content_count(loaded_cloud_to_cloud) > 0
    _assert_dataset_parity(loaded_local, loaded_cloud_to_cloud, key.task)


def test_failed_remote_replace_restores_previous_destination(memory_root, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import datamaite.writers as writer_module
    from datamaite.model import BoxTrackDataset
    from datamaite.writers import Writer

    dest = memory_root / "rollback" / "dataset"
    (dest / "old.txt").parent.mkdir(parents=True)
    (dest / "old.txt").write_text("old", encoding="utf-8")

    class TwoFileWriter(Writer):
        format = DatasetFormat.HMIE

        def write(self, _dataset, output, **_options):  # type: ignore[no-untyped-def]
            first = output / "first.txt"
            second = output / "second.txt"
            first.parent.mkdir(parents=True, exist_ok=True)
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")
            return [first, second]

    original_copy = writer_module.copy_resource
    promotions = 0

    def fail_second_promotion(source, target, **options):  # type: ignore[no-untyped-def]
        nonlocal promotions
        if ".datamaite-stage-" in str(source) and str(target).startswith(str(dest)):
            promotions += 1
            if promotions == 2:
                raise OSError("injected promotion failure")
        return original_copy(source, target, **options)

    monkeypatch.setattr(writer_module, "copy_resource", fail_second_promotion)
    with pytest.raises(OSError, match="injected"):
        writer_module._write_remote_staged_replace(
            TwoFileWriter(), BoxTrackDataset(sequences=(), categories={}), dest, writer_options={}
        )

    assert (dest / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (dest / "first.txt").exists()
    assert not any("datamaite-" in path.name for path in dest.parent.iterdir())


def test_remote_tree_removal_does_not_delete_post_inventory_objects(memory_root, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from datamaite._io import remove_tree

    root = memory_root / "remove-race"
    old = root / "old.txt"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text("old", encoding="utf-8")
    filesystem = root.fs
    original_rm = filesystem.rm

    def create_after_snapshot(paths, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = original_rm(paths, *args, **kwargs)
        if isinstance(paths, list):
            concurrent = root / "concurrent.txt"
            concurrent.write_text("concurrent", encoding="utf-8")
        return result

    monkeypatch.setattr(filesystem, "rm", create_after_snapshot)
    remove_tree(root)

    assert not old.exists()
    assert (root / "concurrent.txt").read_text(encoding="utf-8") == "concurrent"


def test_failed_remote_clear_restores_previous_destination(memory_root, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import datamaite.writers as writer_module
    from datamaite.model import BoxTrackDataset
    from datamaite.writers import Writer

    dest = memory_root / "rollback-clear" / "dataset"
    for name in ("one.txt", "two.txt"):
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")

    class OneFileWriter(Writer):
        format = DatasetFormat.HMIE

        def write(self, _dataset, output, **_options):  # type: ignore[no-untyped-def]
            path = output / "new.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("new", encoding="utf-8")
            return [path]

    original_prepare = writer_module._prepare_destination

    def partial_clear(path, mode):  # type: ignore[no-untyped-def]
        if path == dest:
            (dest / "one.txt").unlink()
            raise OSError("injected clear failure")
        return original_prepare(path, mode)

    monkeypatch.setattr(writer_module, "_prepare_destination", partial_clear)
    with pytest.raises(OSError, match="injected"):
        writer_module._write_remote_staged_replace(
            OneFileWriter(), BoxTrackDataset(sequences=(), categories={}), dest, writer_options={}
        )

    assert (dest / "one.txt").read_text(encoding="utf-8") == "one.txt"
    assert (dest / "two.txt").read_text(encoding="utf-8") == "two.txt"
    assert not (dest / "new.txt").exists()


@pytest.mark.parametrize("key", AUTODETECT_KEYS, ids=lambda key: f"{key.task.value}-{key.format.value}")
def test_remote_autodetection(key, tmp_path: Path, memory_root) -> None:  # type: ignore[no-untyped-def]
    dataset = _dataset_for(key.task, tmp_path / "autodetect-source")
    root = memory_root / "autodetect" / key.task.value / key.format.value
    write(dataset, root, output_format=key.format, output_variant=key.variant)

    detected = load(_reload_root(root, key), dataset_format=None, task=key.task)

    assert _content_count(detected) > 0
    if key.format is DatasetFormat.YOLO:
        wrong_task = Task.OD if key.task is Task.IC else Task.IC
        with pytest.raises(ValueError, match="Could not autodetect"):
            load(_reload_root(root, key), dataset_format=None, task=wrong_task)


@pytest.mark.parametrize("key", WRITER_KEYS, ids=lambda key: f"{key.task.value}-{key.format.value}")
def test_remote_destination_modes_for_every_writer(key, tmp_path: Path, memory_root) -> None:  # type: ignore[no-untyped-def]
    dataset = _dataset_for(key.task, tmp_path / "mode-source")
    dest = memory_root / "destination-modes" / key.task.value / key.format.value

    write(dataset, dest, output_format=key.format, output_variant=key.variant)
    with pytest.raises(FileExistsError):
        write(dataset, dest, output_format=key.format, output_variant=key.variant, mode="error")

    stale = dest / "stale.txt"
    stale.write_text("stale", encoding="utf-8")
    write(dataset, dest, output_format=key.format, output_variant=key.variant, mode="append")
    assert stale.exists()

    write(dataset, dest, output_format=key.format, output_variant=key.variant, mode="replace")
    assert not stale.exists()
