#!/usr/bin/env python
"""Manual S3-compatible/real-S3 benchmarks for datamaite object-storage I/O.

The measured sections exclude fixture generation and cleanup. Results emphasize
S3 operation counts and body bytes; localhost object-store wall time is comparative,
not representative of a managed cloud.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import threading
import time
import urllib.parse
import uuid
from collections.abc import Callable, Iterable
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any

import cv2  # type: ignore[import-untyped]
import fsspec
import numpy as np
import psutil
from metrics import S3CallMeter

from datamaite import (
    ClassificationLabel,
    DatasetMetadata,
    ImageClassificationDataset,
    ImageClassificationSample,
    load,
    write,
)
from datamaite.model import BoxAnnotation, BoxTrackDataset, VideoSequence
from datamaite.taxonomy import CategoryEntry, Taxonomy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.environ.get("OBJECT_STORE_BENCH_ROOT"))
    parser.add_argument("--objects", type=int, default=250)
    parser.add_argument("--object-bytes", type=int, default=64 * 1024)
    parser.add_argument("--video-frames", type=int, default=900)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=("discovery", "transfer", "video"),
        help="Run only selected scenario(s); default runs all three.",
    )
    parser.add_argument("--create-bucket", action="store_true")
    parser.add_argument("--keep", action="store_true", help="Retain generated objects for inspection.")
    parser.add_argument("--output", type=Path)
    return parser


def _storage_options() -> dict[str, Any]:
    endpoint = os.environ.get("DATAMAITE_S3_E2E_ENDPOINT")
    key = os.environ.get("DATAMAITE_S3_E2E_KEY")
    secret = os.environ.get("DATAMAITE_S3_E2E_SECRET")
    options: dict[str, Any] = {}
    if endpoint:
        options["client_kwargs"] = {"endpoint_url": endpoint}
    if key or secret:
        if not key or not secret:
            raise ValueError("DATAMAITE_S3_E2E_KEY and DATAMAITE_S3_E2E_SECRET must be set together")
        options.update(key=key, secret=secret)
    return options


def _measure(name: str, action: Callable[[], Any], **parameters: Any) -> tuple[dict[str, Any], Any]:
    gc.collect()
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sample_rss() -> None:
        while not stop.wait(0.01):
            peak[0] = max(peak[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    with S3CallMeter() as meter:
        started = time.perf_counter()
        sampler.start()
        try:
            value = action()
        finally:
            peak[0] = max(peak[0], process.memory_info().rss)
            stop.set()
            sampler.join()
        elapsed = time.perf_counter() - started
    result = {
        "scenario": name,
        **parameters,
        "wall_seconds": round(elapsed, 4),
        "rss_baseline_mb": round(baseline / (1024 * 1024), 1),
        "rss_peak_mb": round(peak[0] / (1024 * 1024), 1),
        "rss_growth_mb": round((peak[0] - baseline) / (1024 * 1024), 1),
        "versions": {"s3fs": version("s3fs"), "aiobotocore": version("aiobotocore")},
        **meter.as_dict(),
    }
    return result, value


def _uri(path: str) -> str:
    return f"s3://{path.strip('/')}"


def _join(root: str, *parts: str) -> str:
    return str(PurePosixPath(root, *parts))


def _sample_count(dataset: Any) -> int:
    for name in ("sample_count", "sequence_count"):
        value = getattr(dataset, name, None)
        if value is not None:
            return int(value)
    samples = getattr(dataset, "samples", None)
    if samples is not None:
        return len(samples)
    return len(dataset)


def _jpeg_bytes(target_size: int = 0) -> bytes:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :, 1] = np.arange(32, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("OpenCV could not generate the benchmark JPEG")
    payload = encoded.tobytes()
    return payload + b"\0" * max(0, target_size - len(payload))


def _pipe(fs: Any, objects: dict[str, bytes], *, batch_size: int = 500) -> None:
    items = list(objects.items())
    for start in range(0, len(items), batch_size):
        fs.pipe(dict(items[start : start + batch_size]))


def _cold_cache(fs: Any) -> None:
    fs.invalidate_cache()
    dircache = getattr(fs, "dircache", None)
    if dircache is not None:
        dircache.clear()


def _remove_prefix(fs: Any, prefix: str) -> None:
    # Some S3-compatible servers reject modern DeleteObjects checksum
    # negotiation. Individual deletes are portable
    # and cleanup is outside every measured section.
    for path in fs.find(prefix):
        fs.rm_file(path)


def _seed_discovery(fs: Any, root: str, count: int) -> list[tuple[str, str, str, str]]:
    jpeg = _jpeg_bytes()
    yolo = _join(root, "yolo")
    hf_video = _join(root, "hf-video")
    coco = _join(root, "coco")

    objects: dict[str, bytes] = {}
    for index in range(count):
        objects[_join(yolo, "train", f"class_{index % 10:02d}", f"image_{index:06d}.jpg")] = jpeg
        objects[_join(hf_video, "train", f"class_{index % 10:02d}", f"video_{index:06d}.mp4")] = (
            b"benchmark-video-placeholder"
        )
        objects[_join(coco, "images", f"image_{index:06d}.jpg")] = jpeg
    coco_document = {
        "categories": [{"id": 1, "name": "object"}],
        "images": [
            {
                "id": index,
                "file_name": f"images/image_{index:06d}.jpg",
                "width": 32,
                "height": 32,
            }
            for index in range(count)
        ],
        "annotations": [],
    }
    objects[_join(coco, "annotations", "instances.json")] = json.dumps(coco_document).encode()
    _pipe(fs, objects)
    return [
        ("yolo_ic", "yolo", yolo, "ic"),
        ("huggingface_video", "huggingface_video_classification", hf_video, "vc"),
        ("coco", "coco", coco, "od"),
    ]


def benchmark_discovery(fs: Any, root: str, options: dict[str, Any], count: int) -> list[dict[str, Any]]:
    layouts = _seed_discovery(fs, _join(root, "discovery"), count)
    results: list[dict[str, Any]] = []
    for label, dataset_format, path, task in layouts:
        _cold_cache(fs)
        result, dataset = _measure(
            f"discovery/{label}",
            lambda path=path, dataset_format=dataset_format, task=task: load(
                _uri(path), dataset_format=dataset_format, task=task, storage_options=options
            ),
            objects=count,
        )
        actual = _sample_count(dataset)
        if actual != count:
            raise RuntimeError(f"{label}: discovered {actual} objects, expected {count}")
        results.append(result)
    return results


def _classification_dataset(count: int, object_bytes: int) -> ImageClassificationDataset:
    payload = _jpeg_bytes(object_bytes)
    taxonomy = Taxonomy(entries=(CategoryEntry(source_id=0, name="class_00"),), id_density="dense")
    label = ClassificationLabel(category_id=0, source_category_id=0, category_name="class_00")
    return ImageClassificationDataset(
        samples=tuple(
            ImageClassificationSample(
                image_id=f"image-{index}",
                image_bytes=payload,
                file_name=f"image_{index:06d}.jpg",
                split="train",
                labels=(label,),
            )
            for index in range(count)
        ),
        dataset_metadata=DatasetMetadata(taxonomy=taxonomy, splits=("train",)),
    )


def _load_ic(path: str | Path, options: dict[str, Any] | None = None) -> Any:
    return load(path, dataset_format="yolo", task="ic", storage_options=options)


def benchmark_transfer(
    fs: Any, root: str, options: dict[str, Any], count: int, object_bytes: int
) -> list[dict[str, Any]]:
    dataset = _classification_dataset(count, object_bytes)
    source = _join(root, "transfer", "source")
    write(dataset, _uri(source), output_format="yolo", storage_options=options)
    remote_dataset = _load_ic(_uri(source), options)
    results: list[dict[str, Any]] = []

    def remote_write(path: str, mode: str, source_dataset: Any = remote_dataset) -> list[Any]:
        return write(source_dataset, _uri(path), output_format="yolo", mode=mode, storage_options=options)

    local_to_s3 = _join(root, "transfer", "local-to-s3")
    _cold_cache(fs)
    result, _ = _measure(
        "transfer/local_to_s3",
        lambda: write(dataset, _uri(local_to_s3), output_format="yolo", storage_options=options),
        objects=count,
        object_bytes=object_bytes,
    )
    results.append(result)

    with tempfile.TemporaryDirectory() as temporary:
        local = Path(temporary) / "s3-to-local"
        _cold_cache(fs)
        result, _ = _measure(
            "transfer/s3_to_local",
            lambda: write(remote_dataset, local, output_format="yolo"),
            objects=count,
            object_bytes=object_bytes,
        )
        if _sample_count(_load_ic(local)) != count:
            raise RuntimeError("S3-to-local result did not round-trip")
        results.append(result)

    for mode in ("error", "append", "replace"):
        destination = _join(root, "transfer", f"s3-to-s3-{mode}")
        if mode != "error":
            remote_write(destination, "error", dataset)
        _cold_cache(fs)
        result, _ = _measure(
            f"transfer/s3_to_s3/{mode}",
            lambda destination=destination, mode=mode: remote_write(destination, mode),
            objects=count,
            object_bytes=object_bytes,
        )
        expected = count * 2 if mode == "append" else count
        if _sample_count(_load_ic(_uri(destination), options)) != expected:
            raise RuntimeError(f"S3-to-S3 {mode} result did not round-trip")
        results.append(result)

    if _sample_count(_load_ic(_uri(local_to_s3), options)) != count:
        raise RuntimeError("local-to-S3 result did not round-trip")
    return results


def _make_video(path: Path, frame_count: int) -> None:
    import av

    width, height = 320, 240
    container = av.open(str(path), "w", options={"movflags": "+faststart"})
    stream = container.add_stream("mpeg4", rate=30)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    rng = np.random.default_rng(42)
    try:
        for index in range(frame_count):
            pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            pixels[:, :, 0] = np.roll(np.arange(width, dtype=np.uint8), index)
            frame = av.VideoFrame.from_ndarray(pixels, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("PyAV could not generate the benchmark video")


def _video_dataset(
    video_uri: str, options: dict[str, Any], frame_count: int, selected: Iterable[int], size_bytes: int
) -> BoxTrackDataset:
    boxes = tuple(
        BoxAnnotation(
            track_uuid="benchmark-track",
            track_id=0,
            category_id=1,
            category_uri="benchmark/object",
            category_name="object",
            bbox=(1, 2, 10, 20),
            attributes={},
            frame_index=index,
            timestamp=None,
            keyframe_type="start",
            is_inferred=False,
        )
        for index in selected
    )
    sequence = VideoSequence(
        video_id="benchmark-video",
        video_path=video_uri,
        fps=30.0,
        num_frames=frame_count,
        duration=frame_count / 30.0,
        annotation_path="benchmark.json",
        width=320,
        height=240,
        size_bytes=size_bytes,
        boxes=boxes,
        num_frames_exact=True,
    )
    return BoxTrackDataset(
        sequences=(sequence,), categories={"benchmark/object": 1}, _storage_options=options
    ).with_mot_options(empty_frame_policy="annotated")


def benchmark_sparse_video(fs: Any, root: str, options: dict[str, Any], frame_count: int) -> list[dict[str, Any]]:
    video_key = _join(root, "video", "video.mp4")
    with tempfile.TemporaryDirectory() as temporary:
        local_video = Path(temporary) / "video.mp4"
        _make_video(local_video, frame_count)
        fs.put_file(str(local_video), video_key)
        video_size = local_video.stat().st_size

    selections = {
        "first": (0,),
        "middle": (frame_count // 2,),
        "last": (frame_count - 1,),
        "every_100": tuple(range(0, frame_count, 100)),
    }
    profiles = {
        "datamaite_8mb": options,
        "1mb_blocks": {**options, "block_size": 1 << 20},
    }
    results: list[dict[str, Any]] = []
    for profile, profile_options in profiles.items():
        for label, selected in selections.items():
            _cold_cache(fs)
            dataset = _video_dataset(_uri(video_key), profile_options, frame_count, selected, video_size)

            def consume(dataset: BoxTrackDataset = dataset) -> int:
                stream, _target, _metadata = dataset[0]
                consumed = 0
                for frame in stream:
                    _ = frame.pixels.shape
                    consumed += 1
                return consumed

            result, consumed = _measure(
                f"video/{profile}/{label}",
                consume,
                frames=frame_count,
                selected_frames=len(selected),
                video_bytes=video_size,
            )
            if consumed != len(selected):
                raise RuntimeError(
                    f"video/{profile}/{label}: decoded {consumed} selected frames, expected {len(selected)}"
                )
            results.append(result)
    return results


def _print_results(results: list[dict[str, Any]]) -> None:
    print(f"{'scenario':34} {'wall':>8} {'GET':>6} {'HEAD':>6} {'LIST':>6} {'COPY':>6} {'down':>11} {'up':>11}")
    print("-" * 98)
    for result in results:
        operations = result["operations"]
        lists = operations.get("list_objects_v2", 0) + operations.get("list_objects", 0)
        print(
            f"{result['scenario'][:34]:34} {result['wall_seconds']:8.3f} "
            f"{operations.get('get_object', 0):6d} {operations.get('head_object', 0):6d} "
            f"{lists:6d} {operations.get('copy_object', 0):6d} "
            f"{result['download_bytes']:11,d} {result['upload_bytes']:11,d}"
        )


def main() -> None:
    args = _parser().parse_args()
    split = urllib.parse.urlsplit(args.root or "")
    if (
        split.scheme != "s3"
        or not split.netloc
        or split.username is not None
        or split.password is not None
        or split.query
        or split.fragment
    ):
        raise SystemExit(
            "--root (or OBJECT_STORE_BENCH_ROOT) must be a disposable s3:// bucket/prefix without credentials, "
            "query parameters, or fragments"
        )
    if args.objects < 1 or args.object_bytes < 1 or args.video_frames < 2:
        raise SystemExit("object and frame counts must be positive")

    options = _storage_options()
    fs, configured_path = fsspec.core.url_to_fs(args.root, **options)
    configured_path = configured_path.rstrip("/")
    bucket = configured_path.split("/", 1)[0]
    if not fs.exists(bucket):
        if not args.create_bucket:
            raise SystemExit(f"bucket {bucket!r} does not exist; create it or pass --create-bucket")
        fs.mkdir(bucket)

    run_root = _join(configured_path, f"run-{uuid.uuid4().hex[:10]}")
    scenarios = args.scenario or ["discovery", "transfer", "video"]
    results: list[dict[str, Any]] = []
    try:
        if "discovery" in scenarios:
            results.extend(benchmark_discovery(fs, run_root, options, args.objects))
        if "transfer" in scenarios:
            results.extend(benchmark_transfer(fs, run_root, options, args.objects, args.object_bytes))
        if "video" in scenarios:
            results.extend(benchmark_sparse_video(fs, run_root, options, args.video_frames))
    finally:
        if not args.keep:
            _remove_prefix(fs, run_root)

    _print_results(results)
    output = args.output or Path(__file__).with_name(f"results_{int(time.time())}.json")
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nJSON: {output}")


if __name__ == "__main__":
    main()
