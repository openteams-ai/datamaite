"""YOLO image-classification reader/writer tests."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from datamaite import (
    ClassificationLabel,
    DatasetFormat,
    DatasetMetadata,
    ImageClassificationDataset,
    ImageClassificationSample,
    Task,
    convert,
    load,
    load_ic,
    write,
)
from datamaite._formats.yolo.loader import YoloImageClassificationLoader
from datamaite._formats.yolo.writer import YoloImageClassificationWriter
from datamaite.loaders import get_loader
from datamaite.taxonomy import CategoryEntry, Taxonomy
from datamaite.writers import get_writer


def _write_image(path: Path, data: bytes = b"not really an image") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _dataset(root: Path) -> None:
    _write_image(root / "train" / "cat" / "a.jpg", b"cat-a")
    _write_image(root / "train" / "dog" / "b.jpg", b"dog-b")
    _write_image(root / "val" / "cat" / "c.jpg", b"cat-c")
    (root / "data.yaml").write_text("names: ['cat', 'dog']\n", encoding="utf-8")


class TestRegistry:
    def test_loader_and_writer_are_task_aware(self) -> None:
        loader = get_loader(DatasetFormat.YOLO, task=Task.IC, variant="default")
        writer = get_writer("yolo", task="ic", variant="default")

        assert isinstance(loader, YoloImageClassificationLoader)
        assert isinstance(writer, YoloImageClassificationWriter)
        assert loader.task is Task.IC
        assert writer.task is Task.IC


class TestYoloImageClassificationLoader:
    def test_loads_split_class_folders(self, tmp_path: Path) -> None:
        _dataset(tmp_path)

        ds = load_ic(tmp_path, dataset_format="yolo")

        assert isinstance(ds, ImageClassificationDataset)
        assert ds.task is Task.IC
        assert ds.sample_count == 3
        assert ds.index2label() == {0: "cat", 1: "dog"}
        assert ds.dataset_metadata.splits == ("train", "val")
        assert [(sample.split, sample.labels[0].category_name) for sample in ds.samples] == [
            ("train", "cat"),
            ("train", "dog"),
            ("val", "cat"),
        ]

    def test_generic_load_can_disambiguate_with_task(self, tmp_path: Path) -> None:
        _dataset(tmp_path)

        ds = load(tmp_path, dataset_format="yolo", task="ic")

        assert isinstance(ds, ImageClassificationDataset)
        assert ds.sample_count == 3

    def test_sniff_requires_shallow_class_images(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "train" / "MOT17-02" / "img1" / "000001.jpg")

        assert not YoloImageClassificationLoader.sniff(tmp_path)

        _write_image(tmp_path / "train" / "cat" / "a.jpg")
        assert YoloImageClassificationLoader.sniff(tmp_path)

    def test_load_matches_shallow_sniff_nested_images_ignored(self, tmp_path: Path) -> None:
        # An image nested below the class dir is not a class member: load must
        # agree with sniff (which only looks one level deep), not silently pull
        # it in via a recursive walk.
        _write_image(tmp_path / "train" / "cat" / "a.jpg", b"cat-a")
        _write_image(tmp_path / "train" / "cat" / "nested" / "deep.jpg", b"deep")

        ds = load_ic(tmp_path, dataset_format="yolo")

        assert ds.sample_count == 1
        assert [sample.file_name for sample in ds.samples] == ["train/cat/a.jpg"]

    def test_data_yaml_order_is_ignored_folder_names_win(self, tmp_path: Path) -> None:
        # data.yaml deliberately disagrees with the alphabetical folder order;
        # the loader derives class indices from folders, never from data.yaml.
        _write_image(tmp_path / "train" / "cat" / "a.jpg")
        _write_image(tmp_path / "train" / "dog" / "b.jpg")
        (tmp_path / "data.yaml").write_text("names: ['dog', 'cat']\n", encoding="utf-8")

        ds = load_ic(tmp_path, dataset_format="yolo")

        assert ds.index2label() == {0: "cat", 1: "dog"}

    def test_flat_layout_with_split_named_class_is_not_mistaken_for_a_split(self, tmp_path: Path) -> None:
        # Split-less layout where a class folder is legitimately named like a
        # split -- e.g. vehicle classification with a "train" class. The
        # split/class discriminator is structural (a split must hold class
        # subdirs), so "train" here is a class, not a split. Previously the
        # name-based check treated it as the sole split, found no class subdirs
        # inside it, and returned ZERO records with no error -- silent data loss.
        _write_image(tmp_path / "train" / "a.jpg", b"a")
        _write_image(tmp_path / "train" / "b.jpg", b"b")
        _write_image(tmp_path / "car" / "c.jpg", b"c")

        assert YoloImageClassificationLoader.sniff(tmp_path)
        ds = load_ic(tmp_path, dataset_format="yolo")

        assert ds.sample_count == 3
        assert ds.index2label() == {0: "car", 1: "train"}
        assert all(sample.split is None for sample in ds.samples)

    def test_symlinked_image_escaping_root_is_skipped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        secret = tmp_path / "outside" / "secret.bin"
        _write_image(secret, b"top secret")
        root = tmp_path / "dataset"
        _write_image(root / "train" / "cat" / "real.jpg", b"cat-real")
        os.symlink(secret, root / "train" / "cat" / "evil.jpg")

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.yolo"):
            ds = load_ic(root, dataset_format="yolo")

        # The symlink pointing outside the dataset root is dropped, not ingested
        # (and so never copied through on a later write).
        assert [sample.file_name for sample in ds.samples] == ["train/cat/real.jpg"]
        assert "symlinked image escaping" in caplog.text

    def test_in_root_symlink_is_loaded(self, tmp_path: Path) -> None:
        # The containment guard must not over-reach: a symlink resolving to a
        # file *inside* the dataset root is legitimate and still loads.
        _write_image(tmp_path / "train" / "cat" / "real.jpg", b"cat-real")
        os.symlink(tmp_path / "train" / "cat" / "real.jpg", tmp_path / "train" / "cat" / "alias.jpg")

        ds = load_ic(tmp_path, dataset_format="yolo")

        assert [sample.file_name for sample in ds.samples] == ["train/cat/alias.jpg", "train/cat/real.jpg"]


class TestYoloImageClassificationWriter:
    def test_write_and_reload_round_trip(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        out = tmp_path / "out"
        _dataset(src)
        ds = load_ic(src, dataset_format="yolo")

        files = write(ds, out, output_format="yolo", verbose=True)

        assert out / "train" / "cat" / "a.jpg" in files
        assert out / "train" / "dog" / "b.jpg" in files
        assert out / "val" / "cat" / "c.jpg" in files
        assert out / "data.yaml" in files
        assert (out / "train" / "cat" / "a.jpg").read_bytes() == b"cat-a"

        reloaded = load_ic(out, dataset_format="yolo")
        assert reloaded.sample_count == 3
        assert reloaded.index2label() == {0: "cat", 1: "dog"}
        assert reloaded.dataset_metadata.splits == ("train", "val")

    def test_sparse_taxonomy_unknown_id_is_skipped_not_positionally_mapped(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        taxonomy = Taxonomy(
            entries=(CategoryEntry(source_id=2, name="car"), CategoryEntry(source_id=7, name="person")),
            source_dataset="sparse",
            id_density="sparse",
        )
        ds = ImageClassificationDataset(
            samples=(
                ImageClassificationSample(
                    image_id="bad-id",
                    image_bytes=b"image bytes",
                    file_name="bad.jpg",
                    labels=(ClassificationLabel(category_id=1),),
                ),
            ),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy),
        )

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.yolo"):
            files = write(ds, tmp_path, output_format="yolo", verbose=True)

        assert files == [tmp_path / "data.yaml"]
        assert not (tmp_path / "train" / "person" / "bad.jpg").exists()
        assert "unresolved label" in caplog.text

    def test_data_yaml_names_match_on_disk_class_order_not_taxonomy_order(self, tmp_path: Path) -> None:
        # Non-alphabetical taxonomy (zebra=0, ant=1). data.yaml must list class
        # names in the same alphabetical, folder-name order the loader (and
        # Ultralytics) derive indices from -- NOT taxonomy entry order -- or an
        # external YOLO consumer reads scrambled labels.
        taxonomy = Taxonomy(
            entries=(CategoryEntry(source_id=0, name="zebra"), CategoryEntry(source_id=1, name="ant")),
            id_density="dense",
        )
        ds = ImageClassificationDataset(
            samples=(
                ImageClassificationSample(
                    image_id="z",
                    image_bytes=b"z",
                    file_name="z.jpg",
                    split="train",
                    labels=(ClassificationLabel(category_id=0, category_name="zebra"),),
                ),
                ImageClassificationSample(
                    image_id="a",
                    image_bytes=b"a",
                    file_name="a.jpg",
                    split="train",
                    labels=(ClassificationLabel(category_id=1, category_name="ant"),),
                ),
            ),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy),
        )

        write(ds, tmp_path, output_format="yolo")

        names_line = next(
            line for line in (tmp_path / "data.yaml").read_text().splitlines() if line.startswith("names:")
        )
        names = json.loads(names_line.split("names:", 1)[1].strip())
        on_disk = sorted(p.name for p in (tmp_path / "train").iterdir() if p.is_dir())
        assert names == on_disk == ["ant", "zebra"]

    def test_duplicate_file_names_get_unique_targets(self, tmp_path: Path) -> None:
        taxonomy = Taxonomy(entries=(CategoryEntry(source_id=0, name="cat"),), id_density="dense")
        label = ClassificationLabel(category_id=0, category_name="cat")
        ds = ImageClassificationDataset(
            samples=(
                ImageClassificationSample(image_id="a", image_bytes=b"a", file_name="same.jpg", labels=(label,)),
                ImageClassificationSample(image_id="b", image_bytes=b"b", file_name="same.jpg", labels=(label,)),
            ),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy),
        )

        files = write(ds, tmp_path, output_format="yolo", write_data_yaml=False, verbose=True)

        assert files == [tmp_path / "train" / "cat" / "same.jpg", tmp_path / "train" / "cat" / "same_2.jpg"]
        assert (tmp_path / "train" / "cat" / "same.jpg").read_bytes() == b"a"
        assert (tmp_path / "train" / "cat" / "same_2.jpg").read_bytes() == b"b"

    def test_convert_yolo_ic_to_yolo_ic(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        out = tmp_path / "out"
        _dataset(src)

        files = convert(
            src,
            out,
            input_format="yolo",
            output_format="yolo",
            task="ic",
            read_options={"image_extensions": ".jpg"},
            write_options={"write_data_yaml": False},
            verbose=True,
        )

        assert out / "data.yaml" not in files
        assert load_ic(out, dataset_format="yolo").sample_count == 3

    def test_round_trip_preserves_split_class_mapping(self, tmp_path: Path) -> None:
        # Counts and label set alone would not catch a writer that collapsed
        # every sample under train/; assert the full (split, class) mapping and
        # that no unexpected files are emitted.
        src = tmp_path / "src"
        out = tmp_path / "out"
        _dataset(src)
        ds = load_ic(src, dataset_format="yolo")

        write(ds, out, output_format="yolo", write_data_yaml=False)
        reloaded = load_ic(out, dataset_format="yolo")

        assert [(s.split, s.labels[0].category_name) for s in reloaded.samples] == [
            ("train", "cat"),
            ("train", "dog"),
            ("val", "cat"),
        ]
        emitted = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
        assert emitted == {"train/cat/a.jpg", "train/dog/b.jpg", "val/cat/c.jpg"}


def _one_sample_dataset(sample: ImageClassificationSample) -> ImageClassificationDataset:
    taxonomy = Taxonomy(entries=(CategoryEntry(source_id=0, name="cat"),), id_density="dense")
    return ImageClassificationDataset(samples=(sample,), dataset_metadata=DatasetMetadata(taxonomy=taxonomy))


class TestYoloImageClassificationWriterSkipPaths:
    """The writer drops unrepresentable samples with a warning and never crashes,
    and a skipped sample leaves no stray class directory behind."""

    def test_sample_with_no_labels_is_skipped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        ds = _one_sample_dataset(
            ImageClassificationSample(image_id="x", image_bytes=b"img", file_name="x.jpg", labels=())
        )

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.yolo"):
            files = write(ds, tmp_path, output_format="yolo", write_data_yaml=False, verbose=True)

        assert files == []
        assert "no labels" in caplog.text
        assert not (tmp_path / "train").exists()

    def test_region_bearing_crop_sample_is_skipped(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        # A VisDrone-derived IC crop carries a `region`; this image-copying writer
        # cannot crop, so emitting the full source image would mislabel it. It must
        # skip loudly rather than write an incorrect dataset.
        dest = tmp_path / "out"
        label = ClassificationLabel(category_id=0, category_name="cat")
        ds = _one_sample_dataset(
            ImageClassificationSample(
                image_id="crop#1",
                image_bytes=b"img",
                file_name="crop.jpg",
                labels=(label,),
                region=(10.0, 20.0, 30.0, 40.0),
            )
        )

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.yolo"):
            files = write(ds, dest, output_format="yolo", write_data_yaml=False, verbose=True)

        assert files == []
        assert "crop region" in caplog.text
        assert not (dest / "train" / "cat").exists()

    def test_unsafe_split_is_skipped_no_escape(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        dest = tmp_path / "out"
        label = ClassificationLabel(category_id=0, category_name="cat")
        ds = _one_sample_dataset(
            ImageClassificationSample(
                image_id="x", image_bytes=b"img", file_name="x.jpg", split="../escape", labels=(label,)
            )
        )

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.yolo"):
            files = write(ds, dest, output_format="yolo", write_data_yaml=False, verbose=True)

        assert files == []
        assert "unsafe split" in caplog.text
        # Nothing written, and crucially nothing escaped the destination root.
        assert not (tmp_path / "escape").exists()
        assert list(dest.rglob("*")) == []

    def test_missing_source_file_is_skipped_without_stray_dir(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        dest = tmp_path / "out"
        label = ClassificationLabel(category_id=0, category_name="cat")
        ds = _one_sample_dataset(
            ImageClassificationSample(
                image_id="x", path_or_uri=str(tmp_path / "nope.jpg"), file_name="x.jpg", labels=(label,)
            )
        )

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.yolo"):
            files = write(ds, dest, output_format="yolo", write_data_yaml=False, verbose=True)

        assert files == []
        assert "missing image file" in caplog.text
        # The class dir must not be created for a sample that was skipped.
        assert not (dest / "train" / "cat").exists()

    def test_dangling_dest_symlink_is_not_written_through(self, tmp_path: Path) -> None:
        # A pre-planted *dangling* symlink in dest reports exists()==False; the
        # old guard would write through it to the external target. _free_target
        # treats a symlink as occupied, so the real bytes land on a fresh name.
        dest = tmp_path / "out"
        outside = tmp_path / "outside" / "evil.bin"  # deliberately never created
        (dest / "train" / "cat").mkdir(parents=True)
        planted = dest / "train" / "cat" / "x.jpg"
        os.symlink(outside, planted)

        label = ClassificationLabel(category_id=0, category_name="cat")
        ds = _one_sample_dataset(
            ImageClassificationSample(image_id="x", image_bytes=b"real", file_name="x.jpg", labels=(label,))
        )

        write(ds, dest, output_format="yolo", write_data_yaml=False, mode="append", verbose=True)

        assert not outside.exists()  # nothing written through the symlink
        assert planted.is_symlink()  # the planted symlink is left untouched
        assert (dest / "train" / "cat" / "x_2.jpg").read_bytes() == b"real"


class TestConvertOptionGuards:
    """convert()'s option-merging guards reject ambiguous/duplicated kwargs."""

    def test_read_and_load_options_are_mutually_exclusive(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not both"):
            convert(
                tmp_path / "src",
                tmp_path / "out",
                input_format="yolo",
                output_format="yolo",
                task="ic",
                read_options={"image_extensions": ".jpg"},
                load_options={"image_extensions": ".png"},
            )

    def test_writer_option_supplied_twice_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="supplied twice"):
            convert(
                tmp_path / "src",
                tmp_path / "out",
                input_format="yolo",
                output_format="yolo",
                task="ic",
                write_options={"write_data_yaml": False},
                write_data_yaml=True,
            )
