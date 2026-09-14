"""YOLO image-classification reader/writer tests."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from pathlib import Path

import numpy as np
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

    def test_split_option_loads_only_selected_split(self, tmp_path: Path) -> None:
        # (#86) Parity with the OD loader's split option.
        _dataset(tmp_path)

        ds = load_ic(tmp_path, dataset_format="yolo", split="train")

        assert ds.sample_count == 2
        assert all(sample.split == "train" for sample in ds.samples)
        assert ds.dataset_metadata.splits == ("train",)
        assert ds.index2label() == {0: "cat", 1: "dog"}

    def test_split_option_normalizes_aliases(self, tmp_path: Path) -> None:
        _dataset(tmp_path)

        ds = load_ic(tmp_path, dataset_format="yolo", split="validation")

        assert ds.sample_count == 1
        assert ds.samples[0].split == "val"
        # Taxonomy is split-local: val/ declares only cat, so dog is absent —
        # exactly as if val/ had been loaded as its own root.
        assert ds.index2label() == {0: "cat"}

    def test_split_option_includes_empty_class_dirs_of_selected_split(self, tmp_path: Path) -> None:
        # (#81 interplay) An empty class dir in the selected split still enters
        # the taxonomy so dense label indices stay stable.
        _dataset(tmp_path)
        (tmp_path / "train" / "zebra").mkdir()

        ds = load_ic(tmp_path, dataset_format="yolo", split="train")

        assert ds.index2label() == {0: "cat", 1: "dog", 2: "zebra"}
        assert ds.sample_count == 2

    def test_split_option_unknown_selection_selects_nothing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _dataset(tmp_path)

        with caplog.at_level(logging.WARNING):
            ds = load_ic(tmp_path, dataset_format="yolo", split="bogus")

        assert ds.sample_count == 0
        assert any("ignoring unrecognized split" in record.message for record in caplog.records)

    def test_split_option_on_flat_layout_selects_nothing(self, tmp_path: Path) -> None:
        # Flat (split-less) class dirs under an explicit selection must not
        # silently widen back to the whole dataset.
        _write_image(tmp_path / "cat" / "a.jpg", b"cat-a")
        _write_image(tmp_path / "dog" / "b.jpg", b"dog-b")

        assert load_ic(tmp_path, dataset_format="yolo").sample_count == 2
        assert load_ic(tmp_path, dataset_format="yolo", split="train").sample_count == 0

    def test_memory_root_loads_taxonomy_and_decodes_lazily(self, memory_root) -> None:  # type: ignore[no-untyped-def]
        """A second format uses the same backend-neutral path/decode seam."""
        cv2 = pytest.importorskip("cv2")
        source = np.arange(4 * 7 * 3, dtype=np.uint8).reshape(4, 7, 3)
        ok, buf = cv2.imencode(".png", source)
        assert ok

        root = memory_root / "yolo-ic"
        image_path = root / "train" / "cat" / "remote.png"
        image_path.parent.mkdir(parents=True)
        image_path.write_bytes(buf.tobytes())

        ds = load_ic(
            str(root),
            dataset_format="yolo",
            storage_options={"poc_marker": "must-not-appear-in-repr"},
        )

        assert ds.sample_count == 1
        assert ds.index2label() == {0: "cat"}
        assert ds.dataset_metadata.splits == ("train",)
        assert isinstance(ds.samples[0].path_or_uri, str)
        assert ds.samples[0].path_or_uri.startswith("memory://")
        assert "must-not-appear-in-repr" not in repr(ds)

        image, target, metadata = ds[0]
        np.testing.assert_array_equal(np.transpose(image, (1, 2, 0)), source[:, :, ::-1])
        np.testing.assert_array_equal(target, np.array([1.0], dtype=np.float32))
        assert metadata["split"] == "train"

    def test_memory_traversal_uses_detailed_listings_not_per_entry_info(self, memory_root, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        root = memory_root / "yolo-listing-count"
        for index in range(20):
            image_path = root / "train" / "cat" / f"roll-{index}" / "remote.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"encoded later")

        info_calls = 0
        original_info = root.fs.info

        def counted_info(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal info_calls
            info_calls += 1
            return original_info(*args, **kwargs)

        monkeypatch.setattr(root.fs, "info", counted_info)
        ds = load_ic(str(root), dataset_format="yolo")

        assert ds.sample_count == 20
        assert info_calls <= 5

    def test_local_partially_empty_taxonomy_round_trip(self, tmp_path: Path) -> None:
        taxonomy = Taxonomy(
            entries=(CategoryEntry(source_id=0, name="cat"), CategoryEntry(source_id=1, name="dog")),
            id_density="dense",
        )
        dataset = ImageClassificationDataset(
            samples=(
                ImageClassificationSample(
                    image_id="cat",
                    image_bytes=b"image",
                    file_name="cat.jpg",
                    split="train",
                    labels=(ClassificationLabel(category_id=0, source_category_id=0, category_name="cat"),),
                ),
            ),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy, splits=("train",)),
        )

        write(dataset, tmp_path, output_format="yolo")
        restored = load_ic(tmp_path, dataset_format="yolo")

        assert restored.index2label() == {0: "cat", 1: "dog"}

    def test_local_all_empty_classes_round_trip_taxonomy(self, tmp_path: Path) -> None:
        taxonomy = Taxonomy(
            entries=(CategoryEntry(source_id=0, name="cat"), CategoryEntry(source_id=1, name="dog")),
            id_density="dense",
        )
        dataset = ImageClassificationDataset(
            samples=(),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy, splits=("train",)),
        )

        write(dataset, tmp_path, output_format="yolo")
        restored = load_ic(tmp_path, dataset_format="yolo")

        assert restored.index2label() == {0: "cat", 1: "dog"}
        assert restored.dataset_metadata.splits == ("train",)

    def test_remote_all_empty_classes_round_trip_taxonomy(self, memory_root) -> None:  # type: ignore[no-untyped-def]
        taxonomy = Taxonomy(
            entries=(CategoryEntry(source_id=0, name="cat"), CategoryEntry(source_id=1, name="dog")),
            id_density="dense",
        )
        dataset = ImageClassificationDataset(
            samples=(),
            dataset_metadata=DatasetMetadata(taxonomy=taxonomy, splits=("train",)),
        )
        root = memory_root / "empty-yolo-roundtrip"

        write(dataset, root, output_format="yolo")
        restored = load_ic(root, dataset_format="yolo")

        assert restored.sample_count == 0
        assert restored.index2label() == {0: "cat", 1: "dog"}
        assert restored.dataset_metadata.splits == ("train",)

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

    def test_nested_images_are_discovered_recursively(self, tmp_path: Path) -> None:
        # (#90) Images anywhere below a class dir belong to that class: the
        # top-level directory stays the label, the nested relative path stays
        # in the datum ID, and nested dirs never become classes of their own.
        _write_image(tmp_path / "train" / "cat" / "a.jpg", b"cat-a")
        _write_image(tmp_path / "train" / "cat" / "nested" / "deep.jpg", b"deep")
        _write_image(tmp_path / "train" / "cat" / "nested" / "deeper" / "deepest.jpg", b"deepest")
        _write_image(tmp_path / "train" / "dog" / "b.jpg", b"dog-b")

        ds = load_ic(tmp_path, dataset_format="yolo")

        assert ds.index2label() == {0: "cat", 1: "dog"}
        assert [(sample.image_id, sample.labels[0].category_name) for sample in ds.samples] == [
            ("train/cat/a.jpg", "cat"),
            ("train/cat/nested/deep.jpg", "cat"),
            ("train/cat/nested/deeper/deepest.jpg", "cat"),
            ("train/dog/b.jpg", "dog"),
        ]

    def test_nested_only_split_is_not_empty(self, tmp_path: Path) -> None:
        # (#90) The CheckMAITE regression scenario: a split whose class dirs
        # hold only nested images used to load as an empty dataset.
        _write_image(tmp_path / "train" / "cat" / "roll-01" / "a.jpg", b"cat-a")
        _write_image(tmp_path / "train" / "dog" / "roll-02" / "b.jpg", b"dog-b")

        ds = load_ic(tmp_path, dataset_format="yolo")

        assert ds.sample_count == 2
        assert ds.index2label() == {0: "cat", 1: "dog"}
        assert [sample.file_name for sample in ds.samples] == [
            "train/cat/roll-01/a.jpg",
            "train/dog/roll-02/b.jpg",
        ]

    def test_nested_non_images_and_hidden_entries_are_ignored(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "train" / "cat" / "nested" / "a.jpg", b"cat-a")
        (tmp_path / "train" / "cat" / "nested" / "notes.txt").write_text("not an image", encoding="utf-8")
        _write_image(tmp_path / "train" / "cat" / "nested" / ".hidden.jpg", b"hidden")
        _write_image(tmp_path / "train" / "cat" / ".thumbnails" / "b.jpg", b"thumb")

        ds = load_ic(tmp_path, dataset_format="yolo")

        assert [sample.file_name for sample in ds.samples] == ["train/cat/nested/a.jpg"]

    def test_nested_empty_dirs_add_no_classes_or_samples(self, tmp_path: Path) -> None:
        # (#81 interplay) Only direct children of the split declare classes;
        # empty nested dirs neither extend the taxonomy nor break discovery.
        _write_image(tmp_path / "train" / "cat" / "a.jpg", b"cat-a")
        (tmp_path / "train" / "cat" / "empty" / "deeper").mkdir(parents=True)

        ds = load_ic(tmp_path, dataset_format="yolo")

        assert ds.index2label() == {0: "cat"}
        assert [sample.file_name for sample in ds.samples] == ["train/cat/a.jpg"]

    def test_split_option_selects_nested_images(self, tmp_path: Path) -> None:
        # (#86 interplay) Nested discovery composes with the split option.
        _write_image(tmp_path / "train" / "cat" / "roll" / "a.jpg", b"cat-a")
        _write_image(tmp_path / "val" / "cat" / "roll" / "b.jpg", b"cat-b")

        ds = load_ic(tmp_path, dataset_format="yolo", split="val")

        assert [sample.file_name for sample in ds.samples] == ["val/cat/roll/b.jpg"]

    def test_nested_only_root_loads_but_does_not_sniff(self, tmp_path: Path) -> None:
        # (#90) Deliberate load/sniff asymmetry: recursive discovery serves an
        # explicit dataset_format="yolo", while sniff stays shallow so nested
        # trees (e.g. MOT-style video frames) don't ambiguously autodetect.
        _write_image(tmp_path / "train" / "cat" / "roll" / "a.jpg", b"cat-a")

        assert not YoloImageClassificationLoader.sniff(tmp_path)
        assert load_ic(tmp_path, dataset_format="yolo").sample_count == 1

    def test_nested_symlinked_directory_is_not_descended(self, tmp_path: Path) -> None:
        # A symlinked dir below a class dir is not traversed (matching rglob's
        # ** semantics): it can neither smuggle an outside tree into the class
        # nor recurse forever via a link cycle.
        outside = tmp_path / "outside"
        _write_image(outside / "secret.jpg", b"secret")
        root = tmp_path / "dataset"
        _write_image(root / "train" / "cat" / "real.jpg", b"cat-real")
        os.symlink(outside, root / "train" / "cat" / "evil", target_is_directory=True)
        os.symlink(root / "train", root / "train" / "cat" / "cycle", target_is_directory=True)

        ds = load_ic(root, dataset_format="yolo")

        assert [sample.file_name for sample in ds.samples] == ["train/cat/real.jpg"]

    def test_nested_symlinked_image_escaping_root_is_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The escape guard applies at every depth, not just to direct children.
        secret = tmp_path / "outside" / "secret.bin"
        _write_image(secret, b"top secret")
        root = tmp_path / "dataset"
        _write_image(root / "train" / "cat" / "nested" / "real.jpg", b"cat-real")
        os.symlink(secret, root / "train" / "cat" / "nested" / "evil.jpg")

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.yolo"):
            ds = load_ic(root, dataset_format="yolo")

        assert [sample.file_name for sample in ds.samples] == ["train/cat/nested/real.jpg"]
        assert "escaping the dataset root" in caplog.text

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
        assert "escaping the dataset root" in caplog.text

    def test_in_root_symlink_is_loaded(self, tmp_path: Path) -> None:
        # The containment guard must not over-reach: a symlink resolving to a
        # file *inside* the dataset root is legitimate and still loads.
        _write_image(tmp_path / "train" / "cat" / "real.jpg", b"cat-real")
        os.symlink(tmp_path / "train" / "cat" / "real.jpg", tmp_path / "train" / "cat" / "alias.jpg")

        ds = load_ic(tmp_path, dataset_format="yolo")

        assert [sample.file_name for sample in ds.samples] == ["train/cat/alias.jpg", "train/cat/real.jpg"]

    def test_symlinked_class_dir_is_not_descended(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        # (!83 review P1) A class directory that is itself a symlink pointing
        # outside the root previously slipped past the escape guard: its child
        # images are not symlinks, so the per-file check never fired. Split and
        # class dir symlinks now follow the same no-descend policy as nested
        # ones: skipped entirely, no taxonomy entry.
        outside = tmp_path / "outside"
        _write_image(outside / "secret.jpg", b"secret")
        root = tmp_path / "dataset"
        _write_image(root / "train" / "dog" / "real.jpg", b"dog-real")
        os.symlink(outside, root / "train" / "cat", target_is_directory=True)

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.yolo"):
            ds = load_ic(root, dataset_format="yolo")

        assert [sample.file_name for sample in ds.samples] == ["train/dog/real.jpg"]
        assert ds.index2label() == {0: "dog"}
        assert "symlinked directory" in caplog.text

    def test_symlinked_only_class_dir_loads_nothing_and_does_not_sniff(self, tmp_path: Path) -> None:
        # The exact !83 review repro: dataset/train/cat -> /outside with
        # /outside/secret.jpg used to load the outside image.
        outside = tmp_path / "outside"
        _write_image(outside / "secret.jpg", b"secret")
        root = tmp_path / "dataset"
        (root / "train").mkdir(parents=True)
        os.symlink(outside, root / "train" / "cat", target_is_directory=True)

        assert not YoloImageClassificationLoader.sniff(root)
        assert load_ic(root, dataset_format="yolo").sample_count == 0

    def test_symlinked_split_dir_is_not_descended(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        # A whole split symlinked to an outside tree is skipped, not followed.
        outside = tmp_path / "outside"
        _write_image(outside / "cat" / "secret.jpg", b"secret")
        root = tmp_path / "dataset"
        _write_image(root / "val" / "cat" / "real.jpg", b"cat-real")
        os.symlink(outside, root / "train", target_is_directory=True)

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.yolo"):
            ds = load_ic(root, dataset_format="yolo")

        assert [sample.file_name for sample in ds.samples] == ["val/cat/real.jpg"]
        assert ds.dataset_metadata.splits == ("val",)
        assert "symlinked directory" in caplog.text

    def test_ambiguous_nested_split_named_class_warns_and_flat_layout_resolves(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # (!83 review P1) A flat root whose "train" class nests all its images
        # is structurally identical to a split layout, so auto reads it as one
        # -- silently dropping car/ before this fix. Auto now warns, naming the
        # dropped directories and the layout="flat" override that resolves it.
        _write_image(tmp_path / "train" / "roll-01" / "a.jpg", b"a")
        _write_image(tmp_path / "car" / "b.jpg", b"b")

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.yolo"):
            auto = load_ic(tmp_path, dataset_format="yolo")

        assert auto.index2label() == {0: "roll-01"}
        assert "car" in caplog.text
        assert 'layout="flat"' in caplog.text

        flat = load_ic(tmp_path, dataset_format="yolo", layout="flat")

        assert flat.sample_count == 2
        assert flat.index2label() == {0: "car", 1: "train"}
        assert [(sample.file_name, sample.labels[0].category_name) for sample in flat.samples] == [
            ("car/b.jpg", "car"),
            ("train/roll-01/a.jpg", "train"),
        ]
        assert all(sample.split is None for sample in flat.samples)

    def test_layout_flat_treats_split_named_dirs_as_classes(self, tmp_path: Path) -> None:
        _dataset(tmp_path)

        ds = load_ic(tmp_path, dataset_format="yolo", layout="flat")

        assert ds.sample_count == 3
        assert ds.index2label() == {0: "train", 1: "val"}
        assert all(sample.split is None for sample in ds.samples)

    def test_layout_split_forces_split_interpretation(self, tmp_path: Path) -> None:
        # On a proper split root, layout="split" matches auto; on a flat root
        # it finds no split-named directories and loads nothing rather than
        # silently reinterpreting.
        _dataset(tmp_path)
        assert load_ic(tmp_path, dataset_format="yolo", layout="split").sample_count == 3

        flat_root = tmp_path / "flat"
        _write_image(flat_root / "cat" / "a.jpg", b"a")
        assert load_ic(flat_root, dataset_format="yolo", layout="split").sample_count == 0

    def test_layout_rejects_unknown_value(self, tmp_path: Path) -> None:
        _dataset(tmp_path)

        with pytest.raises(ValueError, match="layout"):
            load_ic(tmp_path, dataset_format="yolo", layout="deep")

    def test_each_directory_is_listed_once_per_load(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # (!83 review P2) The layout discriminator and record building share
        # one memoized scan, so no directory tree is listed twice per load.
        import datamaite._formats.yolo.loader as yolo_loader

        _write_image(tmp_path / "train" / "cat" / "roll-01" / "a.jpg", b"a")
        _write_image(tmp_path / "train" / "dog" / "b.jpg", b"b")
        _write_image(tmp_path / "val" / "cat" / "roll-02" / "c.jpg", b"c")

        counts: Counter[Path] = Counter()
        original = yolo_loader.safe_children

        def counting(path: Path) -> list[Path]:
            counts[path] += 1
            return original(path)

        monkeypatch.setattr(yolo_loader, "safe_children", counting)

        ds = load_ic(tmp_path, dataset_format="yolo")

        assert ds.sample_count == 3
        assert counts
        assert max(counts.values()) == 1


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

    def test_missing_remote_source_is_skipped_without_stray_local_dirs(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        dest = tmp_path / "out"
        label = ClassificationLabel(category_id=0, category_name="cat")
        ds = _one_sample_dataset(
            ImageClassificationSample(
                image_id="x", path_or_uri="memory://missing/nope.jpg", file_name="x.jpg", labels=(label,)
            )
        )

        with caplog.at_level(logging.WARNING, logger="datamaite._formats.yolo"):
            files = write(ds, dest, output_format="yolo", write_data_yaml=False, verbose=True)

        assert files == []
        assert "missing image file" in caplog.text
        assert list(dest.rglob("*")) == []

    def test_missing_remote_source_does_not_remove_preexisting_append_directory(self, tmp_path: Path) -> None:
        dest = tmp_path / "out"
        class_dir = dest / "train" / "cat"
        class_dir.mkdir(parents=True)
        label = ClassificationLabel(category_id=0, category_name="cat")
        ds = _one_sample_dataset(
            ImageClassificationSample(
                image_id="x", path_or_uri="memory://missing/nope.jpg", file_name="x.jpg", labels=(label,)
            )
        )

        write(ds, dest, output_format="yolo", mode="append", write_data_yaml=False)

        assert class_dir.is_dir()

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


class TestYoloImageClassificationEmptyClasses:
    """#81: empty class directories are included in the taxonomy so dense label
    indices stay stable across splits."""

    def test_empty_class_dir_included_in_taxonomy(self, tmp_path: Path) -> None:
        # 'bird' exists as a class directory in both splits but has no images.
        _write_image(tmp_path / "train" / "cat" / "a.jpg", b"cat-a")
        _write_image(tmp_path / "train" / "dog" / "b.jpg", b"dog-b")
        (tmp_path / "train" / "bird").mkdir(parents=True)
        _write_image(tmp_path / "val" / "cat" / "c.jpg", b"cat-c")
        (tmp_path / "val" / "bird").mkdir(parents=True)

        ds = load_ic(tmp_path, dataset_format="yolo")

        # bird kept; sorted order bird/cat/dog gives stable dense indices.
        assert ds.index2label() == {0: "bird", 1: "cat", 2: "dog"}
        assert ds.dataset_metadata.taxonomy.dense_index2label() == {0: "bird", 1: "cat", 2: "dog"}
        assert ds.sample_count == 3  # empty class adds no samples

    def test_label_indices_not_shifted_by_empty_class(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "train" / "cat" / "a.jpg", b"cat-a")
        _write_image(tmp_path / "train" / "dog" / "b.jpg", b"dog-b")
        (tmp_path / "train" / "bird").mkdir(parents=True)

        ds = load_ic(tmp_path, dataset_format="yolo")

        by_name = {s.labels[0].category_name: s.labels[0].category_id for s in ds.samples}
        # index 0 is reserved for the empty 'bird'; cat/dog are not pulled down to 0/1.
        assert by_name == {"cat": 1, "dog": 2}
        assert 0 not in by_name.values()

    def test_split_with_only_empty_class_dirs_is_still_discovered(self, tmp_path: Path) -> None:
        # 'val' holds no images at all, so the structural split check does not
        # recognize it on its own; the layout is established by 'train' and every
        # split-named sibling then contributes its class dirs to the union.
        _write_image(tmp_path / "train" / "cat" / "a.jpg", b"cat-a")
        (tmp_path / "val" / "bird").mkdir(parents=True)

        ds = load_ic(tmp_path, dataset_format="yolo")

        assert ds.index2label() == {0: "bird", 1: "cat"}
        assert ds.sample_count == 1
        assert [s.labels[0].category_id for s in ds.samples] == [1]

    def test_empty_split_named_dir_without_class_dirs_adds_no_classes(self, tmp_path: Path) -> None:
        # A split dir that is entirely empty declares no classes -- it must not
        # invent one from its own name.
        _write_image(tmp_path / "train" / "cat" / "a.jpg", b"cat-a")
        (tmp_path / "test").mkdir(parents=True)

        ds = load_ic(tmp_path, dataset_format="yolo")

        assert ds.index2label() == {0: "cat"}
        assert ds.dataset_metadata.splits == ("train",)

    def test_no_empty_dirs_is_unchanged(self, tmp_path: Path) -> None:
        # Regression guard: the common no-empty-class path is unperturbed.
        _dataset(tmp_path)
        ds = load_ic(tmp_path, dataset_format="yolo")
        assert ds.index2label() == {0: "cat", 1: "dog"}
        assert ds.sample_count == 3
