"""Tests for the loader architecture: Loader contract, registry, dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from datamaite import load, load_vc
from datamaite._formats.hmie.loader import HmieLoader
from datamaite._types import DatasetFormat, Task
from datamaite.loaders import (
    Loader,
    LoaderKey,
    available_formats,
    get_loader,
    load_mot,
    register_loader,
)
from datamaite.model import BoxTrackDataset

from ._hmie_factory import SnippetSpec, VideoSpec, default_happy_dataset, single_video_dataset


class TestRegistry:
    def test_hmie_loader_is_registered(self) -> None:
        assert DatasetFormat.HMIE in available_formats()

    def test_get_loader_returns_instance(self) -> None:
        loader = get_loader(DatasetFormat.HMIE)
        assert isinstance(loader, HmieLoader)
        assert isinstance(loader, Loader)
        assert loader.format is DatasetFormat.HMIE

    def test_get_loader_accepts_string(self) -> None:
        assert isinstance(get_loader("hmie"), HmieLoader)
        assert isinstance(get_loader("HMIE"), HmieLoader)

    def test_get_loader_unknown_string_raises(self) -> None:
        with pytest.raises(ValueError, match="not a valid DatasetFormat"):
            get_loader("does-not-exist")

    def test_get_loader_no_registered_loader_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("datamaite.loaders._LOADERS", {})
        with pytest.raises(ValueError, match="No loader registered"):
            get_loader(DatasetFormat.HMIE)


class TestLoaderContract:
    def test_loader_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            Loader()  # type: ignore[abstract]

    def test_subclass_without_load_is_abstract(self) -> None:
        class Incomplete(Loader):
            format = DatasetFormat.HMIE

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_sniff_defaults_to_false(self) -> None:
        assert HmieLoader.sniff("/anything") is False

    def test_register_requires_format(self) -> None:
        class NoFormat(Loader):
            def load(self, root: str | Path, **options: object) -> BoxTrackDataset:
                raise NotImplementedError  # never called; register_loader rejects it first

        with pytest.raises(TypeError, match="must set `format`"):
            register_loader(NoFormat)


class TestRegisterAndDispatch:
    def test_register_loader_and_dispatch(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # Patch the registry to empty *first* so register_loader mutates the
        # throwaway dict, not the real global one (which monkeypatch restores).
        monkeypatch.setattr("datamaite.loaders._LOADERS", {})
        captured: dict[str, object] = {}

        class DummyLoader(Loader):
            format = DatasetFormat.HMIE

            def load(self, root: str | Path, **options: object) -> BoxTrackDataset:
                captured["root"] = str(root)
                captured["options"] = options
                return BoxTrackDataset(sequences=[], categories={})

        register_loader(DummyLoader)
        ds = load(tmp_path, dataset_format=DatasetFormat.HMIE, custom_option=7)

        assert isinstance(ds, BoxTrackDataset)
        assert captured["root"] == str(tmp_path)
        assert captured["options"] == {"custom_option": 7}

    def test_duplicate_key_registration_raises_not_silently_overwrites(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("datamaite.loaders._LOADERS", {})

        class FirstLoader(Loader):
            format = DatasetFormat.HMIE

            def load(self, _root: str | Path, **_options: object) -> BoxTrackDataset:
                return BoxTrackDataset(sequences=[], categories={})

        class SecondLoader(FirstLoader):
            pass  # same (task, format, variant) key

        register_loader(FirstLoader)
        with pytest.raises(ValueError, match="already registered"):
            register_loader(SecondLoader)
        # Re-registering the SAME class stays idempotent (e.g. a module re-import).
        assert register_loader(FirstLoader) is FirstLoader


class TestAutodetect:
    def test_autodetect_uses_sniff(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        class SniffLoader(Loader):
            format = DatasetFormat.HMIE

            @classmethod
            def sniff(cls, _root: str | Path) -> bool:
                return True

            def load(self, _root: str | Path, **_options: object) -> BoxTrackDataset:
                return BoxTrackDataset(sequences=[], categories={})

        monkeypatch.setattr(
            "datamaite.loaders._LOADERS",
            {LoaderKey(task=Task.MOT, format=DatasetFormat.HMIE): SniffLoader},
        )
        assert isinstance(load(tmp_path, dataset_format=None), BoxTrackDataset)

    def test_load_mot_does_not_retry_loader_value_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        calls = 0

        class BadOptionLoader(Loader):
            format = DatasetFormat.HMIE

            def load(self, _root: str | Path, **_options: object) -> BoxTrackDataset:
                nonlocal calls
                calls += 1
                raise ValueError("bad loader option")

        monkeypatch.setattr(
            "datamaite.loaders._LOADERS",
            {LoaderKey(task=Task.MOT, format=DatasetFormat.HMIE): BadOptionLoader},
        )
        with pytest.raises(ValueError, match="bad loader option"):
            load_mot(tmp_path, dataset_format="hmie")
        assert calls == 1

    def test_autodetect_failure_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr("datamaite.loaders._LOADERS", {})
        with pytest.raises(ValueError, match="autodetect"):
            load(tmp_path, dataset_format=None)


class TestEquivalenceWithLoadMot:
    def test_load_dispatch_matches_load_mot(self, tmp_path: Path) -> None:
        default_happy_dataset(tmp_path)
        via_dispatch = load(tmp_path)
        via_helper = load_mot(tmp_path)
        assert len(via_dispatch) == len(via_helper)
        assert via_dispatch.num_boxes == via_helper.num_boxes
        assert via_dispatch.categories == via_helper.categories

    def test_load_forwards_options(self, tmp_path: Path) -> None:
        default_happy_dataset(tmp_path)
        # require_video=True must reach HmieLoader.load and skip the (real) videos
        # only if unreadable; here the factory writes openable videos, so this
        # simply confirms the option is forwarded without error.
        ds = load(tmp_path, dataset_format="hmie", require_video=True)
        assert isinstance(ds, BoxTrackDataset)


class TestLoadVc:
    """The task-first video-classification entry point.

    ``load_vc`` mirrors ``load_mot``: it delegates to ``load`` but pins the
    return type to ``VideoClassificationDataset`` and fails loudly if the
    resolved format produces a different task's dataset.
    """

    def test_load_vc_raises_on_missing_root(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_vc(tmp_path / "nope")

    def test_load_vc_rejects_non_vc_dataset(self, tmp_path: Path) -> None:
        # An HMIE root resolves to a BoxTrackDataset (MOT), not a VC dataset:
        # load_vc must reject it at the call site rather than mistype it.
        default_happy_dataset(tmp_path)
        with pytest.raises(TypeError, match="load_vc expected a VideoClassificationDataset"):
            load_vc(tmp_path, dataset_format="hmie")

    def test_task_first_loaders_preserve_task_error_for_multitask_formats(self, tmp_path: Path) -> None:
        for helper in (load_mot, load_vc):
            with pytest.raises(ValueError, match="No loader registered") as exc_info:
                helper(tmp_path, dataset_format="yolo")

            assert "Multiple loaders registered" not in str(exc_info.value)


class TestCloudRoots:
    def test_generic_load_accepts_memory_url(self, memory_root) -> None:
        single_video_dataset(
            memory_root,
            [SnippetSpec(name="video_001_000001", video=VideoSpec(corrupt=True))],
        )
        ds = load(str(memory_root), dataset_format=DatasetFormat.HMIE)
        assert isinstance(ds, BoxTrackDataset)
        assert ds.sequence_count == 1

    def test_generic_load_missing_memory_root_raises(self, memory_root) -> None:
        with pytest.raises(FileNotFoundError):
            load(str(memory_root / "nope"), dataset_format=DatasetFormat.HMIE)

    def test_load_mot_rejects_non_hmie_cloud_format(self, memory_root) -> None:
        # Cloud roots are HMIE-only; a non-HMIE format must fail loudly rather
        # than crash inside a loader with local-filesystem assumptions.
        root = memory_root / "x"
        root.mkdir()
        with pytest.raises(ValueError, match="HMIE format only"):
            load_mot(str(root), dataset_format="motchallenge")

    def test_load_mot_hmie_cloud_still_works(self, memory_root) -> None:
        single_video_dataset(
            memory_root,
            [SnippetSpec(name="video_001_000001", video=VideoSpec(corrupt=True))],
        )
        ds = load_mot(str(memory_root), dataset_format=DatasetFormat.HMIE)
        assert isinstance(ds, BoxTrackDataset)
        assert ds.sequence_count == 1

    def test_autodetect_rejects_cloud_root(self, memory_root) -> None:
        # No format is cloud-sniffable, so autodetect must fail cleanly instead
        # of letting a loader's sniff() crash on a UPath it can't coerce with
        # Path(). The UPath-object form is the one that used to raise a raw
        # TypeError from inside the yolo loader's sniff; the string form is
        # covered too since it takes a different path through to_dataset_path.
        root = memory_root / "x"
        root.mkdir()
        with pytest.raises(ValueError, match="dataset_format"):
            load(root, dataset_format=None)
        with pytest.raises(ValueError, match="dataset_format"):
            load(str(root), dataset_format=None)
