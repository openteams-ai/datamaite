"""Tests for the loader architecture: Loader contract, registry, dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from databridge import load, load_hmie
from databridge._types import DatasetFormat
from databridge.dataloader import HmieLoader
from databridge.loaders import Loader, available_formats, get_loader, register_loader
from databridge.model import BoxTrackDataset

from ._hmie_factory import default_happy_dataset


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
        monkeypatch.setattr("databridge.loaders._LOADERS", {})
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
        monkeypatch.setattr("databridge.loaders._LOADERS", {})
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


class TestAutodetect:
    def test_autodetect_uses_sniff(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        class SniffLoader(Loader):
            format = DatasetFormat.HMIE

            @classmethod
            def sniff(cls, _root: str | Path) -> bool:
                return True

            def load(self, _root: str | Path, **_options: object) -> BoxTrackDataset:
                return BoxTrackDataset(sequences=[], categories={})

        monkeypatch.setattr("databridge.loaders._LOADERS", {DatasetFormat.HMIE: SniffLoader})
        assert isinstance(load(tmp_path, dataset_format=None), BoxTrackDataset)

    def test_autodetect_failure_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr("databridge.loaders._LOADERS", {})
        with pytest.raises(ValueError, match="autodetect"):
            load(tmp_path, dataset_format=None)


class TestEquivalenceWithLoadHmie:
    def test_load_dispatch_matches_load_hmie(self, tmp_path: Path) -> None:
        default_happy_dataset(tmp_path)
        via_dispatch = load(tmp_path)
        via_helper = load_hmie(tmp_path)
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
