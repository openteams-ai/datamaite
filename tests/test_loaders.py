"""Tests for the loader architecture: Loader contract, registry, dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from databridge import load, load_mot
from databridge._formats.hmie.loader import HmieLoader
from databridge._types import DatasetFormat
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


class TestMissingRoot:
    """A nonexistent or non-directory root is a caller error, not bad data.

    Loaders are best-effort about *data* (skip unparseable items, warn), but a
    root that does not exist is a bad argument -- ``load`` raises rather than
    silently returning an empty dataset (which masked typo'd reload paths).
    """

    def test_nonexistent_root_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist"
        with pytest.raises(FileNotFoundError, match="does-not-exist"):
            load(missing, dataset_format="tao")

    def test_file_root_raises_not_a_directory(self, tmp_path: Path) -> None:
        a_file = tmp_path / "a_file.txt"
        a_file.write_text("not a dataset directory", encoding="utf-8")
        with pytest.raises(NotADirectoryError, match=r"a_file\.txt"):
            load(a_file, dataset_format="tao")

    def test_load_mot_also_raises_on_missing_root(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_mot(tmp_path / "nope", dataset_format="hmie")

    def test_existing_empty_root_returns_empty_without_raising(self, tmp_path: Path) -> None:
        # Root exists but holds no loadable TAO data: best-effort empty, no raise.
        empty = tmp_path / "empty"
        empty.mkdir()
        ds = load(empty, dataset_format="tao")
        assert isinstance(ds, BoxTrackDataset)
        assert len(ds) == 0


class TestEmptyResultWarns:
    """An existing root that yields no loadable items warns (but does not raise).

    Distinct from a missing root (which raises): the root is valid, the load ran
    to completion, but the result is empty -- usually a wrong format or wrong
    subdirectory. The dispatch layer emits one uniform warning so this is never
    silent, closing per-loader gaps (HMIE, MOTChallenge empty splits).
    """

    def test_empty_hmie_load_warns(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        # Directory exists and is a plausible HMIE root, but holds no annotations.
        (tmp_path / "video_001_000000").mkdir()
        with caplog.at_level("WARNING"):
            ds = load(tmp_path, dataset_format="hmie")
        assert len(ds.sequences) == 0
        assert any("empty dataset" in record.message.lower() for record in caplog.records)

    def test_empty_motchallenge_split_warns(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        # train/ exists but contains no sequence directories -> empty, no per-loader warning.
        (tmp_path / "train").mkdir()
        with caplog.at_level("WARNING"):
            ds = load(tmp_path, dataset_format="motchallenge")
        assert len(ds.sequences) == 0
        assert any("empty dataset" in record.message.lower() for record in caplog.records)

    def test_nonempty_load_does_not_warn_empty(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        default_happy_dataset(tmp_path)
        with caplog.at_level("WARNING"):
            ds = load(tmp_path, dataset_format="hmie")
        assert ds.sequence_count > 0
        assert not any("empty dataset" in record.message.lower() for record in caplog.records)


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
