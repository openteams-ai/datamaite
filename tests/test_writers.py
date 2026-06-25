"""Tests for the writer architecture: Writer contract, registry, dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from datamaite._types import DatasetFormat
from datamaite.model import BoxTrackDataset
from datamaite.writers import (
    Writer,
    available_output_formats,
    get_writer,
    register_writer,
    write,
)


class TestRegistry:
    def test_hmie_writer_is_registered(self) -> None:
        # Importing datamaite registers the reference HMIE writer.
        import datamaite  # noqa: F401

        assert DatasetFormat.HMIE in available_output_formats()

    def test_get_writer_returns_instance(self) -> None:
        import datamaite  # noqa: F401

        writer = get_writer(DatasetFormat.HMIE)
        assert isinstance(writer, Writer)
        assert writer.format is DatasetFormat.HMIE

    def test_get_writer_accepts_string(self) -> None:
        import datamaite  # noqa: F401

        assert get_writer("hmie").format is DatasetFormat.HMIE
        assert get_writer("HMIE").format is DatasetFormat.HMIE

    def test_get_writer_unknown_string_raises(self) -> None:
        with pytest.raises(ValueError, match="not a valid DatasetFormat"):
            get_writer("does-not-exist")

    def test_get_writer_no_registered_writer_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("datamaite.writers._WRITERS", {})
        with pytest.raises(ValueError, match="No writer registered"):
            get_writer(DatasetFormat.HMIE)


class TestWriterContract:
    def test_writer_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            Writer()  # type: ignore[abstract]

    def test_subclass_without_write_is_abstract(self) -> None:
        class Incomplete(Writer):
            format = DatasetFormat.HMIE

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_register_requires_format(self) -> None:
        class NoFormat(Writer):
            def write(self, dataset: BoxTrackDataset, dest: str | Path, **options: object) -> list[Path]:
                raise NotImplementedError  # never called; register_writer rejects it first

        with pytest.raises(TypeError, match="must set `format`"):
            register_writer(NoFormat)


class TestRegisterAndDispatch:
    def test_register_writer_and_dispatch(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        # Patch the registry to empty *first* so register_writer mutates the
        # throwaway dict, not the real global one (which monkeypatch restores).
        monkeypatch.setattr("datamaite.writers._WRITERS", {})
        captured: dict[str, object] = {}

        class DummyWriter(Writer):
            format = DatasetFormat.HMIE

            def write(self, _dataset: BoxTrackDataset, dest: str | Path, **options: object) -> list[Path]:
                captured["dest"] = str(dest)
                captured["options"] = options
                out = Path(dest) / "out.txt"
                return [out]

        register_writer(DummyWriter)
        files = write(
            BoxTrackDataset(sequences=[], categories={}),
            tmp_path,
            output_format="hmie",
            verbose=True,
            custom_option=7,
        )

        assert files == [tmp_path / "out.txt"]
        assert captured["dest"] == str(tmp_path)
        # `verbose` is consumed by write(), not forwarded to the writer.
        assert captured["options"] == {"custom_option": 7}


class TestVerboseReturn:
    """`write` returns the file list only when ``verbose=True``.

    The full list is one path per written frame image -- large enough to spam
    an interactive session -- so it is opt-in. Either way the files are written;
    ``verbose`` controls only what is returned.
    """

    def _register_dummy(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
        monkeypatch.setattr("datamaite.writers._WRITERS", {})
        calls = {"writes": 0}

        class DummyWriter(Writer):
            format = DatasetFormat.HMIE

            def write(self, _dataset: BoxTrackDataset, dest: str | Path, **_options: object) -> list[Path]:
                calls["writes"] += 1
                return [Path(dest) / "a.txt", Path(dest) / "b.txt"]

        register_writer(DummyWriter)
        return calls

    def test_default_returns_none_but_still_writes(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        calls = self._register_dummy(monkeypatch)
        result = write(BoxTrackDataset(sequences=[], categories={}), tmp_path, output_format="hmie")
        assert result is None
        assert calls["writes"] == 1  # side effect happened despite the None return

    def test_verbose_true_returns_file_list(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self._register_dummy(monkeypatch)
        result = write(BoxTrackDataset(sequences=[], categories={}), tmp_path, output_format="hmie", verbose=True)
        assert result == [tmp_path / "a.txt", tmp_path / "b.txt"]
