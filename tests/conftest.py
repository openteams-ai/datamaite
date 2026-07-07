"""Shared test fixtures for datamaite."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def valid_annotation(fixtures_dir: Path) -> Path:
    return fixtures_dir / "valid_annotation.json"


@pytest.fixture
def minimal_annotation(fixtures_dir: Path) -> Path:
    return fixtures_dir / "minimal_annotation.json"


@pytest.fixture
def invalid_annotation(fixtures_dir: Path) -> Path:
    return fixtures_dir / "invalid_annotation.json"


@pytest.fixture
def bad_json(fixtures_dir: Path) -> Path:
    return fixtures_dir / "bad_json.json"


@pytest.fixture
def single_snippet_hmie(tmp_path: Path, valid_annotation: Path) -> Path:
    """A minimal HMIE dataset with one valid snippet.

    Structure:
        tmp_path/dataset_000000/
            dataset_000001/
                labeler_a/
                    CDAO_test.json   (copy of the valid_annotation fixture)
                seq_mp4/
                    dataset_000001.mp4   (placeholder bytes, not a real video)

    Shared by test_validation.py and test_cli.py which both want the
    smallest possible pair-matching dataset without the overhead of the
    _hmie_factory (which generates real opencv mp4s).
    """
    root = tmp_path / "dataset_000000"
    root.mkdir()
    snippet = root / "dataset_000001"
    snippet.mkdir()
    labeler = snippet / "labeler_a"
    labeler.mkdir()
    (labeler / "CDAO_test.json").write_text(valid_annotation.read_text())
    (snippet / "seq_mp4").mkdir()
    (snippet / "seq_mp4" / "dataset_000001.mp4").write_bytes(b"fake mp4")
    return root


@pytest.fixture
def memory_root():
    """A clean UPath root on fsspec's in-memory filesystem.

    The memory filesystem is a process-global singleton, so it is cleared
    before and after each test. ``store`` holds the files and ``pseudo_dirs``
    the directory entries -- clearing only the store leaks directories into
    later tests (order-dependent discovery flakes). Tests exercising
    validation against it must pass ``workers=1``: subprocess workers get
    their own empty store.
    """
    import fsspec
    from upath import UPath

    fs = fsspec.filesystem("memory")

    def _wipe() -> None:
        fs.store.clear()
        fs.pseudo_dirs[:] = [""]

    _wipe()
    yield UPath("memory://hmie-cloud-test")
    _wipe()
