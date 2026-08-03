"""Smoke tests for package."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_import() -> None:
    import datamaite

    assert datamaite.__version__


def test_public_api() -> None:
    from datamaite import (
        CategoryEntry,
        CocoLoader,
        CocoWriter,
        DatasetFormat,
        DatasetMetadata,
        Finding,
        FlatMp4Loader,
        HuggingFaceVideoClassificationLoader,
        ImageObjectDetectionSample,
        MotChallengeLoader,
        MotChallengeWriter,
        ObjectDetectionAnnotation,
        ObjectDetectionDataset,
        Severity,
        TaoLoader,
        TaoWriter,
        Task,
        Taxonomy,
        ValidationResult,
        VideoClassificationDataset,
        VideoClassificationSample,
        VisDroneVideoLoader,
        VisDroneVideoWriter,
        VisionDataset,
        WriterCapabilities,
        YoloImageClassificationLoader,
        YoloImageClassificationWriter,
        YoloObjectDetectionLoader,
        YoloObjectDetectionWriter,
        load_mot,
        load_od,
        load_vc,
        validate,
        validate_annotation,
    )

    assert DatasetFormat.COCO.value == "coco"
    assert DatasetFormat.FLAT_MP4.value == "flat_mp4"
    assert DatasetFormat.HUGGINGFACE_VIDEO_CLASSIFICATION.value == "huggingface_video_classification"
    assert DatasetFormat.HMIE.value == "hmie"
    assert DatasetFormat.MOTCHALLENGE.value == "motchallenge"
    assert DatasetFormat.TAO.value == "tao"
    assert DatasetFormat.VISDRONE_VIDEO.value == "visdrone_video"
    assert Severity.ERROR.value == "error"
    assert Severity.WARNING.value == "warning"
    assert Task.MOT.value == "mot"
    assert Task.OD.value == "od"
    assert Task.IC.value == "ic"
    assert Task.VC.value == "vc"
    assert Taxonomy is not None
    assert CategoryEntry is not None
    assert ObjectDetectionDataset is not None
    assert callable(load_od)
    assert ObjectDetectionAnnotation is not None
    assert ImageObjectDetectionSample is not None
    assert DatasetMetadata is not None
    assert Finding is not None
    assert ValidationResult is not None
    assert VideoClassificationDataset is not None
    assert VideoClassificationSample is not None
    assert VisionDataset is not None
    assert FlatMp4Loader is not None
    assert HuggingFaceVideoClassificationLoader is not None
    assert MotChallengeLoader is not None
    assert MotChallengeWriter is not None
    assert TaoLoader is not None
    assert TaoWriter is not None
    assert VisDroneVideoLoader is not None
    assert VisDroneVideoWriter is not None
    assert CocoLoader is not None
    assert CocoWriter is not None
    assert WriterCapabilities is not None
    assert YoloImageClassificationLoader is not None
    assert YoloImageClassificationWriter is not None
    assert YoloObjectDetectionLoader is not None
    assert YoloObjectDetectionWriter is not None
    assert callable(load_mot)
    assert callable(load_vc)
    assert callable(validate)
    assert callable(validate_annotation)


def test_per_format_load_functions_not_in_public_api() -> None:
    """Per-format ``load_*`` helpers are replaced by task-first loaders.

    They live on internally (``datamaite._formats.<format>.loader``) but are no
    longer exposed on the top-level ``datamaite`` namespace.
    """
    import datamaite

    for name in (
        "load_hmie",
        "load_tao",
        "load_motchallenge",
        "load_visdrone_video",
        "load_flat_mp4",
        "load_huggingface_video_classification",
        "load_yolo_image_classification",
        "load_yolo_object_detection",
    ):
        assert not hasattr(datamaite, name), f"{name} should be removed from the public API (use load_<task>)"


def test_validation_import_keeps_loader_and_writer_modules_lazy() -> None:
    """The validation import path should not eagerly load format loaders/writers."""
    code = """
import json
import sys
import datamaite.validation
modules = (
    "datamaite._formats.hmie.loader",
    "datamaite._formats.hmie.writer",
)
print(json.dumps({module: module in sys.modules for module in modules}))
"""
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(  # noqa: S603 - fixed interpreter/code for import isolation
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    loaded = json.loads(completed.stdout)
    assert loaded == {
        "datamaite._formats.hmie.loader": False,
        "datamaite._formats.hmie.writer": False,
    }


def test_version_module_shape() -> None:
    """_version.py exposes the attrs __init__.py imports.

    Regression guard: __init__.py imports __version__ and __version_tuple__
    from datamaite._version, which resolves them from installed package
    metadata (see src/datamaite/_version.py).
    """
    from datamaite import _version

    assert isinstance(_version.__version__, str)
    assert _version.__version__  # non-empty
    assert isinstance(_version.__version_tuple__, tuple)
    assert len(_version.__version_tuple__) >= 1


def test_version_tuple_parsing() -> None:
    """_as_tuple preserves numeric segments and falls back to strings."""
    from datamaite._version import _as_tuple

    assert _as_tuple("0.1.0") == (0, 1, 0)
    assert _as_tuple("0.1.dev59") == (0, 1, "dev59")
    assert _as_tuple("0.0.0+unknown") == (0, 0, 0, "unknown")


def test_version_resolve_prefers_installed_metadata(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Real installed metadata wins; the SCM fallback is never consulted."""
    from datamaite import _version

    monkeypatch.setattr(_version, "_metadata_version", lambda _name: "0.4.0")
    monkeypatch.setattr(_version, "_from_scm", lambda: pytest.fail("_from_scm must not be called"))
    assert _version._resolve() == "0.4.0"


def test_version_resolve_placeholder_falls_back_to_scm(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Poetry's develop-install registers 0.0.0; the SCM derivation replaces it."""
    from datamaite import _version

    monkeypatch.setattr(_version, "_metadata_version", lambda _name: "0.0.0")
    monkeypatch.setattr(_version, "_from_scm", lambda: "0.4.0.post5.dev0+d17db17")
    assert _version._resolve() == "0.4.0.post5.dev0+d17db17"


def test_version_resolve_placeholder_without_scm(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """No dunamai / no git checkout: the placeholder is reported as-is."""
    from datamaite import _version

    monkeypatch.setattr(_version, "_metadata_version", lambda _name: "0.0.0")
    monkeypatch.setattr(_version, "_from_scm", lambda: None)
    assert _version._resolve() == "0.0.0"


def test_version_resolve_not_installed_without_scm(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Not installed and no SCM derivation available: the +unknown marker."""
    from importlib.metadata import PackageNotFoundError

    from datamaite import _version

    def raise_not_found(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(_version, "_metadata_version", raise_not_found)
    monkeypatch.setattr(_version, "_from_scm", lambda: None)
    assert _version._resolve() == "0.0.0+unknown"


def test_from_scm_without_dunamai(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """dunamai is an optional (dev-extra) dependency; its absence is tolerated."""
    from datamaite import _version

    monkeypatch.setitem(sys.modules, "dunamai", None)
    assert _version._from_scm() is None


def test_from_scm_tolerates_git_failure(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A missing git binary / non-repo checkout degrades to None, not an error."""
    import types

    from datamaite import _version

    fake = types.ModuleType("dunamai")
    fake.Pattern = types.SimpleNamespace(DefaultUnprefixed="default-unprefixed")  # type: ignore[attr-defined]

    class _FailingVersion:
        @staticmethod
        def from_git(**_kwargs: object) -> object:
            raise RuntimeError("git not available")

    fake.Version = _FailingVersion  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dunamai", fake)
    assert _version._from_scm() is None


def test_from_scm_returns_pep440_or_none() -> None:
    """Exercise the real dunamai path.

    In a dev environment (git checkout + dev extra) this derives a PEP 440
    version from the tag; in an sdist-installed test run (no git metadata)
    it returns None. Both are correct.
    """
    from packaging.version import Version

    from datamaite._version import _from_scm

    derived = _from_scm()
    if derived is not None:
        Version(derived)  # raises InvalidVersion if malformed


def test_visdrone_static_enum_and_exports() -> None:
    import datamaite
    from datamaite._types import DatasetFormat

    assert DatasetFormat.VISDRONE.value == "visdrone"
    assert datamaite.VisDroneObjectDetectionLoader.__name__ == "VisDroneObjectDetectionLoader"
    assert datamaite.VisDroneImageClassificationLoader.__name__ == "VisDroneImageClassificationLoader"
