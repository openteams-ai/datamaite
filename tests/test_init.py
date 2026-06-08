"""Smoke tests for package."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_import() -> None:
    import databridge

    assert databridge.__version__


def test_public_api() -> None:
    from databridge import (
        CategoryEntry,
        DatasetFormat,
        Finding,
        FlatMp4Loader,
        HuggingFaceVideoClassificationLoader,
        MotChallengeLoader,
        MotChallengeWriter,
        Severity,
        TaoLoader,
        TaoWriter,
        Task,
        Taxonomy,
        ValidationResult,
        VideoClassificationDataset,
        VideoClassificationSample,
        VisDroneVideoLoader,
        VisionDataset,
        load_huggingface_video_classification,
        load_mot,
        validate,
        validate_annotation,
    )

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
    assert callable(load_huggingface_video_classification)
    assert callable(load_mot)
    assert callable(validate)
    assert callable(validate_annotation)


def test_per_format_load_functions_not_in_public_api() -> None:
    """The old per-format ``load_*`` functions are replaced by task-first ``load_mot``.

    They live on internally (``databridge._formats.<format>.loader``) but are no
    longer exposed on the top-level ``databridge`` namespace.
    """
    import databridge

    for name in ("load_hmie", "load_tao", "load_motchallenge", "load_visdrone_video", "load_flat_mp4"):
        assert not hasattr(databridge, name), f"{name} should be removed from the public API (use load_mot)"


def test_validation_import_keeps_loader_and_writer_modules_lazy() -> None:
    """The validation import path should not eagerly load format loaders/writers."""
    code = """
import json
import sys
import databridge.validation
modules = (
    "databridge._formats.hmie.loader",
    "databridge._formats.hmie.writer",
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
        "databridge._formats.hmie.loader": False,
        "databridge._formats.hmie.writer": False,
    }


def test_version_module_shape() -> None:
    """The committed _version.py fallback exposes the attrs __init__.py imports.

    Regression guard: if someone re-adds src/databridge/_version.py to
    .gitignore, `poetry install` + `pytest` on a fresh clone will fail
    because hatch-vcs doesn't run for editable installs.
    """
    from databridge import _version

    assert isinstance(_version.__version__, str)
    assert _version.__version__  # non-empty
    assert isinstance(_version.__version_tuple__, tuple)
    assert len(_version.__version_tuple__) >= 1


def test_version_tuple_parsing() -> None:
    """_as_tuple preserves numeric segments and falls back to strings."""
    from databridge._version import _as_tuple

    assert _as_tuple("0.1.0") == (0, 1, 0)
    assert _as_tuple("0.1.dev59") == (0, 1, "dev59")
    assert _as_tuple("0.0.0+unknown") == (0, 0, 0, "unknown")
