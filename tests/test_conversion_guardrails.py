"""``convert()`` guardrails: no guessing the task."""

from __future__ import annotations

from pathlib import Path

import pytest

from datamaite import convert, write

from .test_image_conversion_matrix import _ic_dataset


class TestTaskIsNotGuessed:
    def test_shared_input_format_still_requires_task(self, tmp_path: Path) -> None:
        # Guessing OD from the output format would read this YOLO
        # classification layout as unlabeled OD images and write a COCO file
        # with no annotations, silently.
        write(_ic_dataset(tmp_path / "raw"), tmp_path / "yolo", output_format="yolo")
        with pytest.raises(ValueError, match="specify task"):
            convert(tmp_path / "yolo", tmp_path / "out", input_format="yolo", output_format="coco")
