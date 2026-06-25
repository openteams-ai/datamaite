"""Tests for Scale annotation Pydantic models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from datamaite._formats.hmie.schema import ScaleAnnotation
from tests._scale_factory import default_frame, one_track_annotation


class TestScaleAnnotation:
    def test_parse_valid(self, valid_annotation: Path) -> None:
        data = json.loads(valid_annotation.read_text())
        ann = ScaleAnnotation.model_validate(data)
        assert ann.task_id == "test-task-001"
        assert ann.status == "completed"
        assert len(ann.response.annotations) == 1

    def test_parse_minimal(self, minimal_annotation: Path) -> None:
        data = json.loads(minimal_annotation.read_text())
        ann = ScaleAnnotation.model_validate(data)
        assert ann.task_id == "minimal-001"
        assert len(ann.response.annotations) == 0

    def test_missing_task_id(self, invalid_annotation: Path) -> None:
        data = json.loads(invalid_annotation.read_text())
        with pytest.raises(ValidationError, match="task_id"):
            ScaleAnnotation.model_validate(data)

    def test_track_fields(self, valid_annotation: Path) -> None:
        data = json.loads(valid_annotation.read_text())
        ann = ScaleAnnotation.model_validate(data)
        track = ann.response.annotations["track-uuid-001"]
        assert track.label == "vehicle"
        assert track.geometry == "box"
        assert len(track.frames) == 2

    def test_frame_fields(self, valid_annotation: Path) -> None:
        data = json.loads(valid_annotation.read_text())
        ann = ScaleAnnotation.model_validate(data)
        frame = ann.response.annotations["track-uuid-001"].frames[0]
        assert frame.key == 0
        assert frame.left == 100
        assert frame.top == 200
        assert frame.height == 50
        assert frame.width == 80
        assert frame.keyframeType == "start"
        assert frame.timestamp_secs == 0.0

    def test_params_frame_rates(self, valid_annotation: Path) -> None:
        data = json.loads(valid_annotation.read_text())
        ann = ScaleAnnotation.model_validate(data)
        assert ann.params is not None
        assert ann.params.annotation_frame_rate == 5
        assert ann.params.videoMetadata is not None
        assert ann.params.videoMetadata.video is not None
        assert ann.params.videoMetadata.video.fps == 30

    def test_extra_fields_allowed(self) -> None:
        data = {
            "task_id": "extra-fields",
            "response": {"annotations": {}},
            "some_unknown_field": "should not fail",
        }
        ann = ScaleAnnotation.model_validate(data)
        assert ann.task_id == "extra-fields"

    def test_missing_response_fails(self) -> None:
        with pytest.raises(ValidationError, match="response"):
            ScaleAnnotation.model_validate({"task_id": "no-response"})

    def test_negative_frame_key_fails(self) -> None:
        data = one_track_annotation(
            task_id="bad-key",
            afr=None,
            fps=None,
            frames=[default_frame(key=-1, left=0, top=0, height=10, width=10)],
        )
        with pytest.raises(ValidationError, match="key"):
            ScaleAnnotation.model_validate(data)
