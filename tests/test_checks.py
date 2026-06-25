"""Tests for validation check functions."""

from __future__ import annotations

import json
from pathlib import Path

from datamaite._formats.hmie.annotation_checks import check_annotation_schema
from datamaite._types import Severity
from tests._scale_factory import default_frame, one_track_annotation


class TestCheckAnnotationSchema:
    def test_valid_annotation(self, valid_annotation: Path) -> None:
        findings, annotation, _labels = check_annotation_schema(valid_annotation)
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) == 0
        assert annotation is not None
        assert annotation.task_id == "test-task-001"

    def test_minimal_annotation_flags_missing_fields(self, minimal_annotation: Path) -> None:
        findings, annotation, _labels = check_annotation_schema(minimal_annotation)
        assert annotation is not None
        # Missing AFR/FPS are errors per Scale spec (required for frame mapping)
        checks = {f.check for f in findings}
        assert "annotation_empty" in checks
        assert "annotation_missing_afr" in checks
        assert "annotation_missing_fps" in checks
        # AFR/FPS are errors, not warnings
        afr_finding = next(f for f in findings if f.check == "annotation_missing_afr")
        assert afr_finding.severity == Severity.ERROR

    def test_invalid_annotation(self, invalid_annotation: Path) -> None:
        findings, annotation, _labels = check_annotation_schema(invalid_annotation)
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) > 0
        assert annotation is None
        assert any("task_id" in f.message for f in errors)

    def test_bad_json(self, bad_json: Path) -> None:
        findings, annotation, _labels = check_annotation_schema(bad_json)
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) == 1
        assert errors[0].check == "annotation_json"
        assert annotation is None

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        findings, annotation, _labels = check_annotation_schema(tmp_path / "nope.json")
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) == 1
        assert errors[0].check == "annotation_readable"
        assert annotation is None

    def test_null_json_root_is_error(self, tmp_path: Path) -> None:
        """Top-level 'null' must produce an ERROR, not a silent pass."""
        p = tmp_path / "null.json"
        p.write_text("null")
        findings, annotation, _labels = check_annotation_schema(p)
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) == 1
        assert errors[0].check == "annotation_schema"
        assert "null" in errors[0].message
        assert annotation is None

    def test_list_json_root_is_error_not_crash(self, tmp_path: Path) -> None:
        """Top-level JSON array must not propagate into _is_unwrapped_annotations."""
        p = tmp_path / "list.json"
        p.write_text("[1, 2, 3]")
        findings, annotation, _labels = check_annotation_schema(p)
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert len(errors) == 1
        assert errors[0].check == "annotation_schema"
        assert "list" in errors[0].message
        assert annotation is None

    def test_scalar_json_root_is_error_not_crash(self, tmp_path: Path) -> None:
        """Top-level scalar (string, int, bool) must produce an ERROR, not crash."""
        for raw, expected_type in [('"hi"', "str"), ("5", "int"), ("true", "bool")]:
            p = tmp_path / f"scalar_{expected_type}.json"
            p.write_text(raw)
            findings, annotation, _labels = check_annotation_schema(p)
            errors = [f for f in findings if f.severity == Severity.ERROR]
            assert len(errors) == 1, f"expected 1 error for {raw}, got {len(errors)}"
            assert errors[0].check == "annotation_schema"
            assert expected_type in errors[0].message
            assert annotation is None

    def test_warns_on_zero_area_bbox(self, tmp_path: Path) -> None:
        data = one_track_annotation(
            task_id="zero-bbox",
            frames=[default_frame(height=0, width=50)],
        )
        p = tmp_path / "zero_bbox.json"
        p.write_text(json.dumps(data))
        findings, _, _labels = check_annotation_schema(p)
        warnings = [f for f in findings if f.severity == Severity.WARNING]
        assert any(f.check == "annotation_bbox_size" for f in warnings)

    def test_warns_on_bad_status(self, tmp_path: Path) -> None:

        data = {
            "task_id": "bad-status",
            "status": "exploded",
            "params": {"annotation_frame_rate": 5, "videoMetadata": {"video": {"fps": 30}}},
            "response": {"annotations": {}},
        }
        p = tmp_path / "bad_status.json"
        p.write_text(json.dumps(data))
        findings, _, _labels = check_annotation_schema(p)
        warnings = [f for f in findings if f.severity == Severity.WARNING]
        assert any(f.check == "annotation_status" for f in warnings)

    def test_errors_on_unknown_geometry(self, tmp_path: Path) -> None:
        """Unknown geometry is now a schema-level ERROR via Pydantic Literal."""
        data = one_track_annotation(
            task_id="bad-geom",
            geometry="hexagon",
            frames=[default_frame(left=0, top=0, height=10, width=10)],
        )
        p = tmp_path / "bad_geom.json"
        p.write_text(json.dumps(data))
        findings, annotation, _labels = check_annotation_schema(p)
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert any(f.check == "annotation_schema" for f in errors)
        assert any("geometry" in f.message for f in errors)
        assert annotation is None  # schema validation stopped here

    def test_warns_on_non_box_geometry(self, tmp_path: Path) -> None:
        """Polygon/line/point/cuboid/ellipse are accepted but not deep-validated."""
        data = one_track_annotation(
            task_id="poly",
            label="road",
            geometry="polygon",
            frames=[default_frame(left=0, top=0, height=10, width=10)],
        )
        p = tmp_path / "poly.json"
        p.write_text(json.dumps(data))
        findings, annotation, _labels = check_annotation_schema(p)
        # Still parses successfully
        assert annotation is not None
        # But emits the not-deep-validated warning
        warnings = [f for f in findings if f.severity == Severity.WARNING]
        assert any(f.check == "annotation_geometry_pending_support" for f in warnings)

    def test_polygon_frames_without_box_fields_parse_cleanly(self, tmp_path: Path) -> None:
        """Polygon frames carry vertices, not left/top/width/height.

        Schema must accept frames missing the box fields when the track
        geometry is non-box; the field-missing check only fires for
        geometry='box' tracks (see test_box_track_with_missing_fields).
        """
        payload = {
            "task_id": "t1",
            "params": {"annotation_frame_rate": 5.0, "videoMetadata": {"video": {"fps": 30.0}}},
            "response": {
                "annotations": {
                    "tr1": {
                        "label": "road",
                        "geometry": "polygon",
                        "frames": [{"key": 0, "vertices": [[0, 0], [10, 0], [10, 10]]}],
                    }
                }
            },
        }
        p = tmp_path / "poly_vertices.json"
        p.write_text(json.dumps(payload))
        findings, annotation, _labels = check_annotation_schema(p)
        assert annotation is not None
        errs = [f for f in findings if f.severity == Severity.ERROR]
        assert not errs, f"expected clean parse for polygon, got errors: {[(f.check, f.message) for f in errs]}"

    def test_box_track_with_missing_fields_flags_annotation_box_missing_fields(self, tmp_path: Path) -> None:
        """A box-geometry track with frames missing left/top/width/height is a data bug."""
        payload = {
            "task_id": "t1",
            "params": {"annotation_frame_rate": 5.0, "videoMetadata": {"video": {"fps": 30.0}}},
            "response": {
                "annotations": {
                    "tr1": {
                        "label": "car",
                        "geometry": "box",
                        "frames": [{"key": 0}],  # no box fields despite geometry=box
                    }
                }
            },
        }
        p = tmp_path / "box_missing.json"
        p.write_text(json.dumps(payload))
        findings, annotation, _labels = check_annotation_schema(p)
        assert annotation is not None
        assert any(f.check == "annotation_box_missing_fields" for f in findings)

    def test_detects_duplicate_track_uuids(self, tmp_path: Path) -> None:
        """Duplicate track UUIDs in raw JSON must be reported as ERROR.

        Python's default json.loads silently drops duplicate keys, so this
        data would otherwise disappear before the validator saw it. We
        write the JSON by hand because json.dumps() can't produce duplicate
        keys from a Python dict.
        """
        raw = """
        {
            "task_id": "dup-uuid",
            "params": {"annotation_frame_rate": 5, "videoMetadata": {"video": {"fps": 30}}},
            "response": {
                "annotations": {
                    "track-abc": {
                        "label": "car",
                        "geometry": "box",
                        "frames": [{"key": 0, "left": 0, "top": 0, "height": 10, "width": 10}]
                    },
                    "track-abc": {
                        "label": "truck",
                        "geometry": "box",
                        "frames": [{"key": 0, "left": 0, "top": 0, "height": 10, "width": 10}]
                    }
                }
            }
        }
        """
        p = tmp_path / "dup.json"
        p.write_text(raw)
        findings, _, _labels = check_annotation_schema(p)
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert any(f.check == "annotation_duplicate_keys" for f in errors)

    def test_errors_on_empty_label(self, tmp_path: Path) -> None:
        data = one_track_annotation(task_id="empty-label", label="")
        p = tmp_path / "empty_label.json"
        p.write_text(json.dumps(data))
        findings, _, labels = check_annotation_schema(p)
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert any(f.check == "annotation_label_empty" for f in errors)
        assert labels["<empty>"] == 1

    def test_errors_on_whitespace_label(self, tmp_path: Path) -> None:
        data = one_track_annotation(task_id="ws-label", label="   ")
        p = tmp_path / "ws_label.json"
        p.write_text(json.dumps(data))
        findings, _, labels = check_annotation_schema(p)
        errors = [f for f in findings if f.severity == Severity.ERROR]
        assert any(f.check == "annotation_label_empty" for f in errors)
        assert labels["<empty>"] == 1

    def test_label_histogram_counts_multiple_tracks(self, tmp_path: Path) -> None:
        one_frame = [default_frame()]
        data = one_track_annotation(
            task_id="hist",
            extra_tracks={
                "t2": {"label": "car", "geometry": "box", "frames": one_frame},
                "t3": {"label": "truck", "geometry": "box", "frames": one_frame},
            },
        )
        p = tmp_path / "hist.json"
        p.write_text(json.dumps(data))
        _findings, _, labels = check_annotation_schema(p)
        assert labels["car"] == 2
        assert labels["truck"] == 1

    def test_label_histogram_normalizes_whitespace(self, tmp_path: Path) -> None:
        """Leading/trailing whitespace on labels should be stripped in the histogram."""
        data = one_track_annotation(task_id="trim", label="  vehicle  ")
        p = tmp_path / "trim.json"
        p.write_text(json.dumps(data))
        _findings, _, labels = check_annotation_schema(p)
        assert labels["vehicle"] == 1
        assert "  vehicle  " not in labels

    def test_warns_on_negative_coords(self, tmp_path: Path) -> None:
        data = one_track_annotation(
            task_id="neg-coords",
            frames=[
                default_frame(key=0, left=-10, top=5),
                default_frame(key=1, left=5, top=-10),
                default_frame(key=2, left=5, top=5),
            ],
        )
        p = tmp_path / "neg_coords.json"
        p.write_text(json.dumps(data))
        findings, _, _labels = check_annotation_schema(p)
        warnings = [f for f in findings if f.severity == Severity.WARNING]
        neg = [f for f in warnings if f.check == "annotation_bbox_negative"]
        assert len(neg) == 1
        assert "2/3" in neg[0].message

    def test_zero_area_bbox_reports_aggregate_count(self, tmp_path: Path) -> None:
        data = one_track_annotation(
            task_id="zero-count",
            frames=[
                default_frame(key=0, height=0, width=50),
                default_frame(key=1, width=0),
                default_frame(key=2),
            ],
        )
        p = tmp_path / "zero_count.json"
        p.write_text(json.dumps(data))
        findings, _, _labels = check_annotation_schema(p)
        warnings = [f for f in findings if f.severity == Severity.WARNING]
        size = [f for f in warnings if f.check == "annotation_bbox_size"]
        assert len(size) == 1
        assert "2/3" in size[0].message

    def test_warns_on_empty_track(self, tmp_path: Path) -> None:
        data = one_track_annotation(task_id="empty-track", frames=[])
        p = tmp_path / "empty_track.json"
        p.write_text(json.dumps(data))
        findings, _, _labels = check_annotation_schema(p)
        warnings = [f for f in findings if f.severity == Severity.WARNING]
        assert any(f.check == "annotation_empty_track" for f in warnings)

    def test_warns_on_nonmonotonic_keys(self, tmp_path: Path) -> None:
        data = one_track_annotation(
            task_id="bad-order",
            frames=[
                default_frame(key=5, left=0, top=0, height=10, width=10),
                default_frame(key=2, left=0, top=0, height=10, width=10),
            ],
        )
        p = tmp_path / "bad_order.json"
        p.write_text(json.dumps(data))
        findings, _, _labels = check_annotation_schema(p)
        warnings = [f for f in findings if f.severity == Severity.WARNING]
        assert any(f.check == "annotation_key_order" for f in warnings)
