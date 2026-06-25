"""Tests for dataset_stats and the `datamaite stats` CLI command."""

from __future__ import annotations

import json
from pathlib import Path

from datamaite import dataset_stats
from datamaite._cli import main
from datamaite._stats import _percentiles, format_stats
from datamaite.model import BoxAnnotation, BoxTrackDataset, VideoSequence

WIDGET = "http://example.com/ontology/a/widget"


def _seq(video_id: int, *, duration: float | None, num_frames: int | None, fps: float, n_boxes: int) -> VideoSequence:
    boxes = [
        BoxAnnotation(
            track_uuid="u",
            track_id=0,
            category_id=1,
            category_uri=WIDGET,
            category_name="widget",
            bbox=(0, 0, 1, 1),
            attributes={},
            frame_index=i,
            timestamp=None,
        )
        for i in range(n_boxes)
    ]
    return VideoSequence(
        video_id=video_id,
        video_path=f"/tmp/v{video_id}.mp4",  # noqa: S108 - synthetic label
        fps=fps,
        num_frames=num_frames,
        duration=duration,
        annotation_path=f"/tmp/a{video_id}.json",  # noqa: S108
        boxes=boxes,
    )


class TestPercentiles:
    def test_empty_returns_none(self) -> None:
        assert _percentiles([]) is None

    def test_single_value(self) -> None:
        out = _percentiles([5.0])
        assert out is not None
        assert out["min"] == out["max"] == out["p50"] == out["mean"] == 5.0
        assert out["count"] == 1

    def test_interpolated_percentiles(self) -> None:
        out = _percentiles([float(i) for i in range(1, 11)])  # 1..10
        assert out is not None
        assert out["min"] == 1.0
        assert out["max"] == 10.0
        assert out["p50"] == 5.5  # midpoint of 5 and 6
        assert out["mean"] == 5.5


class TestDatasetStats:
    def test_distributions_and_counts(self) -> None:
        ds = BoxTrackDataset(
            sequences=[
                _seq(0, duration=10.0, num_frames=300, fps=30.0, n_boxes=5),
                _seq(1, duration=20.0, num_frames=600, fps=30.0, n_boxes=15),
            ],
            categories={WIDGET: 1},
        )
        stats = dataset_stats(ds)
        assert stats["sequences"] == 2
        assert stats["sequences_with_video"] == 2
        assert stats["sequences_with_duration"] == 2
        assert stats["total_boxes"] == 20
        assert stats["duration_s"]["mean"] == 15.0
        assert stats["num_frames"]["max"] == 600
        assert stats["boxes_per_sequence"]["min"] == 5

    def test_missing_duration_is_none_distribution(self) -> None:
        ds = BoxTrackDataset(
            sequences=[_seq(0, duration=None, num_frames=None, fps=0.0, n_boxes=0)],
            categories={},
        )
        stats = dataset_stats(ds)
        assert stats["duration_s"] is None
        assert stats["num_frames"] is None
        assert stats["fps"] is None
        assert stats["sequences_with_duration"] == 0

    def test_format_stats_renders_table_and_no_data(self) -> None:
        ds = BoxTrackDataset(sequences=[_seq(0, duration=None, num_frames=None, fps=0.0, n_boxes=0)], categories={})
        text = format_stats(dataset_stats(ds), root="/data/x")
        assert "/data/x" in text
        assert "Duration (s)" in text
        assert "(no data)" in text


class TestStatsCli:
    def test_stats_command_text(self, tmp_path: Path, capsys) -> None:
        from tests._hmie_factory import SnippetSpec, single_video_dataset

        single_video_dataset(
            tmp_path,
            snippets=[SnippetSpec(name="video_001_000001", source_designator="SRC1", hash_suffix="abc001")],
        )
        rc = main(["stats", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "sequences" in out
        assert "Duration (s)" in out

    def test_stats_command_json(self, tmp_path: Path, capsys) -> None:
        from tests._hmie_factory import SnippetSpec, single_video_dataset

        single_video_dataset(
            tmp_path,
            snippets=[SnippetSpec(name="video_001_000001", source_designator="SRC1", hash_suffix="abc001")],
        )
        rc = main(["stats", str(tmp_path), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["sequences"] >= 1
        assert "duration_s" in payload
