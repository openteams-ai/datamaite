"""Tests for the shared Scale frame-key -> video-frame-index mapping."""

from __future__ import annotations

from datamaite._formats.hmie.frame_mapping import frame_key_to_index, is_mappable


class TestIsMappable:
    def test_true_for_positive_finite_inputs(self) -> None:
        assert is_mappable(30.0, 5.0) is True
        assert is_mappable(30.0, 30.0) is True

    def test_false_when_either_is_none(self) -> None:
        assert is_mappable(None, 5.0) is False
        assert is_mappable(30.0, None) is False

    def test_false_for_nonpositive(self) -> None:
        assert is_mappable(0.0, 5.0) is False
        assert is_mappable(30.0, 0.0) is False
        assert is_mappable(-1.0, 5.0) is False

    def test_false_for_nonfinite(self) -> None:
        assert is_mappable(float("inf"), 5.0) is False
        assert is_mappable(30.0, float("nan")) is False

    def test_agrees_with_frame_key_to_index_fallback(self) -> None:
        # When is_mappable is False, frame_key_to_index returns the raw key.
        assert is_mappable(0.0, 5.0) is False
        assert frame_key_to_index(7, fps=0.0, afr=5.0) == 7
        # When True, it applies the floor mapping.
        assert is_mappable(30.0, 5.0) is True
        assert frame_key_to_index(1, fps=30.0, afr=5.0) == 6


class TestFrameKeyToIndex:
    def test_identity_when_afr_equals_fps(self) -> None:
        # afr == fps: label-space and frame-space coincide.
        assert frame_key_to_index(7, fps=30.0, afr=30.0) == 7

    def test_subsampled_labeling_scales_key(self) -> None:
        # Labeled at 5 fps in a 30 fps video: key k -> floor(k * 6).
        assert frame_key_to_index(4, fps=30.0, afr=5.0) == 24
        assert frame_key_to_index(0, fps=30.0, afr=5.0) == 0
        assert frame_key_to_index(1, fps=30.0, afr=5.0) == 6

    def test_floor_rounding(self) -> None:
        # 23.976 / 5 = 4.7952; key 1 -> floor(4.7952) = 4.
        assert frame_key_to_index(1, fps=23.976, afr=5.0) == 4

    def test_falls_back_to_key_when_fps_missing(self) -> None:
        assert frame_key_to_index(9, fps=None, afr=5.0) == 9

    def test_falls_back_to_key_when_afr_missing(self) -> None:
        assert frame_key_to_index(9, fps=30.0, afr=None) == 9

    def test_falls_back_to_key_when_fps_nonpositive(self) -> None:
        assert frame_key_to_index(9, fps=0.0, afr=5.0) == 9

    def test_falls_back_to_key_when_afr_nonpositive(self) -> None:
        assert frame_key_to_index(9, fps=30.0, afr=0.0) == 9

    def test_falls_back_to_key_when_nonfinite(self) -> None:
        assert frame_key_to_index(9, fps=float("inf"), afr=5.0) == 9
        assert frame_key_to_index(9, fps=float("nan"), afr=5.0) == 9


class TestFloatingPointRobustness:
    def test_undershoot_snaps_to_true_integer(self) -> None:
        # 11 * 29.97 / 14.985 == 22 exactly, but floats compute 21.9999999996;
        # a naive floor would drop it to 21.
        assert frame_key_to_index(11, 29.97, 14.985) == 22

    def test_genuinely_fractional_result_still_floors(self) -> None:
        # 1 * 30 / 7 = 4.2857...; must floor to 4, not snap to a neighbour.
        assert frame_key_to_index(1, 30.0, 7.0) == 4

    def test_exact_integer_ratio_unchanged(self) -> None:
        assert frame_key_to_index(3, 30.0, 5.0) == 18
