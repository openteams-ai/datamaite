"""Tests for canonical bbox geometry and conversions."""

from __future__ import annotations

import math

import pytest

from databridge import geometry as g


class TestCornerConversions:
    def test_to_from_xyxy_round_trip(self) -> None:
        box = (10.0, 20.0, 30.0, 40.0)
        assert g.to_xyxy(box) == (10.0, 20.0, 40.0, 60.0)
        assert g.from_xyxy(*g.to_xyxy(box)) == box

    def test_inclusive_corners_extent(self) -> None:
        # VOC: a box from xmin=10 to xmax=12 inclusive spans 3 pixels.
        assert g.from_xyxy_inclusive(10, 20, 12, 23) == (10, 20, 3, 4)
        assert g.to_xyxy_inclusive((10.0, 20.0, 3.0, 4.0)) == (10.0, 20.0, 12.0, 23.0)

    def test_inclusive_round_trip(self) -> None:
        corners = (5, 7, 25, 37)
        box = g.from_xyxy_inclusive(*corners)
        assert g.to_xyxy_inclusive(box) == corners


class TestCenterConversions:
    def test_to_from_cxcywh_round_trip(self) -> None:
        box = (10.0, 20.0, 30.0, 40.0)
        assert g.to_cxcywh(box) == (25.0, 40.0, 30.0, 40.0)
        assert g.from_cxcywh(*g.to_cxcywh(box)) == box


class TestNormalized:
    def test_to_from_normalized_round_trip(self) -> None:
        box = (50.0, 100.0, 200.0, 50.0)
        norm = g.to_normalized(box, 400.0, 200.0)
        assert norm == (0.125, 0.5, 0.5, 0.25)
        assert g.from_normalized(norm, 400.0, 200.0) == pytest.approx(box)

    def test_yolo_round_trip(self) -> None:
        box = (50.0, 100.0, 200.0, 50.0)
        cx, cy, w, h = g.to_yolo(box, 400.0, 200.0)
        # center normalized: ((50+100)/400, (100+25)/200, 200/400, 50/200)
        assert (cx, cy, w, h) == pytest.approx((0.375, 0.625, 0.5, 0.25))
        assert g.from_yolo(cx, cy, w, h, 400.0, 200.0) == pytest.approx(box)

    @pytest.mark.parametrize("fn", [g.to_normalized, g.to_yolo])
    def test_zero_dims_raise(self, fn) -> None:
        with pytest.raises(ValueError, match="finite and positive"):
            fn((1.0, 1.0, 1.0, 1.0), 0.0, 100.0)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_dims_raise(self, bad) -> None:
        # regression: NaN/inf must fail fast, not slip past `<= 0` and emit NaN boxes
        with pytest.raises(ValueError, match="finite and positive"):
            g.to_normalized((1.0, 1.0, 1.0, 1.0), bad, 100.0)

    def test_from_yolo_zero_dims_raise(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            g.from_yolo(0.5, 0.5, 0.5, 0.5, 100.0, -1.0)


class TestValidatorsAndArea:
    def test_is_finite(self) -> None:
        assert g.is_finite((1.0, 2.0, 3.0, 4.0))
        assert not g.is_finite((1.0, 2.0, math.nan, 4.0))
        assert not g.is_finite((1.0, 2.0, math.inf, 4.0))

    def test_has_positive_area(self) -> None:
        assert g.has_positive_area((0.0, 0.0, 2.0, 3.0))
        assert not g.has_positive_area((0.0, 0.0, 0.0, 3.0))
        assert not g.has_positive_area((0.0, 0.0, 2.0, -1.0))
        assert not g.has_positive_area((0.0, 0.0, math.nan, 3.0))

    def test_area(self) -> None:
        assert g.area((0.0, 0.0, 4.0, 5.0)) == 20.0


class TestClamp:
    def test_clamp_trims_to_image(self) -> None:
        # box overhangs the right/bottom edges
        assert g.clamp_to_image((90.0, 90.0, 40.0, 40.0), 100.0, 100.0) == (90.0, 90.0, 10.0, 10.0)

    def test_clamp_fully_outside_collapses(self) -> None:
        clamped = g.clamp_to_image((200.0, 200.0, 10.0, 10.0), 100.0, 100.0)
        assert not g.has_positive_area(clamped)

    def test_clamp_inside_is_unchanged(self) -> None:
        box = (10.0, 10.0, 20.0, 20.0)
        assert g.clamp_to_image(box, 100.0, 100.0) == box

    def test_clamp_zero_dims_raise(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            g.clamp_to_image((1.0, 1.0, 1.0, 1.0), 0.0, 0.0)
