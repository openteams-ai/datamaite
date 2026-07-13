"""Unit tests for the shared fixed-taxonomy class-id resolver (#55)."""

from __future__ import annotations

import logging

import pytest

from datamaite._formats._fixed_taxonomy import ClassIdResolver, validate_class_map
from datamaite.model import BoxAnnotation

logger = logging.getLogger("test_fixed_taxonomy")


def _box(
    *, category_id: int = 3, category_name: str | None = "person", attributes: dict | None = None
) -> BoxAnnotation:
    return BoxAnnotation(
        track_uuid="t",
        track_id=1,
        category_id=category_id,
        category_uri=f"src/{category_name or category_id}",
        category_name=category_name,
        bbox=(1.0, 2.0, 3.0, 4.0),
        attributes=attributes or {},
        frame_index=0,
        timestamp=None,
    )


def _resolver(class_map: dict | None = None, *, attribute: str = "mot_class_id", minimum: int = 1) -> ClassIdResolver:
    validated = validate_class_map(class_map, minimum=minimum, format_label="MOTChallenge")
    return ClassIdResolver(
        format_label="MOTChallenge", attribute=attribute, class_map=validated, logger=logger, minimum=minimum
    )


class TestValidateClassMap:
    def test_none_passes_through(self) -> None:
        assert validate_class_map(None, minimum=1, format_label="MOTChallenge") is None

    def test_valid_map_is_copied(self) -> None:
        source = {"person": 1, 3: 2}
        validated = validate_class_map(source, minimum=1, format_label="MOTChallenge")
        assert validated == source
        assert validated is not source

    @pytest.mark.parametrize(
        "bad_map",
        [
            {1.5: 1},  # non str/int key
            {True: 1},  # bool key
            {"person": "one"},  # non-int value
            {"person": True},  # bool value
            {"person": 0},  # below MOT minimum of 1
            "person=1",  # not a mapping
        ],
    )
    def test_bad_maps_raise(self, bad_map: object) -> None:
        with pytest.raises(ValueError, match=r"class_map|class ids"):
            validate_class_map(bad_map, minimum=1, format_label="MOTChallenge")

    def test_visdrone_allows_zero(self) -> None:
        assert validate_class_map({"ignored": 0}, minimum=0, format_label="VisDrone") == {"ignored": 0}


class TestResolveWithClassMap:
    def test_name_key_wins_over_id_key(self) -> None:
        resolver = _resolver({"person": 7, 3: 9})
        resolved = resolver.resolve(_box(category_id=3, category_name="person"))
        assert resolved.class_id == 7
        assert resolved.from_generic_fallback is False

    def test_id_key_used_when_name_absent(self) -> None:
        resolver = _resolver({3: 9})
        resolved = resolver.resolve(_box(category_id=3, category_name="person"))
        assert resolved.class_id == 9
        assert resolved.from_generic_fallback is False

    def test_name_present_but_unmatched_falls_through_to_id_key(self) -> None:
        resolver = _resolver({3: 9})
        resolved = resolver.resolve(_box(category_id=3, category_name="unmatched"))
        assert resolved.class_id == 9
        assert resolved.from_generic_fallback is False

    def test_class_map_overrides_target_attribute(self) -> None:
        resolver = _resolver({"person": 7})
        assert resolver.resolve(_box(attributes={"mot_class_id": 1})).class_id == 7

    def test_unmapped_returns_none_and_warns_once_aggregated(self, caplog: pytest.LogCaptureFixture) -> None:
        resolver = _resolver({"vehicle": 3})
        assert resolver.resolve(_box(category_name="person")).class_id is None
        assert resolver.resolve(_box(category_name="person")).class_id is None
        with caplog.at_level(logging.WARNING, logger="test_fixed_taxonomy"):
            resolver.emit_warnings()
        dropped = [r for r in caplog.records if "class_map" in r.message and "person" in r.getMessage()]
        assert len(dropped) == 1
        assert "2" in dropped[0].getMessage()

    def test_empty_class_map_drops_everything(self, caplog: pytest.LogCaptureFixture) -> None:
        resolver = _resolver({})
        assert resolver.has_class_map is True
        assert resolver.resolve(_box()).class_id is None
        with caplog.at_level(logging.WARNING, logger="test_fixed_taxonomy"):
            resolver.emit_warnings()
        assert any("not present in class_map" in r.getMessage() for r in caplog.records)

    def test_name_collision_across_distinct_source_ids_warns_once(self, caplog: pytest.LogCaptureFixture) -> None:
        resolver = _resolver({"person": 7})
        first = resolver.resolve(_box(category_id=3, category_name="person"))
        second = resolver.resolve(_box(category_id=5, category_name="person"))
        assert first.class_id == 7
        assert second.class_id == 7
        with caplog.at_level(logging.WARNING, logger="test_fixed_taxonomy"):
            resolver.emit_warnings()
        ambiguous = [r for r in caplog.records if "ambiguous" in r.getMessage().lower()]
        assert len(ambiguous) == 1
        message = ambiguous[0].getMessage()
        assert "person" in message
        assert "3" in message
        assert "5" in message
        assert "7" in message


class TestResolveWithoutClassMap:
    def test_attribute_wins_and_is_quiet(self, caplog: pytest.LogCaptureFixture) -> None:
        resolver = _resolver(None)
        resolved = resolver.resolve(_box(attributes={"mot_class_id": 5}))
        assert resolved.class_id == 5
        assert resolved.from_generic_fallback is False
        with caplog.at_level(logging.WARNING, logger="test_fixed_taxonomy"):
            resolver.emit_warnings()
        assert not caplog.records

    def test_generic_fallback_returns_id_and_warns_once_aggregated(self, caplog: pytest.LogCaptureFixture) -> None:
        resolver = _resolver(None)
        first = resolver.resolve(_box(category_id=3))
        second = resolver.resolve(_box(category_id=3))
        assert first.class_id == 3
        assert first.from_generic_fallback is True
        assert second.class_id == 3
        assert second.from_generic_fallback is True
        with caplog.at_level(logging.WARNING, logger="test_fixed_taxonomy"):
            resolver.emit_warnings()
        fallback = [r for r in caplog.records if "class_map" in r.getMessage()]
        assert len(fallback) == 1
        message = fallback[0].getMessage()
        assert "generic category_id" in message
        assert "person" in message

    def test_sub_minimum_category_id_not_counted_as_fallback(self, caplog: pytest.LogCaptureFixture) -> None:
        resolver = _resolver(None, minimum=1)
        resolved = resolver.resolve(_box(category_id=0, category_name=None))
        assert resolved.class_id == 0
        assert resolved.from_generic_fallback is True
        with caplog.at_level(logging.WARNING, logger="test_fixed_taxonomy"):
            resolver.emit_warnings()
        assert not [r for r in caplog.records if "class_map" in r.getMessage()]
