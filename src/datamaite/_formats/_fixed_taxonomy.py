"""Fixed-taxonomy class-id resolution shared by the MOTChallenge and VisDrone writers.

Both target formats interpret their class column against a fixed class table
(MOTChallenge classes, VisDrone categories). A generic loader-assigned
``category_id`` is not in that vocabulary, so silently falling back to it can
produce semantically misleading outputs (#55). This module centralises the
precedence (explicit ``class_map`` > target-specific source attribute >
generic ``category_id``) and aggregates the per-write warnings so a large
dataset emits one loud message instead of thousands.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from datamaite._formats._coerce import coerce_int
from datamaite.model import BoxAnnotation


def validate_class_map(
    class_map: object,
    *,
    minimum: int,
    format_label: str,
) -> dict[str | int, int] | None:
    """Validate a user-supplied ``class_map`` before anything is written.

    Keys are source ``category_name`` strings or ``category_id`` ints; values
    are target class ids and must be ``>= minimum`` (1 for MOTChallenge GT,
    0 for VisDrone). Raises ``ValueError`` on the first problem.
    """
    if class_map is None:
        return None
    if not isinstance(class_map, Mapping):
        raise ValueError(f"class_map must map category names/ids to {format_label} class ids; got {class_map!r}")
    validated: dict[str | int, int] = {}
    for key, value in class_map.items():
        if isinstance(key, bool) or not isinstance(key, str | int):
            raise ValueError(f"class_map keys must be category names (str) or category ids (int); got {key!r}")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"class_map values must be int {format_label} class ids; got {value!r} for key {key!r}")
        if value < minimum:
            raise ValueError(f"{format_label} class ids must be >= {minimum}; got {value} for key {key!r}")
        validated[key] = value
    return validated


@dataclass(frozen=True)
class ResolvedClass:
    """A resolved target class id plus provenance for downstream policy checks.

    ``class_id`` is ``None`` **only** to mean "unmapped under an explicit
    ``class_map``" (already counted in the resolver's ``_unmapped`` tally, so
    the caller drops the box without warning itself). ``from_generic_fallback``
    is ``True`` only when ``class_id`` came from the last-resort
    ``category_id`` fallback (no ``class_map``, no target-specific source
    attribute) -- callers use it to avoid treating a fallback-derived
    sentinel value (e.g. VisDrone's category ``0``) as if it were a
    deliberately-assigned source category (#55 B3).
    """

    class_id: int | None
    from_generic_fallback: bool = False


class ClassIdResolver:
    """Resolve target class ids for one ``write()`` call and aggregate warnings.

    Create one per write, call :meth:`resolve` per box, and call
    :meth:`emit_warnings` once after all sequences are serialised.
    """

    def __init__(
        self,
        *,
        format_label: str,
        attribute: str,
        class_map: dict[str | int, int] | None,
        logger: logging.Logger,
        minimum: int,
    ) -> None:
        self._format_label = format_label
        self._attribute = attribute
        self._class_map = class_map
        self._logger = logger
        self._minimum = minimum
        self._fallback: Counter[str] = Counter()
        self._unmapped: Counter[str] = Counter()
        # name -> distinct source category_id values seen for that name key,
        # so emit_warnings can flag a class_map name key that silently
        # conflates two different source categories onto one target (#55 B4).
        self._name_category_ids: dict[str, set[int]] = {}

    @property
    def has_class_map(self) -> bool:
        return self._class_map is not None

    def resolve(self, box: BoxAnnotation) -> ResolvedClass:
        """Target class id (with provenance) for ``box``, or a ``None`` id to drop it."""
        if self._class_map is not None:
            if box.category_name is not None and box.category_name in self._class_map:
                self._name_category_ids.setdefault(box.category_name, set()).add(box.category_id)
                return ResolvedClass(self._class_map[box.category_name])
            if box.category_id in self._class_map:
                return ResolvedClass(self._class_map[box.category_id])
            self._unmapped[_category_label(box)] += 1
            return ResolvedClass(None)
        from_attribute = coerce_int(box.attributes.get(self._attribute))
        if from_attribute is not None:
            return ResolvedClass(from_attribute)
        if box.category_id >= self._minimum:
            self._fallback[_category_label(box)] += 1
        return ResolvedClass(box.category_id, from_generic_fallback=True)

    def emit_warnings(self) -> None:
        """Emit at most one aggregated WARNING per condition for this write."""
        if self._fallback:
            self._logger.warning(
                "%s writer: %d annotation(s) across %d category(ies) had no %r attribute and fell back to "
                "generic category_id values, which are reinterpreted against the fixed %s class table; "
                "pass class_map= to map source categories explicitly. Fallback categories: %s",
                self._format_label,
                sum(self._fallback.values()),
                len(self._fallback),
                self._attribute,
                self._format_label,
                _format_counts(self._fallback),
            )
        if self._unmapped:
            self._logger.warning(
                "%s writer: dropped %d annotation(s) across %d category(ies) not present in class_map: %s",
                self._format_label,
                sum(self._unmapped.values()),
                len(self._unmapped),
                _format_counts(self._unmapped),
            )
        ambiguous = {name: ids for name, ids in self._name_category_ids.items() if len(ids) > 1}
        if ambiguous:
            details = "; ".join(
                f"{name!r} (source category_id {sorted(ids)} -> target {self._class_map[name]})"  # type: ignore[index]
                for name, ids in sorted(ambiguous.items())
            )
            self._logger.warning(
                "%s writer: class_map name key(s) ambiguous -- matched multiple distinct source category_id "
                "values, all collapsed onto a single target class id: %s",
                self._format_label,
                details,
            )


def _category_label(box: BoxAnnotation) -> str:
    if box.category_name:
        return box.category_name
    return f"category_id={box.category_id}"


def _format_counts(counts: Counter[str]) -> str:
    return ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))
