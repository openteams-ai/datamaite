"""Shared numeric coercion helpers used by the fixed-taxonomy writers (#55 B6).

``coerce_int``/``coerce_finite_float`` were duplicated line-for-line across
``_fixed_taxonomy.py``, ``motchallenge/writer.py``, and ``visdrone/writer.py``;
this module centralises the single canonical implementation.
"""

from __future__ import annotations

import math


def coerce_finite_float(value: object) -> float | None:
    """Best-effort coercion of ``value`` to a finite ``float``, else ``None``.

    Rejects ``bool`` (a ``bool`` is technically an ``int``/coerces cleanly to
    ``float``, but ``True``/``False`` are never a meaningful numeric field
    value here) and non-finite results (``inf``/``nan``).
    """
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def coerce_int(value: object) -> int | None:
    """Best-effort coercion of ``value`` to an ``int``, else ``None``.

    Only succeeds when ``value`` coerces to a finite float with no
    fractional part (``coerce_finite_float`` plus an integral check).
    """
    number = coerce_finite_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)
