"""Frame-key to video-frame-index mapping for Scale annotations.

Scale labels frames at the *annotation frame rate* (AFR), which is often
lower than the video FPS. A per-frame ``key`` therefore indexes
label-space, not video-frame-space; the two coincide only when
``afr == fps``. The conversion ``frame# = floor(key * fps / afr)`` maps a
key back to the actual video frame it annotates.

This is the single source of truth for that conversion. It must be applied
wherever annotation frame keys are compared against, or combined with,
video-frame quantities (frame counts, durations) -- the loader uses it for
``BoxAnnotation.frame_index`` and the consistency checks use it for frame
bounds validation.
"""

from __future__ import annotations

import math


def is_mappable(fps: float | None, afr: float | None) -> bool:
    """Whether ``fps`` and ``afr`` permit a key -> video-frame mapping.

    The single predicate for "can we convert frame keys for this sequence?"
    -- both ``fps`` and ``afr`` must be present, finite, and positive.
    Non-finite values (NaN/Inf) silently pass ``<= 0`` comparisons, so they
    are rejected explicitly. Callers that cannot map should decide their own
    fallback (the loader keeps the raw key and logs; the validator skips the
    frame-bounds check).
    """
    if fps is None or afr is None:
        return False
    return math.isfinite(fps) and math.isfinite(afr) and fps > 0 and afr > 0


def frame_key_to_index(key: int, fps: float | None, afr: float | None) -> int:
    """Map a Scale annotation frame ``key`` to a video frame index.

    Returns ``floor(key * fps / afr)``. When the inputs are not mappable
    (see :func:`is_mappable`) the key is returned unchanged -- a best-effort
    identity rather than a crash or a dropped frame. Note the result is in
    *label* space in that fallback case; callers that mix mapped and
    unmapped sequences should surface that (the loader logs a warning).
    """
    # The explicit None check also narrows fps/afr to float for the type
    # checker; is_mappable carries the finite/positive rule (single source).
    if fps is None or afr is None or not is_mappable(fps, afr):
        return key
    value = key * fps / afr
    # Correct floating-point undershoot before flooring: for rates like
    # 29.97/14.985 the exact integer (e.g. 22) is computed as 21.9999999996,
    # and a naive floor would drop it to 21. If ``value`` sits within a tiny
    # tolerance of an integer it almost certainly *is* that integer, so snap
    # to it; genuinely fractional results (e.g. 4.2857) are unaffected.
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=1e-9, abs_tol=1e-9):
        return int(nearest)
    return math.floor(value)
