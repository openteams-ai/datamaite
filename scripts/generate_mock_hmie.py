#!/usr/bin/env python
"""Generate the standalone HMIE mock dataset tree (for the hmie-mock-data repo).

Reuses databridge's HMIE test factory (``tests/_hmie_factory.py``) to write a
set of self-contained, **non-CUI** HMIE datasets -- one directory per scenario,
covering the happy path plus each validation failure mode. The output is what
gets checked into the separate ``hmie-mock-data`` repo (mp4 via git-LFS, JSON
plain); this script itself stays in databridge as the documented regenerator.

Usage:
    poetry run python scripts/generate_mock_hmie.py <output_dir>

``<output_dir>`` gets a ``datasets/<scenario>/`` tree per SCENARIOS below.
All labels/names are generic placeholders -- no real ontology, batch, or
source identifiers.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from pathlib import Path

# Reuse the in-repo factory; run from the databridge repo root via poetry.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests._hmie_factory import (
    AnnotationSpec,
    FullVideoSpec,
    SnippetSpec,
    TrackSpec,
    VideoSpec,
    make_hmie_dataset,
)

# A small valid video/annotation reused as the "good" baseline each scenario
# perturbs. Two tracks so targets are non-trivial.
_GOOD_TRACKS = [
    TrackSpec(label="vehicle", num_frames=5, bbox=(10.0, 10.0, 50.0, 40.0)),
    TrackSpec(label="boat", num_frames=5, bbox=(80.0, 60.0, 30.0, 25.0)),
]


def _good_snippet(name: str, *, src: str, h: str, task: str) -> SnippetSpec:
    return SnippetSpec(
        name=name,
        source_designator=src,
        hash_suffix=h,
        video=VideoSpec(num_frames=30, fps=30.0, width=320, height=240),
        annotation=AnnotationSpec(task_id=task, afr=5.0, video_fps=30.0, tracks=list(_GOOD_TRACKS)),
    )


def _valid() -> list[FullVideoSpec]:
    return [
        FullVideoSpec(
            name="video_001_000000",
            snippets=[
                _good_snippet("video_001_000001", src="SRC1", h="abc001", task="t-v1-s1"),
                _good_snippet("video_001_000002", src="SRC1", h="abc002", task="t-v1-s2"),
            ],
        ),
        FullVideoSpec(
            name="video_002_000000",
            snippets=[_good_snippet("video_002_000001", src="SRC2", h="def001", task="t-v2-s1")],
        ),
    ]


def _one_bad(mutate: Callable[[SnippetSpec], None]) -> list[FullVideoSpec]:
    """A dataset with one good snippet and one mutated (faulty) snippet."""
    good = _good_snippet("video_001_000001", src="SRC1", h="abc001", task="t-v1-s1")
    bad = _good_snippet("video_001_000002", src="SRC1", h="abc002", task="t-v1-s2")
    mutate(bad)
    return [FullVideoSpec(name="video_001_000000", snippets=[good, bad])]


def _corrupt_video(s: SnippetSpec) -> None:
    s.video = VideoSpec(corrupt=True)


def _invalid_json(s: SnippetSpec) -> None:
    s.annotation.valid_json = False


def _missing_task_id(s: SnippetSpec) -> None:
    s.annotation.include_task_id = False


def _orphan_annotation(s: SnippetSpec) -> None:
    s.include_video = False  # annotation present, no video


def _orphan_video(s: SnippetSpec) -> None:
    s.include_annotation = False  # video present, no annotation


def _fps_mismatch(s: SnippetSpec) -> None:
    # Annotation declares 60 fps; the real video is 30 fps -> consistency fail.
    s.annotation.video_fps = 60.0


# scenario dir -> dataset spec. Keep each isolated to one fault.
# ``bad-*`` trip an ERROR (validate() -> FAIL); ``warn-*`` trip only a WARNING
# (validate() -> PASS-with-warnings). Verdicts are pinned in MANIFEST.md.
SCENARIOS: dict[str, list[FullVideoSpec]] = {
    "valid": _valid(),
    "bad-corrupt-video": _one_bad(_corrupt_video),
    "bad-invalid-json": _one_bad(_invalid_json),
    "bad-missing-task-id": _one_bad(_missing_task_id),
    "bad-orphan-annotation": _one_bad(_orphan_annotation),
    "warn-orphan-video": _one_bad(_orphan_video),
    "warn-fps-mismatch": _one_bad(_fps_mismatch),
}


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    out = Path(sys.argv[1]) / "datasets"
    for name, full_videos in SCENARIOS.items():
        target = out / name
        if target.exists():
            shutil.rmtree(target)
        make_hmie_dataset(target, full_videos)
        n_snip = sum(len(fv.snippets) for fv in full_videos)
        print(f"  {name:<24} {len(full_videos)} full-video(s), {n_snip} snippet(s) -> {target}")
    print(f"Done. {len(SCENARIOS)} scenarios written under {out}")


if __name__ == "__main__":
    main()
