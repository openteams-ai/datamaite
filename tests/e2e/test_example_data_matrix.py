"""Cross-format input/output conversion matrix against the example-data repo.

These are **opt-in integration tests**. They run datamaite end-to-end against
a real checkout of the shared example-data repo
(``jatic/orchestration-interoperability/datamaite-example-datasets``) rather than
the synthetic ``tmp_path`` fixtures the unit suite uses.

To run locally, point ``DATAMAITE_DATASETS_ROOT`` at the ``datasets/``
directory of an example-data checkout and select the marker::

    git clone https://gitlab.jatic.net/jatic/orchestration-interoperability/datamaite-example-datasets.git
    export DATAMAITE_DATASETS_ROOT=$PWD/datamaite-example-datasets/datasets
    poetry run pytest -m integration -s     # -s shows the summary report

Skipped by default: the marker is deselected in ``pyproject.toml`` *and* every
test skips when the env var is unset, so the hermetic unit suite and the 90%
coverage gate are unaffected.

What each matrix cell validates
-------------------------------
For every ``input -> output`` conversion the test asserts, after a
``convert() -> reload`` round-trip:

* **counts** -- ``sequence_count`` and ``num_boxes`` match the values pinned in
  the example-data ``MANIFEST.md``; and
* **geometry** -- the *multiset of bounding-box coordinates* ``(left, top,
  width, height)`` is identical before and after. This is the strong,
  format-independent invariant: every box must survive, in the same place,
  with none dropped, moved, or duplicated.

What it deliberately does **not** check: JSON byte/hash equality, frame-index
numbering, or track-id values -- those are legitimately format-specific (e.g.
HMIE remaps frame numbers by fps/afr, and track ids are reassigned per format),
so asserting them would produce false failures, not real ones.

A compact summary table is printed once at the end of the run (visible in CI,
which passes ``-s``).
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest

from datamaite.conversion import convert
from datamaite.loaders import load_mot
from datamaite.model import BoxTrackDataset

pytestmark = pytest.mark.integration

# input format -> (dataset dir relative to DATAMAITE_DATASETS_ROOT,
#                  expected sequence_count, expected num_boxes)
# Counts are pinned to the example-data repo's MANIFEST.md.
MOT_INPUTS: dict[str, tuple[str, int, int]] = {
    "hmie": ("hmie/valid", 3, 30),
    "motchallenge": ("motchallenge/valid", 1, 5),
    "visdrone_video": ("visdrone/valid", 1, 3),
    "tao": ("tao/valid", 1, 2),
    "flat_mp4": ("flat_mp4/valid", 2, 0),
}

# Output formats that produce a reloadable BoxTrackDataset.
BOX_OUTPUTS = ["motchallenge", "tao", "visdrone_video", "hmie"]

# Inputs that actually carry boxes (flat_mp4 has none -> excluded from the
# conversion matrix; still covered by the load test above).
BOX_INPUTS = [name for name, (_, _, boxes) in MOT_INPUTS.items() if boxes > 0]

BBox = tuple[float, float, float, float]

# (in_fmt, out_fmt, seqs, boxes, geometry_preserved) rows for the end-of-run
# report. Module-level so the session fixture can render one table at teardown.
_REPORT: list[tuple[str, str, int, int, bool]] = []


def _bbox_multiset(dataset: BoxTrackDataset) -> Counter[BBox]:
    """Multiset of rounded ``(left, top, width, height)`` over every box.

    The conversion-stable invariant: box geometry must survive a round-trip
    even when frame numbering or track ids are reassigned by the target format.
    """
    counts: Counter[BBox] = Counter()
    for seq in dataset.sequences:
        for box in seq.boxes:
            left, top, width, height = box.bbox
            counts[(round(left, 1), round(top, 1), round(width, 1), round(height, 1))] += 1
    return counts


@pytest.fixture(scope="session")
def datasets_root() -> Path:
    """Path to the example-data ``datasets/`` dir, or skip when unavailable."""
    root = os.environ.get("DATAMAITE_DATASETS_ROOT")
    if not root:
        pytest.skip("DATAMAITE_DATASETS_ROOT not set")
    path = Path(root)
    if not path.is_dir():
        pytest.skip(f"DATAMAITE_DATASETS_ROOT does not exist: {path}")
    return path


@pytest.fixture(scope="session", autouse=True)
def _matrix_report() -> Iterator[None]:
    """Print one compact summary table after the matrix runs (CI passes -s)."""
    yield
    if not _REPORT:
        return
    cells = {(i, o): ok for i, o, _, _, ok in _REPORT}
    meta = {i: (s, b) for i, o, s, b, _ in _REPORT}
    line = "=" * 78
    print(f"\n{line}\nexample-data conversion matrix: counts + box-geometry preservation\n{line}")
    print(f"{'input (seqs/boxes)':24s}" + "".join(f"{'-> ' + o:>18s}" for o in BOX_OUTPUTS))
    for in_fmt in BOX_INPUTS:
        if in_fmt not in meta:
            continue
        seqs, boxes = meta[in_fmt]
        row = "".join(f"{('geom OK' if cells.get((in_fmt, o)) else 'FAIL'):>18s}" for o in BOX_OUTPUTS)
        print(f"{f'{in_fmt} ({seqs}/{boxes})':24s}{row}")
    passed = sum(1 for v in cells.values() if v)
    print("-" * 78)
    print(f"{passed}/{len(cells)} conversions preserved counts + bounding-box geometry (L,T,W,H).")
    print("not asserted: JSON hash, frame-index numbering, track-id values (format-specific).")
    print(line)


@pytest.mark.parametrize("fmt", list(MOT_INPUTS), ids=list(MOT_INPUTS))
def test_input_loads_with_expected_counts(datasets_root: Path, fmt: str) -> None:
    """Each input format loads from the dataset data with its pinned counts."""
    rel, exp_seqs, exp_boxes = MOT_INPUTS[fmt]
    dataset = load_mot(datasets_root / rel, dataset_format=fmt)
    assert dataset.sequence_count == exp_seqs, f"{fmt}: sequence_count {dataset.sequence_count} != {exp_seqs}"
    assert dataset.num_boxes == exp_boxes, f"{fmt}: num_boxes {dataset.num_boxes} != {exp_boxes}"


@pytest.mark.parametrize("out_fmt", BOX_OUTPUTS, ids=lambda v: f"to-{v}")
@pytest.mark.parametrize("in_fmt", BOX_INPUTS, ids=lambda v: f"from-{v}")
def test_conversion_preserves_counts_and_geometry(
    datasets_root: Path, tmp_path: Path, in_fmt: str, out_fmt: str
) -> None:
    """input -> convert -> reload preserves counts AND bounding-box geometry."""
    rel, exp_seqs, exp_boxes = MOT_INPUTS[in_fmt]
    source_geometry = _bbox_multiset(load_mot(datasets_root / rel, dataset_format=in_fmt))

    dest = tmp_path / f"{in_fmt}__to__{out_fmt}"
    convert(datasets_root / rel, dest, input_format=in_fmt, output_format=out_fmt)
    reloaded = load_mot(dest, dataset_format=out_fmt)

    geometry_ok = _bbox_multiset(reloaded) == source_geometry
    _REPORT.append((in_fmt, out_fmt, reloaded.sequence_count, reloaded.num_boxes, geometry_ok))

    assert reloaded.num_boxes == exp_boxes, f"{in_fmt}->{out_fmt}: box count {reloaded.num_boxes} != {exp_boxes}"
    assert reloaded.sequence_count == exp_seqs, (
        f"{in_fmt}->{out_fmt}: seq count {reloaded.sequence_count} != {exp_seqs}"
    )
    assert geometry_ok, f"{in_fmt}->{out_fmt}: bounding-box geometry (L,T,W,H) not preserved through round-trip"
