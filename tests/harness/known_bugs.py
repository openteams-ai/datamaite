"""Known bugs, and the check that a cell's differences are all explained by them."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass

import pytest

#: Prefixes of the cross-checks a cell with a ``relabels_only`` bug adds: the
#: comparison with classes left out, and the class names on appended images.
IGNORING_CLASSES = "ignoring classes: "
APPENDED_IMAGES = "appended images: "
CROSS_CHECKS = (IGNORING_CLASSES, APPENDED_IMAGES)


@dataclass(frozen=True)
class KnownBug:
    """A known bug: the cells it breaks and the differences it causes there.

    A cell with known bugs is an expected failure only if every difference it
    reports matches one of its bugs' ``differences`` patterns (regular
    expressions, searched in each difference), and each of its bugs explains at
    least one difference. Any other difference fails the cell, so a new
    regression in a cell with a known bug is still caught. A cell that no longer
    shows one of its bugs fails too, until the cell is removed from ``cells``.

    A bug that only gives boxes the wrong class sets ``relabels_only``: its
    cells are also compared with every class left out, and nothing may differ
    there, so a box that is lost or moved is not taken for a relabelled one.
    """

    issue: str
    summary: str
    cells: tuple[str, ...]  # fnmatch patterns over cell ids
    differences: tuple[str, ...]
    relabels_only: bool = False

    @property
    def reason(self) -> str:
        return f"{self.issue}: {self.summary}"

    def applies_to(self, cell_id: str) -> bool:
        return any(fnmatch.fnmatchcase(cell_id, pattern) for pattern in self.cells)

    def explains(self, difference: str) -> bool:
        """Whether one of ``differences`` matches.

        A cross-check's difference is matched without its prefix, so anchored
        patterns apply to it too, except that a relabelling bug never explains
        one: the cross-checks exist to catch what it can't.
        """
        for prefix in CROSS_CHECKS:
            if difference.startswith(prefix):
                if self.relabels_only:
                    return False
                difference = difference.removeprefix(prefix)
                break
        return any(re.search(pattern, difference) for pattern in self.differences)


def check_differences(differences: list[str], bugs: list[KnownBug]) -> None:
    """Fail on any difference the cell's known bugs don't explain; mark the cell xfail if they explain all."""
    unexplained = [d for d in differences if not any(bug.explains(d) for bug in bugs)]
    if unexplained:
        lines = "\n".join(f"  - {difference}" for difference in unexplained)
        known = f" (not explained by its known bugs: {'; '.join(b.reason for b in bugs)})" if bugs else ""
        raise AssertionError(f"{len(unexplained)} difference(s) from the expected dataset{known}:\n{lines}")
    fixed = [bug for bug in bugs if not any(bug.explains(d) for d in differences)]
    if fixed:
        raise AssertionError(
            "known bug no longer reproduces here; remove this cell from its KNOWN_BUGS entry: "
            + "; ".join(bug.reason for bug in fixed)
        )
    if bugs:
        pytest.xfail("; ".join(bug.reason for bug in bugs))
