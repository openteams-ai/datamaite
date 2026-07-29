"""Check for, or extract, one version's section of CHANGELOG.md.

The release pipeline's single source of release notes (#85):

* ``python scripts/release_notes.py check X.Y.Z`` -- exit non-zero unless
  CHANGELOG.md has a non-empty ``## [X.Y.Z]`` section. Run first in every tag
  pipeline so a missing changelog entry fails the release before anything is
  built or published.
* ``python scripts/release_notes.py extract X.Y.Z`` -- print the section body
  to stdout. The ``gitlab-release`` job uses this as the GitLab Release
  description.

Sections follow Keep a Changelog headings: ``## [X.Y.Z] - YYYY-MM-DD``.
This file is deliberately dependency-free so release jobs can run it on a
bare ``python:*-slim`` image, and self-contained so sibling projects
(modelmaite) can copy it verbatim.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"


def section_for(version: str, changelog_text: str) -> str | None:
    """Return the body of ``## [version]``, or None if absent/empty."""
    match = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)",
        changelog_text,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        return None
    body = match.group(1).strip()
    return body or None


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in {"check", "extract"}:
        print(f"usage: {argv[0]} check|extract VERSION", file=sys.stderr)
        return 2
    mode, version = argv[1], argv[2].strip()

    body = section_for(version, CHANGELOG.read_text(encoding="utf-8"))
    if body is None:
        print(
            f"CHANGELOG.md has no non-empty '## [{version}]' section. "
            "Merge a changelog MR for this release before tagging (or re-tag after it merges).",
            file=sys.stderr,
        )
        return 1

    if mode == "extract":
        print(body)
    else:
        print(f"CHANGELOG.md has a non-empty section for {version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
