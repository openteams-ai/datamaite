"""Check that a written dataset does not record where its source was (#157).

A dataset written on one machine should not contain absolute paths from that
machine: they expose local user names and folder layouts, and they make the
same conversion give different files on different machines.
:func:`assert_no_source_paths` scans the text files a writer produced for the
absolute path of a local folder, usually the one the source was loaded from.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Files a writer produces as text. Everything else (images, video, parquet) is skipped.
TEXT_SUFFIXES = frozenset({".json", ".jsonl", ".yaml", ".yml", ".txt", ".csv", ".xml", ".ini"})


def source_path_forms(source_dir: Path | str) -> list[str]:
    """The spellings of ``source_dir`` a loader may have stored: as given (made absolute) and resolved.

    On macOS the temporary folders live under ``/var``, which is a link to
    ``/private/var``, so both spellings are checked.
    """
    given = Path(source_dir).absolute()
    resolved = given.resolve()
    forms = {given.as_posix(), resolved.as_posix()}
    unlinked = {form[len("/private") :] for form in forms if form.startswith("/private/")}
    forms |= {form for form in unlinked if Path(form).resolve() == resolved}
    return sorted(forms, key=len, reverse=True)


def find_source_paths(written_dir: Path | str, source_dir: Path | str) -> list[str]:
    """Every place under ``written_dir`` whose text contains the absolute path of ``source_dir``."""
    pattern = _pattern(source_path_forms(source_dir))
    found = []
    root = Path(written_dir)
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if not pattern.search(text):
            continue
        name = path.relative_to(root).as_posix()
        found += [f"{name}: {where}" for where in _places(path.suffix.lower(), text, pattern)]
    return found


def assert_no_source_paths(written_dir: Path | str, source_dir: Path | str) -> None:
    """Fail if any text file under ``written_dir`` contains the absolute path of ``source_dir`` (#157)."""
    found = find_source_paths(written_dir, source_dir)
    if found:
        lines = "\n".join(f"  - {place}" for place in found)
        raise AssertionError(f"written dataset contains the absolute source path {Path(source_dir)}:\n{lines}")


def _pattern(forms: list[str]) -> re.Pattern[str]:
    # A path matches only as a whole folder name: "/data/src" matches "/data/src" and
    # "/data/src/a.png" but not "/data/src2". JSON may escape "/" as "\/".
    spellings = [form for path in forms for form in (path, path.replace("/", "\\/"))]
    alternatives = "|".join(re.escape(spelling) for spelling in sorted(set(spellings), key=len, reverse=True))
    return re.compile(rf"(?:{alternatives})(?![\w.-])")


def _places(suffix: str, text: str, pattern: re.Pattern[str]) -> list[str]:
    """Readable locations of each match: the JSON field when the file parses, else the line."""
    if suffix == ".json":
        try:
            return [f"{field} = {value!r}" for field, value in _json_hits(json.loads(text), "", pattern)]
        except ValueError:
            pass
    places = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not pattern.search(line):
            continue
        if suffix == ".jsonl":
            try:
                hits = list(_json_hits(json.loads(line), "", pattern))
            except ValueError:
                hits = []
            if hits:
                places += [f"line {number}: {field} = {value!r}" for field, value in hits]
                continue
        places.append(f"line {number}: {line.strip()[:200]}")
    return places


def _json_hits(value: Any, field: str, pattern: re.Pattern[str]) -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and pattern.search(key):
                yield (f"{field} (key)" if field else "(key)", key)
            yield from _json_hits(item, f"{field}.{key}" if field else str(key), pattern)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _json_hits(item, f"{field}[{index}]", pattern)
    elif isinstance(value, str) and pattern.search(value):
        yield (field or "(value)", value)
