"""Packaging contract tests."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


def test_all_extra_is_union_of_task_extras() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]

    task_extras = ("fmv", "od", "ic")
    task_union = {dependency for name in task_extras for dependency in extras[name]}

    assert set(extras["all"]) == task_union
