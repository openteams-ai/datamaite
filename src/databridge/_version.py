"""Package version.

This is a committed fallback so that editable installs work before any
build tool has run. At wheel/sdist build time, ``hatch-vcs`` (see
``[tool.hatch.version]`` in ``pyproject.toml``) overwrites this file
with the version derived from git tags. Both paths expose
``__version__`` and ``__version_tuple__``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version

__all__ = ["__version__", "__version_tuple__"]


def _resolve() -> str:
    try:
        return _metadata_version("databridge")
    except PackageNotFoundError:
        return "0.0.0+unknown"


def _as_tuple(v: str) -> tuple[int | str, ...]:
    parts: list[int | str] = []
    for piece in v.replace("+", ".").split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(piece)
    return tuple(parts)


__version__: str = _resolve()
__version_tuple__: tuple[int | str, ...] = _as_tuple(__version__)
