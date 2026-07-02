"""Package version.

The version string is declared statically in ``[project].version`` in
``pyproject.toml`` (the single source of truth) and baked into the
distribution metadata at build time. Here we resolve it at runtime via
``importlib.metadata`` so ``datamaite.__version__`` always reflects the
installed package, including editable installs. Exposes ``__version__``
and ``__version_tuple__``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version

__all__ = ["__version__", "__version_tuple__"]


def _resolve() -> str:
    try:
        return _metadata_version("datamaite")
    except PackageNotFoundError:
        return "0.0.0+unknown"


def _as_tuple(v: str) -> tuple[int | str, ...]:
    return tuple(int(piece) if piece.isdecimal() else piece for piece in v.replace("+", ".").split("."))


__version__: str = _resolve()
__version_tuple__: tuple[int | str, ...] = _as_tuple(__version__)
