"""Package version.

The version string is derived from the git tag at build time
(uv-dynamic-versioning; the tag is the single source of truth, #85) and baked
into the distribution metadata, which we resolve at runtime via
``importlib.metadata``. That is accurate for anything installed through the
hatchling backend -- wheels, sdists, and ``pip install -e .`` (as of the
install; an editable install's metadata goes stale until reinstalled).

Poetry is the exception: ``poetry install`` registers the root project with
its own develop-install machinery, which reads the literal
``[tool.poetry].version`` placeholder (``0.0.0``) instead of building through
the backend. When we see that placeholder (or no metadata at all), we fall
back to deriving the version from git via dunamai -- the same engine
uv-dynamic-versioning uses, available in the ``dev`` extra -- so ordinary
Poetry development environments still report the SCM-derived version. Without
dunamai or a git checkout, the placeholder is reported as-is.

Exposes ``__version__`` and ``__version_tuple__``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version

__all__ = ["__version__", "__version_tuple__"]

# What Poetry's develop-install registers ([tool.poetry].version) and what
# uv-dynamic-versioning falls back to outside a git checkout (fallback-version
# in pyproject.toml). Never a real release: the tag pipeline only accepts
# canonical X.Y.Z tags and 0.0.0 is never tagged.
_PLACEHOLDER = "0.0.0"


def _from_scm() -> str | None:
    """Derive the version from git, mirroring the build backend.

    Returns None when dunamai is not installed (it ships in the ``dev``
    extra) or the package is not running from a git checkout.
    """
    try:
        from dunamai import Pattern, Version
    except ImportError:
        return None
    try:
        from pathlib import Path

        return Version.from_git(
            pattern=Pattern.DefaultUnprefixed,
            path=Path(__file__).resolve().parent,
        ).serialize()
    except Exception:
        return None


def _resolve() -> str:
    try:
        installed: str | None = _metadata_version("datamaite")
    except PackageNotFoundError:
        installed = None
    if installed is not None and installed != _PLACEHOLDER:
        return installed
    return _from_scm() or installed or f"{_PLACEHOLDER}+unknown"


def _as_tuple(v: str) -> tuple[int | str, ...]:
    return tuple(int(piece) if piece.isdecimal() else piece for piece in v.replace("+", ".").split("."))


__version__: str = _resolve()
__version_tuple__: tuple[int | str, ...] = _as_tuple(__version__)
