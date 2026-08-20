"""Package version.

The version string is derived from the git tag at build time
(uv-dynamic-versioning; the tag is the single source of truth, #85) and baked
into the distribution metadata, which we resolve at runtime via
``importlib.metadata``. That covers every install path -- wheels, sdists,
``uv sync``, and ``pip install -e .`` (as of the install; an editable install's
metadata goes stale until reinstalled).

Two corners report a placeholder instead. ``fallback-version`` (``0.0.0``) is
baked when the backend cannot use git at all, e.g. building from a tarball
export; and an uninstalled import (``PYTHONPATH=src``) has no metadata to read,
which reports ``0.0.0+unknown``. Neither is ever a real release: the tag
pipeline only accepts canonical ``X.Y.Z`` tags and ``0.0.0`` is never tagged.

There used to be a dunamai-based re-derivation from git here for the
placeholder case. It was load-bearing under Poetry, whose develop-install
registered ``0.0.0`` on every developer machine, but #60 moved installs to
``uv sync``, which builds through the backend. It is gone because it could no
longer fire: a checkout that can reach git gets a real dunamai-serialized
version baked in (a tagless clone yields ``0.0.0.postN.devN+<sha>``, not
``0.0.0``), and a tree where git is unusable is equally unusable at runtime.
Worse, reading git relative to this file meant a ``0.0.0``-metadata install
sitting inside an unrelated repository reported *that* repository's tag.

Exposes ``__version__`` and ``__version_tuple__``.
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
