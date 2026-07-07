"""Path coercion helpers for local and cloud dataset roots.

datamaite accepts dataset roots either as local filesystem paths or as
cloud object-storage URLs (``s3://...``, ``gs://...``, ``az://...``).
Local inputs stay plain :class:`pathlib.Path` so the local pipeline is
byte-for-byte unchanged; URL inputs become :class:`upath.UPath` backed by
the matching fsspec filesystem, with ``storage_options`` (credentials,
endpoint overrides, ...) forwarded to the filesystem constructor.
``UPath`` implements the pathlib interface (duck-typed; local paths remain
real ``pathlib.Path``), so every downstream ``Path`` annotation and
operation keeps working for both — the casts in :func:`to_dataset_path` are
the single place that difference is papered over.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from upath import UPath

# Protocols that resolve to the local filesystem. UPath reports these for
# plain paths and file:// URLs; anything else is remote object storage.
_LOCAL_PROTOCOLS = frozenset({"", "file", "local"})

# URL schemes accepted from *string* roots. s3 (s3fs), gs (gcsfs), and the
# adlfs family (az/abfs/abfss) are the supported object stores; memory is the
# documented test/tutorial backend; file stays local-classified. Anything
# else -- http/ftp/arbitrary fsspec schemes -- is rejected: aiohttp arrives
# transitively with the aws extra, so an http:// root would otherwise turn a
# "validate this path" call into a live outbound fetch (an SSRF surface).
_ALLOWED_URL_SCHEMES = frozenset({"s3", "gs", "az", "abfs", "abfss", "memory", "file"})


def to_dataset_path(root: str | Path | UPath, storage_options: Mapping[str, Any] | None = None) -> Path:
    """Coerce a dataset root to a concrete path object.

    ``Path`` inputs (including ``UPath``) pass through unchanged -- this is a
    deliberate power-user seam: constructing an ``http://`` or ``memory://``
    UPath directly (as ``tools/probe_bench`` and unit fixtures do) bypasses
    the string-scheme allowlist below. String inputs with a URL scheme are
    validated against ``_ALLOWED_URL_SCHEMES`` and become ``UPath`` with
    ``storage_options`` applied; plain string paths become ``pathlib.Path``.

    A URL string with embedded credentials (``scheme://user:pass@host/...``)
    is rejected: credentials belong in ``storage_options``, never the URL.
    """
    if not isinstance(root, str):
        return cast(Path, root)
    if "://" in root:
        import urllib.parse

        from upath import UPath

        split = urllib.parse.urlsplit(root)
        if split.username is not None or split.password is not None:
            raise ValueError(
                "dataset root URL must not embed credentials; pass them via "
                "storage_options (e.g. storage_options={'key': ..., 'secret': ...}) instead"
            )
        if split.scheme not in _ALLOWED_URL_SCHEMES:
            allowed = ", ".join(sorted(_ALLOWED_URL_SCHEMES))
            raise ValueError(f"unsupported dataset root scheme {split.scheme!r}; allowed schemes are: {allowed}")

        # universal-pathlib >= 0.3 bases remote UPath on pathlib_abc rather
        # than pathlib.Path, but it implements the full pathlib surface. The
        # cast localizes that difference here so downstream code keeps plain
        # ``Path`` annotations for both local and remote roots.
        return cast(Path, UPath(root, **dict(storage_options or {})))
    return Path(root)


def is_remote_path(path: Path) -> bool:
    """True when ``path`` lives on a non-local fsspec filesystem."""
    return getattr(path, "protocol", "") not in _LOCAL_PROTOCOLS


def local_open_target(path: Path) -> str:
    """Plain filesystem string for handing a local path to non-pathlib APIs.

    ``av.open`` (PyAV) accepts filesystem paths only. Plain paths
    stringify as-is; a ``file://`` UPath must be stripped to its bare path
    (``UPath.path``) because ``str()`` would keep the URL scheme.
    """
    return str(getattr(path, "path", path))
