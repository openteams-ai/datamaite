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

    Existing path objects pass through after the same protocol validation used
    for URL strings. URL inputs become configured ``UPath`` instances, except
    ``file://`` URLs, which become native ``Path`` objects to preserve the local
    fast path. Plain strings become ``pathlib.Path``.

    A URL string with embedded credentials (``scheme://user:pass@host/...``)
    is rejected: credentials belong in ``storage_options``, never the URL.
    """
    if not isinstance(root, str):
        protocol = getattr(root, "protocol", "")
        if isinstance(protocol, (tuple, list)):
            protocol = protocol[0] if protocol else ""
        if protocol not in _LOCAL_PROTOCOLS:
            _validate_url(str(root))
            if storage_options:
                from upath import UPath

                return cast(Path, UPath(str(root), **_filesystem_options(str(protocol), storage_options)))
        return cast(Path, root)
    if "://" in root:
        import urllib.parse

        from upath import UPath

        split = _validate_url(root)
        if split.scheme == "file":
            if split.netloc not in ("", "localhost"):
                raise ValueError("file:// dataset roots must refer to the local host")
            return Path(urllib.parse.unquote(split.path))

        # universal-pathlib >= 0.3 bases remote UPath on pathlib_abc rather
        # than pathlib.Path, but it implements the full pathlib surface. The
        # cast localizes that difference here so downstream code keeps plain
        # ``Path`` annotations for both local and remote roots.
        return cast(Path, UPath(root, **_filesystem_options(split.scheme, storage_options)))
    return Path(root)


def _filesystem_options(protocol: str, storage_options: Mapping[str, Any] | None) -> dict[str, Any]:
    """Translate datamaite stream options before constructing a filesystem."""
    options = dict(storage_options or {})
    if protocol == "s3" and "block_size" in options:
        options.setdefault("default_block_size", options.pop("block_size"))
    return options


def _validate_url(value: str) -> Any:
    """Validate a URL's protocol and require credentials outside the URI."""
    import urllib.parse

    split = urllib.parse.urlsplit(value)
    if split.username is not None or split.password is not None or split.query or split.fragment:
        raise ValueError(
            "dataset root URL must not embed credentials, query parameters, or fragments; pass credentials via "
            "storage_options (e.g. storage_options={'key': ..., 'secret': ...}) instead"
        )
    if split.scheme not in _ALLOWED_URL_SCHEMES:
        allowed = ", ".join(sorted(_ALLOWED_URL_SCHEMES))
        raise ValueError(f"unsupported dataset root scheme {split.scheme!r}; allowed schemes are: {allowed}")
    return split


def sanitized_uri(value: str | Path) -> str:
    """Return a URI safe for diagnostics, without userinfo, query, or fragment."""
    import urllib.parse

    text = str(value)
    if "://" not in text:
        return text
    split = urllib.parse.urlsplit(text)
    host = split.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{split.port}" if split.port is not None else host
    return urllib.parse.urlunsplit((split.scheme, netloc, split.path, "", ""))


def is_remote_path(path: Path) -> bool:
    """True when ``path`` lives on a non-local fsspec filesystem."""
    return getattr(path, "protocol", "") not in _LOCAL_PROTOCOLS


def storage_options_for(path: Path) -> dict[str, Any]:
    """Return the fsspec options bound to ``path`` without exposing them in records.

    Local :class:`pathlib.Path` objects have no options and return an empty
    mapping. Remote UPaths retain the options used to construct their fsspec
    filesystem; datasets keep this mapping as private runtime state so a public
    string ``path_or_uri`` can be reopened lazily after loading.

    """
    return dict(getattr(path, "storage_options", {}) or {})


def local_open_target(path: Path) -> str:
    """Plain filesystem string for handing a local path to non-pathlib APIs.

    ``av.open`` (PyAV) accepts filesystem paths only. Plain paths
    stringify as-is; a ``file://`` UPath must be stripped to its bare path
    (``UPath.path``) because ``str()`` would keep the URL scheme.
    """
    return str(getattr(path, "path", path))
