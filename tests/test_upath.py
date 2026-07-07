"""Tests for local/cloud dataset-root coercion helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from upath import UPath

from datamaite._upath import is_remote_path, local_open_target, to_dataset_path


def test_local_string_stays_plain_pathlib() -> None:
    p = to_dataset_path("/data/hmie")
    assert isinstance(p, Path)
    assert not is_remote_path(p)
    # Must be a plain pathlib class, not UPath: the local pipeline stays
    # byte-for-byte identical to the pre-cloud implementation.
    assert not isinstance(p, UPath)


def test_path_instance_passes_through_unchanged() -> None:
    src = Path("/data/hmie")
    assert to_dataset_path(src) is src


def test_upath_instance_passes_through_unchanged() -> None:
    src = UPath("memory://bucket/data")
    assert to_dataset_path(src) is src


def test_url_string_becomes_upath() -> None:
    p = to_dataset_path("memory://bucket/data")
    assert isinstance(p, UPath)
    assert is_remote_path(p)


def test_storage_options_are_threaded() -> None:
    p = to_dataset_path("memory://bucket/data", {"some_option": 1})
    assert isinstance(p, UPath)
    assert p.storage_options.get("some_option") == 1


def test_file_url_is_not_remote() -> None:
    p = to_dataset_path("file:///data/hmie")
    assert not is_remote_path(p)


@pytest.mark.parametrize("url", ["s3://bucket/data", "gs://bucket/data", "az://container/data", "memory://b/data"])
def test_allowed_schemes_become_upath(url: str) -> None:
    p = to_dataset_path(url)
    assert isinstance(p, UPath)


@pytest.mark.parametrize("url", ["http://evil.example/x", "https://evil.example/x", "ftp://host/x"])
def test_disallowed_scheme_is_rejected(url: str) -> None:
    # http/ftp/arbitrary fsspec schemes are an SSRF surface (aiohttp arrives
    # via the aws extra), so a string root outside the allowlist must raise.
    with pytest.raises(ValueError, match="unsupported dataset root scheme"):
        to_dataset_path(url)


def test_url_embedded_credentials_are_rejected() -> None:
    with pytest.raises(ValueError, match="storage_options"):
        to_dataset_path("s3://user:pass@bucket/data")


def test_local_open_target_plain_path() -> None:
    assert local_open_target(Path("/data/v.mp4")) == "/data/v.mp4"


def test_local_open_target_strips_file_scheme() -> None:
    assert local_open_target(UPath("file:///data/v.mp4")) == "/data/v.mp4"
