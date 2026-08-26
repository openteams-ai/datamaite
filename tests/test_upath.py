"""Tests for local/cloud dataset-root coercion helpers."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from upath import UPath

from datamaite._io import open_video_source
from datamaite._upath import is_remote_path, local_open_target, sanitized_uri, to_dataset_path


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


def test_s3_stream_block_size_maps_to_filesystem_option() -> None:
    p = to_dataset_path("s3://bucket/data", {"block_size": 1 << 20})
    assert p.storage_options.get("default_block_size") == 1 << 20
    assert "block_size" not in p.storage_options


@pytest.mark.parametrize(
    ("options", "expected_block_size"),
    [
        ({"block_size": 16, "default_block_size": 32}, 16),
        ({"default_block_size": 32}, 32),
        ({}, 8 * 1024 * 1024),
    ],
)
def test_video_stream_block_size_precedence_and_option_removal(
    options: dict[str, int], expected_block_size: int, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    import datamaite._io as io_module

    resolver_calls: list[dict[str, object]] = []
    open_calls: list[dict[str, object]] = []

    class RemotePath:
        protocol = "memory"

        def open(self, _mode: str, **kwargs: object) -> io.BytesIO:
            open_calls.append(kwargs)
            return io.BytesIO(b"video")

    def resolve(_path, storage_options):  # type: ignore[no-untyped-def]
        resolver_calls.append(storage_options)
        return RemotePath()

    monkeypatch.setattr(io_module, "to_dataset_path", resolve)
    with open_video_source("memory://bucket/video.mp4", {**options, "marker": "kept"}) as stream:
        assert stream.read() == b"video"

    assert resolver_calls == [{"marker": "kept"}]
    assert open_calls == [{"block_size": expected_block_size}]


def test_video_stream_explicit_none_block_size_remains_invalid(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import datamaite._io as io_module

    class RemotePath:
        protocol = "memory"

    monkeypatch.setattr(io_module, "to_dataset_path", lambda *_args: RemotePath())
    with (
        pytest.raises(ValueError, match="positive integer"),
        open_video_source("memory://bucket/video.mp4", {"block_size": None}),
    ):
        pass


def test_file_url_uses_native_local_path() -> None:
    p = to_dataset_path("file:///data/hmie")
    assert isinstance(p, Path)
    assert not isinstance(p, UPath)
    assert p == Path("/data/hmie")
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


@pytest.mark.parametrize(
    "url",
    [
        "s3://user:pass@bucket/data",
        "az://container/data?sig=secret",
        "s3://bucket/data#credential-fragment",
    ],
)
def test_url_embedded_credentials_are_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="storage_options"):
        to_dataset_path(url)


def test_preconstructed_upath_uses_the_same_scheme_allowlist() -> None:
    with pytest.raises(ValueError, match="unsupported dataset root scheme"):
        to_dataset_path(UPath("http://127.0.0.1/private"))


def test_sanitized_uri_removes_credentials_and_query() -> None:
    value = sanitized_uri("https://user:pass@example.test:8443/data?token=secret#fragment")
    assert value == "https://example.test:8443/data"


def test_local_open_target_plain_path() -> None:
    assert local_open_target(Path("/data/v.mp4")) == "/data/v.mp4"


def test_local_open_target_strips_file_scheme() -> None:
    assert local_open_target(UPath("file:///data/v.mp4")) == "/data/v.mp4"
