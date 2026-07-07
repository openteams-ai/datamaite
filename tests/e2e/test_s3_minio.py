"""S3 end-to-end tier: datamaite against a real S3-API object storage server.

Unlike ``tests/test_cloud_integration.py`` (opt-in, but pointed at whatever
cloud bucket the caller already has) and the coverage-gated unit suite's cloud
tests (``tests/test_loaders.py`` et al., which exercise the cloud code paths
against fsspec's in-process ``memory://`` filesystem so they stay hermetic and
count toward the 90% coverage gate), this module talks to an actual S3 API
server -- a MinIO container -- over the network. It proves the ranged-read
video streaming and the ``storage_options``-driven credential/endpoint plumbing
work against real S3 semantics, not just against a filesystem that happens to
share fsspec's interface.

This tier runs with coverage disabled (``--no-cov``) and is never part of the
default ``pytest`` invocation: it is opt-in via the ``integration`` marker
(deselected by default, see ``pyproject.toml``'s ``addopts``) and self-skips
unless ``DATAMAITE_S3_E2E_ENDPOINT`` / ``DATAMAITE_S3_E2E_KEY`` /
``DATAMAITE_S3_E2E_SECRET`` are set. See ``tests/README.md`` for how to run it
locally and how CI wires it up against a MinIO service container.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from datamaite import load_mot, validate
from tests._hmie_factory import (
    AnnotationSpec,
    FullVideoSpec,
    SnippetSpec,
    VideoSpec,
    make_annotation_dict,
    make_hmie_dataset,
    single_video_dataset,
)

try:
    import s3fs
except ImportError:  # pragma: no cover - exercised only when the aws extra is missing
    s3fs = None

_ENDPOINT = os.environ.get("DATAMAITE_S3_E2E_ENDPOINT")
_KEY = os.environ.get("DATAMAITE_S3_E2E_KEY")
_SECRET = os.environ.get("DATAMAITE_S3_E2E_SECRET")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(s3fs is None, reason="s3fs not installed (pip install datamaite[aws])"),
    pytest.mark.skipif(
        not (_ENDPOINT and _KEY and _SECRET),
        reason="DATAMAITE_S3_E2E_ENDPOINT / DATAMAITE_S3_E2E_KEY / DATAMAITE_S3_E2E_SECRET not set",
    ),
]


@pytest.fixture(scope="session")
def storage_options() -> dict[str, Any]:
    """s3fs/boto ``storage_options`` built from the env-gated endpoint + credentials."""
    return {
        "key": _KEY,
        "secret": _SECRET,
        "client_kwargs": {"endpoint_url": _ENDPOINT},
    }


@pytest.fixture(scope="session")
def s3_fs(storage_options: dict[str, Any]) -> Any:
    return s3fs.S3FileSystem(**storage_options)


@pytest.fixture(scope="session")
def bucket(s3_fs: Any) -> Iterator[str]:
    """A uniquely-named bucket for this test session so reruns never collide."""
    name = f"datamaite-e2e-{uuid.uuid4().hex[:12]}"
    s3_fs.mkdir(name)
    yield name
    with contextlib.suppress(OSError):
        s3_fs.rm(name, recursive=True)


def _upload_tree(fs: Any, local_root: Path, bucket_name: str, prefix: str) -> None:
    """Upload every file under a factory-built local HMIE tree to s3://<bucket>/<prefix>/..."""
    for path in sorted(local_root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(local_root).as_posix()
            fs.put_file(str(path), f"{bucket_name}/{prefix}/{rel}")


def test_load_streams_video_probes(tmp_path: Path, s3_fs: Any, bucket: str, storage_options: dict[str, Any]) -> None:
    """load_mot(..., require_video=True) streams real ranged-read probes over S3."""
    local_root = tmp_path / "batch"
    single_video_dataset(
        local_root,
        [
            SnippetSpec(name=f"video_001_00000{i}", source_designator="SRC1", hash_suffix=f"abc00{i}")
            for i in range(1, 5)
        ],
    )
    _upload_tree(s3_fs, local_root, bucket, "load-probe")

    dataset = load_mot(
        f"s3://{bucket}/load-probe",
        dataset_format="hmie",
        require_video=True,
        storage_options=storage_options,
    )

    assert dataset.sequence_count == 4
    assert all(seq.num_frames_exact for seq in dataset.sequences)


def test_validate_full_integrity_parallel(
    tmp_path: Path, s3_fs: Any, bucket: str, storage_options: dict[str, Any]
) -> None:
    """validate(..., workers=4) runs a real worker process pool against S3 with no crashes."""
    local_root = tmp_path / "batch"
    single_video_dataset(
        local_root,
        [
            SnippetSpec(name=f"video_002_00000{i}", source_designator="SRC1", hash_suffix=f"int00{i}")
            for i in range(1, 5)
        ],
        video_name="video_002_000000",
    )
    _upload_tree(s3_fs, local_root, bucket, "validate-integrity")

    result = validate(f"s3://{bucket}/validate-integrity", workers=4, storage_options=storage_options)

    assert result.passed
    assert result.annotation_count == 4
    assert result.finding_counts.get("worker_crash", 0) == 0
    assert all(str(finding.path).startswith("s3://") for finding in result.findings)


def test_validate_flags_corrupt_video(tmp_path: Path, s3_fs: Any, bucket: str, storage_options: dict[str, Any]) -> None:
    """A corrupt video uploaded to S3 is caught as a video_open finding with a logical s3:// path."""
    local_root = tmp_path / "batch"
    single_video_dataset(
        local_root,
        [
            SnippetSpec(
                name="video_003_000001",
                source_designator="SRC1",
                hash_suffix="corrupt1",
                video=VideoSpec(corrupt=True),
            )
        ],
        video_name="video_003_000000",
    )
    _upload_tree(s3_fs, local_root, bucket, "validate-corrupt")

    result = validate(f"s3://{bucket}/validate-corrupt", workers=1, storage_options=storage_options)

    assert not result.passed
    video_open_findings = [finding for finding in result.findings if finding.check == "video_open"]
    assert video_open_findings
    assert all(str(finding.path).startswith("s3://") for finding in video_open_findings)


def test_validate_batches_stress(tmp_path: Path, s3_fs: Any, bucket: str, storage_options: dict[str, Any]) -> None:
    """12 snippets across 2 full-video dirs validate cleanly under a 4-worker pool.

    Video specs stay at the factory default (30 frames) so the whole run stays
    fast (~1-2 min) in CI.
    """
    local_root = tmp_path / "batch"
    make_hmie_dataset(
        local_root,
        [
            FullVideoSpec(
                name="video_004_000000",
                snippets=[
                    SnippetSpec(name=f"video_004_00000{i}", source_designator="SRC1", hash_suffix=f"s4{i:02d}")
                    for i in range(1, 7)
                ],
            ),
            FullVideoSpec(
                name="video_005_000000",
                snippets=[
                    SnippetSpec(
                        name=f"video_005_00000{i}",
                        source_designator="SRC2",
                        hash_suffix=f"s5{i:02d}",
                        labeler="labeler_beta",
                    )
                    for i in range(1, 7)
                ],
            ),
        ],
    )
    _upload_tree(s3_fs, local_root, bucket, "validate-stress")

    result = validate(f"s3://{bucket}/validate-stress", workers=4, storage_options=storage_options)

    assert result.passed
    assert result.annotation_count == 12


def test_generated_large_video_roundtrip(
    tmp_path: Path, s3_fs: Any, bucket: str, storage_options: dict[str, Any]
) -> None:
    """A video from tools/probe_bench/gen_video.py round-trips through the S3 streaming loader.

    Chains the repo's own probe-benchmark tooling end to end: generate a
    noise-heavy ~45 MB / 600-frame clip -- deliberately larger than the
    factory's tiny gradient clips -- upload it, and load it back through
    datamaite's streaming probe path (``require_video=True``).
    """
    repo_root = Path(__file__).resolve().parents[2]
    snippet_dir = tmp_path / "genbatch" / "gen_001_000001"
    mp4_dir = snippet_dir / "seq_mp4"
    mp4_dir.mkdir(parents=True)
    video_path = mp4_dir / "gen_001_000001.mp4"
    # tools/ is not a package -- invoke the generator as a script, not an import.
    subprocess.run(  # noqa: S603 - fixed interpreter/repo-local script, no untrusted input
        [sys.executable, str(repo_root / "tools" / "probe_bench" / "gen_video.py"), str(video_path), "600"],
        check=True,
        cwd=repo_root,
    )

    annotation = make_annotation_dict(AnnotationSpec(video_fps=30.0), VideoSpec(num_frames=600, fps=30.0))
    ann_dir = snippet_dir / "labeler_a"
    ann_dir.mkdir()
    (ann_dir / "CDAO_SRC1_gen_001_000001.mp4_abc.json").write_text(json.dumps(annotation))

    _upload_tree(s3_fs, tmp_path / "genbatch", bucket, "genbatch")

    dataset = load_mot(
        f"s3://{bucket}/genbatch", dataset_format="hmie", require_video=True, storage_options=storage_options
    )

    assert dataset.sequence_count == 1
    seq = dataset.sequences[0]
    assert seq.num_frames_exact
    assert seq.num_frames == 600
    assert seq.fps == 30.0
    assert seq.video_path is not None
    assert seq.video_path.startswith("s3://")


class TestExampleDatasetParity:
    """Cloud (S3) results are identical to local-disk results for a real dataset.

    Complements the synthetic factory-built trees above with the shared
    example-data repo's HMIE dataset (see ``tests/e2e/test_example_data_matrix.py``),
    mirrored into the S3 service and compared against the same tree loaded
    straight off local disk. Gated on top of the module's S3 env-var skip by a
    second, independent skip: ``DATAMAITE_DATASETS_ROOT`` must be set and its
    ``hmie/valid`` dataset must exist. ``datasets/hmie/`` itself is not a
    single loadable dataset -- it's a directory of independent HMIE datasets
    that each isolate one ``validate()`` condition (see the example-data
    repo's ``MANIFEST.md``); ``valid`` is the happy-path one all the other
    e2e tests key off of.
    """

    @pytest.fixture(scope="session")
    def hmie_root(self) -> Path:
        root = os.environ.get("DATAMAITE_DATASETS_ROOT")
        if not root:
            pytest.skip("DATAMAITE_DATASETS_ROOT not set")
        path = Path(root) / "hmie" / "valid"
        if not path.is_dir():
            pytest.skip(f"example-data hmie/valid dataset not found: {path}")
        return path

    def test_s3_results_match_local(
        self, hmie_root: Path, s3_fs: Any, bucket: str, storage_options: dict[str, Any]
    ) -> None:
        """validate() and load_mot() agree between local disk and S3 for the real HMIE example dataset."""
        _upload_tree(s3_fs, hmie_root, bucket, "example-hmie")
        s3_root = f"s3://{bucket}/example-hmie"

        local_result = validate(hmie_root, workers=4)
        s3_result = validate(s3_root, workers=4, storage_options=storage_options)

        assert s3_result.passed == local_result.passed
        assert s3_result.annotation_count == local_result.annotation_count
        assert s3_result.snippet_count == local_result.snippet_count
        assert dict(s3_result.finding_counts) == dict(local_result.finding_counts)

        local_dataset = load_mot(hmie_root, dataset_format="hmie", require_video=True)
        if local_dataset.sequence_count == 0:
            pytest.skip(f"local load of {hmie_root} yielded zero sequences -- dataset layout changed upstream")

        s3_dataset = load_mot(s3_root, dataset_format="hmie", require_video=True, storage_options=storage_options)

        assert s3_dataset.sequence_count == local_dataset.sequence_count

        def _fingerprint(dataset: Any) -> list[tuple[str, int | None, float, int | None, int | None]]:
            return sorted(
                (Path(seq.annotation_path).name, seq.num_frames, seq.fps, seq.width, seq.height)
                for seq in dataset.sequences
            )

        assert _fingerprint(s3_dataset) == _fingerprint(local_dataset)
