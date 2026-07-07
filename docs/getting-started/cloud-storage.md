# Loading datasets from cloud object storage

datamaite can load and validate HMIE datasets directly from S3, Google
Cloud Storage, or Azure Blob Storage — no manual download step. Pass a
cloud URL wherever a dataset root path is accepted.

Cloud URLs are supported for **HMIE loading and validation only**. Other
format loaders (MOTChallenge, COCO, YOLO, ...) raise a clear error on a
cloud root; download the dataset and point them at a local path instead.

## Install the backend extra

Core datamaite ships the cloud plumbing (`fsspec`, `universal-pathlib`);
each provider's filesystem is an optional extra:

```bash
pip install "datamaite[aws]"     # s3://
pip install "datamaite[gcs]"     # gs://
pip install "datamaite[azure]"   # az://
pip install "datamaite[cloud]"   # all three
```

Video integrity checks additionally need the `fmv` extra (OpenCV + PyAV);
combine it with the backend, e.g.:

```bash
pip install "datamaite[aws,fmv]"
```

Without `fmv`, video checks are skipped: each video emits a
`video_dependency` WARNING and nothing is decoded, so validation can look
clean while never touching a single video byte. Install `fmv` whenever you
rely on the integrity findings.

## Load and validate with a cloud URL

```python
import datamaite

ds = datamaite.load_mot("s3://my-bucket/datasets/batch-a", dataset_format="hmie")

result = datamaite.validate("s3://my-bucket/datasets/batch-a", workers=8)
print(result.summary())
```

The CLI accepts the same URLs:

```bash
datamaite validate s3://my-bucket/datasets/batch-a --no-cache
```

## Credentials

Credentials resolve the same way as any fsspec application: provider
environment variables and config files (e.g. `AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY` for S3, `GOOGLE_APPLICATION_CREDENTIALS` for GCS)
work out of the box. For Azure, adlfs resolves credentials via its standard
mechanisms — connection strings or `DefaultAzureCredential`. To pass options
explicitly on any backend, use `storage_options`:

```python
result = datamaite.validate(
    "s3://my-bucket/datasets/batch-a",
    storage_options={"key": "...", "secret": "...", "client_kwargs": {"endpoint_url": "https://..."}},
)
```

The CLI has no credentials flag; configure the environment instead.

## How video integrity checks work on cloud data

Annotation (JSON) checks stream directly from object storage. Video
integrity checks stream too: the probe opens each remote video as a
seekable file object and decodes through PyAV over 1 MiB ranged reads, so
only the byte ranges it actually reads (the container header plus a handful
of sampled frames) are transferred. No full-file download, no temporary
files, no presigned URLs. In practice a probe transfers about 13 MB per
video regardless of file size (see the transport benchmark under
`tools/probe_bench/`), because the cost scales with the number of frames
sampled, not the length of the clip.

The same fsspec code path serves every backend, so behavior is identical on
S3, GCS, and Azure. S3 is exercised end-to-end in CI against a MinIO
service; GCS and Azure are supported but not yet CI-tested.

Validation findings always report the dataset's logical path (the
`s3://...` URL).

## Notes at scale

- The validation cache fingerprints each file by hashing its first 1 MB
  (plus size and mtime). Against a cloud root that means one small ranged
  read per annotation and video on every run — cache hits skip the decode
  and probe work, not the fingerprint read. That is usually a good trade
  for repeated validation of large datasets; use `--no-cache` (CLI) or
  `cache=None` (API) for one-off runs where even those reads are not
  worth it.
- `--skip-video-check` / `check_video_integrity=False` keeps validation
  JSON-only — fastest, and touches no video bytes on any backend. Note that
  a JSON-only load still issues one metadata (stat) request per video to
  record its `size_bytes`.
- The `workers` fan-out multiplies concurrent connections against the
  provider: N workers means up to N in-flight requests. Mind the provider's
  request-rate limits at high worker counts, and dial `workers` down if you
  see throttling.
