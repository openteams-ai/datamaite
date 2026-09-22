# Reading and writing datasets on object storage

datamaite can read and write every registered dataset format directly on S3,
Google Cloud Storage, or Azure Blob Storage. Pass a cloud URL anywhere a
dataset root or write destination is accepted. Format modules use the same
fsspec/UPath path and shared media/copy adapters, so there are no provider
branches in individual readers or writers.

HMIE validation also supports cloud roots. Validation for non-HMIE formats is
not currently implemented, independently of their read/write support.

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

## Load, write, convert, and validate with cloud URLs

```python
import datamaite

ds = datamaite.load_od(
    "s3://my-bucket/datasets/coco",
    dataset_format="coco",
    storage_options={"anon": False},
)

datamaite.write(
    ds,
    "gs://other-bucket/datasets/yolo",
    output_format="yolo",
    storage_options={"token": "google_default"},  # destination options
)

# Source and destination options stay separate during conversion.
datamaite.convert(
    "s3://my-bucket/datasets/coco",
    "az://container/datasets/yolo",
    input_format="coco",
    output_format="yolo",
    read_options={"storage_options": {"anon": False}},
    write_options={"storage_options": {"account_name": "..."}},
)

result = datamaite.validate("s3://my-bucket/datasets/hmie", workers=8)
print(result.summary())
```

The validation CLI accepts cloud URLs:

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

Dataset pickle/cloudpickle state never contains `storage_options`. Workers
reconstruct ambient provider credentials lazily; when explicit process-local
options are required after transfer, call `dataset.with_storage_options(...)`
before lazy media access or writing. `write(..., source_storage_options=...)`
can also explicitly rebind only the source side.

## Destination modes and object-store semantics

`mode="error"`, `"append"`, and `"replace"` work for object-store prefixes.
Remote error/replace writes stage output before promotion, so a writer failure
does not clear an existing destination. Promotion and multi-object replacement
cannot be globally atomic on object storage; rename may be copy + delete.
Replacing an account/bucket/container root is refused. Append retains stale
objects by design and assumes a single writer; concurrent appenders cannot
reserve names atomically across every supported backend.

Cross-filesystem media copies stream in bounded chunks; same-filesystem copies
use the backend's native copy when available. Remote image/video decoding stays
lazy. Empty YOLO class directories are represented by hidden marker objects,
because an empty object-store prefix does not exist.

## Test locally against a MinIO bucket

To develop the S3 path without touching a real AWS bucket, run
[MinIO](https://min.io) — an S3-compatible server — in a container. This is
the same setup the repo's `e2e-s3` CI tier uses. The image tag is pinned in
one place, `MINIO_E2E_IMAGE` in `.gitlab-ci.yml` (the last Apache-2.0 MinIO
release); `tests/README.md` carries the canonical recipe. Read the tag from
there rather than copying it:

```bash
MINIO_E2E_IMAGE=$(sed -n 's/.*MINIO_E2E_IMAGE: *"\(.*\)"/\1/p' .gitlab-ci.yml)
docker run -d --rm --name datamaite-minio -p 9123:9000 \
  -e MINIO_ROOT_USER=datamaite-e2e -e MINIO_ROOT_PASSWORD=datamaite-e2e-secret \
  "$MINIO_E2E_IMAGE" server /data
```

Create a bucket, upload a dataset, and point datamaite at the URL.
`client_kwargs.endpoint_url` is the only thing distinguishing the local
container from AWS:

```python
import datamaite
import s3fs  # installed by datamaite[aws]

storage_options = {
    "key": "datamaite-e2e",
    "secret": "datamaite-e2e-secret",
    "client_kwargs": {"endpoint_url": "http://127.0.0.1:9123"},
}

fs = s3fs.S3FileSystem(**storage_options)
fs.mkdir("my-datasets")
fs.put("path/to/local/hmie-batch-01", "my-datasets/hmie-batch-01", recursive=True)

ds = datamaite.load_mot("s3://my-datasets/hmie-batch-01", storage_options=storage_options)
result = datamaite.validate("s3://my-datasets/hmie-batch-01", storage_options=storage_options)
```

To run the identical code against real AWS, drop `client_kwargs` and let
`key`/`secret` come from the environment or an AWS profile — the `s3://`
URL and everything else stay the same. Stop the container with
`docker stop datamaite-minio` when you're done (`--rm` removes it).

The runnable real-S3 cell in
[HMIE Datasets from Cloud Storage](../tutorials/HMIE_Cloud_Storage.ipynb)
targets this container via the same variables the `e2e-s3` CI tier uses:

```bash
export DATAMAITE_S3_E2E_ENDPOINT=http://127.0.0.1:9123
export DATAMAITE_S3_E2E_KEY=datamaite-e2e
export DATAMAITE_S3_E2E_SECRET=datamaite-e2e-secret
```

## Troubleshooting

- **`CERTIFICATE_VERIFY_FAILED` reaching S3** — managed hosts often preset
  CA-bundle environment variables (`AWS_CA_BUNDLE`, `SSL_CERT_FILE`,
  `REQUESTS_CA_BUNDLE`, …) to an organization-only bundle that lacks the
  Amazon root CAs. Clear those variables (or point them at a bundle that
  includes both the org and Amazon roots) rather than disabling
  verification. For the CLI and your own scripts do that in the shell —
  neither the library nor `datamaite validate s3://…` reads any datamaite
  variable for this:

  ```bash
  env -u AWS_CA_BUNDLE -u SSL_CERT_FILE -u REQUESTS_CA_BUNDLE -u CURL_CA_BUNDLE \
    datamaite validate s3://bucket/prefix
  ```

  The runnable cell in the
  [Cloud Storage tutorial](../tutorials/HMIE_Cloud_Storage.ipynb) is the one
  place that offers a shortcut, `DATAMAITE_S3_CA_BUNDLE=system` (or a
  replacement bundle path): it is a notebook-only convenience that clears or
  replaces those same variables before creating the filesystem. Restart the
  kernel first — an already-created S3 filesystem keeps its old SSL context.
- **`NoCredentialsError` in cluster pods** — Kubernetes pods have no ambient
  AWS identity unless one is wired up (EKS Pod Identity or IRSA on the pod's
  service account, or explicit credentials). Node-role fallback via IMDS is
  typically blocked by a hop limit of 1.

## How video integrity checks work on cloud data

Annotation (JSON) checks stream directly from object storage. Video
integrity checks stream too: the probe opens each remote video as a
seekable file object and decodes through PyAV over bounded ranged reads, so
only the byte ranges it actually reads (the container header plus a handful
of sampled frames) are transferred. The default read-ahead block is 8 MiB;
pass `storage_options={"block_size": 1 << 20}` to tune it to 1 MiB for
seek-heavy workloads. No full-file download, no temporary
files, no presigned URLs. In practice a probe transfers about 13 MB per
video regardless of file size (see the transport benchmark under
`tools/probe_bench/`), because the cost scales with the number of frames
sampled, not the length of the clip.

The same fsspec code path serves every backend, so behavior is identical on
S3, GCS, and Azure. S3 is exercised end-to-end in CI against a SeaweedFS
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
