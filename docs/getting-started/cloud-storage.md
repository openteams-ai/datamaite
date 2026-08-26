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
