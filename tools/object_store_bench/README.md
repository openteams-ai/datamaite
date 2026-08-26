# object_store_bench

Manual object-storage performance diagnostics for datamaite. These scripts are
not CI tests and are not shipped in the package.

The runner measures three high-impact areas:

1. **Discovery amplification** for YOLO IC, Hugging Face video classification,
   and COCO prefixes.
2. **Transfer directions and destination modes** for local→S3, S3→local,
   S3→S3, `error`, `append`, and `replace`.
3. **Sparse remote-video access** for the first, middle, last, and every 100th
   frame, with datamaite's default 8 MiB and tuned 1 MiB read-ahead blocks.

It reports logical S3 SDK operations, requested object-body bytes, wall time,
and sampled scenario-local process RSS. Counts are logical operations, not raw
HTTP attempts, so transport retries are not counted separately. API counts and
bytes are the durable comparison metrics. Wall time against
localhost MinIO is useful only for comparing revisions on the same machine.

## Install

```bash
uv sync --extra dev --extra aws --extra fmv
```

The run command below adds `psutil` ephemerally for scenario-local RSS
sampling; it is a benchmark-tool dependency, not a datamaite package extra.

## Run against MinIO

Start the same pinned Apache-2.0 MinIO version used by the S3 E2E CI job:

```bash
docker run -d --rm --name datamaite-minio-bench -p 9123:9000 \
  -e MINIO_ROOT_USER=datamaite-e2e \
  -e MINIO_ROOT_PASSWORD=datamaite-e2e-secret \
  minio/minio:RELEASE.2021-04-22T15-44-28Z server /data

export DATAMAITE_S3_E2E_ENDPOINT=http://127.0.0.1:9123
export DATAMAITE_S3_E2E_KEY=datamaite-e2e
export DATAMAITE_S3_E2E_SECRET=datamaite-e2e-secret

uv run --with psutil --extra aws --extra fmv python tools/object_store_bench/bench.py \
  --root s3://datamaite-bench/scratch --create-bucket

docker stop datamaite-minio-bench
```

Use `--objects`, `--object-bytes`, and `--video-frames` to change scale. Repeat
`--scenario discovery`, `--scenario transfer`, or `--scenario video` to run a
subset. Fixture creation and cleanup are excluded from measured sections.
Every run uses a unique child prefix and removes it unless `--keep` is passed.

Credentials are read from the three `DATAMAITE_S3_E2E_*` variables above. If
those are absent, s3fs uses its standard AWS credential chain, so the runner can
also target a real disposable S3 prefix. Never put credentials in command-line
arguments or committed result files.

## Interpreting results

- Discovery should use paginated listings rather than one `HEAD` per object,
  and should not download image/video bodies.
- S3→S3 should use `copy_object` and download essentially no object bodies to
  the client.
- `append` and especially rollback-safe `replace` require more metadata/copy
  operations; this benchmark makes that cost explicit.
- Sparse video bytes and wall time reveal whether selecting a late frame scans
  from frame zero and how much 1 MiB read-ahead reduces over-fetch. A localhost
  run does not model cloud latency, but requested bytes and GET counts remain
  useful.

For latency-sensitive experiments, place Toxiproxy between the runner and
MinIO rather than adding sleeps to this script. Run each revision against the
same proxy settings and compare the JSON outputs.

`metrics.py` instruments aiobotocore's private logical API-call boundary. That
is intentional for this developer tool. Each result records the installed
s3fs and aiobotocore versions; compare results from the same code revision and
dependency versions.
