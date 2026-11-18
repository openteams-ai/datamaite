# Tests

Two tiers, by directory:

| Tier | Location | Runs | Needs |
| ------ | ---------- | ------ | ------- |
| **Unit / hermetic** | `tests/*.py` | always (default `pytest`) | nothing — synthetic `tmp_path` fixtures, offline, in-process |
| **End-to-end (e2e)** | `tests/e2e/*.py` | opt-in only | a real external dataset checkout + the `video`/`maite` extras |
| **S3 end-to-end (e2e)** | `tests/e2e/test_s3_compatible.py` | opt-in only | a real S3-API server (SeaweedFS) + the `aws`/`fmv` extras |

The split is intentional: the hermetic suite is the always-on safety net (fast,
offline, runs locally and in every pipeline), and the e2e suite is the
real-data confidence check layered on top. They can look similar — e.g. the
cross-format conversion matrix exists in **both** tiers — but they catch
different classes of bug:

- `tests/test_conversion_matrix.py` (hermetic) proves the conversion **logic**
  is correct, including malformed-input edge cases (fps mismatch,
  drop-with-warning, registry guardrails) that are impossible to express
  against clean real data.
- `tests/e2e/test_example_data_matrix.py` (e2e) proves datamaite loads and
  round-trips the **real on-disk datasets** and that counts match the
  example-data repo's `MANIFEST.md` — a contract check spanning two repos.

Neither replaces the other.

## Running

```bash
# Default: hermetic suite only (e2e is deselected by the `integration` marker)
uv run pytest

# e2e: point at an example-data checkout, then opt in via the marker
git clone https://gitlab.jatic.net/jatic/orchestration-interoperability/datamaite-example-datasets.git
export DATAMAITE_DATASETS_ROOT=$PWD/datamaite-example-datasets/datasets
uv run pytest tests/e2e -m integration -s    # -s shows the summary table
```

## How the opt-in works

Two independent gates keep e2e tests out of the default run:

1. **Marker** — e2e tests are marked `integration`, and `pyproject.toml`'s
   `addopts` deselects `not integration` by default. This is the mechanism;
   the `tests/e2e/` directory is the human-readable signal.
2. **Self-skip** — each e2e test skips when `DATAMAITE_DATASETS_ROOT` is unset
   or missing, so an accidental `-m integration` locally is a clean skip, not a
   failure.

CI runs the e2e tier in its own `integration` job (see `.gitlab-ci.yml`), which
clones the example-data repo over git-LFS and gates the pipeline
(`allow_failure: false`).

## S3 end-to-end tier

`tests/e2e/test_s3_compatible.py` is a separate e2e tier that runs datamaite
against a real S3-API server instead of the shared example-data checkout: it
proves the ranged-read video streaming and `storage_options` plumbing work
against real S3 semantics, not just against fsspec's `memory://` filesystem
(which is what the coverage-gated unit suite's cloud tests use instead, to
stay hermetic — see `tests/e2e/test_cloud_integration.py` and `tests/test_loaders.py`).
Like the rest of `tests/e2e/`, it carries the `integration` marker and
self-skips when its environment variables are unset, so it never affects the
default `pytest` run or the 90% coverage gate.

To run it locally, install the extras the tier needs, then start the same
pinned Apache-2.0 SeaweedFS image used by CI. The image below must match the
`S3_E2E_IMAGE` variable in the `e2e-s3` job in `.gitlab-ci.yml` — keep the two
in sync:

```bash
uv sync --extra dev --extra aws --extra fmv

docker run -d --rm --name datamaite-seaweedfs-e2e -p 9123:8333 \
  -e AWS_ACCESS_KEY_ID=datamaite-e2e \
  -e AWS_SECRET_ACCESS_KEY=datamaite-e2e-secret \
  ghcr.io/chrislusf/seaweedfs:4.46@sha256:08d516132314207d10c8e37cbffc1f32b147d870169688734cc61c6231625b62

export DATAMAITE_S3_E2E_ENDPOINT=http://127.0.0.1:9123
export DATAMAITE_S3_E2E_KEY=datamaite-e2e
export DATAMAITE_S3_E2E_SECRET=datamaite-e2e-secret
uv run pytest tests/e2e/test_s3_compatible.py -m integration --no-cov -v

docker stop datamaite-seaweedfs-e2e
```

CI runs this tier in its own `e2e-s3` job (in the `test` stage, concurrent with
the matrix), against a SeaweedFS service container, with coverage disabled — it
is deliberately kept out of the coverage-gated `test` matrix.

`TestExampleDatasetParity` (in the same file) additionally mirrors the shared
example-data repo's `hmie/valid` dataset into the SeaweedFS bucket and asserts
`validate()`/`load_mot()` give identical results locally and over S3; it needs
`DATAMAITE_DATASETS_ROOT` pointing at an example-datasets checkout and skips
cleanly without it.

## Conversion harness

`tests/harness/` holds one definition of "the same dataset" for object
detection, used by `tests/test_conversion_harness.py`. It covers YOLO and
COCO today.

- `snapshot(dataset)` (`tests/harness/snapshot.py`) walks a loaded
  `ObjectDetectionDataset` from its root with `canonical`, which turns
  everything reachable into plain data: every dataclass at any depth (images,
  boxes, the dataset metadata, the taxonomy and its entries, and any class
  added later) becomes a dict of all its fields; mappings become dicts, lists
  and tuples become lists, NumPy scalars and arrays become Python values,
  enums become their values, and bytes become a hash. Each image's content is
  kept as `image_sha256`, hashed from its bytes or, for a file-backed image,
  from its file. Any other type raises `TypeError` naming where it is, so a
  new kind of value is handled on purpose instead of compared with `==`. A
  field added to any record is therefore compared without any change to the
  harness.
- `EXCLUDED_PATHS` lists the paths left out of every snapshot, each with the
  reason, such as `samples[*].path_or_uri` (where the image is stored, not its
  data), `dataset_id` and `source_dataset` (the name of the loader that ran),
  and the taxonomy's `ordered_names` (its entries' names, so a class change
  would be reported twice). A value that carries the user's data is never
  excluded; a format that cannot store it declares a loss.
  `test_every_excluded_path_matches_a_real_snapshot` checks that each path
  names a value in a loaded dataset, so the list can't go stale;
  `test_a_field_added_to_a_nested_dataclass_is_compared` adds fields to a
  box, a class and the taxonomy and checks that `compare` reports them; and
  `test_snapshot_carries_every_record_field` changes each field of each
  record in turn and checks that `compare` reports it.
- `problems(dataset)` lists where the dataset disagrees with itself: a
  taxonomy entry with no class id, a class id used by two entries, a taxonomy
  whose `ordered_names` are not its entries' names, a box class that is not in
  the labels, a MAITE target whose boxes, labels or scores don't match the
  records (or whose label is not in the public MAITE `index2label`), a public
  `index2label` that is not every class with an integer id (unused classes
  included), and a box outside its image. With `decode=True` it also decodes
  each image and flags a stored size that differs from the decoded one (needs
  the `od` extra). A snapshot keeps its dataset's problems, and `compare`
  reports the actual dataset's as `problem: ...` lines. A snapshot taken with
  `decode=True` also keeps a hash of each image's decoded pixels, so the same
  file bytes decoding to other pixels is a difference.
- `compare(expected, actual, allowed=...)` returns every difference as a
  readable line; `assert_same` fails with all of them. Pairing reads keys from
  the snapshot but takes nothing out of the comparison. Images pair by key
  (the file name as a POSIX path) and split; a missing image that reappears
  under another name is reported as renamed, unless two images shared that
  name, in which case a writer must rename one, or a loss says the format must
  rename it (YOLO's `LABEL_FILE_STEMS`: `a.png` and `a.jpg` in one split would
  share the label file `a.txt`). A renamed image's boxes and other fields are
  still compared. Images with the same pixels
  pair by split, size and boxes. Boxes pair one-to-one on class name, class
  id, crowd flag and position within the box tolerance, as many as possible,
  so the result doesn't depend on their order; boxes whose other fields also
  agree pair first. Classes pair by id. Then every field of each paired
  image, box and class is compared, the fields used for pairing included, and
  so is every field of the dataset, its metadata and its taxonomy. Records
  and mappings are compared field by field and key by key, and each
  difference is reported on its own line, for example:

  ```text
  a.png [split train]: box 'car' (id 1) at [20, 10, 10, 8]: segmentation expected [...], got [...]
  class 'car' (id 1): synset expected 'car.n.01', got None
  dataset: dataset_metadata.taxonomy.id_density expected 'sparse', got 'dense'
  ```

  The labels `{id: name}` are also compared as a summary line, and the class
  order (the class order of formats whose class id is a position) as another.
- Box coordinates (`bbox`) and polygon points (`segmentation`) are pixel
  coordinates, equal within 0.01: YOLO stores boxes normalised to six
  decimals, and a format that rounds boxes rounds polygons the same way.
  Every other number, such as a score, an area or a metadata value, is equal
  only within a relative tolerance of `1e-6` (with a `1e-9` absolute floor),
  so a score of 0.5 is not 0.509. NaN equals NaN, and `True` is not `1`.
- `allowed` is a set of `Loss` values: differences a format allows. Each
  format declares its losses in three groups, so that a choice is never taken
  for a limit of the format. `cannot_store` is what the format has no standard
  place for, such as YOLO renumbering class ids to `0..n-1` in taxonomy order
  or having no polygons, box attributes or image metadata. `chooses` is what
  the format could keep but datamaite does not, such as the YOLO writer
  putting images with no split in `train` rather than a flat folder; each of
  these could be fixed instead. `derives` is a value the format makes up:
  COCO gives a box with no area `width * height` and one with no annotation id
  a new integer id, the YOLO loader records each image's label file and each
  box's label row, and marks the taxonomy dense when its class ids are
  `0..n-1`. A derived number is expected within what the box tolerance
  allows, since the box it comes from may have moved that much. Each loss
  turns the expected snapshot into what the format must give back, by editing
  its tree at paths such as `samples[*].detections[*].iscrowd`; the actual
  snapshot is never changed, so a loss can't hide a change the format did not
  need to make (renumbered ids must still be dense, images with no split may
  go to `train` but not be repartitioned).
  For a conversion, `allowed` is one set per format in pipeline order, from
  `loss_stages(["yolo", "coco"])`, and each format's losses apply in turn:
  images with no split go to `train` in YOLO, then COCO lists that split.
  One format's losses as a set are `FORMATS[fmt].losses`.
- `assert_no_source_paths(written_dir, source_dir)` fails if any text file a
  writer produced (JSON, JSONL, YAML, TXT, CSV, XML, INI) contains the
  absolute path of `source_dir`, resolved or as given, and names the file and
  field. Relative paths pass. Every write, append and convert cell runs
  it, then moves the output and renames the source away before loading, so the
  output must stand on its own. Every write, append and convert cell also
  checks that no file outside the output folder changed: the source files,
  and a neighbouring folder.
- `output_problems(root, fmt)` (`tests/harness/output.py`) reads the written
  COCO or YOLO files directly, without datamaite's loaders: repeated or
  dangling COCO ids, COCO images not on disk, YOLO class indices outside
  `names`, `names` keys that are not `0..n-1`, malformed YOLO rows, and YOLO
  label files with no image, an `nc` that is not the number of `names`, and
  `data.yaml` split paths that are missing, not on disk, not a folder or
  `.txt` list, list an image not on disk, or leave a written image unreached
  (datamaite's own loader finds the folders anyway). Every write, convert and
  append cell runs it.

The test module runs four operations on small synthetic scenarios. One rich,
healthy dataset (several classes, one unused, images with no boxes, fractional
boxes) runs every operation and format pair. Each edge case runs only the
operations it is about: a single `val` or `test` split, split names such as
`validation`, `valid` and `holdout`, nested and aliased file names, two images
whose names differ only in their extension, class ids
of 10 and above with gaps, a taxonomy out of id order, class supercategories
and attributes and crowd boxes, no taxonomy, NumPy class and taxonomy ids,
EXIF-rotated JPEGs, WebP and TIFF. `record_details` runs every operation with
a value in every other record field: a box with a polygon, an area, attributes
and an annotation id, a box with a score and a separate source class id, a
crowd box with an RLE mask, image metadata, class synsets, a class left out of
evaluation, and the dataset's split list, `info`, `licenses` and `extras`.
`test_every_record_field_is_set_by_some_scenario` checks that every record
field is set by some scenario, except the few in `NOT_EXERCISED`, each with
its reason: the snapshot compares every field, but a field no scenario sets is
never tested. Two tests write `record_details` to COCO, change the written
JSON (move one polygon point, delete one box attribute) and check that the
comparison reports exactly that change.

Each cell snapshots the expected dataset before it runs, and reports any
change to its in-memory inputs: most scenarios use one object as input and
expected, so a writer that changed its input in place would otherwise change
the answer too.

| Operation | What it checks |
| --- | --- |
| `memory` | the input dataset as built in memory |
| `load` | an original written by hand (not by datamaite), then loaded |
| `roundtrip` | the input written with datamaite, then loaded |
| `convert` | a hand-written original in format A, converted to B (every pair, same format included), then loaded |

Appends are explicit cases in `APPEND_CASES` (`tests/harness/scenarios.py`),
each a first dataset, a second written into it with `mode="append"`, and the
whole dataset expected back: the same classes, a subset of them, a new class,
conflicting class ids (only the class ids may change, or the writer may refuse
and leave the dataset unchanged), another image with an existing file name
(both must be kept), and an empty append.

Known bugs are listed in `KNOWN_BUGS` in the test module. Each entry names
its issue, the cells it breaks and the differences it causes there, as regular
expressions. A cell with known bugs is an expected failure only if its bugs
explain every difference it reports, so a new regression in that cell still
fails it. When a fix stops a bug showing in a cell, the cell fails until it is
removed from the entry. The source-path check is never covered by a known bug.
An entry for a bug that only gives boxes the wrong class sets `relabels_only`:
its cells are also compared with every class left out, and that bug can't
explain a difference there (the cell's other bugs can), so a lost or moved
box is never taken for a relabelled one.

To compare datasets in another test, for example real datasets in
`tests/e2e/`:

```python
from tests.harness import assert_same, loss_stages, snapshot

original = snapshot(load_od(src, dataset_format="coco"))
converted = snapshot(load_od(out, dataset_format="yolo"))
assert_same(original, converted, allowed=loss_stages(["coco", "yolo"]))
```

### Real-world exports

`tests/e2e/test_real_world_exports.py` runs the harness on real exports in the
example-data repo: COCO and YOLO files written by CVAT, Label Studio,
FiftyOne, Roboflow and Ultralytics, and Hugging Face `metadata.jsonl` and
`metadata.parquet` sets. Each case is one loader call. Its cells check three
things separately:

- `load`: every field reachable from the loaded dataset matches
  `tests/e2e/real_world_expected.json`: classes and their details, images,
  splits, sizes, image metadata, boxes, crowd flags, polygons and masks,
  areas, annotation ids and attributes, and the dataset's `info` and
  `licenses`. That file is read from the exports' own files without
  datamaite; regenerate it with
  `python tests/e2e/real_world_expected.py "$DATAMAITE_DATASETS_ROOT"` when
  the example data changes. The expected snapshot is the snapshot of a
  dataset built from that file, so a field the files don't hold is at its
  default, except values the loader derives, such as the YOLO loader's
  record of each box's label row, the split list and whether the class ids
  are dense (`LOADER_DERIVES`).
- `self-check`: the dataset agrees with itself, its MAITE targets and its
  pixels.
- `roundtrip`, `to-coco`, `to-yolo`: a copy of the export is written or
  converted and reloaded, as in the hermetic harness, and compared after
  what the source loader derives and the target's losses. The source
  writer's losses don't apply, as it never ran. The copy is snapshotted
  before writing and must be unchanged after, and nothing outside the
  output folder, nor the in-memory input, may change.

`test_paired_exports` also checks that two tools' exports of the same data
load the same. Known bugs are `KnownBug` entries in the module, as in the
hermetic harness, including the class-blind comparison for `relabels_only`.
There it leaves out only class names: #154 and #155 keep the class ids, so a
box given another id still fails its cell.

It is part of the e2e tier, so it needs an example-data checkout. A case
whose export is missing is skipped, unless `DATAMAITE_REQUIRE_REAL_EXPORTS=1`
is set, as it is in the `integration` job. The Hugging Face parquet case needs
`pyarrow`, which the `dev` extra installs:

```bash
git lfs install
git clone https://gitlab.jatic.net/jatic/orchestration-interoperability/datamaite-example-datasets.git
export DATAMAITE_DATASETS_ROOT=$PWD/datamaite-example-datasets/datasets
uv run pytest tests/e2e/test_real_world_exports.py -m integration --no-cov -rx
```

### Adding a format to the harness

The formats come from the loader and writer registries: every object-detection
format with both a loader and a writer must be in `FORMATS` or in
`NOT_YET_COVERED` in `tests/harness/formats.py`, or
`test_harness_covers_every_od_format` fails. To add one:

1. Write an `author_<format>(dataset, root)` function in
   `tests/harness/formats.py` that writes the format's on-disk layout by hand,
   without datamaite's writer. Keep it close to what the format's own tools
   produce, and write every field the format can hold.
2. Add a `FormatSpec` to `FORMATS` with its losses in `cannot_store`,
   `derives` and `chooses`, each with a comment. If a needed loss is missing, add
   it to `Loss` in `tests/harness/compare.py`, handle it in
   `_expected_after_losses` (it changes only the expected snapshot), and test
   in `tests/test_harness_compare.py` that it allows its change and nothing
   else. Prefer a few precise losses (one per field) to one broad one. A value
   the format derives, such as an area computed from the box, is a loss that
   fills in the expected value, not one that ignores the field. A bug is not a
   loss.
3. Remove the format from `NOT_YET_COVERED` and update the format list in
   `test_harness_covers_every_od_format`.
4. Add output checks for the format to `tests/harness/output.py`.
5. Run `uv run pytest tests/test_conversion_harness.py --no-cov -rx`. For each
   new failure, open an issue and add a `KNOWN_BUGS` entry with the cells and
   the differences the bug causes, as narrow as you can make them.

A new edge case is a new `Scenario` in `tests/harness/scenarios.py`, run on the
operations it is about (`operations=`), against every format; a new append
contract is a new `AppendCase`. Scenarios that build images with OpenCV set
`needs_image_stack=True`, which marks their cells `optional`.

`tests/test_harness_compare.py` tests the harness itself: that `compare`
reports a lost sample, changed pixels, a relabelled box, a changed class or
taxonomy field and a repartitioned dataset, that each tolerance and each loss
allows only its own change, that `canonical` refuses a type it does not know,
and that the snapshot, problem and output checks flag what they should.

## Required and optional features

Every test carries exactly one of two feature markers (TR-4-H-3 / TR-4-H-4):

- `@pytest.mark.optional`: the test needs an optional extra (`fmv`, `od`,
  `ic`, `maite`, `aws` / `gcs` / `azure`, or the pixel/video stack those pull
  in), either for the feature under test or to synthesise its fixtures (e.g.
  the HMIE factory encodes a real MP4 with OpenCV). Add this marker yourself.
- `@pytest.mark.required`: the test runs on a core install (`datamaite[dev]`
  with no optional extras) and exercises a required feature. Do not add it by
  hand: a collection hook in `tests/conftest.py` applies it to every test that
  is not marked `optional`, and fails collection if a test carries both.

The markers are orthogonal to the tier markers above: an e2e test is also
`required` or `optional`. Mark a whole module `optional` with `pytestmark`, or
individual classes/tests when a module mixes both; parametrized cases can carry
the marker via `pytest.param(..., marks=...)` (see `WRITABLE_FORMAT_PARAMS` in
`test_conversion_matrix.py`). A new test that needs an extra but is not marked
`optional` is treated as `required`, and `pytest -m required` on a core install
then fails, which is how such a test is caught.

```bash
uv run pytest -m required --no-cov   # required features only
uv run pytest -m optional --no-cov   # optional features only
# Every collected test must have exactly one marker; both of these must collect 0:
uv run pytest --co -q -m "not required and not optional"
uv run pytest --co -q -m "required and optional"
```

## Adding tests

- New unit test → `tests/test_<thing>.py`, no tier marker. Must run offline.
- New e2e test → `tests/e2e/test_<thing>.py`, `pytestmark = pytest.mark.integration`,
  and skip cleanly when its external dependency is unavailable.
- Either way, add `required` or `optional` (see above). If it can't run without
  an optional extra installed, it is `optional`.
