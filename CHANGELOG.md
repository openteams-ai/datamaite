# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- YOLO image-classification discovery is now recursive (#90): images anywhere
  below a class directory (`<split>/<class>/**/<image>`) load with the
  top-level directory as the class and the nested relative path preserved in
  the datum ID — nested subdirectories never become classes. The split-vs-flat
  layout discriminator recognises nested-only splits, so they no longer load
  empty. Autodetect `sniff` deliberately stays shallow (a nested-only root
  needs an explicit `dataset_format="yolo"`).
- YOLO image-classification loader gains a `layout` option (`"auto"` /
  `"split"` / `"flat"`, #90): a flat root whose class directory is named like
  a split and nests all its images is structurally identical to a split
  layout, so `"auto"` reads it as one — it now warns about every non-split
  directory that interpretation drops, and `layout="flat"` (or `"split"`)
  makes the intent explicit. Symlink policy is hardened and uniform: symlinked
  directories — split, class, or nested — are never descended into (a
  symlinked class directory could previously smuggle an outside tree in), and
  every discovered image must resolve inside the dataset root whether or not
  it is itself a symlink. Symlinking the dataset *root* still works; in-root
  file symlinks still load. Discovery also lists each directory exactly once
  per load (the discriminator and record building share one memoized scan).
- YOLO image-classification loader gains a native `split` option (#86), matching
  the OD loader's semantics (#78): `load_ic(root, dataset_format="yolo",
  split="validation")` loads only that split, aliases (`validation`/`valid` →
  `val`, `training` → `train`) normalise, and an explicit selection that matches
  nothing selects nothing — it never widens back to all splits. The taxonomy is
  split-local to the selection and still includes empty class directories (#81).

### Changed

- Dependency management, CI, and the developer workflow now run on **uv**
  instead of Poetry (#60). `uv.lock` is the lock file of record and CI fails if
  it has drifted from `pyproject.toml` (`uv lock --check`); `poetry.lock` is
  removed. Every lint / typecheck / build / test / docs job installs with
  `uv sync --locked` and runs with `uv run`. Correctness of the per-job
  interpreter comes from two things together: `UV_PYTHON_DOWNLOADS=never` leaves
  each job only the Python shipped in its own image, and the venv cache key
  carries `${VERSION}` so a wrong-interpreter environment is never restored in
  the first place — an environment-provided interpreter *name* would not have
  overridden one. The docs jobs' editable-reinstall workaround is likewise
  replaced rather than simply unnecessary: `uv sync` installs the root through
  hatchling, and `[tool.uv] cache-keys` re-roots it when the commit or tag moves,
  so executed tutorial notebooks print the current version instead of a stale
  baked one.
  Developer commands are now `uv sync --extra …` /
  `uv run …` / `uv build` throughout the README and docs. The release publish
  job builds with `uv build` too. The PEP 517 build-environment pin is now set
  globally as `UV_BUILD_CONSTRAINT` rather than only on the release job, because
  installing the root through hatchling means *every* job resolves a build
  environment (uv resolves its own and does not read `PIP_CONSTRAINT`); #85's
  byte-identical recovery rebuilds still hold. `twine` and
  `check-wheel-contents` stay pip-installed and pinned, and `packaging` is now
  pinned explicitly wherever the inline release scripts import it. Poetry and
  `[tool.poetry]` are gone entirely.
- `README_DEV.md` now documents where dependencies are declared and which files
  are derived from which (#60): `pyproject.toml` extras are the source,
  `uv.lock` the lock of record, `requirements.txt` a generated compliance
  projection, and `pixi.toml` a hand-maintained duplicate of the dev toolchain
  that nothing in CI verifies. The pixi section is also flagged as known-broken
  and left unrepaired, since #89 proposes removing pixi entirely.
- A generated `requirements.txt` is committed as a dependency-scanning input
  (#60). The DR-compliance component parses the checked-out tree and does not
  understand `uv.lock`, so deleting `poetry.lock` dropped its SBOM from 201
  packages to 0. The file is derived from `uv.lock`
  (`uv export --all-extras --no-emit-project`), carries hashes, excludes the
  project itself, and the `lint` job fails if the two disagree — so `uv.lock`
  stays the single source of truth and the scan cannot silently go stale.
  Verified in CI: Syft now catalogues 213 records / 201 unique packages, all
  attributed to `requirements.txt` and none to `uv.lock`, restoring the same
  201-package coverage the `poetry.lock` scan had. It is not an install path:
  use `uv sync` for development and `pip install datamaite` as a consumer.
- CI caching is fixed rather than removed (#60). A restored `.venv` had two
  failure modes: it silently overrode the requested interpreter (an
  environment-provided `UV_PYTHON` name is only a discovery preference), and it
  was never re-rooted, so `uv sync` kept a stale baked version that the docs jobs
  would publish. The venv cache key now carries the interpreter version
  (`venv-${CI_JOB_NAME}-py${VERSION}`) so a wrong-version environment cannot be
  restored, and `[tool.uv] cache-keys` re-roots the project when the commit or
  tag moves. uv's download cache is kept under a deliberately static key — the
  venv entries are fingerprinted on `uv.lock`, so a dependency change
  invalidates every job's environment at once, and the download cache is what
  lets those jobs rebuild offline instead of all hammering PyPI. Only one job
  writes it (`policy: pull` everywhere, `pull-push` on `lint`), so it is
  uploaded once per pipeline rather than eleven times.
- `datamaite.__version__` no longer re-derives from git via dunamai when the
  installed metadata reports the `0.0.0` fallback (#60), and `dunamai` is dropped
  from the `dev` extra. That path existed for `poetry install`, which registered
  the placeholder on every developer machine; on the uv path it cannot trigger (a
  tagless clone bakes `0.0.0.postN.devN+<sha>`, not `0.0.0`), it read git
  relative to the installed file — so a placeholder install inside an unrelated
  repository reported *that* repository's tag — and it ran a `git` subprocess at
  import time. Placeholder metadata is now reported as-is.
- Releases are tag-driven (#85): pushing an `X.Y.Z` tag runs validation, publishes to TestPyPI automatically with digest verification, gates PyPI behind one manual approval in the same pipeline, and creates the GitLab Release from this changelog's section for the tag. The package version is derived from the git tag (uv-dynamic-versioning); `[project].version` and the version-bump commit are gone, and the manual `RELEASE_TAG` web-form pipeline is retired. Release tags must be canonically spelled — leading zeros in any numeric component (`01.02.03`, `0.5.0-rc01`) are rejected so the published version can never differ from the tag.

### Fixed

- `load_od` and `load_ic` now reject cloud (remote URL) dataset roots with the
  same loud error as `load`/`load_mot`/`load_vc` (#87); previously an `s3://`
  root fell through into loaders with local-filesystem assumptions.


## [0.4.1] - 2026-07-31

### Added

- Python 3.14 support (#82): `requires-python` widens to `<3.15` and CI tests
  the full suite on 3.14. Dependency floors now mirror wheel availability per
  interpreter (numpy 2.3.2+ and pydantic 2.12+ on 3.14; av 15.1+ on 3.14);
  OpenCV needs no split -- its abi3 wheels install on every supported version.

## [0.4.0] - 2026-07-28

### Added

- YOLO object-detection loader gains native `split`, `yaml_file`, and `ann_dir`
  options (no temporary symlink/YAML staging): `split` loads only the given
  split(s) (aliases like `"validation"` normalise); `yaml_file` points at an
  explicit `data.yaml` path; `ann_dir` overrides the label/annotation directory.
  Defaults preserve the previous whole-root behavior, and an unknown/absent
  split warns and returns an empty dataset (datamaite's loader contract) rather
  than raising. Each option fails *closed*: a split selection matching nothing
  loads nothing rather than widening back to every split; an explicit
  `yaml_file` is authoritative, so a missing file or dead image source yields an
  empty dataset instead of silently falling back to a root scan; and a flat
  `ann_dir` label contested by images from several splits is left unassigned
  rather than copied onto each. `ann_dir` also removes the need for a
  conventional `labels/` directory to exist, and mirrors the image's structure
  below the root (minus any `images` component) so equally-named images in
  different splits stay distinct. Relative paths in a nested `yaml_file`
  resolve against the YAML's own directory when it declares no `path:`,
  matching Ultralytics (#78).

### Fixed

- Hugging Face Vision loader/writer no longer lose `ClassLabel` names on
  round trips. The loader now decodes the int→name tables embedded in
  `metadata.parquet` feature schemas (top-level label columns and the OD
  `objects.categories` lists), and the OD writer resolves a detection carrying
  only `category_id` through the dataset taxonomy instead of writing the bare
  int — so HF → datamaite → HF preserves `ClassLabel(names=["cat", "dog"])`
  rather than degrading it to stringified ints. CSV/JSONL metadata files carry
  no name table, so integer labels loaded from those remain name-less (!75).
- YOLO image-classification loader: empty class directories (a class folder
  present on disk but containing no images) are now included in the taxonomy.
  Previously the taxonomy was derived only from classes that contained samples,
  so an empty class in one split shifted dense label indices relative to another
  split (or dropped the class entirely), misaligning labels across splits. This
  covers a split that holds only empty class directories, which was previously
  skipped by split discovery entirely (#81).

## [0.3.1] - 2026-07-24

### Added

- Native IC/OD datasets now implement the MAITE `FieldwiseDataset` protocol
  (`get_input`/`get_target`/`get_metadata`) in addition to `__getitem__`.
  `get_target` and `get_metadata` avoid decoding the image when the target and
  dimensions are known without it, so fieldwise consumers (e.g. dataeval and
  other MAITE tooling) can read targets/metadata without the image extra (#77).

- Object-detection MAITE datum metadata now surfaces the source-preserving
  per-image passthrough (`sample.metadata`) as flat top-level keys, plus
  `file_name`. For COCO this exposes the full `images[]` extras
  (`license`/`date_captured`/`flickr_url`/`coco_url`/...) that MAITE metadata
  consumers (e.g. dataeval bias factors) read; YOLO/VisDrone provenance is
  surfaced the same way. The typed `id`/`height`/`width` keys always win over
  any same-named passthrough value, and a bare sample still yields exactly
  `{id, height, width}` (#79).
- Object-detection MAITE datum metadata now surfaces per-box source attributes
  as flat lists index-aligned to the target boxes, so metadata consumers
  (dataeval, which expands list-valued datum-metadata keys into per-object bias
  factors) can read them. This surfaces VisDrone `truncation`/`occlusion`/
  `visdrone_score` and other entries that detection records retain in their
  attributes. `target.scores` still carries ground-truth confidence; the raw
  VisDrone score stays a distinct `visdrone_score` factor (#80).

### Fixed

- PEP 561 `py.typed` marker: `datamaite` now ships a `py.typed` file in the
  wheel and sdist, declaring the package as typed. Without the marker the
  package was treated as untyped under PEP 561, so `py.typed` consumers (e.g. a
  `pyright --verifytypes` run in `dataeval_flow`) could not use `datamaite`'s
  inline annotations and resolved public types like `ImageClassificationDataset`
  / `ObjectDetectionDataset` to `Unknown` (#83).

## [0.3.0] - 2026-07-23

### Added

- VisDrone Static-Images writers (IR-3.2-S-7): `write(dataset,
  output_format="visdrone")` serialises still-image object-detection and
  image-classification datasets to official `VisDrone2019-DET-<split>/`
  roots (`images/` + eight-field `annotations/*.txt`), with the
  detection/classification selection keyed on the dataset task. Class ids
  resolve through the shared fixed-taxonomy machinery (`class_map` option,
  aggregated #55 warnings); the static loaders now preserve
  `visdrone_category_id` in attributes so VisDrone-to-VisDrone round-trips
  are warning-free (#9).
- Flat-folder still-image loader (IR-3.2-S-1):
  `load_od(root, dataset_format="flat_images")` reads a flat directory of
  label-free `.jpg`/`.png`/`.tif` images as an unlabeled
  object-detection dataset (zero detections, no taxonomy). Explicit opt-in
  only — a bare folder of images is never autodetected. Images keep the
  lazy OpenCV decode (`datamaite[od]`). SafeTensors ingest, also named by
  IR-3.2-S-1, is deferred pending a program-standards change (#74) (#2).
- Hugging Face Vision still-image format (IR-3.2-S-2 loader + IR-3.2-S-6
  writer): `load_ic(root, dataset_format="huggingface_vision")` reads the
  ImageFolder classification convention (class folders, split folders, or
  `metadata.csv`/`metadata.jsonl` with `label`), and
  `load_od(root, dataset_format="huggingface_vision")` reads the
  object-detection convention (metadata `objects` column of parallel
  `bbox`/`categories` lists). `write(dataset,
  output_format="huggingface_vision")` mirrors both back out, with the
  detection/classification selection keyed on the dataset task (#3, #8).

### Fixed

- VisDrone still images: write -> reload is no longer lossy. The static
  loader reads `.tif`/`.tiff` images (the writer copies images verbatim, so
  a `.tif` source — e.g. arriving via `flat_images` — previously produced a
  root that reloaded as zero samples; suffixes the loader cannot read are
  now skipped with a warning at write time instead of silently vanishing on
  reload). The loader infers the official `test-dev`/`test-challenge`
  splits from split-root names instead of collapsing them onto `test`, so
  writer-emitted split roots round-trip their split identity. The writers
  no longer write a generic `category_id` 0 as VisDrone category 0
  ("ignored regions", score 0, excluded from evaluation) — such rows are
  dropped with an aggregated warning pointing at `class_map=` (#55
  provenance rules). The IC writer no longer copies an image whose
  annotation rows all drop, so an emitted root never contains images
  without annotation files (#9).

- Hugging Face Vision: the IC writer now preserves class-folder names with
  spaces/unicode/punctuation (e.g. `traffic light`, `café`) instead of skipping
  those samples, keeping write->reload label identity. The IC/OD loaders order
  integer category ids numerically (not lexically) and preserve original integer
  ids rather than re-indexing them; a metadata file no longer fabricates a label
  from a file_name's parent directory; and class folders literally named
  `train`/`test`/`val` are no longer mistaken for split directories (#3, #8).

- Hugging Face Vision: the OD writer now places `metadata.jsonl`/`.csv`
  *inside* each split directory (`train/metadata.jsonl`,
  `data/metadata.jsonl`) with directory-relative `file_name`s instead of one
  root-level file: `datasets.load_dataset("imagefolder", ...)` only
  associates metadata files within a split's directory tree, so the
  root-level file silently lost the `objects` column on the Hugging Face
  side (verified against `datasets` 5.x). The OD loader correspondingly
  scans all first-level directories (not just split-named ones) for
  metadata files; root-level metadata files remain supported for reading
  (#3, #8).

- Hugging Face Vision: custom split names can no longer break round-trips —
  the writers only emit ImageFolder-recognized split directories
  (`train`/`validation`/`test`; aliases such as `val`/`dev`/`eval` normalise,
  unknown split *options* raise, unknown *sample* splits fall back to the
  default split with a warning), since a directory like `holdout/` would
  reload as a class folder (IC) or lose its split (OD). The loaders now
  recognize the full Hugging Face split keyword set (`dev`, `testing`,
  `eval`, `evaluation`). The OD writer preserves category *names* in
  `objects.categories` when detections carry them (reloads keep `"person"`
  instead of a bare id; the numeric-id drop is declared in
  `lossy_without`). Docs now scope the format as the local
  ImageFolder-compatible layout (not general `datasets`/Hub support) and mark
  CSV OD metadata (JSON-encoded `objects`) as a datamaite extension; an
  optional test module verifies writer output loads with the real
  `datasets.load_dataset("imagefolder", ...)` when `datasets` is installed
  (#3, #8).

## [0.2.2] - 2026-07-16

### Added

- VisDrone Static-Images loaders (IR-3.2-S-3): object detection
  (`load_od(root, dataset_format="visdrone")`) reads the official
  VisDrone-DET layout (`images/` + eight-field `annotations/*.txt`), and
  image classification (`load_ic(root, dataset_format="visdrone")`) derives
  one classification sample per labeled box (the object crop, labeled by its
  VisDrone category) from the same on-disk layout (#4).
- Cloud object storage support for HMIE: dataset roots can be `s3://`,
  `gs://`, or `az://` URLs with the matching extra installed
  (`datamaite[aws]`, `[gcs]`, `[azure]`, or `[cloud]` for all three).
  Cloud roots are HMIE-only; other format loaders raise a clear error on a
  cloud URL. Video integrity checks over cloud data additionally need the
  `fmv` extra — without it they are skipped with a `video_dependency`
  warning.
- MOTChallenge/VisDrone writers warn (aggregated, once per write) when
  falling back from generic `category_id` values to their fixed class
  tables, and accept an explicit `class_map=` option; categories missing
  from `class_map` are dropped with an aggregated warning (#55).
- Python 3.13 support (`requires-python = ">=3.10,<3.14"`), tested in CI
  alongside 3.10/3.11/3.12.

### Changed

- **Breaking:** `write()`/`convert()` now refuse a non-empty destination by
  default (`mode="error"`). Pass `mode="replace"` to clear the destination
  first or `mode="append"` for the previous write-into behavior (#55).

## [0.2.1] - 2026-06-30

### Added

- Object-detection and image-classification formats via task-aware registries: a
  COCO object-detection loader and writer (`load_od(..., dataset_format="coco")`,
  `write(..., output_format="coco")`) and a YOLO/Ultralytics image-classification
  loader and writer (`load_ic(..., dataset_format="yolo")`). A single
  `DatasetFormat` can now back more than one task, dispatched by `(task, format)`.
- YOLO/Ultralytics object-detection loader and writer
  (`load_od(..., dataset_format="yolo")`, `write(..., output_format="yolo")`) with
  standard `images/<split>` + `labels/<split>` and `<split>/images` +
  `<split>/labels` layouts, `data.yaml` discovery, and `load → write → load`
  round-tripping (boxes are clipped to the image on write).
- `datamaite.load_vc(root, dataset_format=…)`: a task-first public entry point
  for video classification, the analogue of `load_mot`. It pins the return type
  to `VideoClassificationDataset` and raises `TypeError` if the resolved format
  produces a different task's dataset. Defaults to
  `dataset_format="huggingface_video_classification"`.
- Hugging Face VideoFolder-style video-classification writer, completing the
  `load → write` round trip for the `VideoClassificationDataset` model.

### Changed

- **Breaking:** the package, distribution, and CLI have been renamed from
  `databridge` to `datamaite`. Update imports (`import datamaite`), the console
  entry point (`datamaite validate …`), and the dependency name; there is no
  `databridge` compatibility shim.
- **Breaking:** `load_huggingface_video_classification` is no longer part of the
  public `datamaite` API. Use the task-first `load_vc(...)` instead (the
  format-specific helper lives on internally in
  `datamaite._formats.huggingface_video_classification.loader`). This makes the
  public loader surface a consistent rule — generic `load` plus one
  `load_<task>` per task (`load_mot`, `load_od`, `load_ic`, `load_vc`) —
  matching how the per-format MOT `load_<format>` helpers were already made
  internal.
- **Breaking:** `load_yolo_image_classification` is no longer part of the public
  `datamaite` API. Use `load_ic(..., dataset_format="yolo")` instead.
- `datamaite.load` (and task-first loaders like `load_mot` / `load_vc`) now fail
  fast on a bad dataset root: a nonexistent path raises `FileNotFoundError` and
  a non-directory path raises
  `NotADirectoryError`, instead of silently returning an empty dataset. A root
  that exists but yields no loadable items still returns an empty dataset, now
  with a `WARNING` so an empty result (e.g. wrong format or wrong subdirectory)
  is never silent.
- `datamaite.load(..., dataset_format=None)` now raises on ambiguous sniff
  matches instead of picking the first registered loader, so multi-task formats
  like YOLO cannot be silently autodetected as the wrong task.
- `datamaite.write` (and `convert`, which forwards to it) now return ``None``
  by default and only return the ``list[Path]`` of files written when called
  with ``verbose=True``. The full list is one path per frame image, which
  floods interactive/REPL output; the file list is now opt-in. Side effects
  (the files written) are unchanged.
- Optional dependencies are declared once via PEP 621
  `[project.optional-dependencies]` instead of duplicated Poetry dependency
  groups, so the extras stay in sync across Poetry, uv, and pip.

### Fixed

- Corrected stale optional-dependency references (e.g. `datamaite[video]`) so
  installs resolve to the current task-oriented extras (`datamaite[fmv]`,
  `datamaite[all]`, …).

### Documentation

- Added a Sphinx documentation build (`docs/`).

## [0.2.0] - 2026-06-16

### Added

- Neutral in-memory dataset model (`datamaite.model`): `Dataset` /
  `VideoSequence` / `BoxAnnotation` form the format-agnostic hub that
  every loader produces and every converter will consume, so any loader
  can feed any output format (N-to-M bridge).
- Loader architecture (`datamaite.loaders`): a `Loader` base class
  defines the input-side contract, `register_loader` is the extension
  point, and `datamaite.load(root, dataset_format=…)` dispatches across
  registered formats (with a `sniff`-based autodetection hook). `HmieLoader`
  is the reference implementation; adding a format is additive (subclass +
  register). See the "Loader architecture" section in `docs/architecture.md`.
- HMIE dataloader (`datamaite.load_mot(..., dataset_format="hmie")`):
  loads an HMIE/Scale dataset into the neutral `BoxTrackDataset` model
  (`VideoSequence` / `BoxAnnotation`
  records with a dataset-wide ontology-URI → category-id map). Reuses the
  existing discovery + Scale-schema layers instead of the hard-coded
  notebook walk; supports `annotation_dir` / `video_dir` overrides for
  flat layouts and an opt-in `require_video` mode that reads true frame
  counts via the `fmv` or `all` extra.
- Writer architecture (`datamaite.writers`): a `Writer` base class defines
  the output-side contract (`BoxTrackDataset` → `list[Path]`), `register_writer`
  is the extension point, and `datamaite.write(ds, dest, output_format=…)`
  dispatches across registered formats. `datamaite.conversion.convert` pairs a
  loader and a writer for end-to-end on-disk → on-disk conversion. See the
  "Writer architecture" section in `docs/architecture.md`.
- HMIE writer (`HmieWriter`): the reference writer that serialises a
  `BoxTrackDataset` back to the HMIE on-disk layout. With the HMIE loader it
  closes a `load → write → load` round trip that recovers the same
  box/category content, proving the writer architecture and that
  `BoxTrackDataset` is a lossless hub.
- Flat-folder MP4 loader
  (`datamaite.load_mot(..., dataset_format="flat_mp4")`, IR-3.3-S-1):
  loads immediate `.mp4` children encoded as H.264 or MPEG-2 into video-backed
  `VideoSequence` records with media metadata and no annotations.
- MOTChallenge loader and writer (`dataset_format="motchallenge"` /
  `MotChallengeWriter`).
- TAO (Tracking Any Object) loader and writer (`dataset_format="tao"` /
  `TaoWriter`); video-backed TAO writes need the `datamaite[fmv]` or
  `datamaite[all]` extra.
- VisDrone video loader and writer (`dataset_format="visdrone"` /
  `VisDroneVideoWriter`).
- Hugging Face VideoFolder-style video-classification loader
  (`HuggingFaceVideoClassificationLoader` /
  `load_huggingface_video_classification`) into the
  `VideoClassificationDataset` model.
- Task / IC / OD foundation: `Task` taxonomy, source-preserving `Taxonomy` /
  `CategoryEntry` (`datamaite.taxonomy`), and canonical `xywh` bbox +
  conversions (`datamaite.geometry`).
- MAITE interoperability (`datamaite.maite`, optional `datamaite[maite]`
  extra): `BoxTrackDataset` conforms to the MAITE MOT protocol structurally;
  `load_mot` returns a MAITE-indexable dataset and `with_mot_options`
  configures the view.

### Changed

- Task-first loader API: `load_mot(root, dataset_format=…)` replaces the
  per-format `load_*` public functions.
- Format loaders/writers reorganised into per-format `datamaite._formats`
  packages.
- Skipped video checks are reported as `SKIPPED` instead of `PASS`.
- Validator notebook exposes the multi-dataset (collection) HTML report.
- `license` set to `Apache-2.0`, replacing the `LicenseRef-TBD` placeholder.

### Fixed

- Batch-level `scale/` discovery now **merges per batch** with the
  snippet-centric pass instead of being an all-or-nothing `root/scale`
  fallback, so per-batch `scale/` under a multi-batch parent and trees mixing
  both layouts are fully discovered. `SnippetPair` carries a `snippet_dir` so
  `snippet_count` no longer collapses centralized-`scale/` annotations onto
  the batch root; non-annotation JSON in a `scale/` dir is skipped; and
  `match_annotation_to_video` returns an orphan instead of guessing when two
  videos share a basename.
- Frame-key mapping snaps to a near integer before flooring (fixes a
  floating-point off-by-one on rates like 29.97/14.985); relative
  `annotation_dir`/`video_dir` overrides resolve against `root` not the CWD;
  non-finite bbox coordinates and NaN/Inf/non-positive fps & duration are
  rejected.

## [0.1.0] - 2026-05-20

### Added

- Initial `datamaite` package scaffold (pyproject, Poetry primary, uv/pixi
  alternatives, hatchling build backend with hatch-vcs versioning).
- GitLab CI pipeline: lint (pre-commit), typecheck (pyright), build (uv build),
  test matrix (Python 3.10 / 3.11 / 3.12 with 90% coverage gate), dr-compliance.
- HMIE / Scale Video Playback dataset validator (`datamaite validate <path>`):
  - Snippet-centric folder-structure discovery for the CDAO SUNet layout.
  - FMV integrity checks (open, frame count, resolution, first/mid/last frame,
    FPS, flat-frame detection) using OpenCV.
  - Annotation coverage checks (orphan annotations, orphan videos).
  - Scale schema conformance via Pydantic models derived from the Scale
    Video Playback reference PDF; handles both full-envelope and
    unwrapped annotation formats.
  - Consistency checks across annotation + video (FPS agreement, frame
    bounds, bbox bounds within image dimensions).
  - Parallel per-pair validation via `ProcessPoolExecutor` with
    `--workers` tuning and main-process cache lookup.
- `docs/schemas/scale-video-playback-v1.schema.json`: machine-readable
  JSON Schema of the Scale Video Playback annotation format.
- SQLite-backed validation cache (`ValidationCache`):
  - Per-file fingerprints (SHA-256 of first 1 MB + size + mtime).
  - WAL journal mode with batched commits (50 writes per flush).
  - `--no-cache` and `--clean` CLI flags; cache hit/miss reporting.
  - Per-user cache at `~/.cache/datamaite/validation.db`.
- CLI dashboard output:
  - 4-check status grid (Folder structure, FMV integrity, Annotation
    coverage, Scale spec compliance) with PASS / WARN / FAIL / N/A states.
  - Multi-batch table view when the path contains sibling batch
    directories; per-batch status indicators and totals.
  - Progress indicators on TTY (pair counts, phase status messages).
  - `--quiet`, `--verbose`, `--debug` modes; `NO_COLOR` env support.
- Output formats:
  - Text summary (default, and `.txt` via `-o`).
  - JSON (`--json` / `.json` extension).
  - JSONL (`--jsonl`) for streaming to `jq` / `grep`.
  - Self-contained HTML report (`.html` extension or default `-o` with
    no filename): inline CSS/JS, no external dependencies, interactive
    dashboard, light/dark theme, search/filter, lazy rendering, print-
    ready, WCAG AA contrast.
- Multi-batch HTML report: aggregated view + clickable batch table that
  swaps to per-batch detail; deep-linkable via `#batch=<name>` URL hash.
- Finding cap (`--max-findings-per-check`) to bound memory on
  pathological datasets; `finding_counts` stays accurate under caps.

### Documentation

- `README.md`: install / quick-start with Poetry.
- `README_DEV.md`: uv and pixi alternatives for contributors.
