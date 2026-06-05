# Architecture

A reviewer's map of the codebase. Read this with the code open.

## What databridge does (today)

One validation pipeline: **walk a dataset root on disk → pair each annotation
JSON with its video → run checks on each pair → aggregate findings into a
`ValidationResult` → render a report**. The only validation format currently
implemented is HMIE (Scale Video Playback JSON + snippet folder layout). On
the loading side, MOTChallenge, TAO, and VisDrone Video are also implemented as
standard image-sequence loaders. Everything is structured so other formats can
be added behind the same public entrypoint without touching the CLI or
reporting layers.

## The bridge — loaders × consumers

The longer-term shape is an **N-to-M bridge**. A *loader* parses an
on-disk dataset into one neutral in-memory model (`model.py`,
`BoxTrackDataset`). That model is then consumed two ways: **directly as a
MAITE dataset** by models and metrics (the model *is* a MAITE
multi-object-tracking dataset — no conversion step), and by
*converters / writers* that serialise it back out to another on-disk
format. Because every loader produces the identical `BoxTrackDataset`,
any input format can reach any consumer — adding a loader gains an input,
adding a converter gains an output, and neither side knows about the
other. Solid = implemented today; dashed = planned.

```mermaid
flowchart LR
    subgraph in [Input on disk]
        HMIE([HMIE / Scale])
        MOTIN([MOTChallenge])
        TAOIN([TAO])
        VISIN([VisDrone Video])
        COCOIN([COCO])
        YOLOIN([YOLO])
    end

    subgraph loaders [Loaders]
        LH["<b>load_hmie</b>"]
        LM["<b>load_motchallenge</b>"]
        LT["<b>load_tao</b>"]
        LV["<b>load_visdrone_video</b>"]
        LC["load_coco"]
        LY["load_yolo"]
    end

    HUB[/"<b>model.py</b><br/>BoxTrackDataset<br/>VideoSequence · BoxAnnotation"/]

    subgraph consumers [Consumers]
        MAITE["<b>MAITE MOT</b><br/>databridge.maite<br/>(the model IS one)"]
        TM["to_mot"]
        TY["to_yolo"]
        TC["to_coco"]
    end

    subgraph out [Output]
        MAITEOUT([model / metric<br/>in-memory])
        MOTOUT([MOTChallenge])
        YOLOUT([YOLO])
        COCOUT([COCO])
    end

    HMIE --> LH
    MOTIN --> LM
    TAOIN --> LT
    VISIN --> LV
    COCOIN -.-> LC
    YOLOIN -.-> LY
    LH --> HUB
    LM --> HUB
    LT --> HUB
    LV --> HUB
    LC -.-> HUB
    LY -.-> HUB
    HUB --> MAITE
    HUB --> TM
    HUB --> TY
    HUB --> TC
    MAITE --> MAITEOUT
    TM -.-> MOTOUT
    TY -.-> YOLOUT
    TC -.-> COCOUT

    classDef hub fill:#fff8e1,stroke:#f57c00,stroke-width:2px;
    classDef impl fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;
    classDef planned fill:#f5f5f5,stroke:#9e9e9e,color:#616161;
    class HUB hub;
    class LH,LM,LT,LV,MAITE impl;
    class MAITEOUT,MOTIN,TAOIN,VISIN impl;
    class LC,LY,TM,TY,TC,COCOIN,YOLOIN,MOTOUT,YOLOUT,COCOUT planned;
```

Today the HMIE loader (`load_hmie`), the MOTChallenge loader
(`load_motchallenge`), the TAO loader (`load_tao`), the VisDrone Video loader
(`load_visdrone_video`), the HMIE validation pipeline, the HMIE reference
writer, and the MAITE surface
(`databridge.maite`) are implemented. See [Loading](#loading--dataloaderpy)
for how loaders build the model, [The model as a MAITE dataset](#the-model-as-a-maite-dataset)
for the MAITE surface, and [Writer architecture](#writer-architecture--writerspy)
for the writer contract.

## Project layout

```
src/databridge/
    __init__.py              Public API surface
    _cli.py                  CLI entrypoint (`databridge validate ...`)
    _types.py                Shared types: Finding, Severity, ValidationResult
    _cache.py                On-disk cache for expensive video probes
    _report.py               Text / JSON / JSONL / HTML report rendering
    _version.py              Package version
    model.py                 Neutral model + MAITE MOT surface: BoxTrackDataset, VideoSequence, BoxAnnotation
    loaders.py               Loader contract (ABC) + registry + load() dispatch
    writers.py               Writer contract (ABC) + registry + write() dispatch
    validation.py            Orchestration: discovery -> checks -> aggregation
    dataloader.py            HmieLoader: the reference loader (on-disk HMIE -> BoxTrackDataset)
    motchallenge.py          MotChallengeLoader: standard MOTChallenge -> BoxTrackDataset
    tao.py                   TaoLoader: official TAO JSON -> BoxTrackDataset
    visdrone.py              VisDroneVideoLoader: VisDrone VID/MOT video -> BoxTrackDataset
    conversion.py            convert(): end-to-end load + write (on-disk -> on-disk)
    maite/                   Optional MAITE surface (databridge[maite] extra)
        __init__.py              package doc; the model is MAITE directly (no adapter)
        _mot.py                  build_mot_item: the MOT view computed from the model
        _decode.py               Decoder protocol + PyAV backend (lazy)
        _common.py               numpy-array + datum-metadata helpers
    _formats/
        __init__.py          Format registry
        hmie/
            __init__.py              HMIE format entrypoint
            discovery.py             Filesystem walk (snippet-centric, seq_mp4 / seq_ts)
            schema.py                Pydantic models for Scale Video Playback JSON
            categories.py            Severity / category taxonomy for findings
            annotation_checks.py     Scale schema + semantic checks on JSONs
            video_checks.py          FMV open / decode / corruption checks
            consistency_checks.py    Annotation <-> video cross-references
            writer.py                HmieWriter: reference writer (BoxTrackDataset -> on-disk HMIE)
docs/
    architecture.md                              This file
    schemas/
        scale-video-playback-v1.schema.json      JSON Schema for Scale format
tests/                       pytest suite (coverage gate 90%)
```

For the on-disk dataset layout that `discovery.py` walks, see the
["Dataset layout on disk" section in the README](../README.md#dataset-layout-on-disk).

## Reading order

The modules form a clean dependency stack. Read them bottom-up; each
layer depends only on layers below it.

1. `_types.py` — the vocabulary (`Finding`, `Severity`, `ValidationResult`, `DatasetFormat`)
2. `_formats/hmie/schema.py` — Pydantic models for the Scale annotation JSON
3. `_formats/hmie/discovery.py` — filesystem walk that produces `SnippetPair`s
4. `_formats/hmie/annotation_checks.py` — per-annotation checks (schema + semantic)
5. `_formats/hmie/video_checks.py` — per-video integrity probe (cv2)
6. `_formats/hmie/consistency_checks.py` — cross-checks between annotation and video
7. `_formats/hmie/categories.py` — maps check names to the 4 requirement categories
8. `_cache.py` — SQLite-backed memo of per-pair results keyed by file fingerprint
9. `validation.py` — orchestration: discovery + fan-out to workers + aggregation
10. `_report.py` — text / JSON / JSONL / HTML rendering of a `ValidationResult`
11. `_cli.py` — argparse wrapper around `validate()` and the renderers

## Data flow CLI

```mermaid
flowchart TD
    ROOT([root path])
    CLI["<b>_cli.py</b><br/>argparse, exit codes"]
    VAL["<b>validation.py</b><br/>validate() — public entrypoint"]
    DISC["<b>_formats/hmie/discovery.py</b><br/>filesystem walk, pair files"]
    DR[/"DiscoveryResult<br/>pairs, orphans, errors"/]
    PAIR["<b>_validate_pair</b><br/>fanned across worker processes"]
    ANN["annotation_checks.py<br/>schema + semantic"]
    VID["video_checks.py<br/>cv2 probe, frame decode"]
    CONS["consistency_checks.py<br/>fps / afr / dims"]
    CACHE[("<b>_cache.py</b><br/>SQLite memo<br/>by file fingerprint")]
    RES[/"ValidationResult<br/>findings, counts,<br/>label_histogram"/]
    REP["<b>_report.py</b><br/>text · json · jsonl · html"]

    ROOT --> CLI
    CLI -->|validate path| VAL
    VAL --> DISC
    DISC --> DR
    DR --> PAIR
    PAIR --> ANN
    PAIR --> VID
    PAIR --> CONS
    ANN --> RES
    VID --> RES
    CONS --> RES
    CACHE -.->|cache hit| PAIR
    PAIR -.->|store| CACHE
    RES --> REP
    REP --> CLI

    classDef entry fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;
    classDef data fill:#fff8e1,stroke:#f57c00;
    classDef store fill:#f3e5f5,stroke:#7b1fa2;
    class ROOT,CLI,VAL entry;
    class DR,RES data;
    class CACHE store;
```

## Data flow notebook

Skipping the CLI — a notebook or script imports `validate` directly,
gets a `ValidationResult` back, and either inspects it in code
(pandas, custom analysis) or passes it to `_report.py` for a rendered
view inside a cell.

```mermaid
flowchart TD
    ROOT([root path])
    NB["<b>notebook.ipynb / script.py</b><br/>from databridge import validate"]
    VAL["<b>validation.py</b><br/>validate()"]
    DISC["<b>_formats/hmie/discovery.py</b>"]
    DR[/"DiscoveryResult"/]
    PAIR["<b>_validate_pair</b><br/>fanned across worker processes"]
    ANN["annotation_checks.py"]
    VID["video_checks.py"]
    CONS["consistency_checks.py"]
    CACHE[("<b>_cache.py</b><br/>SQLite memo")]
    RES[/"ValidationResult<br/>findings, counts,<br/>label_histogram"/]
    REP["<b>_report.py</b><br/>optional renderer"]
    OUT[/"inline output<br/>pandas · HTML cell ·<br/>custom analysis"/]

    ROOT --> NB
    NB -->|validate path| VAL
    VAL --> DISC
    DISC --> DR
    DR --> PAIR
    PAIR --> ANN
    PAIR --> VID
    PAIR --> CONS
    ANN --> RES
    VID --> RES
    CONS --> RES
    CACHE -.->|cache hit| PAIR
    PAIR -.->|store| CACHE
    RES --> NB
    NB -.->|optional| REP
    REP -.->|rendered string| NB
    NB --> OUT

    classDef entry fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;
    classDef data fill:#fff8e1,stroke:#f57c00;
    classDef store fill:#f3e5f5,stroke:#7b1fa2;
    class ROOT,NB,VAL entry;
    class DR,RES,OUT data;
    class CACHE store;
```


## Discovery — how pairs are built

`discovery.py` runs in two phases: a single `os.walk` that *classifies*
every directory it meets (snippet? seq_*? annotation parent? metadata
to skip?), then a pairing pass that matches annotations to videos via
their shared snippet directory. The layout varies between dataset
families (`scale/` vs a labeler subfolder, `seq_mp4/` vs `seq_ts/`,
`mapp_metadata/` vs `0601_metadata/`), so every decision below is a
branch in the real code.

```mermaid
flowchart TD
    START([root: Path])
    WALK{{"os.walk(root) — single pass"}}

    subgraph classify ["<b>Phase 1</b> — for each directory"]
        direction TB
        Q1{"name ends with<br/>_metadata ?"}
        PR["prune — don't descend"]
        Q2{"has seq_*<br/>children?"}
        REG["register as snippet_dir;<br/>non-seq, non-metadata<br/>children → annotation parents"]
        Q3{"is this dir<br/>a seq_* dir?"}
        CV["collect .mp4 / .ts<br/>→ video_dirs"]
        Q4{"is this dir an<br/>annotation parent?"}
        CA["collect .json<br/>→ annotation_files"]
        SKIP(("skip"))

        Q1 -- yes --> PR
        Q1 -- no --> Q2
        Q2 -- yes --> REG --> Q3
        Q2 -- no --> Q3
        Q3 -- yes --> CV
        Q3 -- no --> Q4
        Q4 -- yes --> CA
        Q4 -- no --> SKIP
    end

    INDEX["index annotations by snippet<br/>(ann.parent.parent)"]

    subgraph pair ["<b>Phase 2</b> — for each snippet_dir"]
        direction TB
        PICK["pick best video:<br/>prefer seq_mp4, else seq_ts;<br/>lex-first within"]
        Q5{"annotations found<br/>in subdirs?"}
        NA["no pairs —<br/>snippet unlabeled"]
        EM["emit SnippetPair<br/>per annotation<br/>(all share the video)"]

        PICK --> Q5
        Q5 -- no --> NA
        Q5 -- yes --> EM
    end

    DETECT["record orphans:<br/>videos no annotation matched,<br/>multi-video dirs (extras),<br/>root-level errors"]
    OUT[/"DiscoveryResult<br/>pairs · orphan_annotations<br/>orphan_videos · multi_video_dirs · errors"/]

    START --> WALK
    WALK --> classify
    classify -.->|walk complete| INDEX
    INDEX --> pair
    pair -.->|all snippets done| DETECT
    DETECT --> OUT

    classDef data fill:#fff8e1,stroke:#f57c00;
    classDef terminal fill:#e8f5e9,stroke:#2e7d32;
    class OUT data;
    class SKIP terminal;
```

Key invariants worth remembering while reading `discovery.py`:

- A "snippet dir" is defined by the *presence of a `seq_*/` child*, not
  by name — this is what makes the walker tolerate the family-specific
  layout differences.
- Snippet-level JSONs (right next to `seq_mp4/`) are never annotations —
  they are video metadata. Annotations always live one level deeper, in
  a subdirectory like `scale/` or a labeler folder. The `parent.parent`
  indexing in Phase 2 relies on this.
- A snippet with videos but no annotation subdir is *not* an error here
  — it just produces no pairs. Whether that's acceptable for the
  dataset is a decision made later, in `validation.py`'s coverage check.
- **Batch-level `scale/` merges with the snippet-centric pass** (it is not
  an all-or-nothing fallback). Any `scale/` directory that is *not* inside a
  snippet — i.e. a batch-level `scale/` holding annotations for sibling
  snippets — is discovered per batch directory and its pairs are *added* to
  the per-snippet pairs. So a parent of several batches each with their own
  `scale/`, and trees that mix per-snippet and batch-level annotations, are
  both fully discovered. Each annotation is paired to a video *within its
  batch* by the filename embedded in the Scale annotation name
  (`match_annotation_to_video`, also reused by the loader's override mode);
  that matcher returns ambiguous (orphan) rather than guessing when two
  videos share a basename, and non-annotation JSON (e.g. `metadata.json`) in
  a `scale/` dir is skipped. Batch-level pairs carry the matched video's
  `snippet_dir` so `validation.py`'s `snippet_count` stays correct.



## Inside one pair's validation

The top-level diagrams hide the guards inside `_validate_pair`. This
sequence shows what actually happens for a single `(annotation, video)`
pair — note the early exits and the fact that the parsed annotation
is *reused* (not re-parsed) by the consistency check.

```mermaid
sequenceDiagram
    autonumber
    participant VP as _validate_pair
    participant AC as annotation_checks
    participant VC as video_checks
    participant CC as consistency_checks

    Note over VP: inputs: annotation_path,<br/>video_path, check_video

    VP->>AC: check_annotation_schema(path)
    AC-->>VP: findings, annotation | None,<br/>label_counter

    alt video_path missing OR check_video = False
        Note over VP: early return —<br/>annotation findings only
    else video present
        VP->>VC: probe_video(path)
        VC-->>VP: video_props, findings

        alt annotation parsed AND video opened
            VP->>CC: check_video_annotation_consistency(<br/>path, annotation, video_props)
            CC-->>VP: findings
        else parse failed OR video unreadable
            Note over VP: skip consistency<br/>(no reliable inputs)
        end
    end

    Note over VP: return (combined findings,<br/>label_counter)
```

The consistency step runs even if `video_checks` emitted ERROR findings
(e.g. a bad middle frame), as long as `video_props.opened` is true —
fps / frame_count / dimensions are still authoritative in that case,
and gating would silently hide real annotation-vs-video mismatches.

## Loader architecture — `loaders.py`

The input side of the bridge is a small, explicit contract so that every
format loader looks the same and a new format is additive. Three pieces:

```mermaid
flowchart TD
    LOAD["<b>load(root, dataset_format=…)</b><br/>public dispatch"]
    REG[("<b>registry</b><br/>DatasetFormat → Loader")]
    BASE["<b>Loader (ABC)</b><br/>load(root, **options) → BoxTrackDataset<br/>sniff(root) → bool"]
    HMIE["<b>HmieLoader</b><br/>(dataloader.py)"]
    MOT["<b>MotChallengeLoader</b><br/>(motchallenge.py)"]
    TAO["<b>TaoLoader</b><br/>(tao.py)"]
    VIS["<b>VisDroneVideoLoader</b><br/>(visdrone.py)"]
    NEW["CocoLoader, YoloLoader, …<br/>(future)"]

    LOAD -->|get_loader| REG
    REG --> HMIE
    REG --> MOT
    REG --> TAO
    REG --> VIS
    REG -.-> NEW
    HMIE -->|subclasses| BASE
    MOT -->|subclasses| BASE
    TAO -->|subclasses| BASE
    VIS -->|subclasses| BASE
    NEW -.->|subclasses| BASE
    HMIE -->|@register_loader| REG
    MOT -->|@register_loader| REG
    TAO -->|@register_loader| REG
    VIS -->|@register_loader| REG
    NEW -.->|@register_loader| REG

    classDef entry fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;
    classDef store fill:#f3e5f5,stroke:#7b1fa2;
    classDef impl fill:#e8f5e9,stroke:#2e7d32;
    classDef planned fill:#f5f5f5,stroke:#9e9e9e,color:#616161;
    class LOAD entry;
    class REG store;
    class HMIE,MOT,TAO,VIS impl;
    class NEW planned;
```

- **`Loader` (ABC).** The contract: a concrete loader sets a `format`
  (`DatasetFormat`) class attribute and implements
  `load(self, root, **options) -> BoxTrackDataset`. An optional `sniff(root) -> bool`
  classmethod is the autodetection hook (default `False`).
- **`register_loader`.** A decorator that records `format → loader-class` in
  the registry. This is the extension point — adding a loader does not touch
  any dispatch code.
- **`load(root, *, dataset_format=…, **options)`.** The public entry point.
  Resolves the loader from the registry and calls it. `dataset_format` accepts
  a `DatasetFormat` or its string value; pass `None` to autodetect via
  `sniff` (no format implements detection rules yet, so an explicit format is
  required in practice). `**options` pass through to the loader (e.g. HMIE's
  `require_video`, MOTChallenge's `annotation_source`, TAO's `probe_images`, or
  VisDrone Video's `variant`). `load_hmie(...)`, `load_motchallenge(...)`,
  `load_tao(...)`, and `load_visdrone_video(...)` are thin convenience wrappers.

This mirrors the validator: `validate(path, dataset_format=…)` dispatches the
same way. Loaders and validators are siblings — both are thin, format-specific
consumers of the `_formats/<format>/` layer.

### Loader conventions

Every loader honors the same contract so callers and converters can rely on it:

- **Return, don't raise, on bad data.** Loading is best-effort: an item that
  cannot be parsed is skipped and logged at WARNING; the loader returns a
  (possibly empty) `BoxTrackDataset`. The authoritative "*why* is it bad" answer is a
  separate pass — `validate()`.
- **Keyword-only options, consistent names.** Loader-specific options are
  keyword-only; shared semantics (e.g. `require_video` for any FMV format)
  keep the same name and meaning across loaders.
- **One model out.** Every loader produces the same `BoxTrackDataset`, so any
  converter consumes the result regardless of input format.

### Common data model (temporal sequences vs. still images)

The neutral model (`model.py`) is the agreed common representation, and the
`Loader` contract is intentionally model-shaped, not format-shaped. Today the
model is a temporal box-track IR: a `BoxTrackDataset` holds `VideoSequence`s of
`BoxAnnotation`s. A `VideoSequence` may be backed by a single video file
(`video_path`, HMIE) or by ordered frame images (`frame_dir` / pattern for
MOTChallenge and VisDrone Video, explicit `frame_files` for TAO, plus
`frame_filename()` / `frame_path()` helpers). **Still-image datasets** (COCO,
image-mode YOLO) will be represented by an `ImageSample` sibling of
`VideoSequence`
carrying the same `BoxAnnotation`, added together with the first still-image
loader — at which point `BoxTrackDataset` holds a mix of sequence and image
samples. The `Loader.load() -> BoxTrackDataset` contract does not change when
that lands, which is the whole point of fixing the contract now. (We deliberately
do not define `ImageSample` ahead of a loader that produces it, to avoid a
consumer-less abstraction we'd likely get wrong.)

### Adding a new loader

To support a new input format `foo`:

1. Add `FOO = "foo"` to `DatasetFormat` (`_types.py`).
2. Create `_formats/foo/` with that format's discovery + schema/parse helpers
   (mirrors `_formats/hmie/`), keeping format specifics isolated.
3. Write a `FooLoader(Loader)` with `format = DatasetFormat.FOO` and a `load`
   that returns a `BoxTrackDataset`, following the conventions above. Decorate it with
   `@register_loader`. (Use `HmieLoader` in `dataloader.py` as the template.)
4. Ensure the module is imported so registration runs (export it from the
   package `__init__`).
5. Optionally add format-specific validation under `_formats/foo/` and a
   `DatasetFormat.FOO` branch in `validation.py`.

`MotChallengeLoader` in `motchallenge.py`, `TaoLoader` in `tao.py`, and
`VisDroneVideoLoader` in `visdrone.py` are the image-sequence examples.
MOTChallenge expects a standard benchmark root with `train/` and/or `test/` and
reads `gt/gt.txt` or `det/det.txt`; TAO expects `annotations/train.json`,
`validation.json`, and/or `test.json` / the official
`test_without_annotations.json`; VisDrone Video expects official VID/MOT split
roots with `sequences/<name>/0000001.jpg` and `annotations/<name>.txt`, or a
parent that contains multiple such split roots. All set image-sequence metadata
and helpers (`VideoSequence.frame_dir`, `frame_filename()`, `frame_path()`)
instead of `video_path`.

`databridge.load(root, dataset_format="foo")` then works with no changes to
the dispatcher, and any existing converter accepts the result.

## Loading — `dataloader.py`

`load_hmie(root)` (and the `HmieLoader` behind it) is the other consumer of
the discovery + schema layers.
Where `validate()` runs *checks* on each pair, the loader *parses* each
pair into the neutral in-memory model defined in `model.py`:

```
discover_hmie_pairs(root) ─► [SnippetPair]
                                  │  (per pair)
                                  ▼
              check_annotation_schema(path) ─► ScaleAnnotation
                                  │
                                  ▼
        VideoSequence(boxes=[BoxAnnotation, ...], video_meta, fps, ...)
                                  │
                                  ▼
        BoxTrackDataset(sequences=[...], categories={uri: id})
```

`BoxTrackDataset` / `VideoSequence` / `BoxAnnotation` live in `model.py`, not in
`dataloader.py`, on purpose: the model is the **format-neutral hub** of the
bridge. `load_hmie` is one loader that produces it; future loaders (COCO,
YOLO, ...) produce the same `BoxTrackDataset`, and converters consume `BoxTrackDataset`
without depending on any loader. That is what makes databridge an N-to-M
bridge (loaders × converters) rather than an HMIE-to-X path.

Design points:

- **Reuses, never re-walks.** Pairing comes from `discovery.py` and
  parsing from `annotation_checks.check_annotation_schema` — the same
  robust paths the validator uses (unwrapped-format handling, duplicate
  keys, size limits). It does not reimplement the notebook's
  `rglob("*CDAO*.json")` / `seq_mp4` assumptions.
- **Loading ≠ validating.** Best-effort by design: an unparseable
  annotation is skipped (logged), and a box with any missing
  coordinate is dropped. Callers wanting *why* data is bad run
  `validate()`.
- **Dataset-wide category map.** `category_id`s are assigned once across
  the whole dataset, so a label maps to the same id in every sequence.
- **`require_video`.** Default loading never opens videos
  (`num_frames` comes from the max annotated frame index, core deps
  only). `require_video=True` probes each video via `video_checks`
  (the `video` extra), takes `num_frames` from the true frame count,
  and skips snippets whose video is missing or unreadable.
- **Override mode.** Passing `annotation_dir` / `video_dir` bypasses
  discovery for flat (non-nested) layouts, pairing by matching a
  video's stem against the annotation filename.

## The model as a MAITE dataset

`BoxTrackDataset` does double duty. It is the neutral hub every converter
consumes, **and it natively implements the MAITE multi-object-tracking
protocol** — so `load_hmie(root)` returns an object a MAITE model or metric
can consume directly, with no adapter call:

```python
ds = load_hmie(root)
stream, target, metadata = ds[0]      # MAITE MOT item — one per video
ds = ds.with_mot_options(empty_frame_policy="all")   # configure the MOT view
```

The MAITE surface is a *view computed from the typed records*, which stay
on the object. That is the whole point: a stock MAITE target carries only
boxes / labels / scores / track-ids, but converting to on-disk formats
needs the source detail (ontology URIs, per-box attributes like
truncation / occlusion, keyframe-vs-interpolated, string track UUIDs).
Keeping the typed `VideoSequence` / `BoxAnnotation` records behind the
MAITE view lets the same object serve both consumers without losing
anything on the conversion path.

```mermaid
flowchart TB
    subgraph src [On-disk source — HMIE / Scale]
        direction TB
        S1[ontology<br/>category_uri]
        S2[per-box attrs<br/>truncation /<br/>occlusion]
        S3[keyframe_type<br/>is_inferred]
        S4[track_uuid<br/>track_id]
        S5[bbox<br/>category_id]
    end

    src ==> L["<b>load_hmie</b>"]
    L ==> M[/"<b>BoxTrackDataset</b><br/>typed VideoSequence · BoxAnnotation<br/>— all source detail retained —"/]

    M -. computed view .-> V["MAITE MOT view<br/>ds[i] → (VideoStream,<br/>target, metadata)"]
    M ==> W["converters / writers<br/>read the typed records"]
    V --> MC([MAITE model<br/>/ metric])
    W --> O([MOTChallenge · VisDrone<br/>COCO · YOLO])

    classDef hub fill:#fff8e1,stroke:#f57c00,stroke-width:2px;
    classDef impl fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;
    classDef planned fill:#f5f5f5,stroke:#9e9e9e,color:#616161;
    classDef src fill:#fafafa,stroke:#bdbdbd,color:#424242;
    class M hub;
    class L,V,MC impl;
    class W,O planned;
    class S1,S2,S3,S4,S5 src;
```

Mechanics that keep this honest:

- **`databridge.maite` is optional and lazy.** Core `import databridge`,
  `load`, and `validate` never import `maite`, `numpy`, or a video decoder.
  The view machinery is imported lazily inside `ds[i]`; indexing without the
  `databridge[maite]` extra raises an actionable error. Conformance is
  *structural* (no runtime `maite` import) — `BoxTrackDataset` satisfies
  `maite.protocols.multiobject_tracking.Dataset` by shape. (The `maite`
  package itself is only used in tests; the `[maite]` extra ships it for the
  consumer's convenience since anyone using the MAITE surface has it anyway.)
- **MOT is the format.** For video box-tracks the MOT surface is the whole
  surface (`ds[i]` is one video). Object-detection is a *different AI task*,
  not a conversion target, so there is no OD projection here.
- **Two length / iteration views.** `len(ds)` / `ds[i]` / `for x in ds` are
  the MAITE **item** view — one item per *video-bearing* sequence. The
  **record** view is `ds.sequence_count` / `ds.iter_sequences()` /
  `ds.sequences` — every loaded sequence, including video-less ones (which
  the validator and converters walk). They differ when a sequence has
  annotations but no video. `sequences` is stored as a tuple, so the cached
  video-bearing item list (`_mot_sequences`) is O(1) and never stale.
- **`ds.with_mot_options(...)` configures the MOT view** (`empty_frame_policy`,
  `decoder`, `dataset_id`) by returning a copy — it is *not* an
  adapter/conversion call (the model is already MAITE).
- **`empty_frame_policy="all"` needs an exact frame count.** It only streams
  every frame when `VideoSequence.num_frames_exact` is set (the loader sets
  it under `require_video=True`); otherwise the count is an estimate and the
  view falls back to annotated frames with a warning.

Verification: beyond `isinstance`, the suite drives the dataset through
MAITE's own `maite.tasks.predict` with a stub model — proving the object is
actually consumable by MAITE tooling (dataloader + collation + iteration),
not merely shaped right.

`databridge.maite` layout: `_mot.py` (`build_mot_item` — the MOT view),
`_decode.py` (the pluggable `Decoder` protocol + PyAV backend), `_common.py`
(numpy-array + datum-metadata helpers).

## Writer architecture — `writers.py`

The output side mirrors the loader architecture: a small, explicit contract so
every format writer looks the same and a new output format is additive. A
*writer* takes the neutral `BoxTrackDataset` and serialises it to one on-disk
format; `conversion.convert` pairs a loader and a writer for end-to-end
on-disk → on-disk conversion.

```mermaid
flowchart TD
    CONVERT["<b>convert(src, dest, input_format=…, output_format=…)</b><br/>conversion.py — load + write"]
    WRITE["<b>write(dataset, dest, output_format=…)</b><br/>writers.py — public dispatch"]
    REG[("<b>registry</b><br/>DatasetFormat → Writer")]
    BASE["<b>Writer (ABC)</b><br/>write(dataset, dest, **options) → list[Path]"]
    HMIE["<b>HmieWriter</b><br/>(_formats/hmie/writer.py)"]
    NEW["MotWriter, YoloWriter, …<br/>(future)"]

    CONVERT -->|load → write| WRITE
    WRITE -->|get_writer| REG
    REG --> HMIE
    REG -.-> NEW
    HMIE -->|subclasses| BASE
    NEW -.->|subclasses| BASE
    HMIE -->|@register_writer| REG
    NEW -.->|@register_writer| REG

    classDef entry fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;
    classDef store fill:#f3e5f5,stroke:#7b1fa2;
    classDef impl fill:#e8f5e9,stroke:#2e7d32;
    classDef planned fill:#f5f5f5,stroke:#9e9e9e,color:#616161;
    class CONVERT,WRITE entry;
    class REG store;
    class HMIE impl;
    class NEW planned;
```

- **`Writer` (ABC).** A concrete writer sets a `format` (`DatasetFormat`) class
  attribute and implements `write(self, dataset, dest, **options) -> list[Path]`
  (the files it created).
- **`register_writer`.** A decorator that records `format → writer-class` in the
  registry. This is the extension point — adding a writer touches no dispatch code.
- **`write(dataset, dest, *, output_format, **options)`.** The public entry
  point; resolves the writer from the registry and calls it.
- **`convert(src, dest, *, input_format, output_format, read_options=…, **write_options)`**
  (`conversion.py`). End-to-end: `write(load(src, input_format), dest, output_format)`.
  It binds to the neutral model on both sides, so any registered input format
  can be converted to any registered output format.

### Writer conventions

- **Consume the neutral model, never a loader or raw format.** A writer's only
  inputs are a `BoxTrackDataset` and a destination.
- **Map best-effort; drop with a warning, don't crash.** Data the target format
  cannot represent is dropped and logged at WARNING; destination/IO failures raise.
- **Keyword-only options.** Format variants (e.g. MOT16 vs MOT20 columns) are a
  writer option, not a separate `DatasetFormat`.

### Reference writer: HMIE (round-trip proof)

`HmieWriter` (`_formats/hmie/writer.py`) is the reference writer that proves the
architecture. Because databridge also has the HMIE *loader*, it closes a full
round trip:

```
load_hmie(src) → BoxTrackDataset → write(…, output_format="hmie") → load_hmie(dest)
```

recovers the same box/category content — verifying both the writer contract and
that `BoxTrackDataset` is a lossless hub. The writer emits annotations with
`annotation_frame_rate == video fps` (so `key == frame_index` maps straight
back) and labels as ontology URIs (so categories re-resolve to the same names);
the integer `category_id` is reassigned on reload, so round-trip equivalence is
by `category_uri`, not by id.

### Adding a new writer

1. Add the format to `DatasetFormat` (`_types.py`) if it isn't there yet.
2. Create `_formats/<fmt>/writer.py` with a `Writer` subclass
   (`format = DatasetFormat.<FMT>`), decorated with `@register_writer`.
3. Import it from the package `__init__` so registration runs.
4. `databridge.write(ds, dest, output_format="<fmt>")` and `convert(...)` then
   work with no changes to the dispatcher.

### What the model already gives writers

- **Frame indices are video-frame-space.** `BoxAnnotation.frame_index` is the
  mapped index (not the raw label key), so frame-indexed targets like
  MOTChallenge map straight across without re-deriving the clock.
- **Tracks and dataset-wide category ids** (`track_id`, `category_id`) are
  already in the model, which the track-centric (MOT) and class-indexed (YOLO)
  formats need.
