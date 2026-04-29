# Architecture

A reviewer's map of the codebase. Read this with the code open.

## What databridge does (today)

One pipeline: **walk a dataset root on disk → pair each annotation JSON
with its video → run checks on each pair → aggregate findings into a
`ValidationResult` → render a report**. The only format currently
implemented is HMIE (Scale Video Playback JSON + snippet folder layout).
Everything is structured so other formats can be added behind the same
public entrypoint without touching the CLI or reporting layers.

## Project layout

```
src/databridge/
    __init__.py              Public API surface
    _cli.py                  CLI entrypoint (`databridge validate ...`)
    _types.py                Shared types: Finding, Severity, ValidationResult
    _cache.py                On-disk cache for expensive video probes
    _report.py               Text / JSON / JSONL / HTML report rendering
    _version.py              Package version
    validation.py            Orchestration: discovery -> checks -> aggregation
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
