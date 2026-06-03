"""Databridge's neutral in-memory **box-track** dataset model.

This is the format-agnostic intermediate representation at the center of
the bridge: every *loader* produces it and every *converter* consumes it.
A loader (e.g. :func:`databridge.load_hmie`) turns an on-disk dataset into
a :class:`BoxTrackDataset`; a converter turns a :class:`BoxTrackDataset`
into an output format (MOTChallenge, YOLO, ...) or a MAITE-protocol view
(:mod:`databridge.maite`). Because the model is tied to neither a specific
input nor a specific output format, databridge is an N-to-M bridge -- add a
loader to gain an input, add a converter to gain an output -- rather than a
one-off HMIE-to-X path.

Scope: this is deliberately a **box-track** IR -- a set of videos, each with
per-frame *bounding-box* tracks. It is not a universal model for every future
databridge format. Tasks whose labels are not boxes (semantic segmentation
masks, image-level classification, keypoints, ...) need their own
source-preserving records and adapters; do not stretch this model to cover
them. The name is ``BoxTrackDataset`` (not ``Dataset``) to keep that scope
honest and to avoid colliding with MAITE's own task-specific ``Dataset``
protocols.

The dataclasses are intentionally plain (no external protocol dependency).
:class:`BoxTrackDataset` itself implements the MAITE multi-object-tracking
protocol via lazily-imported view machinery in :mod:`databridge.maite`; no
separate adapter call is required. MOT-view options are configured with
:meth:`BoxTrackDataset.with_mot_options`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

# Bounding box as (left, top, width, height) in pixels.
BBox = tuple[float, float, float, float]


def category_name_from_uri(category_uri: str) -> str:
    """Return the final path segment of an ontology URI.

    ``http://example.com/ontology/a/FOO_000`` -> ``FOO_000``. A trailing
    slash is ignored; a plain string with no ``/`` passes through unchanged.
    This is the single definition of "category display name" shared by the
    loader (when resolving categories) and :meth:`BoxTrackDataset.index2label`.
    """
    return category_uri.rstrip("/").split("/")[-1]


@dataclass(frozen=True)
class BoxAnnotation:
    """One bounding box on one frame of one track.

    ``bbox`` is ``(left, top, width, height)`` in pixels, matching the
    per-frame fields of the source annotation. ``category_id`` is assigned
    per dataset (stable across all sequences in the same
    :class:`BoxTrackDataset`), and ``category_name`` is the final path segment
    of the ontology URI.

    ``keyframe_type`` (start/middle/end) and ``is_inferred`` come straight
    from the source frame: ``is_inferred=True`` marks a tool-interpolated
    box rather than a human-placed keyframe, which a downstream consumer
    may want to weight or filter.
    """

    track_uuid: str
    track_id: int
    category_id: int
    category_uri: str
    category_name: str | None
    bbox: BBox
    attributes: dict[str, Any]
    frame_index: int
    timestamp: float | None
    keyframe_type: str | None = None
    is_inferred: bool | None = None


@dataclass(frozen=True)
class VideoSequence:
    """One snippet: a video plus all of its box annotations.

    ``video_path`` is ``None`` when the snippet has an annotation but no
    discoverable video (and the loader was not asked to require one).

    ``num_frames`` is the video's *true* frame count only when the loader
    probed the video. Otherwise it is a lower-bound *estimate* -- the
    maximum annotated ``frame_index`` plus one (``None`` when the snippet
    has no boxes). Because labeling usually stops before the end of a
    snippet, this estimate understates the true video length; do not treat
    the non-probed value as the real frame count. ``duration``
    (``num_frames / fps``) inherits the same caveat.

    ``status`` is the source task status (``completed``/``pending``/etc.),
    so a consumer can filter non-final tasks. ``video_meta`` is video-level
    metadata (e.g. codec, dimensions, plus any global attributes);
    ``metadata`` is the source task-level ``metadata`` object (which may
    carry the original video filename).

    ``width``/``height`` (pixels) and ``size_bytes`` (file size) are the
    video-level fields a MAITE datum-metadata record needs. They are
    ``None`` when neither the source annotation metadata nor a video probe
    supplied them; the MAITE surface then probes the video lazily.
    """

    video_id: int
    video_path: str | None
    fps: float
    num_frames: int | None
    duration: float | None
    annotation_path: str
    status: str | None = None
    video_meta: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    boxes: list[BoxAnnotation] = field(default_factory=list)
    width: int | None = None
    height: int | None = None
    size_bytes: int | None = None
    # True only when ``num_frames`` is the video's *probed* frame count (the
    # loader sets it under ``require_video=True``). When False, ``num_frames``
    # is the lower-bound estimate (max annotated frame + 1) and must not be
    # treated as the true length -- e.g. ``empty_frame_policy="all"`` refuses
    # to emit every frame against a merely-estimated count.
    num_frames_exact: bool = False

    def boxes_by_frame(self) -> dict[int, list[BoxAnnotation]]:
        """Group this sequence's boxes by video frame index.

        ``boxes`` is stored flat (all tracks, all frames); per-frame
        consumers -- the MOTChallenge writer, the MAITE MOT/OD surfaces --
        need the boxes belonging to each frame. Keys are the frame indices
        that actually carry at least one box, in ascending order; a frame
        with no boxes is absent rather than mapped to an empty list. Box
        order within a frame follows the flat ``boxes`` order (track order).
        """
        by_frame: dict[int, list[BoxAnnotation]] = {}
        for box in self.boxes:
            by_frame.setdefault(box.frame_index, []).append(box)
        return {key: by_frame[key] for key in sorted(by_frame)}


@dataclass(frozen=True)
class BoxTrackDataset:
    """A loaded box-track dataset that *is* a MAITE multi-object-tracking dataset.

    ``sequences`` + ``categories`` are the source-preserving records every
    converter consumes. The object also implements the MAITE MOT protocol:
    ``len(ds)`` is the number of video-bearing sequences and ``ds[i]`` yields
    ``(VideoStream, MultiobjectTrackingTarget, DatumMetadata)`` for video ``i``
    (see :func:`databridge.maite._mot.build_mot_item`). The MAITE surface is
    computed lazily; importing/indexing it requires the ``databridge[maite]``
    extra, but ``import databridge`` / ``load`` / ``validate`` never touch it.

    Two distinct "size" views, because the object wears two hats:

    * ``len(ds)`` / ``ds[i]`` / iterating ``ds`` are the **MAITE item** view --
      one item per *video-bearing* sequence (MOT needs pixels). This is what
      MAITE tooling consumes.
    * ``ds.sequence_count`` / ``ds.iter_sequences()`` / ``ds.sequences`` are the
      **record** view -- every loaded sequence, including those with no video.
      This is what the validator and converters walk.

    ``sequences`` is stored as a tuple (coerced in ``__post_init__``) so the
    record set is immutable and the cached MAITE item list cannot go stale.

    MOT-view options (``empty_frame_policy``, decoder, ``dataset_id``) are set
    at construction or copied onto a new instance via :meth:`with_mot_options`.
    """

    sequences: tuple[VideoSequence, ...]
    categories: dict[str, int]
    dataset_id: str = "databridge"
    empty_frame_policy: str = "annotated"
    # MOT decode backend override (a databridge.maite._decode.Decoder). None ->
    # the default PyAV decoder, resolved lazily inside build_mot_item so the
    # core never imports the decoder. Annotated Any to keep model.py import-clean.
    _decoder: Any = field(default=None, compare=False, repr=False)
    # Per-instance memo store (e.g. probed VideoInfo). Mutated in place, never
    # rebound, so it is compatible with frozen=True.
    _caches: dict[str, Any] = field(default_factory=dict, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        # Accept any iterable of sequences but store an immutable tuple, so a
        # caller cannot ``ds.sequences.append(...)`` and invalidate the cached
        # ``_mot_sequences`` list. object.__setattr__ is the frozen-safe rebind.
        if not isinstance(self.sequences, tuple):
            object.__setattr__(self, "sequences", tuple(self.sequences))

    @cached_property
    def _mot_sequences(self) -> tuple[VideoSequence, ...]:
        """Video-bearing sequences (MOT needs pixels) -- the MAITE item list.

        Computed once and cached on the instance so ``len(ds)`` is O(1) and
        full iteration is O(N), not O(N^2). Safe because ``sequences`` is an
        immutable tuple (see :meth:`__post_init__`).
        """
        return tuple(seq for seq in self.sequences if seq.video_path is not None)

    def __len__(self) -> int:
        # MAITE item count == number of video-bearing sequences (O(1), cached).
        return len(self._mot_sequences)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        seq = self._mot_sequences[index]  # IndexError past the end -> stops iteration
        try:
            from databridge.maite._mot import build_mot_item
        except ImportError as exc:
            # Narrow guard: a missing build_mot_item *symbol* is an internal
            # error (re-raise as-is); only a missing module/dependency
            # (numpy/av not installed) maps to the "install the extra" hint.
            if "build_mot_item" in str(exc):
                raise
            raise ImportError(
                "Indexing a databridge dataset as a MAITE dataset requires the optional "
                "video stack. Install it with: pip install databridge[maite]"
            ) from exc
        return build_mot_item(self, seq)

    @property
    def metadata(self) -> dict[str, Any]:
        """MAITE ``DatasetMetadata``: dataset id + ``index2label`` map."""
        return {"id": self.dataset_id, "index2label": self.index2label()}

    def with_mot_options(
        self,
        *,
        empty_frame_policy: str | None = None,
        decoder: Any = None,
        dataset_id: str | None = None,
    ) -> BoxTrackDataset:
        """Return a copy of this dataset with MOT-view options applied.

        Replaces the adapter-style ``to_maite_*`` call: the dataset is already
        a MAITE MOT dataset, so this only configures *how* the MAITE surface
        decodes/labels it. Omitted (``None``) options keep their current value.

        Parameters
        ----------
        empty_frame_policy
            ``"annotated"`` (default) emits only annotated frames; ``"all"``
            emits every video frame (requires an exact, probed frame count --
            see ``VideoSequence.num_frames_exact`` -- else it falls back to
            ``"annotated"`` with a warning).
        decoder
            Video-decode backend (a ``databridge.maite._decode.Decoder``);
            ``None`` keeps the current one (default PyAV when never set).
        dataset_id
            Value for the dataset-level ``metadata['id']``.
        """
        from dataclasses import replace

        if empty_frame_policy is not None and empty_frame_policy not in ("annotated", "all"):
            raise ValueError(f"empty_frame_policy must be 'annotated' or 'all', got {empty_frame_policy!r}")
        return replace(
            self,
            empty_frame_policy=empty_frame_policy if empty_frame_policy is not None else self.empty_frame_policy,
            dataset_id=dataset_id if dataset_id is not None else self.dataset_id,
            _decoder=decoder if decoder is not None else self._decoder,
        )

    @property
    def sequence_count(self) -> int:
        """Number of loaded sequences (the **record** count, incl. video-less).

        Distinct from ``len(self)``, which is the MAITE item count
        (video-bearing sequences only).
        """
        return len(self.sequences)

    def iter_sequences(self) -> Iterator[VideoSequence]:
        """Iterate the typed source records.

        Iterating the dataset itself (``for x in ds``) yields MAITE
        ``(stream, target, metadata)`` items; use this to walk the
        :class:`VideoSequence` records instead.
        """
        return iter(self.sequences)

    @property
    def num_boxes(self) -> int:
        """Total box annotations across all sequences."""
        return sum(len(seq.boxes) for seq in self.sequences)

    def index2label(self) -> dict[int, str]:
        """Map ``category_id`` to a human-readable label.

        Inverts ``categories`` (URI -> id) to ``{id: name}`` where ``name``
        is the final path segment of the ontology URI
        (:func:`category_name_from_uri`). This is the form MAITE's
        ``DatasetMetadata.index2label`` expects. Unlabeled tracks
        (``category_id == -1``) are not in ``categories`` and so are absent here.
        """
        return {cid: category_name_from_uri(uri) for uri, cid in self.categories.items()}
