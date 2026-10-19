# Glossary

Key terms used in the DataMAITE documentation. Longer background on many of them
is in [Explanations](explanations/index.md).

```{glossary}
Box track
  The sequence of bounding boxes that follows one object through a video, one
  box per labeled frame, under a single track identity. DataMAITE's
  `BoxTrackDataset` holds box tracks. See
  [Box tracks and multi-object tracking](explanations/box-tracks.md).

Empty-frame policy
  The setting that decides which frames a tracking dataset yields: `annotated`
  (the default) yields only frames with at least one box, and `all` yields every
  frame. `all` needs the video's true frame count. See
  [Empty frames in tracking data](explanations/empty-frames.md).

FMV
  Full-motion video: continuous video, as opposed to still images. See
  [Full-motion video and snippets](explanations/full-motion-video.md).

Fixed-taxonomy writer
  A writer for a format whose class ids come from a fixed class table:
  MOTChallenge and VisDrone. It translates each box's class into that table and
  warns when the translation is not explicit. See
  [Category taxonomies and ID density](explanations/category-taxonomies.md).

Generic-id reinterpretation
  Writing a loader-assigned class id into a fixed class table as if it already
  used that table's numbering. The output can look valid but carry the wrong
  labels, so DataMAITE warns when it happens; pass `class_map=` to translate
  explicitly. See
  [Category taxonomies and ID density](explanations/category-taxonomies.md).

ID density
  Whether a taxonomy's class ids are contiguous integers starting at zero
  (dense) or sparse, fixed, or non-numeric. DataMAITE keeps source ids as given
  and derives dense ids as a separate view for formats, such as YOLO, that need
  them. See [Category taxonomies and ID density](explanations/category-taxonomies.md).

MAITE
  Modular AI Trustworthy Engineering: a JATIC library of protocols that let
  datasets and models work with T&E tools without adapters. See
  [Dataset protocols and structural compatibility](explanations/maite-dataset-protocols.md).

MOT
  Multi-object tracking: following several objects through a video at once,
  each with a persistent identity. Also the MAITE task DataMAITE's video datasets
  implement. See [Box tracks and multi-object tracking](explanations/box-tracks.md).

Snippet
  One short labeled video clip with its annotations, the unit of a labeled
  video delivery. On disk, a directory holding a `seq_mp4/` or `seq_ts/` video
  folder and its annotation JSON. See
  [Full-motion video and snippets](explanations/full-motion-video.md).

Taxonomy
  The table that gives a dataset's class ids their meaning, such as which id
  means "car". See
  [Category taxonomies and ID density](explanations/category-taxonomies.md).
```
