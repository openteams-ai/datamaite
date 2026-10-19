# Box tracks and multi-object tracking

## Tracks

{term}`Multi-object tracking <MOT>` follows several objects through a video at
once. Each object gets a *track*: an identity that persists across frames, with
one bounding box per frame where the object is labeled. A {term}`box track` is
that sequence of boxes for one object.

Tracks answer questions single-frame detection cannot: is the same vehicle seen
in frame 10 and frame 200, how long does an object stay in view, and does a model
keep an object's identity or swap it with another.

## Keyframes and interpolated boxes

Labelers rarely draw a box on every frame. They place boxes on *keyframes*, and
the labeling tool fills in the frames between them. DataMAITE keeps that
distinction from the source: each box records its keyframe type and whether it
was interpolated (`is_inferred`) rather than placed by a person.

## Box coordinates

Boxes are stored once, in absolute-pixel `(left, top, width, height)` form, and
converted to other shapes, such as MAITE's `(x1, y1, x2, y2)` or YOLO's normalized
center form, only when they are read or written in that format.

## Two ways to count a video dataset

A tracking dataset has two natural sizes. The *item* view counts sequences that
carry pixels, whether as a video file, a table of frame files, or a frame
directory plus a filename pattern, because a MAITE tracking item needs pixels to
decode. MOTChallenge, TAO, and VisDrone Video sequences store frames rather than
a video file and are still items. The *record* view counts every loaded
sequence, including ones that have annotations and no pixels, which validation
and conversion still need to see. The two counts differ only when a sequence has
annotations and no pixels.

## Further reading

- [Empty frames in tracking data](empty-frames.md)
- [MAITE documentation](https://mit-ll-ai-technology.github.io/maite/)
- [Working with FMV datasets tutorial](../tutorials/Working_with_FMV_datasets.ipynb)
