# Empty frames in tracking data

A video has many more frames than labeled frames. An *empty frame* is a frame
with no boxes: nothing to track there, or nothing labeled. Whether an evaluation
should see those frames depends on what it measures. Scoring a tracker only where
labels exist hides false positives on frames where nothing is present; feeding it
every frame shows them.

## Two policies

DataMAITE's tracking view has an {term}`empty-frame policy` with two settings:

- **`annotated`** (the default) yields only frames that have at least one box.
- **`all`** yields every frame of the video, including empty ones.

## Why `all` needs the true frame count

To yield every frame, DataMAITE has to know how many frames the video has.
Annotations alone only give a lower bound: the last labeled frame. Labeling often
stops before the video ends, so that estimate can be short. An authoritative
count can come from any of three places: probing the video (the HMIE loader does
this when loaded with `require_video=True`), the sequence length in a
MOTChallenge `seqinfo.ini`, or counting the frame files on disk (TAO, VisDrone
Video). Without an authoritative count, asking for `all` falls back to
`annotated` with a warning rather than silently dropping the frames after the
last label.

## Further reading

- [Box tracks and multi-object tracking](box-tracks.md)
- [Architecture](../reference/architecture.md): the MOT view and its options.
