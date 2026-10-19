# Full-motion video and snippets

## Full-motion video

{term}`Full-motion video <FMV>` (FMV) is continuous video, as opposed to still
images. For T&E, FMV datasets usually pair video with tracks: boxes that follow
each object from frame to frame (see [Box tracks](box-tracks.md)). Video brings
concerns that still images do not have: files that fail to decode, frame counts
that differ from what the annotations imply, and annotation files that do not
match their video.

## Snippets

Long recordings are cut into short clips for labeling. Each clip is a
{term}`snippet`: one video plus the annotations made for it. A labeled delivery
is a batch of snippets, and checking a delivery means checking every snippet's
video and annotations and that each one has the other.

## Why validate before use

A delivery can look complete and still be unusable: a video that will not open,
an annotation with no video, or JSON that does not match the annotation format.
Validation checks folder structure, media integrity, annotation completeness,
and required fields, and reports PASS or FAIL with the findings, so problems are
found at intake rather than in the middle of an evaluation.

## Further reading

- [Annotation schemas](../reference/schemas.md)
- [HMIE Validation tutorial](../tutorials/HMIE_Validation.ipynb)
