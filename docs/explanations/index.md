# Explanations

Background on the concepts behind DataMAITE: what the data looks like, how it is
modeled, and why DataMAITE behaves the way it does. These pages explain ideas
rather than steps; for steps, see the [Tutorials](../tutorials/index.md), and for
exact APIs, see the [Reference](../reference/index.md). Terms are defined in the
[Glossary](../glossary.md).

- **[Dataset protocols and structural compatibility](maite-dataset-protocols.md)**:
  how a dataset can satisfy MAITE without an adapter.
- **[Full-motion video and snippets](full-motion-video.md)**:
  how labeled video arrives as snippets and why it is validated.
- **[Box tracks and multi-object tracking](box-tracks.md)**: following objects
  across frames, keyframes, and the two ways to count a video dataset.
- **[Empty frames in tracking data](empty-frames.md)**: which frames a tracking
  dataset yields, and why the frame count matters.
- **[Category taxonomies and ID density](category-taxonomies.md)**: what a class
  id means, and what happens when you move between class tables.

```{toctree}
:maxdepth: 1
:hidden:

maite-dataset-protocols
full-motion-video
box-tracks
empty-frames
category-taxonomies
```
