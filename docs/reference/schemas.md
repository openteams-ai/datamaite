# Schemas

## Scale Video Annotation Schema

JSON Schema for Scale AI Video annotations. It covers the completed task response, including annotation tracks and per-frame bounding boxes.

```{note}
This schema is a human-readable description of the Scale Video Playback format.
It is **not** what the validator loads at runtime — `datamaite`'s HMIE "Scale
spec compliance" check enforces the equivalent rules through Pydantic models in
`datamaite._formats.hmie.schema` (see [Architecture](architecture.md)). Treat
this file as reference documentation for the format; the Pydantic models are the
source of truth for what the validator accepts.
```

```{literalinclude} schemas/scale-video-playback-v1.schema.json
:language: json
:caption: scale-video-playback-v1.schema.json
```
