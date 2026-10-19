# Dataset protocols and structural compatibility

Test and evaluation (T&E) tools need to read datasets and run models without
being written for each one. [MAITE](https://mit-ll-ai-technology.github.io/maite/)
(Modular AI Trustworthy Engineering), developed under the JATIC program, solves
this with *protocols*: shared descriptions of what a dataset or model must look
like, written as Python structural types.

## Structural, not inherited

A protocol says "anything with these methods and these return types counts".
An object does not have to inherit from a MAITE base class or be wrapped by an
adapter; if it has the right shape, any MAITE-aware tool can use it. For a
dataset that shape is small: `len(dataset)` gives the number of items, and
`dataset[i]` returns one item as an `(input, target, metadata)` tuple whose
contents depend on the task.

## Tasks

MAITE defines a protocol per machine-learning task. DataMAITE uses three:

| Task | One item is | DataMAITE dataset |
| --- | --- | --- |
| Multi-object tracking ({term}`MOT`) | One video: a frame stream and its tracks | `BoxTrackDataset` |
| Object detection | One image and its boxes | `ObjectDetectionDataset` |
| Image classification | One image (or image region) and its label | `ImageClassificationDataset` |

## How DataMAITE applies it

DataMAITE's loaded datasets *are* MAITE datasets: there is no conversion step
between loading data and handing it to a MAITE tool. The same object also keeps
the source records (sequences, tracks, categories, provenance) that writers use
to save the data in another format. Pixels are decoded only when an item is
read, so a dataset can be loaded and inspected without decoding any images or
video.

Hugging Face video classification data loads as a `VideoClassificationDataset`
with no MAITE surface, because MAITE has no video-classification protocol yet.

## Further reading

- [MAITE documentation](https://mit-ll-ai-technology.github.io/maite/)
- [Architecture](../reference/architecture.md): how the loaders, the in-memory
  models, and the writers fit together.
