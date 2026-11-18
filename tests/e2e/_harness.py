"""The conversion harness snapshot, adjusted for datasets on disk.

Real datasets do not always store image sizes (Hugging Face never does, and
older COCO files written by datamaite may not). A writer that fills a missing
size in is not losing anything, so :func:`disk_snapshot` takes a size the
dataset does not store from the decoded image. The missing size is still
listed in the snapshot's ``problems``.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import ImageObjectDetectionSample

from ..harness import DatasetSnapshot, snapshot

# Formats whose file names start with the split folder (``train/a.jpg``).
_SPLIT_FOLDER_FORMATS = frozenset({"huggingface_vision"})


def split_folders_hold_file_names(fmt: str) -> bool:
    """Whether ``fmt``'s file names start with the image's split folder (``train/a.jpg``)."""
    return fmt in _SPLIT_FOLDER_FORMATS


def split_folders(dataset: ObjectDetectionDataset, fmt: str) -> frozenset[str]:
    """The split folders that start ``dataset``'s file names: its split names for Hugging Face, else none.

    Pass the source's folders when snapshotting a conversion's output too: a
    COCO writer keeps the file names it is given, split folder and all.
    """
    if not split_folders_hold_file_names(fmt):
        return frozenset()
    return frozenset(sample.split for sample in dataset.samples if sample.split)


def sample_key(sample: ImageObjectDetectionSample, folders: frozenset[str] = frozenset()) -> str:
    """The image's file name, without a leading split folder from ``folders``."""
    name = PurePosixPath(str(sample.file_name or sample.path_or_uri or sample.image_id).replace("\\", "/"))
    if len(name.parts) > 1 and name.parts[0] in folders:
        name = PurePosixPath(*name.parts[1:])
    return name.as_posix()


def disk_snapshot(dataset: ObjectDetectionDataset, folders: frozenset[str] = frozenset()) -> DatasetSnapshot:
    """``snapshot(dataset, decode=True)`` keyed by :func:`sample_key`, with missing sizes taken from the pixels."""
    snap = snapshot(dataset, decode=True, key=lambda sample: sample_key(sample, folders))
    tree = snap.tree
    for fields in tree["samples"]:
        decoded_width, decoded_height = fields["decoded_size"]
        if fields["width"] is None:
            fields["width"] = decoded_width
        if fields["height"] is None:
            fields["height"] = decoded_height
    return snap.with_tree(tree)
