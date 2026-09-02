"""Flat-folder still-image loader (IR-3.2-S-1).

IR-3.2-S-1 requires JATIC products that consume label-free CV image datasets
to accept flat folders of images in ``.jpg``, ``.png``, ``.tif``, and
SafeTensors formats. This loader intentionally models that narrow contract: it
reads only the immediate image children of ``root`` (no recursive discovery, no
annotations) and returns an *unlabeled* object-detection dataset -- every sample
has zero detections and there is no taxonomy.

Per the dataset-structures policy (#40), this format is **explicit opt-in
only**: ``sniff`` stays False so a bare folder of images is never
autodetected as a dataset. Load it with
``load_od(root, dataset_format="flat_images")``.

Like the other loaders, this is best-effort: files whose magic bytes do not
match their suffix, and safetensors files without a decodable image tensor,
are skipped with warnings rather than aborting the whole load. Encoded images
(``.jpg``/``.png``/``.tif``) are validated by magic bytes only at load time
and decoded lazily by the MAITE surface (``pip install datamaite[od]``);
safetensors files are validated by a header-only parse and their tensors
decode with numpy alone. :mod:`datamaite._safetensors` documents the tensor
layout conventions we accept, since safetensors itself defines none.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from datamaite._io import list_files, read_resource_prefix
from datamaite._safetensors import (
    SAFETENSORS_SUFFIX,
    SafeTensorEntry,
    image_entries,
    layout_is_ambiguous,
    read_entries,
)
from datamaite._types import DatasetFormat, Task
from datamaite._upath import storage_options_for, to_dataset_path
from datamaite.loaders import Loader, register_loader
from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import DatasetMetadata, ImageObjectDetectionSample

logger = logging.getLogger(__name__)

#: IR-3.2-S-1 names .jpg/.png/.tif/SafeTensors; the long-suffix aliases are
#: accepted because they are the same wire formats.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", SAFETENSORS_SUFFIX})

#: Magic bytes for the encoded-image suffixes. Suffixes outside this table
#: (user-supplied via ``image_extensions``) pass through unchecked; the real
#: validation happens at MAITE decode time.
_MAGIC_BYTES: dict[str, tuple[bytes, ...]] = {
    ".jpg": (b"\xff\xd8\xff",),
    ".jpeg": (b"\xff\xd8\xff",),
    ".png": (b"\x89PNG\r\n\x1a\n",),
    ".tif": (b"II*\x00", b"MM\x00*"),
    ".tiff": (b"II*\x00", b"MM\x00*"),
}
_MAGIC_PROBE_BYTES = 8


@register_loader
class FlatImagesLoader(Loader):
    """Loader for a flat directory of label-free still images."""

    task: ClassVar[Task] = Task.OD
    format = DatasetFormat.FLAT_IMAGES
    variant: ClassVar[str] = "default"
    supports_remote: ClassVar[bool] = True

    # No sniff override: any folder containing images would match, so this
    # format is explicit opt-in only and never participates in autodetect (#40).

    def load(
        self,
        root: str | Path,
        *,
        image_extensions: Any = None,
        storage_options: Any = None,
        **_: Any,
    ) -> ObjectDetectionDataset:
        """Read immediate image children under ``root`` into an unlabeled OD dataset.

        Parameters
        ----------
        root
            Directory whose immediate children are image files. The loader
            does **not** recurse into subdirectories; nested images are
            ignored by design because IR-3.2-S-1 is the flat-folder standard.
        image_extensions
            Optional extension spec overriding the defaults
            (``.jpg``/``.jpeg``/``.png``/``.tif``/``.tiff``/``.safetensors``).
            Accepts a string or an iterable of strings, with or without the
            leading dot, case-insensitive.

        Returns
        -------
        ObjectDetectionDataset
            One sample per accepted image, each with zero detections and no
            taxonomy, because this format carries no annotations. A
            ``.safetensors`` file contributes one sample per image-shaped
            tensor it stores.
        """
        root_path = to_dataset_path(root, storage_options)
        if not root_path.is_dir():
            logger.warning("Flat images root is not a directory: %s", root_path)
            return ObjectDetectionDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="flat_images"))

        extensions = _normalize_extensions(image_extensions)
        files = _flat_image_files(root_path, extensions)
        if not files:
            logger.warning("No immediate image files found in flat images root: %s", root_path)
            return ObjectDetectionDataset(samples=(), dataset_metadata=DatasetMetadata(source_dataset="flat_images"))

        samples: list[ImageObjectDetectionSample] = []
        for path in files:
            if path.suffix.lower() == SAFETENSORS_SUFFIX:
                samples.extend(_safetensors_samples(path))
            else:
                sample = _encoded_image_sample(path)
                if sample is not None:
                    samples.append(sample)

        logger.info("Loaded %d flat image(s) from %s", len(samples), root_path)
        return ObjectDetectionDataset(
            samples=tuple(samples),
            dataset_metadata=DatasetMetadata(source_dataset="flat_images"),
            dataset_id="flat_images",
            _storage_options=storage_options_for(root_path),
        )


def load_flat_images(
    root: str | Path, *, image_extensions: Any = None, storage_options: Any = None
) -> ObjectDetectionDataset:
    """Load a flat folder of label-free still images.

    Equivalent to ``datamaite.load_od(root, dataset_format="flat_images")``.
    See :meth:`FlatImagesLoader.load` for semantics.
    """
    return FlatImagesLoader().load(root, image_extensions=image_extensions, storage_options=storage_options)


def _normalize_extensions(image_extensions: Any) -> frozenset[str]:
    """Coerce a user-supplied extension spec to a lowercased, dot-prefixed set.

    ``None`` yields the built-in defaults (the encoded suffixes plus
    ``.safetensors``). A bare string (``".jpg"`` or ``"jpg"``)
    is treated as a single extension, not iterated into characters; any other
    iterable of strings is normalized element-wise. Blank entries are dropped;
    an all-blank spec falls back to the defaults so a stray ``""`` never
    silently loads zero images.
    """
    if image_extensions is None:
        return IMAGE_EXTENSIONS
    items = [image_extensions] if isinstance(image_extensions, str) else list(image_extensions)
    normalized: set[str] = set()
    for raw in items:
        ext = str(raw).strip().lower()
        if not ext:
            continue
        normalized.add(ext if ext.startswith(".") else f".{ext}")
    return frozenset(normalized) if normalized else IMAGE_EXTENSIONS


def _flat_image_files(root: Path, extensions: frozenset[str]) -> list[Path]:
    """Return immediate image files in deterministic order; never recurse."""
    try:
        return [path for path in list_files(root) if path.suffix.lower() in extensions]
    except OSError as exc:
        logger.warning("Could not list flat images root %s: %s", root, exc)
        return []


def _encoded_image_sample(path: Path) -> ImageObjectDetectionSample | None:
    """Build one sample after the same bounded header check on every backend."""
    try:
        head = read_resource_prefix(path, _MAGIC_PROBE_BYTES)
    except OSError as exc:
        logger.warning("Skipping unreadable flat image %s: %s", path, exc)
        return None
    if not head:
        logger.warning("Skipping empty flat image file: %s", path)
        return None
    magics = _MAGIC_BYTES.get(path.suffix.lower())
    if magics is not None and not any(head.startswith(magic) for magic in magics):
        logger.warning("Skipping flat image whose content does not match its %s suffix: %s", path.suffix, path)
        return None
    return ImageObjectDetectionSample(
        image_id=path.name,
        path_or_uri=str(path),
        file_name=path.name,
        metadata={"source_format": "flat_images", "source_file_name": path.name},
    )


def _safetensors_samples(path: Path) -> list[ImageObjectDetectionSample]:
    """Build one sample per image-shaped tensor in a safetensors file.

    Header-only parsing: dimensions come for free from the header, and tensor
    bytes are only range-read at MAITE decode time. Each image tensor is
    ``<file>#<tensor>`` even when the file holds only one.
    """
    try:
        entries = read_entries(path)
    except (OSError, ValueError) as exc:
        logger.warning("Skipping malformed safetensors file %s: %s", path, exc)
        return []
    images = image_entries(entries)
    if not images:
        logger.warning(
            "Skipping safetensors file with no image-shaped tensor of a supported dtype within the size cap: %s",
            path,
        )
        return []
    return [_safetensors_sample(path, entry, layout) for entry, layout in images]


def _safetensors_sample(path: Path, entry: SafeTensorEntry, layout: tuple[int, int, str]) -> ImageObjectDetectionSample:
    """Build one sample for an image tensor, taking its dimensions from the header.

    Dimensions are set here (unlike encoded images, whose headers this loader
    never parses) so MAITE metadata never needs a decode to learn them. The
    layout is the one ``image_entries`` already resolved, not a second guess.
    """
    height, width, _ = layout
    if layout_is_ambiguous(entry.shape):
        logger.warning(
            "SafeTensors tensor %r in %s has an ambiguous shape %s; assuming HWC. "
            "safetensors has no image-layout convention, so this is a guess",
            entry.name,
            path,
            entry.shape,
        )
    return ImageObjectDetectionSample(
        image_id=f"{path.name}#{entry.name}",
        path_or_uri=str(path),
        file_name=path.name,
        width=width,
        height=height,
        metadata={
            "source_format": "flat_images",
            "source_file_name": path.name,
            "safetensors_key": entry.name,
        },
    )
