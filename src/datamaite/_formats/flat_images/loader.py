"""Flat-folder still-image loader (IR-3.2-S-1).

IR-3.2-S-1 requires JATIC products that consume label-free CV image datasets
to accept flat folders of images in ``.jpg``, ``.png``, and ``.tif`` formats.
This loader intentionally models that narrow contract: it reads only the
immediate image children of ``root`` (no recursive discovery, no annotations)
and returns an *unlabeled* object-detection dataset -- every sample has zero
detections and there is no taxonomy.

The standard also names SafeTensors as an accepted image format; that is
deliberately not implemented (see the tracking issue) because safetensors is
a tensor container with no image-layout convention, not an image interchange
format (#74).

Per the dataset-structures policy (#40), this format is **explicit opt-in
only**: ``sniff`` stays False so a bare folder of images is never
autodetected as a dataset. Load it with
``load_od(root, dataset_format="flat_images")``.

Like the other loaders, this is best-effort: files whose magic bytes do not
match their suffix are skipped with warnings rather than aborting the whole
load. Images are validated by magic bytes only at load time and decoded
lazily by the MAITE surface (``pip install datamaite[od]``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from datamaite._types import DatasetFormat, Task
from datamaite.loaders import Loader, register_loader
from datamaite.object_detection import ObjectDetectionDataset
from datamaite.records import DatasetMetadata, ImageObjectDetectionSample

logger = logging.getLogger(__name__)

#: IR-3.2-S-1 names .jpg/.png/.tif (SafeTensors is deferred, #74); the
#: long-suffix aliases are accepted because they are the same wire formats.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff"})

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

    # No sniff override: any folder containing images would match, so this
    # format is explicit opt-in only and never participates in autodetect (#40).

    def load(self, root: str | Path, *, image_extensions: Any = None, **_: Any) -> ObjectDetectionDataset:
        """Read immediate image children under ``root`` into an unlabeled OD dataset.

        Parameters
        ----------
        root
            Directory whose immediate children are image files. The loader
            does **not** recurse into subdirectories; nested images are
            ignored by design because IR-3.2-S-1 is the flat-folder standard.
        image_extensions
            Optional extension spec overriding the defaults
            (``.jpg``/``.jpeg``/``.png``/``.tif``/``.tiff``). Accepts a
            string or an iterable of strings, with or without the leading
            dot, case-insensitive.

        Returns
        -------
        ObjectDetectionDataset
            One sample per accepted image, each with zero detections and no
            taxonomy, because this format carries no annotations.
        """
        root_path = Path(root)
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
            sample = _encoded_image_sample(path)
            if sample is not None:
                samples.append(sample)

        logger.info("Loaded %d flat image(s) from %s", len(samples), root_path)
        return ObjectDetectionDataset(
            samples=tuple(samples),
            dataset_metadata=DatasetMetadata(source_dataset="flat_images"),
            dataset_id="flat_images",
        )


def load_flat_images(root: str | Path, *, image_extensions: Any = None) -> ObjectDetectionDataset:
    """Load a flat folder of label-free still images.

    Equivalent to ``datamaite.load_od(root, dataset_format="flat_images")``.
    See :meth:`FlatImagesLoader.load` for semantics.
    """
    return FlatImagesLoader().load(root, image_extensions=image_extensions)


def _normalize_extensions(image_extensions: Any) -> frozenset[str]:
    """Coerce a user-supplied extension spec to a lowercased, dot-prefixed set.

    ``None`` yields the built-in defaults. A bare string (``".jpg"`` or ``"jpg"``)
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
        return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    except OSError as exc:
        logger.warning("Could not list flat images root %s: %s", root, exc)
        return []


def _encoded_image_sample(path: Path) -> ImageObjectDetectionSample | None:
    """Build one sample for an encoded image file, or skip it with a warning.

    Validation here is magic-bytes only (cheap, dependency-free); pixel
    decoding stays lazy in the MAITE surface, mirroring the other still-image
    loaders. Dimensions are left unset -- MAITE indexing fills them from the
    decoded array.
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(_MAGIC_PROBE_BYTES)
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
