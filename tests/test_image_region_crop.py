"""decode_image honors ImageRecord.region for lazy object crops."""

from __future__ import annotations

import numpy as np
import pytest

from datamaite.records import ImageClassificationSample, ImageRecord

cv2 = pytest.importorskip("cv2")


def _write_rgb(path, width: int, height: int) -> None:
    # Encode source coordinates per pixel so a crop's offset AND axis order are
    # verifiable (not just its shape): blue = x (column), green = y (row).
    # cv2 stores BGR; after decode's BGR->RGB + HWC->CHW, blue lands in channel 2
    # and green in channel 1, so arr[2,i,j]==src_x and arr[1,i,j]==src_y.
    bgr = np.zeros((height, width, 3), dtype=np.uint8)
    bgr[:, :, 0] = np.arange(width, dtype=np.uint8)[None, :]
    bgr[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None]
    cv2.imwrite(str(path), bgr)


def test_region_none_returns_full_image(tmp_path) -> None:
    from datamaite.maite._image import decode_image

    img = tmp_path / "full.png"
    _write_rgb(img, 8, 6)
    arr = decode_image(ImageRecord(image_id="x", path_or_uri=str(img)))
    assert arr.shape == (3, 6, 8)


def test_region_crops_and_clamps(tmp_path) -> None:
    from datamaite.maite._image import decode_image

    img = tmp_path / "c.png"
    _write_rgb(img, 10, 10)
    # Region partly outside the image (w=6 starting at x=6 -> clamps to x in [6,10]).
    sample = ImageClassificationSample(image_id="crop", path_or_uri=str(img), region=(6.0, 2.0, 6.0, 4.0))
    arr = decode_image(sample)
    assert arr.shape == (3, 4, 4)  # height 4 (rows 2..6), width 4 (cols 6..10 after clamp)
    # Content, not just shape: columns must read the source x-coords 6..9 (offset +
    # width axis) and rows the source y-coords 2..5 (offset + height axis). An
    # axis swap or wrong origin yields the same shape but different content.
    assert arr[2].tolist() == [[6, 7, 8, 9]] * 4  # channel 2 == source x
    assert arr[1].tolist() == [[y] * 4 for y in (2, 3, 4, 5)]  # channel 1 == source y


def test_region_fully_outside_raises(tmp_path) -> None:
    from datamaite.maite._image import decode_image

    img = tmp_path / "o.png"
    _write_rgb(img, 5, 5)
    sample = ImageClassificationSample(image_id="oob", path_or_uri=str(img), region=(50.0, 50.0, 4.0, 4.0))
    with pytest.raises(ValueError, match="oob"):  # error names the offending sample id
        decode_image(sample)
