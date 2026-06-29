"""MAITE interoperability for datamaite.

Optional subpackage: ``import datamaite`` never imports this module. Core
installs include ``numpy`` for target arrays, while pixel decoding is enabled by
task extras (``datamaite[fmv]`` for MOT video, ``datamaite[od]`` /
``datamaite[ic]`` for still images). For all task surfaces::

    pip install datamaite[all]

:class:`~datamaite.model.BoxTrackDataset`,
:class:`~datamaite.object_detection.ObjectDetectionDataset`, and
:class:`~datamaite.image_classification.ImageClassificationDataset` implement
their MAITE protocols directly -- there is no adapter/conversion call.
``load_mot`` returns one; index it::

    from datamaite import load_mot

    ds = load_mot(root)
    video_stream, target, metadata = ds[0]   # one MAITE MOT item per video

To configure the MOT view (decoder, ``empty_frame_policy``, ``dataset_id``)
use :meth:`~datamaite.model.BoxTrackDataset.with_mot_options`.

MAITE protocols are structural, so datamaite datasets conform by shape and
this package does not import ``maite`` at runtime; the runtime dependencies are
``numpy`` and lazy media decoders (PyAV for MOT video, OpenCV for IC/OD images).
The view machinery lives in :mod:`datamaite.maite._mot`,
:mod:`datamaite.maite._od`, and :mod:`datamaite.maite._ic` (imported lazily
from the core dataset classes).
"""

from __future__ import annotations
