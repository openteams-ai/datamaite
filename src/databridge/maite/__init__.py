"""MAITE interoperability for databridge.

Optional subpackage: ``import databridge`` never imports this module, so the
core loader/validator stays usable without MAITE installed. Install the extra
to use it::

    pip install databridge[maite]

:class:`~databridge.model.BoxTrackDataset` already implements the MAITE
multi-object-tracking protocol directly -- there is no adapter/conversion
call. ``load_hmie`` returns one; index it::

    from databridge import load_hmie

    ds = load_hmie(root)
    video_stream, target, metadata = ds[0]   # one MAITE MOT item per video

To configure the MOT view (decoder, ``empty_frame_policy``, ``dataset_id``)
use :meth:`~databridge.model.BoxTrackDataset.with_mot_options`.

MAITE protocols are structural, so ``BoxTrackDataset`` conforms by shape and
this package does not import ``maite`` at runtime; the runtime dependencies are
``numpy`` and a video decoder (PyAV by default). The view machinery lives in
:mod:`databridge.maite._mot` (imported lazily from the core model).
"""

from __future__ import annotations
