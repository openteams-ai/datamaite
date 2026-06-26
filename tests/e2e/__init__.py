"""End-to-end tests that run datamaite against real external datasets.

Unlike the hermetic unit suite in ``tests/`` (synthetic ``tmp_path`` fixtures,
always-on, offline), these exercise the full pipeline against a real checkout
of the shared example-data repo. They are opt-in: marked ``integration`` (so the
default ``pytest`` run deselects them) and they self-skip unless
``DATAMAITE_DATASETS_ROOT`` points at an example-data checkout. See
``tests/README.md`` for the two-tier layout.
"""
