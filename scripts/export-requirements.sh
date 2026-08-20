#!/bin/sh
# Regenerate requirements.txt, the dependency-scanning (SCA/SBOM) input for the
# DR-compliance component, from uv.lock.
#
# The component parses the checked-out tree and does not understand uv.lock, so
# deleting poetry.lock in #60 dropped its SBOM from 201 packages to 0. uv.lock
# stays the single source of truth; this file is derived from it, and the `lint`
# job runs this script and fails if the committed copy has drifted.
#
# Usage: scripts/export-requirements.sh [output-path]   (default: requirements.txt)
set -eu

OUT="${1:-requirements.txt}"

{
    cat <<'HEADER'
# GENERATED FILE -- do not edit, and do not install from it.
#
# Dependency-scanning (SCA/SBOM) input for the DR-compliance component, which
# parses the checked-out tree and does not understand uv.lock. uv.lock remains
# the single source of truth; this file is derived from it and the `lint` job
# fails if the two disagree.
#
# Regenerate with:
#   scripts/export-requirements.sh
#
# Install paths are `uv sync` (development) and `pip install datamaite`
# (consumers) -- never `pip install -r requirements.txt`.
HEADER
    uv export --all-extras --no-emit-project --no-header --format requirements-txt -q
} > "$OUT"
