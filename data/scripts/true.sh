#!/usr/bin/env bash

set -euo pipefail

command -v tigger-restore >/dev/null || {
    echo "tigger-restore is required" >&2
    exit 127
}

# Run from the directory containing the image and sky model.

SKYMODEL="${SKYMODEL:-skymodel-nassp.lsm.html}"
tigger-restore --clear KAT-7_6h60s_dec-30_10MHz_10chans_uniform-image.fits "$SKYMODEL" KAT-7_6h60s_dec-30_10MHz_10chans_true.fits
