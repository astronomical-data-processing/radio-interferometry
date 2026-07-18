#!/bin/bash

# Run from the directory containing the image and sky model.

SKYMODEL="${SKYMODEL:-skymodel-nassp.lsm.html}"
tigger-restore --clear KAT-7_6h60s_dec-30_10MHz_10chans_uniform-image.fits "$SKYMODEL" KAT-7_6h60s_dec-30_10MHz_10chans_true.fits
