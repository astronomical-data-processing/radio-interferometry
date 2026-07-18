#!/usr/bin/env bash

set -euo pipefail

#original data: fundamentals_of_interferometry/data/simulated_kat_7_vis/simulated_KAT-7_ms.tar.gz

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UVSCRIPT="$SCRIPT_DIR/plotUVcoverage.py"
PYTHON="${PYTHON:-python3}"

command -v "$PYTHON" >/dev/null || {
    echo "$PYTHON is required" >&2
    exit 127
}

#Obs length
"$PYTHON" "$UVSCRIPT" -l 1000 -f KAT-7_0.167h60s_dec-30_10MHz_10chans.ms -s KAT-7_0.167h60s_dec-30_10MHz_10chans.png
"$PYTHON" "$UVSCRIPT" -l 1000 -f KAT-7_2h60s_dec-30_10MHz_10chans.ms     -s KAT-7_2h60s_dec-30_10MHz_10chans.png
"$PYTHON" "$UVSCRIPT" -l 1000 -f KAT-7_4h60s_dec-30_10MHz_10chans.ms     -s KAT-7_4h60s_dec-30_10MHz_10chans.png
"$PYTHON" "$UVSCRIPT" -l 1000 -f KAT-7_6h60s_dec-30_10MHz_10chans.ms     -s KAT-7_6h60s_dec-30_10MHz_10chans.png
"$PYTHON" "$UVSCRIPT" -l 1000 -f KAT-7_8h60s_dec-30_10MHz_10chans.ms     -s KAT-7_8h60s_dec-30_10MHz_10chans.png
"$PYTHON" "$UVSCRIPT" -l 1000 -f KAT-7_10h60s_dec-30_10MHz_10chans.ms    -s KAT-7_10h60s_dec-30_10MHz_10chans.png
"$PYTHON" "$UVSCRIPT" -l 1000 -f KAT-7_12h60s_dec-30_10MHz_10chans.ms    -s KAT-7_12h60s_dec-30_10MHz_10chans.png

#channels
"$PYTHON" "$UVSCRIPT" -l 1000 -f KAT-7_6h60s_dec-30_10MHz_1chans.ms   -s KAT-7_6h60s_dec-30_10MHz_1chans.png
"$PYTHON" "$UVSCRIPT" -l 1000 -f KAT-7_6h60s_dec-30_10MHz_100chans.ms -s KAT-7_6h60s_dec-30_10MHz_100chans.png

#declination
"$PYTHON" "$UVSCRIPT" -l 1000 -f KAT-7_6h60s_dec30_10MHz_10chans.ms  -s KAT-7_6h60s_dec30_10MHz_10chans.png
"$PYTHON" "$UVSCRIPT" -l 1000 -f KAT-7_6h60s_dec0_10MHz_10chans.ms   -s KAT-7_6h60s_dec0_10MHz_10chans.png
"$PYTHON" "$UVSCRIPT" -l 1000 -f KAT-7_6h60s_dec-60_10MHz_10chans.ms -s KAT-7_6h60s_dec-60_10MHz_10chans.png
"$PYTHON" "$UVSCRIPT" -l 1000 -f KAT-7_6h60s_dec-90_10MHz_10chans.ms -s KAT-7_6h60s_dec-90_10MHz_10chans.png
