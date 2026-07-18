# BIMA NGC 4826 Measurement Set replay

This package contains the unmodified compressed Measurement Set used by the
casacore field-selection tests at commit
`ef9a25f41cd2c7edfe7a2d0eee549becb55a6403`. Git history first adds the file
in commit `54c10722cb1a35342cbe6b132063fbf6ff9a002d`. Casacore TaQL identifies
14,985 main-table rows, ten BIMA antennas, a 3C273 calibrator field and a
seven-field NGC 4826 mosaic observed on 1998-04-16.

The repository-level casacore license is reproduced in `LICENSE-casacore`.
No separate license statement or original archive identifier was found for
this Measurement Set fixture. The exact source path and SHA-256 checksum are
recorded in `manifests/data_manifest.yaml`.

`derived/` contains compact NumPy extracts of the 3C273 calibrator, one NGC
4826 target field, and a coverage-only extract for all seven target fields and
four 64-channel data descriptions. The visibility extracts preserve `DATA`,
`FLAG`, `UVW`, `WEIGHT`, antenna, time, field and data-description columns;
the coverage extract preserves `UVW`, row metadata and per-row flag fractions.
These extracts allow the core notebook to run without CASA or casacore.
`extract_visibility.py` documents and reproduces the selection when
`python-casacore` is available:

```bash
python extract_visibility.py --output /tmp/bima-extract
```

Run checksum, structure, visibility and scalar-calibration validation with:

```bash
python analyze_ms.py
```

This is a genuine Measurement Set table with real-valued visibility products,
but it is a casacore test fixture rather than a complete archive delivery. It
supports table inspection, flag/coverage QA and a normalized point-calibrator
gain experiment. It does not contain a complete observing log, independent
flux model, modern calibration tables or a documented path back to the
original archive. The scalar solve therefore demonstrates relative gain
calibration; it does not establish an absolute flux scale. The calibrator and
target data are separated by about 989 seconds, so the notebook does not claim
that the calibrator gains can be transferred synchronously to the target
mosaic; doing that would require a time-dependent calibration model and
additional observing metadata.
