# VLA 3C391 archive calibration lab

This optional course lab replays the NRAO/CASA 3C391 continuum tutorial from
the downloadable 4.6 GHz Measurement Set. The observation is VLA project
`TDEM0001`, execution block `TDEM0001_sb1218006_1.55310.33439732639`, observed
in D configuration on 2010-04-24. The target is a seven-field mosaic; 3C286
sets the Perley-Butler 2017 flux scale and J1822-0938 transfers the complex
gain calibration.

The 3.1 GB archive and its expanded Measurement Set are not distributed by
this repository. NRAO makes post-proprietary-period VLA data publicly
accessible, but no standalone redistribution license was found for this
tutorial archive. Public access is not treated here as permission to relicense
the data under the repository's GPLv2. `manifests/data_manifest.yaml` records
the source, archive identity, checksum, data contract, and this licensing
boundary.

## Stages

The base textbook environment can inspect the manifests, evaluate the 3C286
flux model, and test the audit code. Downloading 3.1 GB or importing CASA is
never part of the default notebook run.

Download, verify, and expand the external archive into a scratch directory:

```bash
python download_data.py --output-dir /scratch/vla_3c391
```

The downloader pins the expected length and SHA-256. An interrupted transfer
is retained as a `.part` file and resumed when the server supports byte ranges.
Use `--verify-only` to check an existing archive or `--download-only` to skip
expansion.

Run the calibration stage in a clean work directory:

```bash
python run_casa_pipeline.py \
  --input-ms /scratch/vla_3c391/3c391_ctm_mosaic_10s_spw0.ms \
  --work-dir /scratch/vla_3c391/run \
  --stop-after calibration
```

Run the complete calibration and reference imaging baseline with
`--stop-after imaging`. Add `--export-fits` only when FITS products are needed.
The pipeline was validated with `casatasks` and `casatools` 6.7.0.31 and
`casadata` 2025.9.22 in Python 3.10. CASA data paths must be configured before
the process imports CASA, for example through `CASASITECONFIG` in a pip-based
installation. CASA is deliberately absent from the repository
`requirements.txt`.

Audit the machine-readable results with:

```bash
python audit_results.py --results-dir /scratch/vla_3c391/run
python audit_results.py --results-dir /scratch/vla_3c391/run --require-imaging
```

Compare the validated mask choices without repeating calibration:

```bash
python run_imaging_sensitivity.py \
  --science-ms /scratch/vla_3c391/run/3c391_ctm_mosaic_spw0.ms \
  --work-dir /scratch/vla_3c391/sensitivity
```

The script runs three mask choices with common imaging parameters, followed by
one diagnostic that keeps the broad mask but removes the 45-pixel CLEAN scale.
It records mask area, model positive and negative sums, residual statistics and
spatial correlation, restoring beam, CASA stopping code, and relevant log
warnings in `imaging_sensitivity.json`. Its work directory must be empty; use
`--variants` to run a selected subset in a separate directory.

## Validated sensitivity result

All statistics below use the common region with primary-beam response at least
0.2. The beam remains 17.01 by 14.91 arcsec in every run.

| Variant | Model sum (Jy) | Negative absolute fraction | Residual RMS (mJy/beam) | Divergence warnings |
| --- | ---: | ---: | ---: | ---: |
| Broad 130-pixel circle | 3.788 | 25.6% | 0.570 | 11 |
| Tight 100-pixel circle | 5.539 | 12.9% | 0.840 | 8 |
| Conservative auto-mask | 8.070 | 0.7% | 0.645 | 17 |
| Broad circle, scales `[0,5,15]` | 3.540 | 24.9% | 0.737 | 0 |

No tested variant supports a stronger scientific product. The tight mask
creates strong boundary residuals. The auto-mask follows the source more
closely and suppresses negative model components, but nearly doubles the model
sum relative to the regression baseline, leaves correlated rings, and produces
more divergence warnings. Removing the largest scale eliminates those warnings
but leaves much stronger beam-scale residual correlation. These are distinct
failure trade-offs, not a ranking based on residual RMS alone.

## Interpretation boundary

The fixed imaging recipe is a reproducible course baseline, not a publication
recipe. It uses Briggs robust 0.5, mosaic gridding, multiscale CLEAN, and a
fixed circular mask. In the validation run CASA stopped minor cycles when a
large-scale component became negative or divergent; correlated rings remain
in the residual. A controlled sensitivity study did not identify a better
mask/scale recipe, so the broad-mask result remains only the regression
baseline. A scientific release still requires additional visibility and PSF
diagnosis, an independently reviewed source-support model, primary-beam-aware
measurements, uncertainty propagation, and a documented stopping decision.
