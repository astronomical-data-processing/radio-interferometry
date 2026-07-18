# PyBDSF Abell 2255 product replay

This package contains an exact, checksum-pinned subset of the PyBDSF
reference test fixture at commit
`c70103be3ae9ae9908286f144e6ce956acc0ce5c`. The FITS header identifies the
input as a cropped 2006 WSRT image of Abell 2255. The package includes the
four-Stokes input cube, collapsed Stokes-I image, RMS image, Gaussian model,
Gaussian residual and source catalogue.

The files are redistributed under the GPLv3 license at the root of the
upstream repository, reproduced in `LICENSE-PyBDSF`. No separate license
statement or original archive identifier was found for the FITS fixture. Git
history first records it in commit `7e407c27019b71dfeca4d5690ee700ae637deea5`
as test data. Original repository paths and SHA-256 checksums are recorded in
`manifests/product_manifest.yaml`. The upstream `parameters_used` file is not
copied because it contains a developer's absolute path; the relevant scientific
settings are transcribed in `configs/pybdsf_parameters.yaml`.

Run the offline validation with:

```bash
python analyze_products.py
```

The FITS header identifies this as a real-data product, but the fixture is not a
Measurement Set or a complete archive package. It supports FITS/header
inspection, product identity checks, model-residual closure and catalogue QA.
It does not support recalibration, re-imaging, independent flux-scale
verification or reconstruction of the original observing setup. In particular,
the cropped input header does not contain a physical observing frequency or
`BUNIT`; the PyBDSF test configuration supplies the beam and processing
assumptions externally.
