"""Validate the bundled PyBDSF Abell 2255 reference products."""

import hashlib
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifests" / "product_manifest.yaml"


def load_manifest():
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums():
    verified = {}
    for record in load_manifest()["files"]:
        path = ROOT / record["path"]
        actual = sha256(path)
        if actual != record["sha256"]:
            raise ValueError(f"checksum mismatch: {record['path']}")
        verified[record["path"]] = actual
    return verified


def _image(relative):
    return np.asarray(fits.getdata(ROOT / relative), dtype=float).squeeze()


def product_summary():
    verified = verify_checksums()
    raw_path = ROOT / "data" / "abell2255_wsrt_stokes_cube.fits"
    raw = np.asarray(fits.getdata(raw_path), dtype=float)
    header = fits.getheader(raw_path)
    image = _image("products/mean_stokes_i.fits")
    rms = _image("products/local_rms.fits")
    model = _image("products/gaussian_model.fits")
    residual = _image("products/gaussian_residual.fits")
    catalog = fits.getdata(ROOT / "products" / "source_catalog.fits", 1)

    average_error = float(np.max(np.abs(raw[0].mean(axis=0) - image)))
    closure_error = float(np.max(np.abs(model + residual - image)))
    if raw.shape != (4, 10, 129, 129):
        raise ValueError(f"unexpected input shape: {raw.shape}")
    if average_error >= 2e-8 or closure_error >= 2e-8:
        raise ValueError("reference products do not close against the input image")
    if len(catalog) != 43:
        raise ValueError(f"unexpected catalogue length: {len(catalog)}")

    codes, counts = np.unique(catalog["S_Code"], return_counts=True)
    robust_rms = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    return {
        "verified_files": len(verified),
        "object": header["OBJECT"],
        "instrument": header["INSTRUME"],
        "input_shape": tuple(raw.shape),
        "catalog_sources": len(catalog),
        "source_codes": {str(code): int(count) for code, count in zip(codes, counts)},
        "image_peak_jy_per_beam": float(np.max(image)),
        "rms_map_jy_per_beam": float(np.median(rms)),
        "residual_robust_rms_jy_per_beam": float(robust_rms),
        "residual_peak_over_rms": float(np.max(np.abs(residual)) / np.median(rms)),
        "frequency_average_max_error": average_error,
        "model_residual_max_error": closure_error,
    }


if __name__ == "__main__":
    for key, value in product_summary().items():
        print(f"{key}: {value}")
