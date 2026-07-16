import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import yaml


PACKAGE_DIR = Path(__file__).resolve().parent
ELLIPSE_PATTERN = re.compile(
    r"ellipse\(\s*([+-]?[0-9.]+)\s*,\s*([+-]?[0-9.]+)\s*,"
    r"\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([+-]?[0-9.]+)\s*\)"
)


def load_yaml(relative_path):
    with (PACKAGE_DIR / relative_path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def sha256(relative_path):
    digest = hashlib.sha256()
    with (PACKAGE_DIR / relative_path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_checksums():
    data_manifest = load_yaml("manifests/data_manifest.yaml")
    product_manifest = load_yaml("manifests/product_manifest.yaml")
    checksums = {
        item["path"]: item["checksum"].removeprefix("sha256:")
        for item in data_manifest["products"]
    }
    for role, expected in product_manifest["checksums"].items():
        checksums[product_manifest["products"][role]] = expected.removeprefix("sha256:")
    return checksums


def verify_checksums():
    failures = {}
    for relative_path, expected in expected_checksums().items():
        actual = sha256(relative_path)
        if actual != expected:
            failures[relative_path] = {"expected": expected, "actual": actual}
    return failures


def load_catalog(relative_path="products/catalogs/source_catalog.csv"):
    with (PACKAGE_DIR / relative_path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def ellipse_mask(shape, relative_path, coordinate_origin=1):
    text = (PACKAGE_DIR / relative_path).read_text(encoding="utf-8")
    match = ELLIPSE_PATTERN.search(text)
    if match is None:
        raise ValueError(f"No image-coordinate ellipse found in {relative_path}")

    center_x, center_y, radius_x, radius_y, angle_deg = map(float, match.groups())
    center_x -= coordinate_origin
    center_y -= coordinate_origin
    yy, xx = np.indices(shape)
    dx = xx - center_x
    dy = yy - center_y
    angle = np.deg2rad(angle_deg)
    rotated_x = np.cos(angle) * dx + np.sin(angle) * dy
    rotated_y = -np.sin(angle) * dx + np.cos(angle) * dy
    return (rotated_x / radius_x) ** 2 + (rotated_y / radius_y) ** 2 <= 1.0


def compact_source_mask(shape, catalog, radius_pix, coordinate_origin=1):
    yy, xx = np.indices(shape)
    mask = np.zeros(shape, dtype=bool)
    for source in catalog:
        if source["flag"] != "science_masked_compact_source":
            continue
        center_x = float(source["x_pix"]) - coordinate_origin
        center_y = float(source["y_pix"]) - coordinate_origin
        mask |= (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius_pix**2
    return mask


def measurement_summary():
    failures = verify_checksums()
    if failures:
        raise ValueError(f"Checksum validation failed: {failures}")

    imaging = load_yaml("configs/imaging_parameters.yaml")
    measurement = load_yaml("configs/measurement_parameters.yaml")
    image = np.load(PACKAGE_DIR / "products/images/target_tapered_image.npy")
    residual = np.load(PACKAGE_DIR / "products/images/target_residual_image.npy")
    local_rms = np.load(PACKAGE_DIR / "products/images/local_rms_map.npy")
    if image.shape != residual.shape or image.shape != local_rms.shape:
        raise ValueError("Image, residual, and local RMS arrays must have identical shapes")

    origin = int(measurement["region_coordinate_origin"])
    region = ellipse_mask(image.shape, measurement["measurement_region"], origin)
    compact = compact_source_mask(
        image.shape,
        load_catalog(),
        float(measurement["compact_source_mask_radius_pix"]),
        origin,
    )
    analysis_mask = region & ~compact

    beam_major, beam_minor, _ = imaging["restored_beam_arcsec"]
    pixel_scale = float(imaging["pixel_scale_arcsec"])
    pixels_per_beam = 1.1331 * beam_major * beam_minor / pixel_scale**2
    independent_beams = analysis_mask.sum() / pixels_per_beam
    integrated_flux = image[analysis_mask].sum() / pixels_per_beam
    residual_integral = residual[analysis_mask].sum() / pixels_per_beam
    local_noise = float(np.median(local_rms[analysis_mask]))

    if measurement["include_correlated_noise"]:
        noise_model = "independent_synthesized_beams"
        random_uncertainty = local_noise * np.sqrt(independent_beams)
    else:
        noise_model = "independent_pixels"
        random_uncertainty = (
            local_noise * np.sqrt(analysis_mask.sum()) / pixels_per_beam
        )
    model_uncertainty = abs(residual_integral)
    flux_scale_uncertainty = (
        float(measurement["flux_scale_fractional_error"]) * abs(integrated_flux)
        if measurement["include_flux_scale_error"]
        else 0.0
    )
    total_uncertainty = np.sqrt(
        random_uncertainty**2 + model_uncertainty**2 + flux_scale_uncertainty**2
    )

    return {
        "checksum_validation": "passed",
        "analysis_pixels": int(analysis_mask.sum()),
        "pixels_per_beam": float(pixels_per_beam),
        "independent_beams": float(independent_beams),
        "integrated_flux_jy": float(integrated_flux),
        "residual_integral_jy": float(residual_integral),
        "local_rms_jy_beam": local_noise,
        "noise_model": noise_model,
        "random_uncertainty_jy": float(random_uncertainty),
        "model_uncertainty_jy": float(model_uncertainty),
        "flux_scale_uncertainty_jy": float(flux_scale_uncertainty),
        "total_uncertainty_jy": float(total_uncertainty),
        "signal_to_noise": float(integrated_flux / total_uncertainty),
        "three_sigma_sensitivity_jy": float(
            measurement["sensitivity_sigma"] * total_uncertainty
        ),
    }


def main():
    print(json.dumps(measurement_summary(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
