#!/usr/bin/env python3
"""Compare controlled mask choices for the calibrated 3C391 mosaic."""

import argparse
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VARIANTS = {
    "fixed_broad": {
        "usemask": "user",
        "mask": "circle[[240pix,240pix],130pix]",
    },
    "fixed_tight": {
        "usemask": "user",
        "mask": "circle[[240pix,240pix],100pix]",
    },
    "auto_conservative": {
        "usemask": "auto-multithresh",
        "sidelobethreshold": 2.0,
        "noisethreshold": 4.5,
        "lownoisethreshold": 1.5,
        "minbeamfrac": 0.3,
        "growiterations": 75,
        "negativethreshold": 0.0,
    },
    "fixed_broad_restricted_scales": {
        "usemask": "user",
        "mask": "circle[[240pix,240pix],130pix]",
        "scales": [0, 5, 15],
    },
}


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def common_parameters():
    return {
        "specmode": "mfs",
        "gridder": "mosaic",
        "imsize": [480, 480],
        "cell": ["2.5arcsec", "2.5arcsec"],
        "stokes": "I",
        "weighting": "briggs",
        "robust": 0.5,
        "deconvolver": "multiscale",
        "scales": [0, 5, 15, 45],
        "smallscalebias": 0.9,
        "niter": 20_000,
        "gain": 0.1,
        "threshold": "1.0mJy",
        "pblimit": 0.2,
    }


def load_casa():
    try:
        import casatasks
        from casatools import image
    except ImportError as error:
        raise RuntimeError(
            "CASA is required only for this optional sensitivity study; "
            "configure a separate CASA 6.7 environment before running it"
        ) from error
    return {
        "casalog": casatasks.casalog,
        "image": image,
        "imstat": casatasks.imstat,
        "tclean": casatasks.tclean,
    }


def package_versions():
    versions = {}
    for name in ("casatasks", "casatools", "casadata"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed-as-package"
    return versions


def read_image(image_factory, path):
    tool = image_factory()
    tool.open(str(path))
    try:
        values = np.squeeze(np.asarray(tool.getchunk(), dtype=float))
        beam = plain(tool.restoringbeam()) if path.name.endswith(".image") else None
    finally:
        tool.close()
    if values.ndim != 2:
        raise ValueError(f"expected a two-dimensional image at {path}: {values.shape}")
    return values, beam


def lag_correlation(values, valid, dx=0, dy=0):
    if dx == 0 and dy == 0:
        return 1.0
    x1 = slice(max(dx, 0), values.shape[0] + min(dx, 0))
    x2 = slice(max(-dx, 0), values.shape[0] - max(dx, 0))
    y1 = slice(max(dy, 0), values.shape[1] + min(dy, 0))
    y2 = slice(max(-dy, 0), values.shape[1] - max(dy, 0))
    pairs = valid[x1, y1] & valid[x2, y2]
    first = values[x1, y1][pairs]
    second = values[x2, y2][pairs]
    if first.size < 2:
        return None
    first = first - first.mean()
    second = second - second.mean()
    denominator = np.sqrt(np.dot(first, first) * np.dot(second, second))
    return None if denominator == 0 else float(np.dot(first, second) / denominator)


def residual_metrics(values, valid, beam_pixels):
    sample = values[valid]
    if sample.size == 0:
        raise ValueError("the common primary-beam-valid region is empty")
    median = float(np.median(sample))
    robust_sigma = float(1.4826 * np.median(np.abs(sample - median)))
    center = np.asarray(values.shape, dtype=float) / 2.0
    x, y = np.indices(values.shape)
    radius = np.hypot(x - center[0], y - center[1])
    annuli = []
    for inner in range(0, 161, 10):
        selected = valid & (radius >= inner) & (radius < inner + 10)
        if np.any(selected):
            mean = float(np.mean(values[selected]))
            annuli.append(
                {
                    "inner_radius_pixels": inner,
                    "outer_radius_pixels": inner + 10,
                    "mean_jy_per_beam": mean,
                    "mean_over_robust_sigma": (
                        None if robust_sigma == 0 else mean / robust_sigma
                    ),
                }
            )
    return {
        "valid_pixels": int(sample.size),
        "mean_jy_per_beam": float(sample.mean()),
        "median_jy_per_beam": median,
        "rms_jy_per_beam": float(np.sqrt(np.mean(sample**2))),
        "robust_sigma_jy_per_beam": robust_sigma,
        "minimum_jy_per_beam": float(sample.min()),
        "maximum_jy_per_beam": float(sample.max()),
        "lag_correlation": {
            "one_pixel_x": lag_correlation(values, valid, dx=1),
            "one_pixel_y": lag_correlation(values, valid, dy=1),
            "one_beam_x": lag_correlation(values, valid, dx=beam_pixels),
            "one_beam_y": lag_correlation(values, valid, dy=beam_pixels),
        },
        "radial_annuli": annuli,
        "maximum_absolute_annular_mean_over_robust_sigma": max(
            abs(item["mean_over_robust_sigma"])
            for item in annuli
            if item["mean_over_robust_sigma"] is not None
        ),
    }


def model_metrics(values, valid):
    sample = values[valid]
    positive = sample[sample > 0]
    negative = sample[sample < 0]
    absolute_flux = positive.sum() - negative.sum()
    return {
        "nonzero_pixels": int(np.count_nonzero(sample)),
        "negative_pixels": int(negative.size),
        "sum_jy": float(sample.sum()),
        "positive_sum_jy": float(positive.sum()),
        "negative_sum_jy": float(negative.sum()),
        "negative_fraction_of_absolute_flux": (
            0.0 if absolute_flux == 0 else float(-negative.sum() / absolute_flux)
        ),
    }


def log_summary(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    warning = "MSClean minor cycle stopped at large scale negative or diverging"
    stopping_lines = [
        line.rsplit("\t", 1)[-1]
        for line in text.splitlines()
        if "Reached global stopping criterion" in line
    ]
    return {
        "large_scale_divergence_warnings": text.count(warning),
        "failed_threshold_messages": text.count("Failed to reach stopping threshold"),
        "ignored_scale_warnings": text.count("since it is too large to fit within the mask"),
        "global_stopping_messages": stopping_lines,
    }


def run_variant(casa, science, root, name, parameters):
    prefix = root / name
    log_path = root / f"{name}.log"
    casa["casalog"].setlogfile(str(log_path))
    tclean_parameters = {**common_parameters(), **parameters}
    result = casa["tclean"](
        vis=str(science),
        imagename=str(prefix),
        pbcor=False,
        interactive=False,
        parallel=False,
        fullsummary=True,
        savemodel="none",
        **tclean_parameters,
    )
    arrays = {}
    beam = None
    for suffix in ("image", "residual", "model", "pb", "mask"):
        path = Path(f"{prefix}.{suffix}")
        if path.exists():
            arrays[suffix], found_beam = read_image(casa["image"], path)
            beam = found_beam or beam
    required = {"image", "residual", "model", "pb", "mask"}
    if missing := required - arrays.keys():
        raise RuntimeError(f"{name} did not produce: {', '.join(sorted(missing))}")

    valid = (
        np.isfinite(arrays["pb"])
        & (arrays["pb"] >= common_parameters()["pblimit"])
        & np.isfinite(arrays["residual"])
        & np.isfinite(arrays["image"])
        & np.isfinite(arrays["model"])
    )
    beam_pixels = max(1, round(beam["major"]["value"] / 2.5))
    mask = arrays["mask"] > 0.5
    image_stats = plain(casa["imstat"](imagename=str(Path(f"{prefix}.image"))))
    return {
        "parameters": parameters,
        "tclean": {
            "iterations": plain(result.get("iterdone")),
            "major_cycles": plain(result.get("nmajordone")),
            "stop_code": plain(result.get("stopcode")),
            "stop_description": plain(
                result.get("stopDescription", result.get("stopdescription"))
            ),
        },
        "log": log_summary(log_path),
        "restoring_beam": beam,
        "restored_peak_jy_per_beam": float(np.nanmax(arrays["image"][valid])),
        "restored_integrated_flux_jy": plain(image_stats.get("flux")),
        "mask": {
            "pixels": int(np.count_nonzero(mask)),
            "fraction_of_pb_valid_region": float(np.count_nonzero(mask & valid) / valid.sum()),
        },
        "model": model_metrics(arrays["model"], valid),
        "residual": residual_metrics(arrays["residual"], valid, beam_pixels),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--science-ms", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=list(VARIANTS),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    science = args.science_ms.resolve()
    root = args.work_dir.resolve()
    if not science.is_dir():
        raise SystemExit(f"calibrated target Measurement Set not found: {science}")
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"work directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    casa = load_casa()
    results = {
        "schema_version": 1,
        "completed_utc": None,
        "science_measurement_set": str(science),
        "environment": package_versions(),
        "common_parameters": common_parameters(),
        "comparison_design": (
            "Three variants change only mask strategy; the restricted-scale "
            "variant holds the broad mask fixed and removes the 45-pixel scale"
        ),
        "variants": {},
        "interpretation": {
            "status": "requires_scientific_review",
            "publication_claim_supported": False,
            "caution": (
                "Lower residual RMS alone does not establish a better image; "
                "review model flux, negative components, mask area, residual "
                "correlation, radial structure, and stopping behavior together"
            ),
        },
    }
    for name in args.variants:
        print(f"Running {name}", flush=True)
        results["variants"][name] = run_variant(
            casa, science, root, name, VARIANTS[name]
        )
        (root / "imaging_sensitivity.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="ascii"
        )
    results["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (root / "imaging_sensitivity.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(f"Completed mask sensitivity study in {root}")


if __name__ == "__main__":
    main()
