"""Small, deterministic CLEAN utilities used by the chapter notebooks."""

from pathlib import Path

import numpy as np
from scipy import signal


def _fits_image(path):
    from astropy.io import fits

    with fits.open(path) as hdus:
        image = np.squeeze(hdus[0].data).astype(float, copy=True)
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D FITS image in {path}")
    return image


def _real_image(data, name):
    image = np.asarray(data, dtype=float)
    if image.ndim != 2 or not np.all(np.isfinite(image)):
        raise ValueError(f"{name} must be a finite two-dimensional array")
    return image


def _normalized_psf(psf):
    psf = _real_image(psf, "psf")
    peak_position = np.unravel_index(np.argmax(np.abs(psf)), psf.shape)
    peak = psf[peak_position]
    if peak == 0:
        raise ValueError("psf must have a non-zero peak")
    return psf / peak, peak_position


def subtract_psf(image, psf, position, amplitude, *, in_place=False):
    """Subtract a PSF centred on one image pixel, clipping at image edges."""
    image = _real_image(image, "image")
    psf, center = _normalized_psf(psf)
    output = image if in_place else image.copy()
    position = np.asarray(position, dtype=int)
    if position.shape != (2,) or np.any(position < 0) or np.any(position >= output.shape):
        raise IndexError("position is outside the image")

    image_start = np.maximum(position - center, 0)
    image_stop = np.minimum(position + np.asarray(psf.shape) - center, output.shape)
    psf_start = image_start - (position - center)
    psf_stop = psf_start + image_stop - image_start
    image_slice = tuple(slice(start, stop) for start, stop in zip(image_start, image_stop))
    psf_slice = tuple(slice(start, stop) for start, stop in zip(psf_start, psf_stop))
    output[image_slice] -= amplitude * psf[psf_slice]
    return output


def hogbom_clean(dirty, psf, gain=0.1, niter=1000, threshold=0.0, mask=None):
    """Run a compact Högbom CLEAN and return model, residual, and components."""
    dirty = _real_image(dirty, "dirty")
    psf, _ = _normalized_psf(psf)
    if not 0 < gain <= 1:
        raise ValueError("gain must satisfy 0 < gain <= 1")
    if not isinstance(niter, (int, np.integer)) or niter < 0:
        raise ValueError("niter must be a non-negative integer")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    if mask is None:
        mask = np.ones(dirty.shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != dirty.shape or not np.any(mask):
            raise ValueError("mask must select at least one dirty-image pixel")

    residual = dirty.copy()
    model = np.zeros_like(dirty)
    components = []
    peak_history = []

    for _ in range(niter):
        search = np.where(mask, np.abs(residual), -np.inf)
        position = np.unravel_index(np.argmax(search), residual.shape)
        peak = residual[position]
        peak_history.append(float(np.abs(peak)))
        if np.abs(peak) <= threshold:
            break
        amplitude = gain * peak
        model[position] += amplitude
        residual = subtract_psf(residual, psf, position, amplitude)
        components.append((int(position[0]), int(position[1]), float(amplitude)))

    return {
        "model": model,
        "residual": residual,
        "components": components,
        "iterations": len(components),
        "peak_history": np.asarray(peak_history),
    }


def gaussian_clean_beam(psf, radius=None):
    """Estimate a peak-normalized Gaussian restoring beam from a PSF main lobe."""
    psf, peak_position = _normalized_psf(psf)
    if radius is None:
        radius = max(3, min(psf.shape) // 12)
    yy, xx = np.indices(psf.shape, dtype=float)
    rr2 = (yy - peak_position[0]) ** 2 + (xx - peak_position[1]) ** 2
    weights = np.where(rr2 <= radius**2, np.clip(psf, 0.0, None), 0.0)
    if weights.sum() == 0:
        raise ValueError("psf main lobe has no positive support")

    y_mean = np.sum(weights * yy) / weights.sum()
    x_mean = np.sum(weights * xx) / weights.sum()
    dy, dx = yy - y_mean, xx - x_mean
    covariance = np.array(
        [
            [np.sum(weights * dy * dy), np.sum(weights * dy * dx)],
            [np.sum(weights * dy * dx), np.sum(weights * dx * dx)],
        ]
    ) / weights.sum()
    covariance += np.eye(2) * 1e-6

    output_center = (np.asarray(psf.shape) - 1) / 2
    dy, dx = yy - output_center[0], xx - output_center[1]
    inverse = np.linalg.inv(covariance)
    exponent = -0.5 * (
        inverse[0, 0] * dy**2
        + 2 * inverse[0, 1] * dy * dx
        + inverse[1, 1] * dx**2
    )
    beam = np.exp(exponent)
    return beam / beam.max()


def restore_image(model, residual, clean_beam, residual_scale=1.0):
    """Convolve a component model with the clean beam and add scaled residuals."""
    model = _real_image(model, "model")
    residual = _real_image(residual, "residual")
    clean_beam, _ = _normalized_psf(clean_beam)
    if model.shape != residual.shape or not np.isfinite(residual_scale):
        raise ValueError("model and residual must match and residual_scale must be finite")
    restored_model = signal.fftconvolve(model, clean_beam, mode="same")
    return restored_model + residual_scale * residual


def load_clean_data(dirty_path, psf_path, size=101, seed=11):
    """Load historical FITS inputs or generate a deterministic small example."""
    dirty_path, psf_path = Path(dirty_path), Path(psf_path)
    if dirty_path.exists() and psf_path.exists():
        return _fits_image(dirty_path), _fits_image(psf_path), "historical FITS data"

    axis = np.arange(size) - size // 2
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(xx, yy)
    psf = np.exp(-0.5 * (radius / 2.2) ** 2)
    psf += 0.18 * np.cos(0.75 * radius) * np.exp(-0.5 * (radius / 13.0) ** 2)
    psf /= psf[size // 2, size // 2]

    sky = np.zeros((size, size))
    center = size // 2
    sky[center, center] = 18.0
    sky[center - 17, center + 21] = 10.0
    sky[center + 23, center - 14] = 6.0
    dirty = signal.fftconvolve(sky, psf, mode="same")
    dirty += np.random.default_rng(seed).normal(0.0, 0.15, dirty.shape)
    return dirty, psf, "deterministic synthetic data"
