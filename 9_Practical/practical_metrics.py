"""Small, unit-explicit numerical helpers for Chapter 9 exercises."""

import numpy as np


def gaussian_beam_area(fwhm_major, fwhm_minor):
    """Return a Gaussian beam area in the squared unit of its FWHM axes."""
    major = np.asarray(fwhm_major, dtype=float)
    minor = np.asarray(fwhm_minor, dtype=float)
    if np.any(major <= 0) or np.any(minor <= 0):
        raise ValueError("Beam FWHM axes must be positive")
    return np.pi * major * minor / (4 * np.log(2))


def pixels_per_beam(fwhm_major, fwhm_minor, pixel_scale):
    """Return pixels per beam when all angular inputs use the same unit."""
    if pixel_scale <= 0:
        raise ValueError("Pixel scale must be positive")
    return gaussian_beam_area(fwhm_major, fwhm_minor) / pixel_scale**2


def aperture_flux_jy(image_jy_beam, mask, beam_area_pixels, background=0.0):
    """Integrate a background-subtracted Jy/beam image over a Boolean mask."""
    image = np.asarray(image_jy_beam, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if image.shape != mask.shape or not mask.any():
        raise ValueError("Image and non-empty mask must have identical shapes")
    if beam_area_pixels <= 0:
        raise ValueError("Beam area in pixels must be positive")
    return float(np.sum(image[mask] - background) / beam_area_pixels)


def averaging_attenuation(phase_rate, interval):
    """Return sinc attenuation for phase rate in cycles/unit over an interval."""
    if interval < 0:
        raise ValueError("Averaging interval must be non-negative")
    return np.abs(np.sinc(np.asarray(phase_rate, dtype=float) * interval))


def polarization_angle(q, u):
    """Return EVPA in radians, preserving the Q/U quadrant."""
    return 0.5 * np.arctan2(u, q)


def debiased_linear_polarization(q, u, sigma_qu):
    """Apply the high-S/N equal-noise Q/U debiasing approximation."""
    if np.any(np.asarray(sigma_qu) < 0):
        raise ValueError("Q/U noise must be non-negative")
    amplitude_squared = np.asarray(q) ** 2 + np.asarray(u) ** 2
    return np.sqrt(np.maximum(amplitude_squared - np.asarray(sigma_qu) ** 2, 0))


def rm_synthesis(complex_polarization, lambda_squared, faraday_depth, weights=None):
    """Return normalized dirty Faraday spectrum, RMSF, and weighted lambda0^2."""
    polarization = np.asarray(complex_polarization, dtype=complex)
    lambda_squared = np.asarray(lambda_squared, dtype=float)
    faraday_depth = np.asarray(faraday_depth, dtype=float)
    if lambda_squared.ndim != 1 or faraday_depth.ndim != 1:
        raise ValueError("lambda_squared and faraday_depth must be one-dimensional")
    if polarization.shape[-1] != lambda_squared.size:
        raise ValueError("Polarization's last axis must match lambda_squared")
    if not np.all(np.isfinite(lambda_squared)) or not np.all(np.isfinite(faraday_depth)):
        raise ValueError("Coordinates must be finite")

    weights = np.ones_like(lambda_squared) if weights is None else np.asarray(weights, float)
    if weights.shape != lambda_squared.shape or np.any(weights < 0):
        raise ValueError("Weights must be non-negative and match lambda_squared")
    weight_sum = weights.sum()
    if weight_sum <= 0:
        raise ValueError("At least one RM-synthesis weight must be positive")

    lambda0_squared = float(np.dot(weights, lambda_squared) / weight_sum)
    kernel = np.exp(
        -2j * np.outer(lambda_squared - lambda0_squared, faraday_depth)
    )
    dirty_spectrum = np.tensordot(
        polarization * weights, kernel, axes=([-1], [0])
    ) / weight_sum
    rmsf = np.sum(weights[:, None] * kernel, axis=0) / weight_sum
    return dirty_spectrum, rmsf, lambda0_squared


def primary_beam_mosaic(apparent_images, primary_beams, noise_jy_beam):
    """Combine uncorrected pointing images with primary-beam inverse variance."""
    images = np.asarray(apparent_images, dtype=float)
    beams = np.asarray(primary_beams, dtype=float)
    noise = np.asarray(noise_jy_beam, dtype=float)
    if images.shape != beams.shape or images.ndim < 2:
        raise ValueError("Images and primary beams must share a pointing-first shape")
    if noise.shape != (images.shape[0],) or np.any(noise <= 0):
        raise ValueError("Provide one positive noise value per pointing")
    if np.any(beams < 0):
        raise ValueError("Power primary beams must be non-negative")

    reshape = (noise.size,) + (1,) * (images.ndim - 1)
    inverse_variance = 1 / noise.reshape(reshape) ** 2
    denominator = np.sum(beams**2 * inverse_variance, axis=0)
    numerator = np.sum(beams * images * inverse_variance, axis=0)
    mosaic = np.full(denominator.shape, np.nan)
    uncertainty = np.full(denominator.shape, np.inf)
    np.divide(numerator, denominator, out=mosaic, where=denominator > 0)
    np.divide(1, np.sqrt(denominator), out=uncertainty, where=denominator > 0)
    return mosaic, uncertainty


def hi_mass_solar(integrated_flux_jy_km_s, luminosity_distance_mpc, redshift=0.0):
    """Return optically thin H I mass from observed velocity-integrated flux."""
    if integrated_flux_jy_km_s < 0 or luminosity_distance_mpc < 0 or redshift <= -1:
        raise ValueError("Flux and distance must be non-negative and redshift > -1")
    return (
        2.356e5
        * luminosity_distance_mpc**2
        * integrated_flux_jy_km_s
        / (1 + redshift)
    )
