"""Small numerical checks for Chapter 7 observing-system estimates."""

import numpy as np


C = 299_792_458.0
K_B = 1.380_649e-23
JY = 1e-26


def sefd_jy(diameter_m, aperture_efficiency, system_temperature_k):
    """Return the single-antenna SEFD in Jy."""
    if diameter_m <= 0 or system_temperature_k <= 0:
        raise ValueError("diameter and system temperature must be positive")
    if not 0 < aperture_efficiency <= 1:
        raise ValueError("aperture efficiency must satisfy 0 < eta <= 1")
    effective_area = aperture_efficiency * np.pi * (diameter_m / 2) ** 2
    return 2 * K_B * system_temperature_k / effective_area / JY


def image_noise_jy(
    sefd,
    n_antennas,
    n_polarizations,
    bandwidth_hz,
    integration_s,
    system_efficiency=1.0,
):
    """Return ideal natural-weight image noise for identical antennas."""
    if n_antennas < 2 or n_polarizations < 1:
        raise ValueError("at least two antennas and one polarization are required")
    if min(sefd, bandwidth_hz, integration_s, system_efficiency) <= 0:
        raise ValueError("SEFD, bandwidth, time, and efficiency must be positive")
    samples = (
        n_polarizations
        * n_antennas
        * (n_antennas - 1)
        * bandwidth_hz
        * integration_s
    )
    return sefd / (system_efficiency * np.sqrt(samples))


def correlator_output_rates(
    n_antennas,
    n_channels,
    n_products,
    integration_s,
    bytes_per_visibility=8,
    include_autocorrelations=False,
):
    """Return baseline count, visibility rate, and byte rate."""
    if min(n_antennas, n_channels, n_products, integration_s) <= 0:
        raise ValueError("correlator dimensions and integration time must be positive")
    offset = 1 if include_autocorrelations else -1
    baselines = n_antennas * (n_antennas + offset) // 2
    visibility_rate = baselines * n_channels * n_products / integration_s
    return baselines, visibility_rate, visibility_rate * bytes_per_visibility


def dterm_crosshand_ratio(d_x_p, d_y_q):
    """Return first-order V_xy/(I/2) leakage for an unpolarized source."""
    return complex(d_x_p) + np.conj(complex(d_y_q))


def gaussian_power_beam(offset_rad, frequency_hz, diameter_m, kappa=1.02):
    """Return FWHM and response of a circular Gaussian power beam."""
    if min(frequency_hz, diameter_m, kappa) <= 0:
        raise ValueError("frequency, diameter, and kappa must be positive")
    fwhm = kappa * C / (frequency_hz * diameter_m)
    response = np.exp(-4 * np.log(2) * (np.asarray(offset_rad) / fwhm) ** 2)
    return fwhm, response


def apparent_spectral_index(alpha, nu_low, nu_high, beam_low, beam_high):
    """Return the two-frequency spectral index after beam attenuation."""
    if min(nu_low, nu_high, beam_low, beam_high) <= 0 or nu_low == nu_high:
        raise ValueError("frequencies and beam responses must be positive and distinct")
    return alpha + np.log(beam_high / beam_low) / np.log(nu_high / nu_low)
