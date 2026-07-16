"""Small direction-independent complex-gain calibration experiment."""

import numpy as np


def baselines(n_antennas):
    """Return all unique cross-correlation antenna pairs."""
    if n_antennas < 2:
        raise ValueError("at least two antennas are required")
    return np.column_stack(np.triu_indices(n_antennas, 1))


def _hermitian_matrix(values, pairs, n_antennas, diagonal=0.0):
    matrix = np.zeros(values.shape[:-1] + (n_antennas, n_antennas), complex)
    p, q = pairs.T
    matrix[..., p, q] = values
    matrix[..., q, p] = values.conj()
    matrix[..., np.arange(n_antennas), np.arange(n_antennas)] = diagonal
    return matrix


def point_source_model(
    positions_m, hour_angle, source_l, source_flux, frequency_hz=1.4e9
):
    """Build a scalar visibility matrix for a one-dimensional point-source sky."""
    positions_m = np.asarray(positions_m, float)
    hour_angle = np.asarray(hour_angle, float)
    pairs = baselines(positions_m.size)
    wavelength = 299_792_458.0 / frequency_hz
    projected_u = (
        np.cos(hour_angle[:, None])
        * (positions_m[pairs[:, 1]] - positions_m[pairs[:, 0]])
        / wavelength
    )
    phase = -2j * np.pi * projected_u[..., None] * np.asarray(source_l)
    values = np.sum(np.asarray(source_flux) * np.exp(phase), axis=-1)
    return _hermitian_matrix(values, pairs, positions_m.size, np.sum(source_flux))


def apply_gains(model, gains):
    """Apply d_pq = g_p m_pq conjugate(g_q)."""
    return gains[..., :, None] * model * gains[..., None, :].conj()


def solve_gains(data, model, reference=0, max_iterations=200, tolerance=1e-10):
    """Solve scalar antenna gains by alternating least-squares updates."""
    data, model = np.asarray(data), np.asarray(model)
    if data.shape != model.shape or data.shape[-1] != data.shape[-2]:
        raise ValueError("data and model must be matching square visibility matrices")

    n_antennas = data.shape[-1]
    if not 0 <= reference < n_antennas:
        raise ValueError("reference antenna is out of range")
    gains = np.ones(data.shape[:-1], complex)

    for _ in range(max_iterations):
        previous = gains.copy()
        for p in range(n_antennas):
            q = np.arange(n_antennas) != p
            numerator = np.sum(
                data[..., p, q] * model[..., p, q].conj() * gains[..., q], axis=-1
            )
            denominator = np.sum(
                abs(model[..., p, q]) ** 2 * abs(gains[..., q]) ** 2, axis=-1
            )
            gains[..., p] = np.divide(
                numerator,
                denominator,
                out=gains[..., p].copy(),
                where=denominator > 0,
            )

        gains *= np.exp(-1j * np.angle(gains[..., reference]))[..., None]
        if np.max(abs(gains - previous)) < tolerance:
            break
    return gains


def correct_visibilities(data, gains):
    """Remove antenna gains from a visibility matrix."""
    response = gains[..., :, None] * gains[..., None, :].conj()
    return np.divide(data, response, out=np.zeros_like(data), where=abs(response) > 0)


def rms_residual(data, model):
    """Return complex RMS over unique cross-correlations."""
    if data.shape != model.shape:
        raise ValueError("data and model shapes do not match")
    p, q = baselines(data.shape[-1]).T
    return float(np.sqrt(np.mean(abs(data[..., p, q] - model[..., p, q]) ** 2)))


def generate_synthetic_data(n_times=64, noise_std=0.02, seed=7):
    """Create the five-antenna data set used by the chapter exercises."""
    positions = np.array([0.0, 36.0, 102.0, 210.0, 348.0])
    hour_angle = np.linspace(-3.5, 3.5, n_times) * np.pi / 12.0
    source_l = np.array([0.0, 0.012, -0.021])
    source_flux = np.array([1.0, 0.35, 0.18])
    model = point_source_model(positions, hour_angle, source_l, source_flux)
    incomplete_model = point_source_model(positions, hour_angle, source_l[:2], source_flux[:2])

    antenna = np.arange(positions.size)
    amplitude = 1.0 + 0.10 * np.sin(1.3 * hour_angle[:, None] + 0.7 * antenna)
    phase = 0.25 * np.sin(0.9 * hour_angle[:, None] + 0.6 * antenna) + 0.04 * antenna
    true_gains = amplitude * np.exp(1j * phase)
    data = apply_gains(model, true_gains)

    if noise_std:
        rng = np.random.default_rng(seed)
        pairs = baselines(positions.size)
        noise = noise_std / np.sqrt(2.0) * (
            rng.normal(size=(n_times, pairs.shape[0]))
            + 1j * rng.normal(size=(n_times, pairs.shape[0]))
        )
        data += _hermitian_matrix(noise, pairs, positions.size)

    return {
        "hour_angle": hour_angle,
        "model": model,
        "incomplete_model": incomplete_model,
        "data": data,
        "true_gains": true_gains,
    }
