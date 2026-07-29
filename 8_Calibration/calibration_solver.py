"""Small direction-independent complex-gain calibration experiment."""

import numpy as np


C = 299_792_458.0


def baselines(n_antennas):
    """Return all unique cross-correlation antenna pairs with ``p < q``."""
    if not isinstance(n_antennas, (int, np.integer)) or n_antennas < 2:
        raise ValueError("n_antennas must be an integer of at least 2")
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
    positions_m = np.asarray(positions_m, dtype=float)
    hour_angle = np.asarray(hour_angle, dtype=float)
    source_l = np.asarray(source_l, dtype=float)
    source_flux = np.asarray(source_flux, dtype=float)
    if positions_m.ndim != 1 or positions_m.size < 2:
        raise ValueError("positions_m must contain at least two antenna positions")
    if hour_angle.ndim != 1:
        raise ValueError("hour_angle must be one-dimensional")
    if source_l.ndim != 1 or source_flux.ndim != 1 or source_l.shape != source_flux.shape:
        raise ValueError("source_l and source_flux must be matching one-dimensional arrays")
    if source_l.size == 0:
        raise ValueError("at least one source is required")
    if not all(
        np.isfinite(value).all()
        for value in (positions_m, hour_angle, source_l, source_flux)
    ):
        raise ValueError("model inputs must be finite")
    if np.any(abs(source_l) > 1.0):
        raise ValueError("source_l values must be direction cosines in [-1, 1]")
    if not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive and finite")

    pairs = baselines(positions_m.size)
    wavelength = C / frequency_hz
    projected_u = (
        np.cos(hour_angle[:, None])
        * (positions_m[pairs[:, 0]] - positions_m[pairs[:, 1]])
        / wavelength
    )
    phase = -2j * np.pi * projected_u[..., None] * source_l
    values = np.sum(source_flux * np.exp(phase), axis=-1)
    return _hermitian_matrix(values, pairs, positions_m.size, np.sum(source_flux))


def apply_gains(model, gains):
    """Apply ``d_pq = g_p m_pq conjugate(g_q)``."""
    model = np.asarray(model, dtype=complex)
    gains = np.asarray(gains, dtype=complex)
    if model.ndim < 2 or model.shape[-1] != model.shape[-2]:
        raise ValueError("model must be a square visibility matrix")
    if gains.shape != model.shape[:-1]:
        raise ValueError("gains must have shape model.shape[:-1]")
    if not np.isfinite(model).all() or not np.isfinite(gains).all():
        raise ValueError("model and gains must be finite")
    return gains[..., :, None] * model * gains[..., None, :].conj()


def _prepare_solver_inputs(data, model, weights):
    data = np.asarray(data, dtype=complex)
    model = np.asarray(model, dtype=complex)
    if data.ndim < 2 or data.shape != model.shape or data.shape[-1] != data.shape[-2]:
        raise ValueError("data and model must be matching square visibility matrices")

    if weights is None:
        weights = np.ones(data.shape, dtype=float)
    else:
        try:
            weights = np.broadcast_to(np.asarray(weights, dtype=float), data.shape).copy()
        except ValueError as error:
            raise ValueError("weights must be broadcastable to the data shape") from error
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative")
    if not np.allclose(weights, np.swapaxes(weights, -1, -2)):
        raise ValueError("weights must be symmetric across each antenna pair")

    diagonal = np.arange(data.shape[-1])
    weights[..., diagonal, diagonal] = 0.0
    valid = weights > 0
    if not np.isfinite(data[valid]).all() or not np.isfinite(model[valid]).all():
        raise ValueError("unflagged data and model values must be finite")
    data = np.where(valid, data, 0.0)
    model = np.where(valid, model, 0.0)
    if not np.allclose(data, np.swapaxes(data.conj(), -1, -2)):
        raise ValueError("data must be Hermitian after flags are applied")
    if not np.allclose(model, np.swapaxes(model.conj(), -1, -2)):
        raise ValueError("model must be Hermitian after flags are applied")
    return data, model, weights


def _require_connected_constraints(model, weights):
    n_antennas = model.shape[-1]
    adjacency = weights * abs(model) ** 2 > 0
    for graph in adjacency.reshape(-1, n_antennas, n_antennas):
        reached = {0}
        pending = [0]
        while pending:
            antenna = pending.pop()
            neighbours = set(np.flatnonzero(graph[antenna])) - reached
            reached.update(neighbours)
            pending.extend(neighbours)
        if len(reached) != n_antennas:
            raise ValueError(
                "weighted non-zero model baselines must connect every antenna"
            )


def solve_gains(
    data,
    model,
    weights=None,
    reference=0,
    max_iterations=200,
    tolerance=1e-10,
):
    """Solve scalar antenna gains with weighted alternating least squares.

    ``data`` and ``model`` are full Hermitian visibility matrices. ``weights``
    contains inverse variances; set both entries of a baseline to zero to flag
    it. Each leading-axis sample is solved independently.
    """
    data, model, weights = _prepare_solver_inputs(data, model, weights)
    n_antennas = data.shape[-1]
    if not isinstance(reference, (int, np.integer)) or not 0 <= reference < n_antennas:
        raise ValueError("reference antenna is out of range")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")
    _require_connected_constraints(model, weights)

    gains = np.ones(data.shape[:-1], complex)
    for _ in range(max_iterations):
        previous = gains.copy()
        for p in range(n_antennas):
            q = np.arange(n_antennas) != p
            numerator = np.sum(
                weights[..., p, q]
                * data[..., p, q]
                * model[..., p, q].conj()
                * gains[..., q],
                axis=-1,
            )
            denominator = np.sum(
                weights[..., p, q]
                * abs(model[..., p, q]) ** 2
                * abs(gains[..., q]) ** 2,
                axis=-1,
            )
            if np.any(denominator <= np.finfo(float).tiny):
                raise RuntimeError("gain update became unconstrained")
            gains[..., p] = numerator / denominator

        gains *= np.exp(-1j * np.angle(gains[..., reference]))[..., None]
        if np.max(abs(gains - previous)) < tolerance:
            break
    else:
        raise RuntimeError("gain solver did not converge")
    return gains


def correct_visibilities(data, gains):
    """Remove non-zero antenna gains from a visibility matrix."""
    data = np.asarray(data, dtype=complex)
    gains = np.asarray(gains, dtype=complex)
    if data.ndim < 2 or data.shape[-1] != data.shape[-2]:
        raise ValueError("data must be a square visibility matrix")
    if gains.shape != data.shape[:-1]:
        raise ValueError("gains must have shape data.shape[:-1]")
    if not np.isfinite(data).all() or not np.isfinite(gains).all():
        raise ValueError("data and gains must be finite")
    if np.any(abs(gains) <= np.finfo(float).tiny):
        raise ValueError("cannot correct with a zero gain")
    response = gains[..., :, None] * gains[..., None, :].conj()
    return data / response


def rms_residual(data, model, weights=None):
    """Return weighted complex RMS over unique cross-correlations."""
    data = np.asarray(data, dtype=complex)
    model = np.asarray(model, dtype=complex)
    if data.ndim < 2 or data.shape != model.shape or data.shape[-1] != data.shape[-2]:
        raise ValueError("data and model must be matching square visibility matrices")
    p, q = baselines(data.shape[-1]).T
    residual_squared = abs(data[..., p, q] - model[..., p, q]) ** 2
    if weights is None:
        pair_weights = np.ones(residual_squared.shape)
    else:
        try:
            full_weights = np.broadcast_to(np.asarray(weights, dtype=float), data.shape)
        except ValueError as error:
            raise ValueError("weights must be broadcastable to the data shape") from error
        pair_weights = full_weights[..., p, q]
        if not np.isfinite(pair_weights).all() or np.any(pair_weights < 0):
            raise ValueError("weights must be finite and non-negative")
    total_weight = np.sum(pair_weights)
    if total_weight <= 0:
        raise ValueError("at least one cross-correlation must have positive weight")
    valid = pair_weights > 0
    if not np.isfinite(residual_squared[valid]).all():
        raise ValueError("unflagged residuals must be finite")
    return float(np.sqrt(np.sum(pair_weights[valid] * residual_squared[valid]) / total_weight))


def calibration_diagnostics(
    data,
    model,
    fit_weights=None,
    validation_weights=None,
    reference=0,
    comparison_reference=None,
    **solver_options,
):
    """Fit gains and report training, validation, and gauge-invariance RMS.

    For a true holdout test, samples selected by ``validation_weights`` must
    have zero ``fit_weights``. The returned gains use ``reference``.
    """
    gains = solve_gains(
        data,
        model,
        weights=fit_weights,
        reference=reference,
        **solver_options,
    )
    corrected = correct_visibilities(data, gains)
    metrics = {
        "fit_rms_before": rms_residual(data, model, fit_weights),
        "fit_rms_after": rms_residual(corrected, model, fit_weights),
    }

    if validation_weights is not None:
        metrics["validation_rms_before"] = rms_residual(
            data, model, validation_weights
        )
        metrics["validation_rms_after"] = rms_residual(
            corrected, model, validation_weights
        )

    if comparison_reference is not None:
        if comparison_reference == reference:
            raise ValueError("comparison_reference must differ from reference")
        comparison_gains = solve_gains(
            data,
            model,
            weights=fit_weights,
            reference=comparison_reference,
            **solver_options,
        )
        comparison_corrected = correct_visibilities(data, comparison_gains)
        metrics["reference_rms_difference"] = rms_residual(
            corrected, comparison_corrected
        )

    return gains, metrics


def generate_synthetic_data(n_times=64, noise_std=0.02, seed=7):
    """Create the five-antenna data set used by the chapter exercises."""
    if not isinstance(n_times, (int, np.integer)) or n_times < 1:
        raise ValueError("n_times must be a positive integer")
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and non-negative")

    positions = np.array([0.0, 36.0, 102.0, 210.0, 348.0])
    hour_angle = np.linspace(-3.5, 3.5, n_times) * np.pi / 12.0
    source_l = np.array([0.0, 0.012, -0.021])
    source_flux = np.array([1.0, 0.35, 0.18])
    model = point_source_model(positions, hour_angle, source_l, source_flux)
    incomplete_model = point_source_model(
        positions, hour_angle, source_l[:2], source_flux[:2]
    )

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
