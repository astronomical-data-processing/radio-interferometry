"""Deterministic visibility experiment for the Chapter 9 foundation chain."""

import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np

C = 299_792_458.0
ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _calibration_solver():
    return _load_module(
        "foundation_calibration_solver", "8_Calibration/calibration_solver.py"
    )


@lru_cache(maxsize=1)
def _clean_tools():
    return _load_module("foundation_clean_tools", "6_Deconvolution/clean_demo.py")


def _sky_visibility(uvw_m, frequency_hz, directions_lm, flux_jy):
    uv_lambda = uvw_m[:, None, :, :2] * frequency_hz[None, :, None, None] / C
    phase = -2j * np.pi * np.einsum("tcbd,sd->tcbs", uv_lambda, directions_lm)
    return np.sum(flux_jy * np.exp(phase), axis=-1)


def _apply_row_gains(model, gains, pairs):
    response = gains[:, pairs[:, 0]] * gains[:, pairs[:, 1]].conj()
    return model * response[:, None, :]


def _complex_noise(rng, shape, sigma):
    return sigma / np.sqrt(2) * (rng.normal(size=shape) + 1j * rng.normal(size=shape))


def make_lab(seed=23):
    """Return a small controlled observation with calibrator and target scans.

    The arrays are generated in memory. They are not a Measurement Set and are
    not evidence for the performance of any real telescope or pipeline.
    """
    rng = np.random.default_rng(seed)
    antenna_xy_m = np.array(
        [[0, 0], [32, 12], [71, -19], [113, 38], [158, -31], [203, 18]],
        dtype=float,
    )
    pairs = np.column_stack(np.triu_indices(len(antenna_xy_m), 1))
    baseline_xy = antenna_xy_m[pairs[:, 0]] - antenna_xy_m[pairs[:, 1]]
    hour_angle = np.linspace(-2, 2, 81) * np.pi / 12
    declination = np.deg2rad(32)
    sin_h, cos_h = np.sin(hour_angle), np.cos(hour_angle)
    u_m = sin_h[:, None] * baseline_xy[:, 0] + cos_h[:, None] * baseline_xy[:, 1]
    v_m = (
        -np.sin(declination) * cos_h[:, None] * baseline_xy[:, 0]
        + np.sin(declination) * sin_h[:, None] * baseline_xy[:, 1]
    )
    uvw_m = np.stack((u_m, v_m, np.zeros_like(u_m)), axis=-1)
    frequency_hz = 1.4e9 + np.arange(-3.5, 4.5) * 4e6

    directions_lm = np.array([[0, 0], [0.0045, -0.003], [-0.006, 0.004]])
    flux_jy = np.array([1.0, 0.32, 0.09])
    target_model = _sky_visibility(uvw_m, frequency_hz, directions_lm, flux_jy)
    bright_model = _sky_visibility(uvw_m, frequency_hz, directions_lm[:2], flux_jy[:2])
    calibrator_model = np.ones_like(target_model)

    antenna = np.arange(len(antenna_xy_m))
    amplitude = 1 + 0.06 * np.sin(1.4 * hour_angle[:, None] + 0.7 * antenna)
    phase = 0.24 * np.sin(1.1 * hour_angle[:, None] + 0.8 * antenna)
    transfer_gains = amplitude * np.exp(1j * phase)
    target_phase = 0.10 * np.sin(2.2 * hour_angle[:, None] + 0.9 * antenna)
    target_residual_gains = np.exp(1j * target_phase)

    noise_jy = 0.018
    baseline_scale = 1 + 0.15 * np.arange(len(pairs)) / len(pairs)
    baseline_noise_jy = noise_jy * baseline_scale
    calibrator_data = _apply_row_gains(calibrator_model, transfer_gains, pairs)
    target_data = _apply_row_gains(
        target_model, transfer_gains * target_residual_gains, pairs
    )
    calibrator_data += _complex_noise(rng, calibrator_data.shape, baseline_noise_jy)
    target_data += _complex_noise(rng, target_data.shape, baseline_noise_jy)

    flag = rng.random(target_data.shape) < 0.008
    flag[:, 1, :] = True
    antenna_four = np.any(pairs == 4, axis=1)
    flag[20:35, 2:6, antenna_four] = True
    baseline_14 = np.flatnonzero(np.all(pairs == [1, 4], axis=1))[0]
    flag[46:51, :, baseline_14] = True
    calibrator_data[:, 1, :] += 4 * np.exp(0.4j)
    target_data[:, 1, :] += 3 * np.exp(-0.7j)
    calibrator_data[0, 1, 0] = np.nan + 1j * np.nan
    target_data[0, 1, 0] = np.nan + 1j * np.nan

    weight = np.broadcast_to(
        1 / baseline_noise_jy[None, None, :] ** 2,
        target_data.shape,
    ).copy()
    weight[flag] = 0
    return {
        "schema": "controlled_foundation_visibility_v1",
        "seed": seed,
        "time_s": np.arange(len(hour_angle)) * 180.0,
        "frequency_hz": frequency_hz,
        "channel_width_hz": 4e6,
        "antenna_xy_m": antenna_xy_m,
        "pairs": pairs,
        "uvw_m": uvw_m,
        "directions_lm": directions_lm,
        "flux_jy": flux_jy,
        "calibrator_model": calibrator_model,
        "target_model": target_model,
        "bright_model": bright_model,
        "calibrator_data": calibrator_data,
        "target_data": target_data,
        "flag": flag,
        "weight": weight,
        "noise_jy": noise_jy,
        "transfer_gains": transfer_gains,
        "target_residual_gains": target_residual_gains,
    }


def inspection_summary(lab=None):
    """Summarize structure, validity and flag evidence before calibration."""
    lab = make_lab() if lab is None else lab
    flag, weight, pairs = lab["flag"], lab["weight"], lab["pairs"]
    antenna_fraction = []
    for antenna in range(len(lab["antenna_xy_m"])):
        selected = np.any(pairs == antenna, axis=1)
        antenna_fraction.append(float(flag[:, :, selected].mean()))
    return {
        "schema": lab["schema"],
        "data_shape_time_channel_baseline": lab["target_data"].shape,
        "cross_correlations": len(pairs),
        "flag_fraction": float(flag.mean()),
        "channel_flag_fraction": flag.mean(axis=(0, 2)),
        "antenna_flag_fraction": np.asarray(antenna_fraction),
        "unflagged_nonfinite_values": int(
            np.count_nonzero(~np.isfinite(lab["target_data"]) & ~flag)
        ),
        "unflagged_nonpositive_weights": int(np.count_nonzero((weight <= 0) & ~flag)),
        "fully_flagged_channel_indices": np.flatnonzero(flag.all(axis=(0, 2))),
        "highest_flag_antenna": int(np.argmax(antenna_fraction)),
    }


def _weighted_channel_average(data, model, weight):
    safe_data = np.where(weight > 0, data, 0)
    safe_model = np.where(weight > 0, model, 0)
    combined_weight = weight.sum(axis=1)
    averaged_data = np.divide(
        (safe_data * weight).sum(axis=1),
        combined_weight,
        out=np.zeros_like(combined_weight, dtype=complex),
        where=combined_weight > 0,
    )
    averaged_model = np.divide(
        (safe_model * weight).sum(axis=1),
        combined_weight,
        out=np.zeros_like(combined_weight, dtype=complex),
        where=combined_weight > 0,
    )
    return averaged_data, averaged_model, combined_weight


def _row_to_matrix(values, pairs, n_antennas, fill=0):
    matrix = np.full(values.shape[:-1] + (n_antennas, n_antennas), fill, values.dtype)
    p, q = pairs.T
    matrix[..., p, q] = values
    matrix[..., q, p] = values.conj() if np.iscomplexobj(values) else values
    return matrix


def _baseline_masks(pairs, n_antennas):
    holdout_pairs = {(0, 4), (1, 3), (2, 5)}
    held_out = np.array([tuple(pair) in holdout_pairs for pair in pairs])
    holdout = _row_to_matrix(held_out, pairs, n_antennas)
    training = _row_to_matrix(~held_out, pairs, n_antennas)
    return training, holdout


def _weighted_rms(data, model, weight):
    valid = weight > 0
    return float(
        np.sqrt(
            np.sum(weight[valid] * abs(data[valid] - model[valid]) ** 2)
            / weight[valid].sum()
        )
    )


def calibrate(lab=None):
    """Solve transferable scalar gains and apply them with weight propagation."""
    lab = make_lab() if lab is None else lab
    data, model, weight = _weighted_channel_average(
        lab["calibrator_data"], lab["calibrator_model"], lab["weight"]
    )
    n_antennas = len(lab["antenna_xy_m"])
    data_matrix = _row_to_matrix(data, lab["pairs"], n_antennas)
    model_matrix = _row_to_matrix(model, lab["pairs"], n_antennas)
    weight_matrix = _row_to_matrix(weight, lab["pairs"], n_antennas)
    training, holdout = _baseline_masks(lab["pairs"], n_antennas)
    solution_support = np.sqrt(
        (weight_matrix * training * abs(model_matrix) ** 2).sum(axis=-1)
    )
    solver = _calibration_solver()
    validation_gains = solver.solve_gains(
        data_matrix, model_matrix, weights=weight_matrix * training
    )
    gains = solver.solve_gains(data_matrix, model_matrix, weights=weight_matrix)

    response = gains[:, lab["pairs"][:, 0]] * gains[:, lab["pairs"][:, 1]].conj()
    corrected_calibrator = data / response
    corrected_calibrator_weight = weight * abs(response) ** 2
    validation_response = (
        validation_gains[:, lab["pairs"][:, 0]]
        * validation_gains[:, lab["pairs"][:, 1]].conj()
    )
    validation_corrected = data / validation_response
    validation_weight = weight * abs(validation_response) ** 2
    row_holdout = holdout[lab["pairs"][:, 0], lab["pairs"][:, 1]].astype(bool)

    target_response = response[:, None, :]
    corrected_target = lab["target_data"] / target_response
    corrected_weight = lab["weight"] * abs(target_response) ** 2
    return {
        "lab": lab,
        "gains": gains,
        "corrected_calibrator": corrected_calibrator,
        "corrected_target": corrected_target,
        "corrected_weight": corrected_weight,
        "calibrator_rms_before": _weighted_rms(data, model, weight),
        "calibrator_rms_after": _weighted_rms(
            corrected_calibrator, model, corrected_calibrator_weight
        ),
        "holdout_rms": _weighted_rms(
            validation_corrected[:, row_holdout],
            model[:, row_holdout],
            validation_weight[:, row_holdout],
        ),
        "minimum_antenna_solution_snr": float(solution_support.min()),
        "weight_scale_range": (
            float(np.min(abs(target_response) ** 2)),
            float(np.max(abs(target_response) ** 2)),
        ),
    }


def calibration_summary(result=None):
    result = calibrate() if result is None else result
    gains = result["gains"]
    return {
        "reference_antenna": 0,
        "gain_amplitude_range": (float(abs(gains).min()), float(abs(gains).max())),
        "calibrator_rms_before": result["calibrator_rms_before"],
        "calibrator_rms_after": result["calibrator_rms_after"],
        "connected_holdout_rms": result["holdout_rms"],
        "minimum_antenna_solution_snr": result["minimum_antenna_solution_snr"],
        "corrected_weight_scale_range": result["weight_scale_range"],
        "absolute_flux_scale_claim": "controlled calibrator model only",
    }


def _direct_image(data, weight, lab, npix=64, cell_rad=3e-4):
    valid = weight > 0
    u = (lab["uvw_m"][:, None, :, 0] * lab["frequency_hz"][None, :, None] / C)[valid]
    v = (lab["uvw_m"][:, None, :, 1] * lab["frequency_hz"][None, :, None] / C)[valid]
    visibility, sample_weight = data[valid], weight[valid]
    axis = (np.arange(npix) - npix // 2) * cell_rad
    ll, mm = np.meshgrid(axis, axis)
    dirty = np.zeros((npix, npix), float)
    psf = np.zeros_like(dirty)
    for start in range(0, len(visibility), 128):
        block = slice(start, start + 128)
        phase = np.exp(
            2j
            * np.pi
            * (u[block, None, None] * ll[None] + v[block, None, None] * mm[None])
        )
        dirty += np.sum(
            sample_weight[block, None, None]
            * np.real(visibility[block, None, None] * phase),
            axis=0,
        )
        psf += np.sum(sample_weight[block, None, None] * phase.real, axis=0)
    normalization = sample_weight.sum()
    return dirty / normalization, psf / normalization, cell_rad


def make_imaging_bundle(lab=None, data=None, weight=None):
    """Form a transparent dirty/PSF/CLEAN/residual/restored product bundle."""
    if (data is None) != (weight is None):
        raise ValueError("data and weight must be supplied together")
    if data is None or weight is None:
        calibration = calibrate(lab)
        lab = calibration["lab"]
        data = calibration["corrected_target"]
        weight = calibration["corrected_weight"]
    else:
        lab = make_lab() if lab is None else lab
    dirty, psf, cell_rad = _direct_image(data, weight, lab)
    clean = _clean_tools()
    edge = np.ones(dirty.shape, bool)
    edge[12:-12, 12:-12] = False
    noise = 1.4826 * np.median(abs(dirty[edge] - np.median(dirty[edge])))
    result = clean.hogbom_clean(dirty, psf, gain=0.12, niter=350, threshold=3 * noise)
    beam_fit_radius = 4
    clean_beam = clean.gaussian_clean_beam(psf, radius=beam_fit_radius)
    yy, xx = np.indices(psf.shape)
    center = np.array(psf.shape) // 2
    beam_fit = np.hypot(yy - center[0], xx - center[1]) <= beam_fit_radius
    dirty_beam_area = float(psf[beam_fit].sum())
    residual_scale = float(clean_beam.sum() / dirty_beam_area)
    restored = clean.restore_image(
        result["model"], result["residual"], clean_beam, residual_scale
    )
    return {
        "lab": lab,
        "dirty": dirty,
        "psf": psf,
        "model": result["model"],
        "residual": result["residual"],
        "restored": restored,
        "primary_beam": np.ones_like(dirty),
        "sensitivity": np.full_like(dirty, 1 / np.sqrt(2 * weight.sum())),
        "clean_beam": clean_beam,
        "beam_area_pixels": float(clean_beam.sum()),
        "dirty_beam_area_pixels": dirty_beam_area,
        "cell_rad": cell_rad,
        "weight_sum": float(weight.sum()),
        "iterations": result["iterations"],
        "threshold_jy_per_beam": float(3 * noise),
        "residual_scale": residual_scale,
        "residual_scaling": "clean-beam area / central dirty-beam aperture area",
    }


def imaging_summary(bundle=None):
    bundle = make_imaging_bundle() if bundle is None else bundle
    center = tuple(np.array(bundle["dirty"].shape) // 2)
    edge = np.ones(bundle["residual"].shape, bool)
    edge[12:-12, 12:-12] = False
    residual_median = np.median(bundle["residual"][edge])
    return {
        "products": ["dirty", "psf", "model", "residual", "restored", "primary_beam"],
        "psf_peak": float(bundle["psf"][center]),
        "dirty_peak_jy_per_beam": float(bundle["dirty"].max()),
        "restored_peak_jy_per_beam": float(bundle["restored"].max()),
        "residual_rms_jy_per_dirty_beam": float(
            1.4826 * np.median(abs(bundle["residual"][edge] - residual_median))
        ),
        "clean_iterations": bundle["iterations"],
        "beam_area_pixels": bundle["beam_area_pixels"],
        "residual_scale": bundle["residual_scale"],
        "residual_scaling": bundle["residual_scaling"],
    }


def _selfcal_ratios(data, model, weight):
    valid_model = (
        (weight > 0) & np.isfinite(data) & np.isfinite(model) & (abs(model) > 1e-8)
    )
    ratio_weight = np.where(valid_model, weight * abs(model) ** 2, 0)
    ratio = np.divide(data, model, out=np.zeros_like(data), where=valid_model)
    combined_weight = ratio_weight.sum(axis=1)
    combined_ratio = np.divide(
        (ratio * ratio_weight).sum(axis=1),
        combined_weight,
        out=np.zeros_like(combined_weight, dtype=complex),
        where=combined_weight > 0,
    )
    return combined_ratio, combined_weight


def self_calibrate(lab=None):
    """Apply one phase-only target self-calibration with connected holdout baselines."""
    calibration = calibrate(lab)
    lab = calibration["lab"]
    ratio, ratio_weight = _selfcal_ratios(
        calibration["corrected_target"],
        lab["bright_model"],
        calibration["corrected_weight"],
    )
    n_antennas = len(lab["antenna_xy_m"])
    ratio_matrix = _row_to_matrix(ratio, lab["pairs"], n_antennas)
    weight_matrix = _row_to_matrix(ratio_weight, lab["pairs"], n_antennas)
    unit_model = np.ones_like(ratio_matrix)
    diagonal = np.arange(n_antennas)
    unit_model[..., diagonal, diagonal] = 0
    training, holdout = _baseline_masks(lab["pairs"], n_antennas)
    solution_support = np.sqrt((weight_matrix * training).sum(axis=-1))
    solver = _calibration_solver()
    solved = solver.solve_gains(
        ratio_matrix, unit_model, weights=weight_matrix * training
    )
    phase_gains = np.exp(1j * np.angle(solved))
    response = (
        phase_gains[:, lab["pairs"][:, 0]] * phase_gains[:, lab["pairs"][:, 1]].conj()
    )[:, None, :]
    corrected = calibration["corrected_target"] / response
    row_holdout = holdout[lab["pairs"][:, 0], lab["pairs"][:, 1]].astype(bool)
    selected_weight = calibration["corrected_weight"][:, :, row_holdout]
    before = _weighted_rms(
        calibration["corrected_target"][:, :, row_holdout],
        lab["target_model"][:, :, row_holdout],
        selected_weight,
    )
    after = _weighted_rms(
        corrected[:, :, row_holdout],
        lab["target_model"][:, :, row_holdout],
        selected_weight,
    )
    return {
        "lab": lab,
        "phase_gains": phase_gains,
        "corrected_target": corrected,
        "corrected_weight": calibration["corrected_weight"],
        "holdout_rms_before": before,
        "holdout_rms_after": after,
        "model_flux_fraction": float(lab["flux_jy"][:2].sum() / lab["flux_jy"].sum()),
        "minimum_antenna_solution_snr": float(solution_support.min()),
        "accepted": bool(after < before),
        "holdout_reference": "known full input sky; controlled simulation only",
    }


def selfcal_summary(result=None):
    result = self_calibrate() if result is None else result
    return {
        "mode": "phase-only",
        "model_flux_fraction": result["model_flux_fraction"],
        "minimum_antenna_solution_snr": result["minimum_antenna_solution_snr"],
        "gain_amplitude_deviation": float(np.max(abs(abs(result["phase_gains"]) - 1))),
        "holdout_rms_before": result["holdout_rms_before"],
        "holdout_rms_after": result["holdout_rms_after"],
        "holdout_reference": result["holdout_reference"],
        "decision": "accept" if result["accepted"] else "rollback",
        "absolute_flux_preservation_proved": False,
    }


def measurement_summary(lab=None):
    """Measure the central source in a self-calibrated restored image."""
    selfcal = self_calibrate(lab)
    bundle = make_imaging_bundle(
        selfcal["lab"], selfcal["corrected_target"], selfcal["corrected_weight"]
    )
    yy, xx = np.indices(bundle["restored"].shape)
    center = np.array(bundle["restored"].shape) // 2
    radius = np.hypot(yy - center[0], xx - center[1])
    aperture = radius <= 6
    background = (radius >= 10) & (radius <= 14)
    background_level = float(np.median(bundle["restored"][background]))
    flux = float(
        np.sum(bundle["restored"][aperture] - background_level)
        / bundle["beam_area_pixels"]
    )
    scaled_residual = bundle["residual"] * bundle["residual_scale"]
    residual_rms = float(
        1.4826
        * np.median(
            abs(scaled_residual[background] - np.median(scaled_residual[background]))
        )
    )
    independent_beams = float(aperture.sum() / bundle["beam_area_pixels"])
    random_uncertainty = residual_rms * np.sqrt(independent_beams)
    return {
        "product": "self-calibrated restored image",
        "units": "Jy/beam",
        "aperture_pixels": int(aperture.sum()),
        "beam_area_pixels": bundle["beam_area_pixels"],
        "background_jy_per_beam": background_level,
        "integrated_flux_jy": flux,
        "random_uncertainty_jy": float(random_uncertainty),
        "input_central_source_flux_jy": float(selfcal["lab"]["flux_jy"][0]),
        "truth_available_only_because_controlled_simulation": True,
    }


def averaging_summary(lab=None, time_bin=5, channel_bin=4):
    """Compare direct time/frequency coherence loss with a separable estimate."""
    lab = make_lab() if lab is None else lab
    middle = len(lab["time_s"]) // 2
    time_indices = np.arange(middle - time_bin // 2, middle + time_bin // 2 + 1)
    if len(time_indices) != time_bin or not 1 <= channel_bin <= len(
        lab["frequency_hz"]
    ):
        raise ValueError("time_bin must be odd and channel_bin must fit the sample")
    channel_start = (len(lab["frequency_hz"]) - channel_bin) // 2
    channel_indices = np.arange(channel_start, channel_start + channel_bin)
    projected = np.linalg.norm(lab["uvw_m"][middle, :, :2], axis=1)
    baseline = int(np.argmax(projected))
    direction = np.array([0.03, 0.02])
    uv_lambda = (
        lab["uvw_m"][time_indices, None, baseline, :2]
        * lab["frequency_hz"][None, channel_indices, None]
        / C
    )
    phasor = np.exp(-2j * np.pi * np.einsum("tcd,d->tc", uv_lambda, direction))
    window_weight = lab["weight"][
        time_indices[:, None], channel_indices[None, :], baseline
    ]

    def retention(values, weights):
        weight_sum = weights.sum()
        if weight_sum <= 0:
            raise ValueError("averaging window contains no positive-weight samples")
        return float(abs(np.sum(weights * values) / weight_sum))

    frequency_retention = retention(phasor[time_bin // 2], window_weight[time_bin // 2])
    time_retention = retention(
        phasor[:, channel_bin // 2], window_weight[:, channel_bin // 2]
    )
    combined_retention = retention(phasor, window_weight)
    return {
        "integration_s": float(np.median(np.diff(lab["time_s"]))),
        "channel_width_hz": float(lab["channel_width_hz"]),
        "time_bin": time_bin,
        "channel_bin": channel_bin,
        "direction_radius_deg": float(np.rad2deg(np.linalg.norm(direction))),
        "projected_baseline_m": float(projected[baseline]),
        "frequency_retention": frequency_retention,
        "time_retention": time_retention,
        "separable_product": frequency_retention * time_retention,
        "direct_combined_retention": combined_retention,
        "input_samples": int(window_weight.size),
        "effective_unflagged_samples": int(np.count_nonzero(window_weight)),
        "output_inverse_variance": float(window_weight.sum()),
        "flag_weight_rule": "exclude zero-weight samples; sum inverse variances",
        "coherence_is_recoverable_after_averaging": False,
    }
