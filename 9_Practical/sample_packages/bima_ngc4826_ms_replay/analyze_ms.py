"""Offline QA and scalar calibration for the bundled BIMA MS extracts."""

import argparse
import hashlib
import importlib.util
import tarfile
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifests" / "data_manifest.yaml"
ACTIVE_CALIBRATOR_ANTENNAS = np.array([0, 1, 2, 3, 4, 6, 7])


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


def archive_subtables():
    archive = ROOT / "data" / "mssel_test_small_multifield_spw.ms.tgz"
    prefix = "mssel_test_small_multifield_spw.ms/"
    with tarfile.open(archive, "r:gz") as stream:
        names = [member.name for member in stream.getmembers()]
    if any(name.startswith("/") or ".." in Path(name).parts for name in names):
        raise ValueError("unsafe archive member")
    subtables = {
        Path(name.removeprefix(prefix)).parts[0]
        for name in names
        if name.startswith(prefix) and len(Path(name.removeprefix(prefix)).parts) > 1
    }
    required = {"ANTENNA", "DATA_DESCRIPTION", "FIELD", "POLARIZATION", "SPECTRAL_WINDOW"}
    if not required <= subtables:
        raise ValueError("Measurement Set is missing required subtables")
    return subtables


def _load_extract(name):
    with np.load(ROOT / "derived" / name) as archive:
        return {key: archive[key] for key in archive.files}


def _time_range(values):
    return float(values.min()), float(values.max())


def visibility_summary():
    verify_checksums()
    subtables = archive_subtables()
    calibrator = _load_extract("calibrator_3c273_ddid0.npz")
    target = _load_extract("target_ngc4826_field2_ddid2.npz")
    frequency = np.load(ROOT / "derived" / "target_ddid2_channel_frequency_hz.npy")
    return {
        "archive_subtables": len(subtables),
        "calibrator_rows": len(calibrator["time_s"]),
        "calibrator_times": len(np.unique(calibrator["time_s"])),
        "calibrator_flag_fraction": float(np.mean(calibrator["flag"])),
        "calibrator_median_amplitude": float(np.median(np.abs(calibrator["data"][~calibrator["flag"]]))),
        "target_rows": len(target["time_s"]),
        "target_channels": target["data"].shape[1],
        "target_flag_fraction": float(np.mean(target["flag"])),
        "target_frequency_hz": (float(frequency.min()), float(frequency.max())),
        "target_uv_range_m": (
            float(np.linalg.norm(target["uvw_m"], axis=1).min()),
            float(np.linalg.norm(target["uvw_m"], axis=1).max()),
        ),
    }


def metadata_summary():
    path = ROOT / "derived" / "ms_metadata.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def coverage_summary():
    coverage = _load_extract("target_ngc4826_mosaic_coverage.npz")
    calibrator = _load_extract("calibrator_3c273_ddid0.npz")
    fields, field_counts = np.unique(coverage["field_id"], return_counts=True)
    data_descriptions = np.unique(coverage["data_desc_id"])
    uv_distance = np.linalg.norm(coverage["uvw_m"], axis=1)
    return {
        "target_rows": len(coverage["time_s"]),
        "target_fields": fields.tolist(),
        "field_row_counts": {
            int(field): int(count) for field, count in zip(fields, field_counts)
        },
        "data_descriptions": data_descriptions.tolist(),
        "time_range_s": _time_range(coverage["time_s"]),
        "calibrator_target_gap_s": float(
            coverage["time_s"].min() - calibrator["time_s"].max()
        ),
        "uv_range_m": (float(uv_distance.min()), float(uv_distance.max())),
        "mean_row_flag_fraction": float(np.mean(coverage["flag_fraction"])),
    }


def calibrator_matrices():
    values = _load_extract("calibrator_3c273_ddid0.npz")
    active = ACTIVE_CALIBRATOR_ANTENNAS
    selected = np.isin(values["antenna1"], active) & np.isin(values["antenna2"], active)
    times = np.unique(values["time_s"])
    remap = {antenna: index for index, antenna in enumerate(active)}
    shape = (len(times), len(active), len(active))
    data = np.zeros(shape, complex)
    weights = np.zeros(shape, float)

    for index, time in enumerate(times):
        rows = selected & (values["time_s"] == time)
        antenna1 = np.array([remap[value] for value in values["antenna1"][rows]])
        antenna2 = np.array([remap[value] for value in values["antenna2"][rows]])
        visibility = values["data"][rows, 0]
        weight = values["weight"][rows] * ~values["flag"][rows, 0]
        data[index, antenna1, antenna2] = visibility
        data[index, antenna2, antenna1] = visibility.conj()
        weights[index, antenna1, antenna2] = weight
        weights[index, antenna2, antenna1] = weight
    return data, weights


def _load_solver():
    solver_path = ROOT.parents[2] / "8_Calibration" / "calibration_solver.py"
    spec = importlib.util.spec_from_file_location("calibration_solver", solver_path)
    solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)
    return solver


def _point_model(shape):
    model = np.ones(shape, complex)
    diagonal = np.arange(shape[-1])
    model[..., diagonal, diagonal] = 0.0
    return model


def _baseline_split(n_antennas):
    antenna1, antenna2 = np.triu_indices(n_antennas, 1)
    held_out = (antenna1 + antenna2) % 3 == 0
    holdout_mask = np.zeros((n_antennas, n_antennas), bool)
    holdout_mask[antenna1[held_out], antenna2[held_out]] = True
    holdout_mask |= holdout_mask.T
    training_mask = ~holdout_mask
    np.fill_diagonal(training_mask, False)
    pairs = np.column_stack((antenna1, antenna2))
    return training_mask, holdout_mask, pairs[~held_out], pairs[held_out]


def _solve_interval_gains(data, weights, model, baseline_mask, interval_samples):
    if not isinstance(interval_samples, (int, np.integer)) or interval_samples < 1:
        raise ValueError("interval_samples must be a positive integer")

    solver = _load_solver()
    gains = np.empty(data.shape[:-1], complex)
    for start in range(0, data.shape[0], interval_samples):
        stop = min(start + interval_samples, data.shape[0])
        block_weights = weights[start:stop] * baseline_mask
        combined_weights = block_weights.sum(axis=0)
        combined_data = np.divide(
            (data[start:stop] * block_weights).sum(axis=0),
            combined_weights,
            out=np.zeros(data.shape[-2:], complex),
            where=combined_weights > 0,
        )
        gains[start:stop] = solver.solve_gains(
            combined_data, model[0], weights=combined_weights
        )
    return gains


def _rms_by_time(data, model, weights):
    antenna1, antenna2 = np.triu_indices(data.shape[-1], 1)
    pair_weights = weights[:, antenna1, antenna2]
    residual_squared = abs(
        data[:, antenna1, antenna2] - model[:, antenna1, antenna2]
    ) ** 2
    total_weight = pair_weights.sum(axis=1)
    if np.any(total_weight <= 0):
        raise ValueError("every solution must contain a positive-weight baseline")
    return np.sqrt((pair_weights * residual_squared).sum(axis=1) / total_weight)


def calibration_interval_summary(interval_samples=(1, 5, 65)):
    data, weights = calibrator_matrices()
    model = _point_model(data.shape)
    training_mask, holdout_mask, _, _ = _baseline_split(data.shape[-1])
    solver = _load_solver()
    safe_data = np.where(weights > 0, data, 1.0)
    summaries = []
    for samples in interval_samples:
        gains = _solve_interval_gains(
            data, weights, model, training_mask, samples
        )
        corrected = solver.correct_visibilities(safe_data, gains)
        summaries.append(
            {
                "interval_samples": int(samples),
                "solutions": int(np.ceil(data.shape[0] / samples)),
                "training_rms": solver.rms_residual(
                    corrected, model, weights * training_mask
                ),
                "holdout_rms": solver.rms_residual(
                    corrected, model, weights * holdout_mask
                ),
            }
        )
    return summaries


def calibration_table():
    values = _load_extract("calibrator_3c273_ddid0.npz")
    times = np.unique(values["time_s"])
    integration_s = np.array(
        [np.median(values["interval_s"][values["time_s"] == time]) for time in times]
    )
    data, weights = calibrator_matrices()
    model = _point_model(data.shape)
    training_mask, holdout_mask, training_pairs, holdout_pairs = _baseline_split(
        data.shape[-1]
    )
    solver = _load_solver()
    full_mask = ~np.eye(data.shape[-1], dtype=bool)
    gains = _solve_interval_gains(data, weights, model, full_mask, 1)
    validation_gains = _solve_interval_gains(
        data, weights, model, training_mask, 1
    )
    safe_data = np.where(weights > 0, data, 1.0)
    corrected = solver.correct_visibilities(safe_data, gains)
    validation_corrected = solver.correct_visibilities(safe_data, validation_gains)

    return {
        "schema_version": np.array(1),
        "time_s": times,
        "interval_s": integration_s,
        "antenna_id": ACTIVE_CALIBRATOR_ANTENNAS.copy(),
        "reference_antenna_id": np.array(ACTIVE_CALIBRATOR_ANTENNAS[0]),
        "model_flux_native": np.array(1.0),
        "gain": gains,
        "input_weight": weights.sum(axis=-1),
        "gain_flag": weights.sum(axis=-1) <= 0,
        "training_baseline_pairs": ACTIVE_CALIBRATOR_ANTENNAS[training_pairs],
        "holdout_baseline_pairs": ACTIVE_CALIBRATOR_ANTENNAS[holdout_pairs],
        "full_fit_rms": _rms_by_time(corrected, model, weights),
        "training_rms": _rms_by_time(
            validation_corrected, model, weights * training_mask
        ),
        "holdout_rms": _rms_by_time(
            validation_corrected, model, weights * holdout_mask
        ),
    }


def write_calibration_table(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **calibration_table())
    return path


def calibration_summary():
    solver = _load_solver()

    data, weights = calibrator_matrices()
    model = _point_model(data.shape)
    gains = solver.solve_gains(data, model, weights=weights)
    corrected = solver.correct_visibilities(np.where(weights > 0, data, 1.0), gains)
    before = solver.rms_residual(data, model, weights)
    after = solver.rms_residual(corrected, model, weights)
    return {
        "times": data.shape[0],
        "active_antennas": ACTIVE_CALIBRATOR_ANTENNAS.tolist(),
        "gain_amplitude_range": (float(abs(gains).min()), float(abs(gains).max())),
        "normalized_rms_before": before,
        "normalized_rms_after": after,
        "rms_ratio": after / before,
        "gains": gains,
        "corrected": corrected,
        "weights": weights,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-gain-table", type=Path)
    args = parser.parse_args()
    if args.write_gain_table:
        print(f"gain table: {write_calibration_table(args.write_gain_table)}")
        return

    summaries = (
        ("visibility", visibility_summary()),
        ("metadata evidence", metadata_summary()),
        ("mosaic coverage", coverage_summary()),
        ("relative calibration", calibration_summary()),
        ("solution intervals", calibration_interval_summary()),
    )
    for title, summary in summaries:
        print(f"[{title}]")
        if isinstance(summary, list):
            for record in summary:
                print(record)
        else:
            for key, value in summary.items():
                if not isinstance(value, np.ndarray):
                    print(f"{key}: {value}")


if __name__ == "__main__":
    main()
