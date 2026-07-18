"""Offline QA and scalar calibration for the bundled BIMA MS extracts."""

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


def calibration_summary():
    solver_path = ROOT.parents[2] / "8_Calibration" / "calibration_solver.py"
    spec = importlib.util.spec_from_file_location("calibration_solver", solver_path)
    solver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solver)

    data, weights = calibrator_matrices()
    model = np.ones_like(data)
    model[..., np.arange(model.shape[-1]), np.arange(model.shape[-1])] = 0.0
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


if __name__ == "__main__":
    summaries = (
        ("visibility", visibility_summary()),
        ("mosaic coverage", coverage_summary()),
        ("relative calibration", calibration_summary()),
    )
    for title, summary in summaries:
        print(f"[{title}]")
        for key, value in summary.items():
            if not isinstance(value, np.ndarray):
                print(f"{key}: {value}")
