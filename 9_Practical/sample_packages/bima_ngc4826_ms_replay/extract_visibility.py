"""Extract compact NumPy views from the bundled BIMA Measurement Set."""

import argparse
import tarfile
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "data" / "mssel_test_small_multifield_spw.ms.tgz"


def _safe_extract(archive, destination):
    with tarfile.open(archive, "r:gz") as stream:
        destination = destination.resolve()
        for member in stream.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"unsafe archive member: {member.name}")
        stream.extractall(destination)


def _columns(table):
    data = table.getcol("DATA")
    flag = table.getcol("FLAG")
    return {
        "data": data[..., 0],
        "flag": flag[..., 0],
        "uvw_m": table.getcol("UVW"),
        "weight": table.getcol("WEIGHT")[..., 0],
        "antenna1": table.getcol("ANTENNA1"),
        "antenna2": table.getcol("ANTENNA2"),
        "time_s": table.getcol("TIME"),
        "interval_s": table.getcol("INTERVAL"),
        "field_id": table.getcol("FIELD_ID"),
        "data_desc_id": table.getcol("DATA_DESC_ID"),
    }


def _coverage_columns(table):
    flag = table.getcol("FLAG")
    return {
        "uvw_m": table.getcol("UVW"),
        "antenna1": table.getcol("ANTENNA1"),
        "antenna2": table.getcol("ANTENNA2"),
        "time_s": table.getcol("TIME"),
        "field_id": table.getcol("FIELD_ID"),
        "data_desc_id": table.getcol("DATA_DESC_ID"),
        "flag_fraction": flag.mean(axis=(1, 2)),
    }


def extract(output_dir):
    try:
        from casacore.tables import table
    except ImportError as error:
        raise RuntimeError("python-casacore is required to regenerate extracts") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bima-ms-") as temp:
        temp_path = Path(temp)
        _safe_extract(ARCHIVE, temp_path)
        ms_path = temp_path / "mssel_test_small_multifield_spw.ms"

        with table(str(ms_path), ack=False) as measurement_set:
            calibrator = measurement_set.query("FIELD_ID==0 && DATA_DESC_ID==0")
            target = measurement_set.query("FIELD_ID==2 && DATA_DESC_ID==2")
            try:
                np.savez_compressed(output_dir / "calibrator_3c273_ddid0.npz", **_columns(calibrator))
                np.savez_compressed(output_dir / "target_ngc4826_field2_ddid2.npz", **_columns(target))
            finally:
                calibrator.close()
                target.close()
            target_mosaic = measurement_set.query("FIELD_ID>=2 && DATA_DESC_ID>=2")
            try:
                np.savez_compressed(
                    output_dir / "target_ngc4826_mosaic_coverage.npz",
                    **_coverage_columns(target_mosaic),
                )
            finally:
                target_mosaic.close()

        with table(str(ms_path / "SPECTRAL_WINDOW"), ack=False) as spectral_window:
            channel_frequency_hz = spectral_window.getcell("CHAN_FREQ", 2)
        np.save(output_dir / "target_ddid2_channel_frequency_hz.npy", channel_frequency_hz)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(tempfile.gettempdir()) / "bima-ngc4826-extract",
    )
    args = parser.parse_args()
    extract(args.output)


if __name__ == "__main__":
    main()
