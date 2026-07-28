#!/usr/bin/env python3
"""Run the pinned 3C391 calibration and reference imaging workflow in CASA."""

import argparse
import importlib.metadata
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

INPUT_NAME = "3c391_ctm_mosaic_10s_spw0.ms"
SCIENCE_NAME = "3c391_ctm_mosaic_spw0.ms"
DATASET_ID = "vla-3c391-casa-tutorial-spw0-v1"
SOURCE_ARCHIVE_SHA256 = (
    "9152b1ce8603a3b0ffd50ce3d3c57f53a82e6fde89fd6a4b5fa4f8c7d8481910"
)
FLUX_CALIBRATOR = "J1331+3030"
GAIN_CALIBRATOR = "J1822-0938"
REFERENCE_ANTENNA = "ea21"
ANTENNA_POSITION_OFFSETS_M = {
    "ea01": [0.0, 0.0030, 0.0],
    "ea02": [-0.0008, 0.0, 0.0],
    "ea03": [-0.0028, 0.0, 0.0],
    "ea05": [0.0, 0.0028, 0.0],
    "ea11": [0.0009, 0.0, 0.0],
    "ea12": [-0.0100, 0.0045, -0.0017],
    "ea13": [0.0, -0.0008, 0.0],
    "ea17": [-0.0012, 0.0, 0.0],
    "ea18": [0.0004, -0.0008, 0.0004],
    "ea22": [-0.0257, 0.0027, -0.0190],
    "ea23": [-0.0014, 0.0, 0.0],
    "ea24": [-0.0015, 0.0, 0.0],
    "ea26": [-0.0019, 0.0, 0.0021],
    "ea27": [0.0, 0.0019, -0.0016],
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


def pipeline_parameters():
    return {
        "reference_antenna": REFERENCE_ANTENNA,
        "initial_delay_channels": "0:5~58",
        "initial_phase_channels": "0:27~36",
        "target_fields": "2~8",
        "target_correlations": "RR,LL",
        "imaging": {
            "gridder": "mosaic",
            "imsize": [480, 480],
            "cell": ["2.5arcsec", "2.5arcsec"],
            "weighting": "briggs",
            "robust": 0.5,
            "scales": [0, 5, 15, 45],
            "threshold": "1.0mJy",
        },
    }


def load_casa():
    try:
        import casatasks
        from casatools import image, table
    except ImportError as error:
        raise RuntimeError(
            "CASA is required only for this optional pipeline; configure a "
            "separate CASA 6.7 environment before running it"
        ) from error
    task_names = (
        "applycal",
        "bandpass",
        "exportfits",
        "flagdata",
        "fluxscale",
        "gaincal",
        "gencal",
        "imstat",
        "setjy",
        "split",
        "statwt",
        "tclean",
    )
    tasks = {name: getattr(casatasks, name) for name in task_names}
    return {**tasks, "image": image, "table": table}


def package_versions():
    versions = {}
    for name in ("casatasks", "casatools", "casadata"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed-as-package"
    return versions


def delay_summary(table_factory, path):
    table = table_factory()
    table.open(str(path))
    try:
        delay = np.asarray(table.getcol("FPARAM"))
        flag = np.asarray(table.getcol("FLAG"), dtype=bool)
    finally:
        table.close()
    valid = delay[~flag]
    return {
        "valid_solutions": int(valid.size),
        "minimum_ns": float(valid.min()),
        "maximum_ns": float(valid.max()),
        "median_ns": float(np.median(valid)),
    }


def run_calibration(casa, root, measurement_set):
    flagdata = casa["flagdata"]
    gaincal = casa["gaincal"]
    prefix = root / INPUT_NAME.removesuffix(".ms")
    summary = {
        "flag_before": plain(flagdata(vis=str(measurement_set), mode="summary")),
        "antenna_position_offsets_m": ANTENNA_POSITION_OFFSETS_M,
        "parameters": pipeline_parameters(),
    }

    flagdata(vis=str(measurement_set), mode="manual", scan="1", flagbackup=False)
    flagdata(
        vis=str(measurement_set),
        mode="manual",
        antenna="ea13,ea15",
        flagbackup=False,
    )
    flagdata(
        vis=str(measurement_set),
        mode="quack",
        quackinterval=10.0,
        quackmode="beg",
        flagbackup=False,
    )

    antpos = Path(f"{prefix}.antpos")
    casa["gencal"](
        vis=str(measurement_set),
        caltable=str(antpos),
        caltype="antpos",
        antenna=",".join(ANTENNA_POSITION_OFFSETS_M),
        parameter=[
            value
            for offsets in ANTENNA_POSITION_OFFSETS_M.values()
            for value in offsets
        ],
    )
    summary["setjy"] = plain(
        casa["setjy"](
            vis=str(measurement_set),
            field=FLUX_CALIBRATOR,
            standard="Perley-Butler 2017",
            model="3C286_C.im",
            usescratch=True,
            scalebychan=True,
        )
    )

    g0all = Path(f"{prefix}.G0all")
    gaincal(
        vis=str(measurement_set),
        caltable=str(g0all),
        field=f"{FLUX_CALIBRATOR},{GAIN_CALIBRATOR}",
        refant=REFERENCE_ANTENNA,
        spw="0:27~36",
        gaintype="G",
        calmode="p",
        solint="int",
        minsnr=5,
        gaintable=[str(antpos)],
    )
    flagdata(
        vis=str(measurement_set),
        mode="manual",
        antenna="ea05",
        flagbackup=False,
    )

    g0 = Path(f"{prefix}.G0")
    gaincal(
        vis=str(measurement_set),
        caltable=str(g0),
        field=FLUX_CALIBRATOR,
        refant=REFERENCE_ANTENNA,
        spw="0:27~36",
        calmode="p",
        solint="int",
        minsnr=5,
        gaintable=[str(antpos)],
    )
    k0 = Path(f"{prefix}.K0")
    gaincal(
        vis=str(measurement_set),
        caltable=str(k0),
        field=FLUX_CALIBRATOR,
        refant=REFERENCE_ANTENNA,
        spw="0:5~58",
        gaintype="K",
        solint="inf",
        combine="scan",
        minsnr=5,
        gaintable=[str(antpos), str(g0)],
    )
    summary["delay_solution_ns"] = delay_summary(casa["table"], k0)

    b0 = Path(f"{prefix}.B0")
    casa["bandpass"](
        vis=str(measurement_set),
        caltable=str(b0),
        field=FLUX_CALIBRATOR,
        refant=REFERENCE_ANTENNA,
        combine="scan",
        solint="inf",
        bandtype="B",
        gaintable=[str(antpos), str(g0), str(k0)],
    )
    g1 = Path(f"{prefix}.G1")
    common_gain = {
        "vis": str(measurement_set),
        "caltable": str(g1),
        "spw": "0:5~58",
        "solint": "inf",
        "refant": REFERENCE_ANTENNA,
        "gaintype": "G",
        "calmode": "ap",
        "gaintable": [str(antpos), str(k0), str(b0)],
    }
    gaincal(
        field=FLUX_CALIBRATOR,
        solnorm=False,
        interp=["", "", "nearest"],
        **common_gain,
    )
    gaincal(field=GAIN_CALIBRATOR, append=True, **common_gain)

    flux = Path(f"{prefix}.fluxscale1")
    summary["fluxscale"] = plain(
        casa["fluxscale"](
            vis=str(measurement_set),
            caltable=str(g1),
            fluxtable=str(flux),
            reference=FLUX_CALIBRATOR,
            transfer=[GAIN_CALIBRATOR],
            incremental=False,
        )
    )
    tables = [str(antpos), str(flux), str(k0), str(b0)]
    for field, gain_field, interpolation in (
        (FLUX_CALIBRATOR, FLUX_CALIBRATOR, "nearest"),
        (GAIN_CALIBRATOR, GAIN_CALIBRATOR, "nearest"),
        ("2~8", GAIN_CALIBRATOR, "linear"),
    ):
        casa["applycal"](
            vis=str(measurement_set),
            field=field,
            gaintable=tables,
            gainfield=["", gain_field, "", ""],
            interp=["", interpolation, "", ""],
            calwt=False,
        )
    summary["flag_after_applycal"] = plain(
        flagdata(vis=str(measurement_set), mode="summary")
    )

    science = root / SCIENCE_NAME
    casa["split"](
        vis=str(measurement_set),
        outputvis=str(science),
        datacolumn="corrected",
        field="2~8",
        correlation="RR,LL",
    )
    summary["statwt"] = plain(casa["statwt"](vis=str(science), datacolumn="data"))
    summary["science_flag"] = plain(flagdata(vis=str(science), mode="summary"))
    (root / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return science


def run_imaging(casa, root, science, export_fits=False):
    dirty = root / "3c391_dirty"
    clean = root / "3c391_multiscale"
    common = {
        "vis": str(science),
        "specmode": "mfs",
        "gridder": "mosaic",
        "imsize": [480, 480],
        "cell": ["2.5arcsec", "2.5arcsec"],
        "stokes": "I",
        "weighting": "briggs",
        "robust": 0.5,
        "pbcor": False,
        "interactive": False,
        "parallel": False,
    }
    casa["tclean"](imagename=str(dirty), niter=0, **common)
    casa["tclean"](
        imagename=str(clean),
        niter=20_000,
        gain=0.1,
        threshold="1.0mJy",
        deconvolver="multiscale",
        scales=[0, 5, 15, 45],
        smallscalebias=0.9,
        usemask="user",
        mask="circle[[240pix,240pix],130pix]",
        savemodel="none",
        **common,
    )

    summary = {"parameters": pipeline_parameters()["imaging"]}
    for label, prefix in (("dirty", dirty), ("clean", clean)):
        summary[label] = {}
        for suffix in ("image", "residual", "psf", "pb", "model"):
            image_path = Path(f"{prefix}.{suffix}")
            if not image_path.exists():
                continue
            summary[label][suffix] = plain(casa["imstat"](imagename=str(image_path)))
            if (
                export_fits
                and label == "clean"
                and suffix
                in {
                    "image",
                    "residual",
                    "psf",
                    "pb",
                }
            ):
                casa["exportfits"](
                    imagename=str(image_path),
                    fitsimage=str(root / f"{prefix.name}.{suffix}.fits"),
                    overwrite=True,
                )
    image_tool = casa["image"]()
    image_tool.open(str(Path(f"{clean}.image")))
    try:
        summary["restoring_beam"] = plain(image_tool.restoringbeam())
    finally:
        image_tool.close()
    summary["interpretation"] = {
        "status": "reproducible_course_baseline",
        "known_failure_mode": (
            "A broad fixed mask can admit negative or divergent large-scale "
            "minor cycles; inspect correlated residual structure manually"
        ),
        "publication_claim_supported": False,
    }
    (root / "imaging_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ms", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--stop-after", choices=("calibration", "imaging"), default="imaging"
    )
    parser.add_argument("--export-fits", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    input_ms = args.input_ms.resolve()
    root = args.work_dir.resolve()
    if not input_ms.is_dir():
        raise SystemExit(f"Measurement Set not found: {input_ms}")
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"work directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    casa = load_casa()
    working_ms = root / INPUT_NAME
    shutil.copytree(input_ms, working_ms)
    science = run_calibration(casa, root, working_ms)
    stages = ["calibration"]
    if args.stop_after == "imaging":
        run_imaging(casa, root, science, export_fits=args.export_fits)
        stages.append("imaging")
    metadata = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_measurement_set": str(input_ms),
        "working_measurement_set": str(working_ms),
        "stages": stages,
        "environment": package_versions(),
        "parameters": pipeline_parameters(),
    }
    (root / "pipeline_run.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(f"Completed {', '.join(stages)} in {root}")


if __name__ == "__main__":
    main()
