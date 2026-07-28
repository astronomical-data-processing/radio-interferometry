#!/usr/bin/env python3
"""Audit 3C391 CASA outputs against the validated course baseline."""

import argparse
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
REFERENCE_PATH = ROOT / "manifests" / "reference_metrics.yaml"


def load_reference(path=REFERENCE_PATH):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def perley_butler_2017_3c286(frequency_hz, coefficients=None):
    if coefficients is None:
        coefficients = load_reference()["flux_model"]["log10_polynomial_coefficients"]
    x = math.log10(float(frequency_hz) / 1e9)
    log_flux = sum(
        coefficient * x**order for order, coefficient in enumerate(coefficients)
    )
    return 10**log_flux


def _scalar(value):
    while isinstance(value, list):
        value = value[0]
    return float(value)


def _check(name, measured, expected, passed):
    return {
        "name": name,
        "measured": measured,
        "expected": expected,
        "passed": bool(passed),
    }


def _close_check(name, measured, specification):
    expected = float(specification["value"])
    atol = float(specification["atol"])
    return _check(name, measured, specification, abs(measured - expected) <= atol)


def audit_calibration(summary, reference):
    specification = reference["calibration"]
    setjy = _scalar(summary["setjy"]["0"]["0"]["fluxd"])
    fluxscale = summary["fluxscale"]["1"]["0"]
    science_flag = summary["science_flag"]
    flag_fraction = science_flag["flagged"] / science_flag["total"]
    parameters = summary["parameters"]
    checks = [
        _check(
            "calibration parameters",
            {
                "reference_antenna": parameters["reference_antenna"],
                "target_fields": parameters["target_fields"],
                "target_correlations": parameters["target_correlations"],
            },
            specification["parameters"],
            all(
                parameters[key] == expected
                for key, expected in specification["parameters"].items()
            ),
        ),
        _close_check("3C286 setjy flux", setjy, specification["setjy_3c286_flux_jy"]),
        _close_check(
            "J1822-0938 flux",
            _scalar(fluxscale["fluxd"]),
            specification["gain_calibrator_flux_jy"],
        ),
        _close_check(
            "J1822-0938 flux error",
            _scalar(fluxscale["fluxdErr"]),
            specification["gain_calibrator_flux_error_jy"],
        ),
        _check(
            "valid fluxscale solutions",
            int(_scalar(fluxscale["numSol"])),
            specification["valid_fluxscale_solutions"],
            int(_scalar(fluxscale["numSol"]))
            == specification["valid_fluxscale_solutions"],
        ),
        _close_check(
            "science flag fraction",
            flag_fraction,
            specification["science_flag_fraction"],
        ),
    ]
    lower, upper = specification["target_field_flag_fraction_range"]
    field_fractions = {
        field: record["flagged"] / record["total"]
        for field, record in science_flag["field"].items()
    }
    checks.append(
        _check(
            "target fields and flag fractions",
            field_fractions,
            {"fields": specification["target_fields"], "range": [lower, upper]},
            set(field_fractions) == set(specification["target_fields"])
            and all(lower <= value <= upper for value in field_fractions.values()),
        )
    )
    fully_flagged = sorted(
        antenna
        for antenna, record in science_flag["antenna"].items()
        if record["flagged"] == record["total"]
    )
    expected_antennas = sorted(specification["fully_flagged_antennas"])
    checks.append(
        _check(
            "fully flagged antennas",
            fully_flagged,
            expected_antennas,
            fully_flagged == expected_antennas,
        )
    )
    delay = summary["delay_solution_ns"]
    checks.extend(
        [
            _check(
                "valid delay solutions",
                delay["valid_solutions"],
                specification["delay_solution_ns"]["valid_solutions"],
                delay["valid_solutions"]
                == specification["delay_solution_ns"]["valid_solutions"],
            ),
            _close_check(
                "minimum delay",
                delay["minimum_ns"],
                specification["delay_solution_ns"]["minimum"],
            ),
            _close_check(
                "maximum delay",
                delay["maximum_ns"],
                specification["delay_solution_ns"]["maximum"],
            ),
        ]
    )
    return checks


def audit_imaging(summary, reference):
    specification = reference["imaging"]
    parameters = summary["parameters"]
    beam = summary["restoring_beam"]
    dirty_peak = _scalar(summary["dirty"]["image"]["max"])
    restored_peak = _scalar(summary["clean"]["image"]["max"])
    residual = summary["clean"]["residual"]
    residual_extreme = max(abs(_scalar(residual["min"])), abs(_scalar(residual["max"])))
    return [
        _check(
            "imaging parameters",
            parameters,
            specification["parameters"],
            parameters["gridder"] == specification["parameters"]["gridder"]
            and parameters["imsize"] == specification["parameters"]["imsize"]
            and parameters["cell"]
            == [
                f"{specification['parameters']['cell_arcsec']}arcsec",
                f"{specification['parameters']['cell_arcsec']}arcsec",
            ]
            and parameters["weighting"] == specification["parameters"]["weighting"]
            and parameters["robust"] == specification["parameters"]["robust"]
            and parameters["scales"] == specification["parameters"]["scales_pixels"]
            and parameters["threshold"]
            == f"{specification['parameters']['threshold_mjy_per_beam']}mJy",
        ),
        _close_check(
            "beam major axis",
            beam["major"]["value"],
            specification["restoring_beam"]["major_arcsec"],
        ),
        _close_check(
            "beam minor axis",
            beam["minor"]["value"],
            specification["restoring_beam"]["minor_arcsec"],
        ),
        _close_check(
            "beam position angle",
            beam["positionangle"]["value"],
            specification["restoring_beam"]["position_angle_deg"],
        ),
        _close_check("dirty peak", dirty_peak, specification["dirty_peak_jy_per_beam"]),
        _close_check(
            "restored peak", restored_peak, specification["restored_peak_jy_per_beam"]
        ),
        _close_check(
            "residual RMS",
            _scalar(residual["rms"]),
            specification["residual_rms_jy_per_beam"],
        ),
        _check(
            "residual absolute extreme",
            residual_extreme,
            {"maximum": specification["residual_absolute_extreme_max_jy_per_beam"]},
            residual_extreme
            <= specification["residual_absolute_extreme_max_jy_per_beam"],
        ),
    ]


def audit_results(results_dir, require_imaging=False):
    results_dir = Path(results_dir)
    reference = load_reference()
    calibration = json.loads(
        (results_dir / "calibration_summary.json").read_text(encoding="ascii")
    )
    checks = audit_calibration(calibration, reference)
    imaging_path = results_dir / "imaging_summary.json"
    if imaging_path.exists():
        imaging = json.loads(imaging_path.read_text(encoding="ascii"))
        checks.extend(audit_imaging(imaging, reference))
        stages = ["calibration", "imaging"]
    elif require_imaging:
        checks.append(_check("imaging summary present", False, True, False))
        stages = ["calibration"]
    else:
        stages = ["calibration"]
    return {
        "schema_version": 1,
        "status": "passed" if all(item["passed"] for item in checks) else "failed",
        "stages": stages,
        "checks": checks,
        "interpretation": reference["interpretation"],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--require-imaging", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    report = audit_results(args.results_dir, require_imaging=args.require_imaging)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
