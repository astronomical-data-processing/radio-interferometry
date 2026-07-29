import hashlib
import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "9_Practical" / "archive_labs" / "vla_3c391"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, LAB / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DOWNLOAD = load_module("download_data")
PIPELINE = load_module("run_casa_pipeline")
AUDIT = load_module("audit_results")
SENSITIVITY = load_module("run_imaging_sensitivity")


class VLA3C391ArchiveLabTests(unittest.TestCase):
    def test_manifest_matches_pinned_download(self):
        manifest = yaml.safe_load(
            (LAB / "manifests" / "data_manifest.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["archive"]["filename"], DOWNLOAD.ARCHIVE_NAME)
        self.assertEqual(manifest["archive"]["bytes"], DOWNLOAD.EXPECTED_BYTES)
        self.assertEqual(manifest["archive"]["sha256"], DOWNLOAD.EXPECTED_SHA256)
        self.assertEqual(manifest["source"]["data_url"], DOWNLOAD.SOURCE_URL)
        self.assertEqual(manifest["sample_id"], PIPELINE.DATASET_ID)
        self.assertEqual(manifest["archive"]["sha256"], PIPELINE.SOURCE_ARCHIVE_SHA256)
        self.assertIn("not bundled", manifest["source"]["redistribution_status"])

    def test_archive_verification(self):
        payload = b"3C391 archive verification fixture"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.tgz"
            path.write_bytes(payload)
            with (
                mock.patch.object(DOWNLOAD, "EXPECTED_BYTES", len(payload)),
                mock.patch.object(
                    DOWNLOAD, "EXPECTED_SHA256", hashlib.sha256(payload).hexdigest()
                ),
            ):
                self.assertEqual(
                    DOWNLOAD.verify_archive(path), DOWNLOAD.EXPECTED_SHA256
                )

    def test_unsafe_archive_member_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                member = tarfile.TarInfo("../outside")
                member.size = 1
                stream.addfile(member, io.BytesIO(b"x"))
            with self.assertRaisesRegex(ValueError, "unsafe archive member"):
                DOWNLOAD.extract_archive(archive, Path(temporary) / "output")

    def test_complete_partial_download_is_promoted_without_network(self):
        payload = b"complete partial archive"
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / DOWNLOAD.ARCHIVE_NAME
            partial = destination.with_name(destination.name + ".part")
            partial.write_bytes(payload)
            with (
                mock.patch.object(DOWNLOAD, "EXPECTED_BYTES", len(payload)),
                mock.patch.object(
                    DOWNLOAD, "EXPECTED_SHA256", hashlib.sha256(payload).hexdigest()
                ),
                mock.patch.object(
                    DOWNLOAD, "urlopen", side_effect=AssertionError("network used")
                ),
            ):
                self.assertEqual(DOWNLOAD.download_archive(destination), destination)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertFalse(partial.exists())

    def test_pipeline_is_importable_without_casa(self):
        parameters = PIPELINE.pipeline_parameters()
        self.assertEqual(parameters["reference_antenna"], "ea21")
        self.assertEqual(parameters["target_fields"], "2~8")
        self.assertEqual(parameters["imaging"]["gridder"], "mosaic")

    def test_sensitivity_study_is_controlled(self):
        common = SENSITIVITY.common_parameters()
        reference = PIPELINE.pipeline_parameters()["imaging"]
        self.assertEqual(common["gridder"], reference["gridder"])
        self.assertEqual(common["imsize"], reference["imsize"])
        self.assertEqual(common["cell"], reference["cell"])
        self.assertEqual(common["weighting"], reference["weighting"])
        self.assertEqual(common["robust"], reference["robust"])
        self.assertEqual(common["scales"], reference["scales"])
        self.assertEqual(common["threshold"], reference["threshold"])
        self.assertEqual(
            set(SENSITIVITY.VARIANTS),
            {
                "fixed_broad",
                "fixed_tight",
                "auto_conservative",
                "fixed_broad_restricted_scales",
            },
        )

    def test_sensitivity_metrics_detect_negative_model_and_correlation(self):
        values = np.arange(25, dtype=float).reshape(5, 5)
        valid = np.ones_like(values, dtype=bool)
        model = SENSITIVITY.model_metrics(values - 12, valid)
        residual = SENSITIVITY.residual_metrics(values, valid, beam_pixels=2)
        self.assertEqual(model["sum_jy"], 0.0)
        self.assertEqual(model["negative_pixels"], 12)
        self.assertAlmostEqual(residual["lag_correlation"]["one_pixel_x"], 1.0)
        self.assertAlmostEqual(residual["lag_correlation"]["one_pixel_y"], 1.0)

    def test_perley_butler_flux_checkpoints(self):
        reference = AUDIT.load_reference()
        for point in reference["flux_model"]["checkpoints"]:
            measured = AUDIT.perley_butler_2017_3c286(point["frequency_ghz"] * 1e9)
            with self.subTest(frequency_ghz=point["frequency_ghz"]):
                self.assertAlmostEqual(measured, point["flux_jy"], places=6)

    def test_reference_calibration_passes_audit(self):
        reference = AUDIT.load_reference()
        fields = {
            f"3C391 C{index}": {"flagged": 34.5, "total": 100.0}
            for index in range(1, 8)
        }
        antennas = {"ea01": {"flagged": 20.0, "total": 100.0}}
        antennas.update(
            {
                name: {"flagged": 100.0, "total": 100.0}
                for name in ("ea05", "ea13", "ea15")
            }
        )
        summary = {
            "setjy": {"0": {"0": {"fluxd": [7.6685524, 0.0, 0.0, 0.0]}}},
            "fluxscale": {
                "1": {
                    "0": {
                        "fluxd": [2.2960073, 0.0, 0.0, 0.0],
                        "fluxdErr": [0.0069219, 0.0, 0.0, 0.0],
                        "numSol": [46.0, 0.0, 0.0, 0.0],
                    }
                }
            },
            "science_flag": {
                "flagged": 34.55225,
                "total": 100.0,
                "field": fields,
                "antenna": antennas,
            },
            "parameters": PIPELINE.pipeline_parameters(),
            "delay_solution_ns": {
                "valid_solutions": 46,
                "minimum_ns": -3.8506012,
                "maximum_ns": 4.5584397,
            },
        }
        checks = AUDIT.audit_calibration(summary, reference)
        self.assertTrue(all(check["passed"] for check in checks))

    def test_reference_imaging_passes_audit(self):
        reference = AUDIT.load_reference()
        summary = {
            "parameters": PIPELINE.pipeline_parameters()["imaging"],
            "restoring_beam": {
                "major": {"value": 17.0065, "unit": "arcsec"},
                "minor": {"value": 14.9104, "unit": "arcsec"},
                "positionangle": {"value": 21.0034, "unit": "deg"},
            },
            "dirty": {"image": {"max": [0.122880]}},
            "clean": {
                "image": {"max": [0.130091]},
                "residual": {
                    "rms": [0.0005699],
                    "minimum": [-0.0022741],
                    "maximum": [0.0022422],
                    "min": [-0.0022741],
                    "max": [0.0022422],
                },
            },
        }
        checks = AUDIT.audit_imaging(summary, reference)
        self.assertTrue(all(check["passed"] for check in checks))

    def test_sensitivity_reference_preserves_failure_boundary(self):
        sensitivity = AUDIT.load_reference()["imaging_sensitivity"]
        conclusion = sensitivity["conclusion"]
        self.assertEqual(conclusion["regression_baseline"], "fixed_broad")
        self.assertIsNone(conclusion["scientifically_preferred_variant"])
        self.assertFalse(conclusion["publication_claim_supported"])
        self.assertEqual(
            sensitivity["fixed_broad_restricted_scales"]["divergence_warnings"],
            0,
        )
        self.assertGreater(
            sensitivity["fixed_broad_restricted_scales"][
                "residual_one_beam_correlation_xy"
            ][0],
            sensitivity["fixed_broad"]["residual_one_beam_correlation_xy"][0],
        )


if __name__ == "__main__":
    unittest.main()
