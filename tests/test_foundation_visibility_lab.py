import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "9_Practical" / "foundation_visibility_lab.py"
SPEC = importlib.util.spec_from_file_location("foundation_visibility_lab", MODULE_PATH)
LAB = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAB)


class FoundationVisibilityLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lab = LAB.make_lab()

    def test_data_contract_and_audit(self):
        summary = LAB.inspection_summary(self.lab)
        self.assertEqual(summary["data_shape_time_channel_baseline"], (81, 8, 15))
        self.assertEqual(summary["cross_correlations"], 15)
        self.assertEqual(summary["fully_flagged_channel_indices"].tolist(), [1])
        self.assertEqual(summary["highest_flag_antenna"], 4)
        self.assertEqual(summary["unflagged_nonfinite_values"], 0)
        self.assertEqual(summary["unflagged_nonpositive_weights"], 0)

    def test_calibration_reduces_residual_and_propagates_weights(self):
        result = LAB.calibrate(self.lab)
        self.assertLess(
            result["calibrator_rms_after"], 0.08 * result["calibrator_rms_before"]
        )
        self.assertLess(result["holdout_rms"], 0.03)
        self.assertGreater(result["minimum_antenna_solution_snr"], 100)
        self.assertNotEqual(result["weight_scale_range"], (1.0, 1.0))
        self.assertTrue(np.all(result["corrected_weight"][self.lab["flag"]] == 0))

    def test_imaging_bundle_has_consistent_products(self):
        bundle = LAB.make_imaging_bundle(self.lab)
        shape = bundle["dirty"].shape
        for key in ("psf", "model", "residual", "restored", "primary_beam"):
            self.assertEqual(bundle[key].shape, shape)
            self.assertTrue(np.all(np.isfinite(bundle[key])))
        center = tuple(np.array(shape) // 2)
        self.assertAlmostEqual(bundle["psf"][center], 1.0)
        self.assertGreater(bundle["beam_area_pixels"], 1.0)
        self.assertGreater(bundle["residual_scale"], 0)
        self.assertNotAlmostEqual(bundle["residual_scale"], 1.0)
        self.assertAlmostEqual(
            bundle["sensitivity"][0, 0], 1 / np.sqrt(2 * bundle["weight_sum"])
        )
        with self.assertRaisesRegex(ValueError, "data and weight"):
            LAB.make_imaging_bundle(self.lab, data=self.lab["target_data"])

    def test_phase_selfcal_improves_connected_holdout(self):
        result = LAB.self_calibrate(self.lab)
        self.assertTrue(result["accepted"])
        self.assertLess(result["holdout_rms_after"], result["holdout_rms_before"])
        self.assertGreater(result["minimum_antenna_solution_snr"], 100)
        np.testing.assert_allclose(abs(result["phase_gains"]), 1.0)
        self.assertIn("controlled simulation", result["holdout_reference"])

    def test_measurement_and_averaging_boundaries(self):
        measurement = LAB.measurement_summary(self.lab)
        self.assertGreater(measurement["integrated_flux_jy"], 0.8)
        self.assertLess(measurement["integrated_flux_jy"], 1.2)
        self.assertGreater(measurement["random_uncertainty_jy"], 0)
        averaging = LAB.averaging_summary(self.lab)
        self.assertLess(averaging["direct_combined_retention"], 1.0)
        self.assertAlmostEqual(
            averaging["direct_combined_retention"],
            averaging["separable_product"],
            delta=0.01,
        )
        self.assertFalse(averaging["coherence_is_recoverable_after_averaging"])
        flagged_window = LAB.averaging_summary(self.lab, time_bin=5, channel_bin=8)
        self.assertLess(
            flagged_window["effective_unflagged_samples"],
            flagged_window["input_samples"],
        )
        self.assertGreater(flagged_window["output_inverse_variance"], 0)


if __name__ == "__main__":
    unittest.main()
